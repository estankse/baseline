from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from cl_fcl_baseline.algorithms.fcl import FCLExperiment, NaiveContinualStrategy
from cl_fcl_baseline.algorithms.FL_robust.own import (
    RobustLociClient,
    RobustLociServer,
    evaluate_task_aware_pgd,
)
from cl_fcl_baseline.algorithms.FL_robust.PGD import PGDConfig
from cl_fcl_baseline.contracts import TaskDefinition
from cl_fcl_baseline.datasets.build import (
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.experiments.FCL_robust.fcl_robust import (
    EXPERIMENTS_LOG_DIR,
    NORMALIZATION_STATS,
)
from cl_fcl_baseline.experiments.FCL.run_Loci import (
    _build_kd_model,
    _build_model,
    _build_optimizer,
    _build_task_stream,
)
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import move_to_device, set_seed

try:
    from ..args import parse_own_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import parse_own_args


def _attack_config(
    args: argparse.Namespace,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    random_start: bool,
    normalized_space: bool,
) -> PGDConfig:
    """Translate raw-pixel radii to the normalized model input space."""
    stats = NORMALIZATION_STATS.get(str(args.dataset).lower())
    if stats is None:
        return PGDConfig(
            epsilon=float(epsilon),
            step_size=float(step_size),
            steps=int(steps),
            random_start=bool(random_start),
        )
    mean, std = stats
    clip_min = [
        (0.0 - channel_mean) / channel_std
        for channel_mean, channel_std in zip(mean, std)
    ]
    clip_max = [
        (1.0 - channel_mean) / channel_std
        for channel_mean, channel_std in zip(mean, std)
    ]
    if normalized_space:
        attack_epsilon: float | list[float] = float(epsilon)
        attack_step: float | list[float] = float(step_size)
    else:
        attack_epsilon = [float(epsilon) / channel_std for channel_std in std]
        attack_step = [float(step_size) / channel_std for channel_std in std]
    return PGDConfig(
        epsilon=attack_epsilon,
        step_size=attack_step,
        steps=int(steps),
        random_start=bool(random_start),
        clip_min=clip_min,
        clip_max=clip_max,
    )


def _clean_task_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    class_ids: Sequence[int] | None,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    ids = (
        torch.tensor(class_ids, device=device, dtype=torch.long)
        if class_ids
        else None
    )
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = move_to_device(inputs, device)
            targets = move_to_device(targets, device)
            logits = model(inputs)
            uses_full_head = ids is None or (
                int(ids.numel()) == int(logits.shape[1])
                and torch.equal(ids, torch.arange(logits.shape[1], device=device))
            )
            if uses_full_head:
                task_logits = logits
                local_targets = targets
            else:
                task_logits = logits.index_select(1, ids)
                local_targets = torch.zeros_like(targets)
                for local_index, class_id in enumerate(ids.tolist()):
                    local_targets = torch.where(
                        targets == int(class_id),
                        torch.full_like(targets, local_index),
                        local_targets,
                    )
            loss = F.cross_entropy(task_logits, local_targets)
            batch_size = int(targets.shape[0])
            total_examples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_correct += int(
                (task_logits.argmax(dim=1) == local_targets).sum().item()
            )
    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
    }


def main() -> None:
    args = parse_own_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"CUDA is unavailable; falling back from {args.device} to CPU.")
            device = torch.device("cpu")

    tasks, train_datasets, test_datasets, input_shape, num_classes = _build_task_stream(args)
    train_attack = _attack_config(
        args,
        epsilon=args.own_epsilon,
        step_size=args.own_step_size,
        steps=args.own_steps,
        random_start=args.own_random_start,
        normalized_space=args.own_normalized_space,
    )
    selector_attack = _attack_config(
        args,
        epsilon=args.own_epsilon,
        step_size=args.own_step_size,
        steps=args.own_selector_steps,
        random_start=args.own_random_start,
        normalized_space=args.own_normalized_space,
    )
    eval_attack = _attack_config(
        args,
        epsilon=args.pgd_epsilon,
        step_size=args.pgd_step_size,
        steps=args.pgd_steps,
        random_start=args.pgd_random_start,
        normalized_space=args.pgd_normalized_space,
    )

    client_task_loaders: list[dict[str, torch.utils.data.DataLoader]] = [
        {} for _ in range(args.num_clients)
    ]
    for task_index, task in enumerate(tasks):
        dataset = train_datasets[task.task_id]
        partition_seed = int(args.seed) + task_index
        if args.partition == "iid":
            partitions = partition_dataset_iid(
                dataset, num_clients=args.num_clients, seed=partition_seed
            )
        elif args.noniid_method == "dirichlet":
            partitions = partition_dataset_dirichlet(
                dataset,
                num_clients=args.num_clients,
                beta=args.dirichlet_beta,
                num_classes=task.num_classes,
                seed=partition_seed,
            )
        else:
            partitions = partition_dataset_noniid(
                dataset,
                num_clients=args.num_clients,
                num_shards=args.noniid_shards,
                seed=partition_seed,
            )
        for client_index, partition in enumerate(partitions):
            client_task_loaders[client_index][task.task_id] = build_dataloader(
                partition, batch_size=args.batch_size, shuffle=True
            )

    test_loaders = {
        task_id: build_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
        for task_id, dataset in test_datasets.items()
    }
    task_classes = {
        task.task_id: list(task.metadata.get("classes", range(task.num_classes)))
        for task in tasks
    }

    # LOCI supports heterogeneous private models.  All clients nevertheless
    # exchange the same compact KD architecture, which RobustLoci explicitly
    # trains to carry both clean semantics and adversarial boundary behavior.
    client_model_names = list(args.loci_client_models) or [args.model]
    client_learning_rates = list(args.loci_client_lrs) or [args.lr]
    client_local_epochs = list(args.loci_client_local_epochs) or [args.local_epochs]
    clients: list[RobustLociClient] = []
    for client_index in range(args.num_clients):
        model_name = client_model_names[client_index % len(client_model_names)]
        client_lr = float(
            client_learning_rates[client_index % len(client_learning_rates)]
        )
        local_epochs = int(
            client_local_epochs[client_index % len(client_local_epochs)]
        )
        model = _build_model(
            args,
            input_shape=input_shape,
            num_classes=num_classes,
            model_name=model_name,
        )
        trainer = BaseTrainer(
            model=model,
            optimizer=_build_optimizer(args, model, lr=client_lr),
            device=device,
        )
        clients.append(
            RobustLociClient(
                client_id=f"client_{client_index}",
                trainer=trainer,
                kd_model=_build_kd_model(args, input_shape, num_classes),
                task_loaders=client_task_loaders[client_index],
                epochs=local_epochs,
                kd_epochs=args.loci_kd_epochs,
                kd_lr=args.loci_kd_lr,
                temperature=args.loci_temperature,
                kd_alpha=args.loci_kd_alpha,
                integrator_weight=args.loci_integrator_weight,
                continual_method=args.loci_continual_method,
                ewc_lambda=args.loci_ewc_lambda,
                fisher_batches=args.loci_fisher_batches,
                gem_memory_size=args.loci_gem_memory_size,
                gem_memory_strength=args.loci_gem_memory_strength,
                gem_qp_eps=args.loci_gem_qp_eps,
                knowledge_ratio=args.loci_knowledge_ratio,
                knowledge_finetune_epochs=args.loci_knowledge_finetune_epochs,
                task_classes=task_classes,
                optimizer_name=args.optimizer,
                lr=client_lr,
                weight_decay=args.loci_weight_decay,
                pgd_config=train_attack,
                variant=args.own_variant,
                clean_weight=args.own_clean_weight,
                adversarial_weight=args.own_adversarial_weight,
                trades_weight=args.own_trades_weight,
                robust_kd_weight=args.own_robust_kd_weight,
                boundary_weight=args.own_boundary_weight,
                robust_gradient_ratio=args.own_robust_gradient_ratio,
                teacher_clean_weight=args.own_teacher_clean_weight,
                teacher_eval_batches=args.own_teacher_eval_batches,
                teacher_weight_floor=args.own_teacher_weight_floor,
                robust_warmup_rounds=args.own_warmup_rounds,
                fisher_adversarial_weight=args.own_fisher_adversarial_weight,
                importance_weight=args.own_importance_weight,
                knowledge_robust_weight=args.own_knowledge_robust_weight,
                class_balance_power=args.own_class_balance_power,
                class_balance_smoothing=args.own_class_balance_smoothing,
                class_weight_max=args.own_class_weight_max,
                replay_budget=args.own_replay_budget,
                replay_batch_size=args.own_replay_batch_size,
                replay_selection_batches=args.own_replay_selection_batches,
                replay_weight=args.own_replay_weight,
                robust_memory_batch_size=args.own_robust_memory_batch_size,
                public_refine_epochs=args.own_public_refine_epochs,
                public_refine_lr_scale=args.own_public_refine_lr_scale,
            )
        )

    public_loaders: dict[str, torch.utils.data.DataLoader] = {}
    for task_index, (task_id, dataset) in enumerate(train_datasets.items()):
        public_dataset: Dataset = dataset
        if 0 < int(args.loci_public_samples) < len(dataset):
            generator = torch.Generator().manual_seed(args.seed + 20_000 + task_index)
            indices = torch.randperm(len(dataset), generator=generator)[
                : args.loci_public_samples
            ]
            public_dataset = Subset(dataset, indices.tolist())
        public_loaders[task_id] = build_dataloader(
            public_dataset, batch_size=args.batch_size, shuffle=False
        )

    server = RobustLociServer(
        kd_model=_build_kd_model(args, input_shape, num_classes).to(device),
        clients=clients,
        public_loaders=public_loaders,
        task_classes=task_classes,
        client_sample_ratio=args.client_sample_ratio,
        similar_tasks=args.loci_similar_tasks,
        similarity=args.loci_similarity,
        selector_candidates=args.loci_selector_candidates,
        selector_batches=args.loci_selector_batches,
        temperature=args.loci_temperature,
        ot_regularization=args.loci_ot_regularization,
        ot_iterations=args.loci_ot_iterations,
        selector_pgd_config=selector_attack,
        robust_similarity=args.own_robust_similarity,
        fusion_clean_tolerance=args.own_fusion_clean_tolerance,
        fusion_clean_loss_tolerance=args.own_fusion_clean_loss_tolerance,
        fusion_min_robust_gain=args.own_fusion_min_robust_gain,
    )

    eval_trainers: dict[str, BaseTrainer] = {}
    for client in clients:
        eval_model = copy.deepcopy(client.trainer.model)
        eval_trainers[client.client_id] = BaseTrainer(
            model=eval_model,
            optimizer=_build_optimizer(args, eval_model, lr=client.lr),
            device=device,
        )

    log_path = args.log_file.strip()
    if not log_path:
        EXPERIMENTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(EXPERIMENTS_LOG_DIR / f"own_{args.own_variant}_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    pgd_max_batches = (
        None if int(args.pgd_max_batches) <= 0 else int(args.pgd_max_batches)
    )
    with open(log_path, "a", encoding="utf-8") as handle:
        algorithm_name = "RAMP-LOCI" if args.own_variant == "ramp" else "RobustLoci"
        handle.write(
            json.dumps(
                {
                    "type": "config",
                    "algorithm": algorithm_name,
                    "args": vars(args),
                    "device": str(device),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def _eval_round(task_id: str, round_index: int) -> None:
        groups = experiment.evaluation_task_groups
        task_metrics: dict[str, dict[str, float]] = {}
        for eval_task_id, client_task_ids in groups:
            sums = {
                "accuracy": 0.0,
                "loss": 0.0,
                "robust_accuracy": 0.0,
                "robust_loss": 0.0,
            }
            evaluated_clients = 0
            eval_samples = 0
            pgd_batches = 0.0
            pgd_samples = 0.0
            for client in server.clients:
                client_task_id = client_task_ids.get(client.client_id)
                if client_task_id is None or client.current_state is None:
                    continue
                loader = test_loaders[client_task_id]
                eval_trainer = eval_trainers[client.client_id]
                eval_trainer.model.load_state_dict(
                    server.build_eval_state(client_task_id, client_id=client.client_id),
                    strict=True,
                )
                classes = task_classes.get(client_task_id)
                clean = _clean_task_metrics(
                    eval_trainer.model,
                    loader,
                    device=device,
                    class_ids=classes,
                )
                robust = evaluate_task_aware_pgd(
                    eval_trainer.model,
                    loader,
                    eval_attack,
                    device=device,
                    class_ids=classes,
                    max_batches=pgd_max_batches,
                )
                evaluated_clients += 1
                eval_samples += len(loader.dataset)
                sums["accuracy"] += float(clean["accuracy"])
                sums["loss"] += float(clean["loss"])
                sums["robust_accuracy"] += float(robust["accuracy"])
                sums["robust_loss"] += float(robust["loss"])
                pgd_batches += float(robust["num_batches"])
                pgd_samples += float(robust["num_samples"])
            denominator = max(1, evaluated_clients)
            task_metrics[eval_task_id] = {
                name: value / denominator for name, value in sums.items()
            }
            task_metrics[eval_task_id].update(
                {
                    "num_eval_clients": float(evaluated_clients),
                    "num_eval_samples": float(eval_samples / denominator),
                    "num_pgd_batches": pgd_batches / denominator,
                    "num_pgd_samples": pgd_samples / denominator,
                }
            )

        metric_names = ("accuracy", "loss", "robust_accuracy", "robust_loss")
        averages = {
            name: sum(metrics[name] for metrics in task_metrics.values())
            / max(1, len(task_metrics))
            for name in metric_names
        }
        summary = " ".join(
            f"{name}={metrics['accuracy']:.4f}/robust={metrics['robust_accuracy']:.4f}"
            for name, metrics in task_metrics.items()
        )
        print(
            f"[eval] task={task_id} round={round_index}: "
            f"avg_acc={averages['accuracy']:.4f} "
            f"avg_robust_acc={averages['robust_accuracy']:.4f} {summary}"
        )
        record = {
            "type": "eval",
            "algorithm": algorithm_name,
            "variant": args.own_variant,
            "task_id": task_id,
            "round": round_index,
            "avg_metrics": averages,
            "task_metrics": task_metrics,
            "heterogeneous_eval_mode": experiment.effective_eval_mode,
            "evaluation_task_groups": dict(groups),
            "training_attack": {
                "epsilon": args.own_epsilon,
                "step_size": args.own_step_size,
                "steps": args.own_steps,
                "normalized_space": args.own_normalized_space,
            },
            "evaluation_attack": {
                "epsilon": args.pgd_epsilon,
                "step_size": args.pgd_step_size,
                "steps": args.pgd_steps,
                "normalized_space": args.pgd_normalized_space,
            },
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    experiment = FCLExperiment(
        server=server,
        strategy=NaiveContinualStrategy(),
        tasks=tasks,
        rounds_per_task=args.rounds_per_task,
        heterogeneous_task_order=args.heterogeneous_task_order,
        heterogeneous_eval_mode=args.heterogeneous_eval_mode,
        seed=args.seed,
        log_each_round=True,
        eval_every=args.eval_every,
        eval_fn=_eval_round if args.eval_every and args.eval_every > 0 else None,
        log_path=log_path,
    )
    experiment.run()
    print(f"{algorithm_name} ({args.own_variant}) finished.")


if __name__ == "__main__":
    main()
