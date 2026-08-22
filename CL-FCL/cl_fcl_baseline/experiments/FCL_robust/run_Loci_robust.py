from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.utils.data import Dataset, Subset

from cl_fcl_baseline.algorithms.fcl import FCLExperiment, NaiveContinualStrategy
from cl_fcl_baseline.algorithms.FL_robust.loci_fat import AdversarialLociClient
from cl_fcl_baseline.algorithms.FL_robust.own import evaluate_task_aware_pgd
from cl_fcl_baseline.algorithms.FL_robust.PGD import PGDConfig
from cl_fcl_baseline.algorithms.loci import LociServer
from cl_fcl_baseline.contracts import TaskDefinition
from cl_fcl_baseline.datasets import (
    build_class_incremental_tasks,
    build_torchvision_dataset,
    dataset_info,
)
from cl_fcl_baseline.datasets.build import (
    RandomClassificationDataset,
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.models import build_model
from cl_fcl_baseline.experiments.FCL_robust.fcl_robust import (
    EXPERIMENTS_LOG_DIR,
    NORMALIZATION_STATS,
)
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import move_to_device, set_seed

try:
    from ..args import parse_loci_robust_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import parse_loci_robust_args


def _build_pgd_config(args: argparse.Namespace) -> PGDConfig:
    stats = NORMALIZATION_STATS.get(str(args.dataset).lower())
    if stats is None:
        return PGDConfig(
            epsilon=float(args.pgd_epsilon),
            step_size=float(args.pgd_step_size),
            steps=int(args.pgd_steps),
            random_start=bool(args.pgd_random_start),
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
    if bool(args.pgd_normalized_space):
        epsilon: float | list[float] = float(args.pgd_epsilon)
        step_size: float | list[float] = float(args.pgd_step_size)
    else:
        epsilon = [float(args.pgd_epsilon) / channel_std for channel_std in std]
        step_size = [float(args.pgd_step_size) / channel_std for channel_std in std]
    return PGDConfig(
        epsilon=epsilon,
        step_size=step_size,
        steps=int(args.pgd_steps),
        random_start=bool(args.pgd_random_start),
        clip_min=clip_min,
        clip_max=clip_max,
    )


def _build_model(
    args: argparse.Namespace,
    input_shape: tuple[int, int, int],
    num_classes: int,
    model_name: str | None = None,
) -> torch.nn.Module:
    requested_model = str(args.model if model_name is None else model_name)
    if requested_model.lower() in {"resnet", "resnet18"}:
        requested_model = "LociResNet18"
    return build_model(
        model_name=requested_model,
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        vit_patch_size=args.vit_patch_size,
        vit_dropout=args.vit_dropout,
        vit_attention_dropout=args.vit_attention_dropout,
        vit_mlp_ratio=args.vit_mlp_ratio,
    )


def _build_kd_model(
    args: argparse.Namespace,
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> torch.nn.Module:
    model_name = str(args.loci_kd_model)
    if model_name.lower() in {"resnet", "resnet18"}:
        model_name = "LociResNet18"
    return build_model(
        model_name=model_name,
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dim=args.loci_kd_hidden_dim,
    )


def _build_optimizer(
    args: argparse.Namespace,
    model: torch.nn.Module,
    lr: float | None = None,
) -> torch.optim.Optimizer:
    learning_rate = float(args.lr if lr is None else lr)
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(args.loci_weight_decay),
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=float(args.loci_weight_decay),
    )


def _build_task_stream(
    args: argparse.Namespace,
) -> tuple[list[TaskDefinition], dict[str, Dataset], dict[str, Dataset], tuple[int, int, int], int]:
    if args.dataset == "random_classification":
        input_shape = tuple(args.input_shape)
        task_num_classes = int(args.num_classes)
        num_tasks = int(args.num_tasks) if int(args.num_tasks) > 0 else 2
        tasks: list[TaskDefinition] = []
        train_datasets: dict[str, Dataset] = {}
        test_datasets: dict[str, Dataset] = {}
        for task_idx in range(num_tasks):
            task_id = f"task_{task_idx}"
            tasks.append(
                TaskDefinition(task_id=task_id, name=task_id, num_classes=task_num_classes)
            )
            train_datasets[task_id] = RandomClassificationDataset(
                num_samples=args.num_samples if args.num_samples > 0 else 256,
                input_shape=input_shape,
                num_classes=task_num_classes,
                seed=args.seed + task_idx,
            )
            test_datasets[task_id] = RandomClassificationDataset(
                num_samples=args.num_samples if args.num_samples > 0 else 256,
                input_shape=input_shape,
                num_classes=task_num_classes,
                seed=args.seed + 10_000 + task_idx,
            )
        return tasks, train_datasets, test_datasets, input_shape, task_num_classes

    input_shape, total_num_classes = dataset_info(args.dataset)
    if args.dataset not in {"mnist", "cifar10", "cifar100"}:
        input_shape = (input_shape[0], int(args.loci_image_size), int(args.loci_image_size))
    train_dataset = build_torchvision_dataset(
        name=args.dataset,
        train=True,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        image_size=args.loci_image_size,
        download=args.download,
    )
    test_dataset = build_torchvision_dataset(
        name=args.dataset,
        train=False,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed + 1,
        image_size=args.loci_image_size,
        download=args.download,
    )

    requested_tasks = int(args.num_tasks)
    classes_per_task = int(args.classes_per_task)
    if classes_per_task <= 0:
        default_task_counts = {
            "cifar100": 10,
            "miniimagenet": 10,
            "mini-imagenet": 10,
            "tinyimagenet": 20,
            "tiny-imagenet-200": 20,
            "fc100": 10,
            "core50": 11,
            "imagenet": 100,
        }
        num_tasks = requested_tasks or default_task_counts.get(args.dataset, 1)
        if total_num_classes % num_tasks != 0:
            raise ValueError(
                f"The {args.dataset} class count ({total_num_classes}) is not divisible "
                f"by --task/--num-tasks={num_tasks}; set --classes-per-task explicitly."
            )
        classes_per_task = total_num_classes // num_tasks
    else:
        default_num_tasks = total_num_classes // classes_per_task
        num_tasks = requested_tasks or default_num_tasks
        if num_tasks * classes_per_task > total_num_classes:
            raise ValueError("num_tasks * classes_per_task cannot exceed the dataset class count.")
    train_task_datasets = build_class_incremental_tasks(
        train_dataset,
        classes_per_task=classes_per_task,
        num_tasks=num_tasks,
        seed=args.seed,
        shuffle_classes=args.task_order_shuffle,
        remap_labels=False,
    )
    test_task_datasets = build_class_incremental_tasks(
        test_dataset,
        classes_per_task=classes_per_task,
        num_tasks=num_tasks,
        seed=args.seed,
        shuffle_classes=args.task_order_shuffle,
        remap_labels=False,
    )

    tasks: list[TaskDefinition] = []
    train_datasets: dict[str, Dataset] = {}
    test_datasets: dict[str, Dataset] = {}
    for task_idx, (train_split, test_split) in enumerate(
        zip(train_task_datasets, test_task_datasets)
    ):
        task_id = f"task_{task_idx}"
        tasks.append(
            TaskDefinition(
                task_id=task_id,
                name=task_id,
                num_classes=total_num_classes,
                metadata={"classes": list(train_split.class_ids)},
            )
        )
        train_datasets[task_id] = train_split
        test_datasets[task_id] = test_split
    return tasks, train_datasets, test_datasets, input_shape, total_num_classes


def main() -> None:
    args = parse_loci_robust_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"CUDA is unavailable; falling back from {args.device} to CPU.")
            device = torch.device("cpu")

    tasks, train_datasets, test_datasets, input_shape, task_num_classes = _build_task_stream(args)
    pgd_config = _build_pgd_config(args)

    client_task_loaders = [dict() for _ in range(args.num_clients)]
    for task in tasks:
        dataset = train_datasets[task.task_id]
        if args.partition == "iid":
            partitions = partition_dataset_iid(
                dataset, num_clients=args.num_clients, seed=args.seed
            )
        elif args.noniid_method == "dirichlet":
            partitions = partition_dataset_dirichlet(
                dataset,
                num_clients=args.num_clients,
                beta=args.dirichlet_beta,
                num_classes=task.num_classes,
                seed=args.seed,
            )
        else:
            partitions = partition_dataset_noniid(
                dataset,
                num_clients=args.num_clients,
                num_shards=args.noniid_shards,
                seed=args.seed,
            )
        for client_idx, partition in enumerate(partitions):
            client_task_loaders[client_idx][task.task_id] = build_dataloader(
                partition,
                batch_size=args.batch_size,
                shuffle=True,
            )

    test_loaders = {
        task_id: build_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
        for task_id, dataset in test_datasets.items()
    }
    task_classes = {
        task.task_id: list(task.metadata.get("classes", range(task.num_classes))) for task in tasks
    }

    client_model_names = list(args.loci_client_models) or [args.model]
    client_learning_rates = list(args.loci_client_lrs) or [args.lr]
    client_local_epochs = list(args.loci_client_local_epochs) or [args.local_epochs]
    clients = []
    for idx in range(args.num_clients):
        model_name = client_model_names[idx % len(client_model_names)]
        client_lr = float(client_learning_rates[idx % len(client_learning_rates)])
        local_epochs = int(client_local_epochs[idx % len(client_local_epochs)])
        model = _build_model(
            args,
            input_shape=input_shape,
            num_classes=task_num_classes,
            model_name=model_name,
        )
        optimizer = _build_optimizer(args, model, lr=client_lr)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device)
        clients.append(
            AdversarialLociClient(
                client_id=f"client_{idx}",
                trainer=trainer,
                kd_model=_build_kd_model(
                    args,
                    input_shape=input_shape,
                    num_classes=task_num_classes,
                ),
                task_loaders=client_task_loaders[idx],
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
                optimizer_name=args.optimizer,
                lr=client_lr,
                weight_decay=args.loci_weight_decay,
                task_classes=task_classes,
                pgd_config=pgd_config,
                adversarial_ratio=args.loci_adversarial_ratio,
                warmup_rounds=args.loci_warmup_rounds,
                warmup_adversarial_ratio=args.loci_warmup_adversarial_ratio,
            )
        )

    public_loaders = {}
    for task_index, (task_id, dataset) in enumerate(train_datasets.items()):
        public_dataset: Dataset = dataset
        if 0 < int(args.loci_public_samples) < len(dataset):
            generator = torch.Generator().manual_seed(args.seed + 20_000 + task_index)
            indices = torch.randperm(len(dataset), generator=generator)[: args.loci_public_samples]
            public_dataset = Subset(dataset, indices.tolist())
        public_loaders[task_id] = build_dataloader(
            public_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

    server = LociServer(
        kd_model=_build_kd_model(
            args,
            input_shape=input_shape,
            num_classes=task_num_classes,
        ).to(device),
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
    )

    eval_trainers = {}
    for client in clients:
        eval_model = copy.deepcopy(client.trainer.model)
        eval_trainers[client.client_id] = BaseTrainer(
            model=eval_model,
            optimizer=_build_optimizer(args, eval_model, lr=client.lr),
            device=device,
        )

    log_path = args.log_file.strip()
    if not log_path:
        log_dir = EXPERIMENTS_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"loci_at_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    pgd_max_batches = None if int(args.pgd_max_batches) <= 0 else int(args.pgd_max_batches)

    def _evaluate_task(task_id: str, eval_trainer: BaseTrainer) -> dict[str, float]:
        task_loader = test_loaders[task_id]
        class_ids = task_classes.get(task_id, [])
        class_tensor = None
        if class_ids and len(class_ids) < task_num_classes:
            class_tensor = torch.tensor(
                [int(class_id) for class_id in class_ids], device=device, dtype=torch.long
            )
        eval_trainer.model.to(device)
        eval_trainer.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for inputs, targets in task_loader:
                inputs = move_to_device(inputs, device)
                targets = move_to_device(targets, device)
                logits = eval_trainer.model(inputs)
                if class_tensor is None:
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    predictions = logits.argmax(dim=1)
                else:
                    task_logits = logits.index_select(dim=1, index=class_tensor)
                    local_targets = torch.zeros_like(targets)
                    for local_idx, class_id in enumerate(class_tensor.tolist()):
                        local_targets = torch.where(
                            targets == int(class_id),
                            torch.full_like(targets, local_idx),
                            local_targets,
                        )
                    loss = torch.nn.functional.cross_entropy(task_logits, local_targets)
                    predictions = class_tensor[task_logits.argmax(dim=1)]
                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int((predictions == targets).sum().item())
        if total_examples == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }

    def _eval_round(task_id: str, round_idx: int) -> None:
        evaluation_groups = experiment.evaluation_task_groups
        task_metrics: dict[str, dict[str, float]] = {}
        avg_accuracy = 0.0
        avg_loss = 0.0
        avg_robust_accuracy = 0.0
        avg_robust_loss = 0.0
        for eval_task_id, client_task_ids in evaluation_groups:
            evaluated_clients = 0
            client_eval_samples = 0
            client_accuracy = 0.0
            client_loss = 0.0
            client_robust_accuracy = 0.0
            client_robust_loss = 0.0
            client_robust_batches = 0.0
            client_robust_samples = 0.0
            for client in server.clients:
                client_task_id = client_task_ids.get(client.client_id)
                if client_task_id is None:
                    continue
                if client.current_state is None:
                    continue
                task_loader = test_loaders[client_task_id]
                eval_state = server.build_eval_state(client_task_id, client_id=client.client_id)
                eval_trainer = eval_trainers[client.client_id]
                eval_trainer.model.load_state_dict(eval_state, strict=True)
                local_metrics = _evaluate_task(client_task_id, eval_trainer)
                robust_metrics = evaluate_task_aware_pgd(
                    eval_trainer.model,
                    task_loader,
                    pgd_config,
                    device=device,
                    class_ids=task_classes.get(client_task_id),
                    max_batches=pgd_max_batches,
                )
                evaluated_clients += 1
                client_eval_samples += len(task_loader.dataset)
                client_accuracy += float(local_metrics.get("accuracy", 0.0))
                client_loss += float(local_metrics.get("loss", 0.0))
                client_robust_accuracy += float(robust_metrics.get("accuracy", 0.0))
                client_robust_loss += float(robust_metrics.get("loss", 0.0))
                client_robust_batches += float(robust_metrics.get("num_batches", 0.0))
                client_robust_samples += float(robust_metrics.get("num_samples", 0.0))
            metrics = {
                "accuracy": client_accuracy / max(1, evaluated_clients),
                "loss": client_loss / max(1, evaluated_clients),
                "robust_accuracy": client_robust_accuracy / max(1, evaluated_clients),
                "robust_loss": client_robust_loss / max(1, evaluated_clients),
                "num_eval_clients": float(evaluated_clients),
                "num_eval_samples": float(client_eval_samples / max(1, evaluated_clients)),
                "num_pgd_batches": client_robust_batches / max(1, evaluated_clients),
                "num_pgd_samples": client_robust_samples / max(1, evaluated_clients),
            }
            task_metrics[eval_task_id] = metrics
            avg_accuracy += float(metrics.get("accuracy", 0.0))
            avg_loss += float(metrics.get("loss", 0.0))
            avg_robust_accuracy += float(metrics.get("robust_accuracy", 0.0))
            avg_robust_loss += float(metrics.get("robust_loss", 0.0))
        avg_accuracy /= max(1, len(evaluation_groups))
        avg_loss /= max(1, len(evaluation_groups))
        avg_robust_accuracy /= max(1, len(evaluation_groups))
        avg_robust_loss /= max(1, len(evaluation_groups))
        per_task_accuracy = " ".join(
            f"{seen_task_id}={metrics.get('accuracy', 0.0):.4f}/robust={metrics.get('robust_accuracy', 0.0):.4f}"
            for seen_task_id, metrics in task_metrics.items()
        )
        print(
            f"[eval] task={task_id} round={round_idx}: "
            f"avg_acc={avg_accuracy:.4f} avg_robust_acc={avg_robust_accuracy:.4f} "
            f"{per_task_accuracy}"
        )
        record = {
            "type": "eval",
            "algorithm": "LOCI-AT",
            "task_id": task_id,
            "round": round_idx,
            "avg_metrics": {
                "accuracy": avg_accuracy,
                "loss": avg_loss,
                "robust_accuracy": avg_robust_accuracy,
                "robust_loss": avg_robust_loss,
            },
            "task_metrics": task_metrics,
            "eval_mode": "task_aware",
            "task_classes": {
                seen_task_id: list(task_classes.get(seen_task_id, []))
                for seen_task_id in experiment.seen_task_ids
            },
            "heterogeneous_eval_mode": experiment.effective_eval_mode,
            "evaluation_task_groups": {
                eval_task_id: client_task_ids for eval_task_id, client_task_ids in evaluation_groups
            },
            "pgd": {
                "epsilon": args.pgd_epsilon,
                "step_size": args.pgd_step_size,
                "steps": args.pgd_steps,
                "random_start": args.pgd_random_start,
                "normalized_space": args.pgd_normalized_space,
                "max_batches": args.pgd_max_batches,
            },
            "adversarial_training": {
                "adversarial_ratio": args.loci_adversarial_ratio,
                "warmup_rounds": args.loci_warmup_rounds,
                "warmup_adversarial_ratio": args.loci_warmup_adversarial_ratio,
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

    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "config",
                    "algorithm": "LOCI-AT",
                    "args": vars(args),
                    "resolved_device": str(device),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    experiment.run()
    print("LOCI-AT finished.")


if __name__ == "__main__":
    main()
