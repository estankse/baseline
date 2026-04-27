from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import torch

from cl_fcl_baseline.algorithms.PGD import evaluate_pgd_robustness
from cl_fcl_baseline.algorithms.RBN import (
    FedWeITRBNClient,
    FedWeITRBNServer,
    enable_dual_batch_norm,
    set_dual_bn_mode,
)
from cl_fcl_baseline.algorithms.fcl import FCLExperiment, NaiveContinualStrategy
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import set_seed

try:
    from .args import _parse_fedweit_rbn_args
    from .run_FedWeIT_FAT import _build_model, _build_optimizer, _build_pgd_config, _build_task_stream
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import _parse_fedweit_rbn_args
    from cl_fcl_baseline.experiments.run_FedWeIT_FAT import (
        _build_model,
        _build_optimizer,
        _build_pgd_config,
        _build_task_stream,
    )

from cl_fcl_baseline.datasets.build import (
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)


def _build_rbn_model(args, input_shape: tuple[int, int, int], num_classes: int) -> torch.nn.Module:
    model = _build_model(args, input_shape=input_shape, num_classes=num_classes)
    num_replaced = enable_dual_batch_norm(model)
    if num_replaced <= 0:
        raise ValueError(
            f"FedWeIT-RBN requires a BatchNorm-based backbone, but model '{args.model}' exposes no BatchNorm layers."
        )
    return model


def _select_at_client_ids(num_clients: int, at_ratio: float, seed: int) -> set[str]:
    clamped_ratio = min(max(float(at_ratio), 0.0), 1.0)
    num_at_clients = max(1, int(round(num_clients * clamped_ratio))) if num_clients > 0 and clamped_ratio > 0.0 else 0
    indices = list(range(num_clients))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return {f"client_{idx}" for idx in sorted(indices[:num_at_clients])}


def main() -> None:
    args = _parse_fedweit_rbn_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tasks, train_datasets, test_datasets, input_shape, task_num_classes = _build_task_stream(args)
    pgd_config = _build_pgd_config(args)
    at_client_ids = _select_at_client_ids(args.num_clients, args.rbn_at_ratio, args.seed)

    client_task_loaders = [dict() for _ in range(args.num_clients)]
    for task in tasks:
        dataset = train_datasets[task.task_id]
        if args.partition == "iid":
            partitions = partition_dataset_iid(dataset, num_clients=args.num_clients, seed=args.seed)
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

    clients = []
    for idx in range(args.num_clients):
        client_id = f"client_{idx}"
        model = _build_rbn_model(args, input_shape=input_shape, num_classes=task_num_classes)
        optimizer = _build_optimizer(args, model)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device)
        clients.append(
            FedWeITRBNClient(
                client_id=client_id,
                trainer=trainer,
                task_loaders=client_task_loaders[idx],
                epochs=args.local_epochs,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                lambda_mask=args.lambda_mask,
                mask_init=args.mask_init,
                mask_threshold=args.mask_threshold,
                adaptive_threshold=None if args.adaptive_threshold < 0 else args.adaptive_threshold,
                client_sparsity=args.client_sparsity,
                optimizer_name=args.optimizer,
                lr=args.lr,
                weight_decay=5e-4 if args.optimizer == "sgd" else 0.0,
                pgd_config=pgd_config,
                is_at_client=client_id in at_client_ids,
                adv_lambda=args.rbn_adv_lambda,
                pnc_coef=args.rbn_pnc,
                pnc_warmup=args.rbn_pnc_warmup,
                src_weight_mode=args.rbn_src_weight_mode,
                attack_noised_bn=args.rbn_attack_noised_bn,
            )
        )

    server_model = _build_rbn_model(args, input_shape=input_shape, num_classes=task_num_classes)
    server = FedWeITRBNServer(
        model=server_model,
        clients=clients,
        client_sample_ratio=args.client_sample_ratio,
        kb_sample_size=args.kb_sample_size,
        src_weight_mode=args.rbn_src_weight_mode,
        propagate_before_training=args.rbn_pnc >= 0.0,
    )

    eval_model = _build_rbn_model(args, input_shape=input_shape, num_classes=task_num_classes)
    eval_optimizer = _build_optimizer(args, eval_model)
    eval_trainer = BaseTrainer(model=eval_model, optimizer=eval_optimizer, device=device)

    log_path = args.log_file.strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"fedweit_rbn_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    task_order = [task.task_id for task in tasks]
    pgd_max_batches = None if int(args.pgd_max_batches) <= 0 else int(args.pgd_max_batches)

    def _eval_round(task_id: str, round_idx: int) -> None:
        seen_task_ids = task_order[: task_order.index(task_id) + 1]
        task_metrics: dict[str, dict[str, float]] = {}
        avg_accuracy = 0.0
        avg_loss = 0.0
        avg_robust_accuracy = 0.0
        avg_robust_loss = 0.0
        for seen_task_id in seen_task_ids:
            task_loader = test_loaders[seen_task_id]
            evaluated_clients = 0
            client_accuracy = 0.0
            client_loss = 0.0
            client_robust_accuracy = 0.0
            client_robust_loss = 0.0
            client_robust_batches = 0.0
            client_robust_samples = 0.0
            for client in server.clients:
                if seen_task_id not in client.mask_logits:
                    continue
                eval_state = server.build_eval_state(seen_task_id, client_id=client.client_id)
                eval_trainer.model.load_state_dict(eval_state, strict=True)
                set_dual_bn_mode(eval_trainer.model, False)
                local_metrics = eval_trainer.evaluate(task_loader)
                set_dual_bn_mode(eval_trainer.model, True)
                robust_metrics = evaluate_pgd_robustness(
                    eval_trainer.model,
                    task_loader,
                    pgd_config,
                    device=eval_trainer.device,
                    max_batches=pgd_max_batches,
                )
                evaluated_clients += 1
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
                "num_eval_samples": float(len(task_loader.dataset)),
                "num_pgd_batches": client_robust_batches / max(1, evaluated_clients),
                "num_pgd_samples": client_robust_samples / max(1, evaluated_clients),
            }
            task_metrics[seen_task_id] = metrics
            avg_accuracy += float(metrics.get("accuracy", 0.0))
            avg_loss += float(metrics.get("loss", 0.0))
            avg_robust_accuracy += float(metrics.get("robust_accuracy", 0.0))
            avg_robust_loss += float(metrics.get("robust_loss", 0.0))
        avg_accuracy /= max(1, len(seen_task_ids))
        avg_loss /= max(1, len(seen_task_ids))
        avg_robust_accuracy /= max(1, len(seen_task_ids))
        avg_robust_loss /= max(1, len(seen_task_ids))
        per_task_accuracy = " ".join(
            f"{seen_task_id}={metrics.get('accuracy', 0.0):.4f}/robust={metrics.get('robust_accuracy', 0.0):.4f}"
            for seen_task_id, metrics in task_metrics.items()
        )
        print(
            f"[eval] task={task_id} round={round_idx}: "
            f"avg_acc={avg_accuracy:.4f} avg_robust_acc={avg_robust_accuracy:.4f} {per_task_accuracy}"
        )
        record = {
            "type": "eval",
            "task_id": task_id,
            "round": round_idx,
            "avg_metrics": {
                "accuracy": avg_accuracy,
                "loss": avg_loss,
                "robust_accuracy": avg_robust_accuracy,
                "robust_loss": avg_robust_loss,
            },
            "task_metrics": task_metrics,
            "pgd": {
                "epsilon": args.pgd_epsilon,
                "step_size": args.pgd_step_size,
                "steps": args.pgd_steps,
                "random_start": args.pgd_random_start,
                "normalized_space": args.pgd_normalized_space,
                "max_batches": args.pgd_max_batches,
            },
            "rbn": {
                "at_ratio": args.rbn_at_ratio,
                "at_clients": sorted(at_client_ids),
                "adv_lambda": args.rbn_adv_lambda,
                "src_weight_mode": args.rbn_src_weight_mode,
                "pnc": args.rbn_pnc,
                "pnc_warmup": args.rbn_pnc_warmup,
                "attack_noised_bn": args.rbn_attack_noised_bn,
            },
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    experiment = FCLExperiment(
        server=server,
        strategy=NaiveContinualStrategy(),
        tasks=tasks,
        rounds_per_task=args.rounds_per_task,
        log_each_round=True,
        eval_every=args.eval_every,
        eval_fn=_eval_round if args.eval_every and args.eval_every > 0 else None,
        log_path=log_path,
    )

    experiment.run()
    if tasks:
        _eval_round(tasks[-1].task_id, args.rounds_per_task)
    print("FedWeIT-RBN finished.")


if __name__ == "__main__":
    main()
