from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset

from cl_fcl_baseline.algorithms.PGD import PGDConfig, evaluate_pgd_robustness
from cl_fcl_baseline.algorithms.fcl import FCLExperiment, NaiveContinualStrategy
from cl_fcl_baseline.algorithms.fedweit import FedWeITClient, FedWeITServer
from cl_fcl_baseline.contracts import TaskDefinition
from cl_fcl_baseline.datasets import build_class_incremental_tasks, build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    RandomClassificationDataset,
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.models import ResNet18, ResNet20, ResNet32, VGG11
from cl_fcl_baseline.models.simple_model import MLPClassifier, SimpleCNN
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import set_seed

try:
    from .args import build_fedweit_parser, _parse_fedweit_pgd_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import build_fedweit_parser, _parse_fedweit_pgd_args


_NORMALIZATION = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}




def _build_pgd_config(args: argparse.Namespace) -> PGDConfig:
    key = str(args.dataset).lower()
    if key not in _NORMALIZATION:
        return PGDConfig(
            epsilon=float(args.pgd_epsilon),
            step_size=float(args.pgd_step_size),
            steps=int(args.pgd_steps),
            random_start=bool(args.pgd_random_start),
        )

    mean, std = _NORMALIZATION[key]
    clip_min = [(0.0 - channel_mean) / channel_std for channel_mean, channel_std in zip(mean, std)]
    clip_max = [(1.0 - channel_mean) / channel_std for channel_mean, channel_std in zip(mean, std)]
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


def _build_model(args: argparse.Namespace, input_shape: tuple[int, int, int], num_classes: int) -> torch.nn.Module:
    input_channels = int(input_shape[0])
    if args.model == "mlp":
        return MLPClassifier(input_shape=input_shape, hidden_dim=args.hidden_dim, num_classes=num_classes)
    if args.model == "simplecnn":
        return SimpleCNN(input_shape=input_shape, num_classes=num_classes)
    if args.model == "VGG11":
        return VGG11(input_channels=input_channels, num_classes=num_classes)
    if args.model == "ResNet18":
        return ResNet18(input_channels=input_channels, num_classes=num_classes)
    if args.model == "ResNet20":
        return ResNet20(input_channels=input_channels, num_classes=num_classes)
    if args.model == "ResNet32":
        return ResNet32(input_channels=input_channels, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {args.model}")


def _build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


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
            tasks.append(TaskDefinition(task_id=task_id, name=task_id, num_classes=task_num_classes))
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
    train_dataset = build_torchvision_dataset(
        name=args.dataset,
        train=True,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    test_dataset = build_torchvision_dataset(
        name=args.dataset,
        train=False,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed + 1,
    )

    classes_per_task = int(args.classes_per_task)
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive.")

    default_num_tasks = total_num_classes // classes_per_task
    num_tasks = int(args.num_tasks) if int(args.num_tasks) > 0 else default_num_tasks
    train_task_datasets = build_class_incremental_tasks(
        train_dataset,
        classes_per_task=classes_per_task,
        num_tasks=num_tasks,
        seed=args.seed,
        shuffle_classes=args.task_order_shuffle,
        remap_labels=True,
    )
    test_task_datasets = build_class_incremental_tasks(
        test_dataset,
        classes_per_task=classes_per_task,
        num_tasks=num_tasks,
        seed=args.seed,
        shuffle_classes=args.task_order_shuffle,
        remap_labels=True,
    )

    tasks: list[TaskDefinition] = []
    train_datasets: dict[str, Dataset] = {}
    test_datasets: dict[str, Dataset] = {}
    for task_idx, (train_split, test_split) in enumerate(zip(train_task_datasets, test_task_datasets)):
        task_id = f"task_{task_idx}"
        tasks.append(
            TaskDefinition(
                task_id=task_id,
                name=task_id,
                num_classes=classes_per_task,
                metadata={"classes": list(train_split.class_ids)},
            )
        )
        train_datasets[task_id] = train_split
        test_datasets[task_id] = test_split
    return tasks, train_datasets, test_datasets, input_shape, classes_per_task


def main() -> None:
    args = _parse_fedweit_pgd_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tasks, train_datasets, test_datasets, input_shape, task_num_classes = _build_task_stream(args)

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
        model = _build_model(args, input_shape=input_shape, num_classes=task_num_classes)
        optimizer = _build_optimizer(args, model)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device)
        clients.append(
            FedWeITClient(
                client_id=f"client_{idx}",
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
            )
        )

    server_model = _build_model(args, input_shape=input_shape, num_classes=task_num_classes)
    server = FedWeITServer(
        model=server_model,
        clients=clients,
        client_sample_ratio=args.client_sample_ratio,
        kb_sample_size=args.kb_sample_size,
    )

    eval_model = _build_model(args, input_shape=input_shape, num_classes=task_num_classes)
    eval_optimizer = _build_optimizer(args, eval_model)
    eval_trainer = BaseTrainer(model=eval_model, optimizer=eval_optimizer, device=device)

    log_path = args.log_file.strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"fedweit_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    task_order = [task.task_id for task in tasks]
    pgd_config = _build_pgd_config(args)
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
            for client_idx, client in enumerate(server.clients):
                if seen_task_id not in client.mask_logits:
                    continue
                eval_state = server.build_eval_state(seen_task_id, client_id=client.client_id)
                eval_trainer.model.load_state_dict(eval_state, strict=True)
                local_metrics = eval_trainer.evaluate(task_loader)
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
    print("FedWeIT finished.")


if __name__ == "__main__":
    main()
