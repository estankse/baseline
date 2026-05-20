from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

from cl_fcl_baseline.contracts import TaskDefinition
from cl_fcl_baseline.datasets import build_class_incremental_tasks, build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    RandomClassificationDataset,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.models import ResNet18, ResNet20, ResNet32, VGG11
from cl_fcl_baseline.models.simple_model import MLPClassifier, SimpleCNN


EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_LOG_DIR = EXPERIMENTS_DIR / "logs"

NORMALIZATION_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def build_pgd_config(args: argparse.Namespace):
    from cl_fcl_baseline.algorithms.PGD import PGDConfig

    key = str(args.dataset).lower()
    if key not in NORMALIZATION_STATS:
        return PGDConfig(
            epsilon=float(args.pgd_epsilon),
            step_size=float(args.pgd_step_size),
            steps=int(args.pgd_steps),
            random_start=bool(args.pgd_random_start),
        )

    mean, std = NORMALIZATION_STATS[key]
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


def build_model(args: argparse.Namespace, input_shape: tuple[int, int, int], num_classes: int) -> torch.nn.Module:
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


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def build_task_stream(
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


__all__ = [
    "EXPERIMENTS_LOG_DIR",
    "NORMALIZATION_STATS",
    "build_model",
    "build_optimizer",
    "build_pgd_config",
    "build_task_stream",
    "partition_dataset_dirichlet",
    "partition_dataset_iid",
    "partition_dataset_noniid",
]
