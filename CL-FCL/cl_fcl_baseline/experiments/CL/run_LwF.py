from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset

from cl_fcl_baseline.algorithms.CL.lwf import LwFLearner
from cl_fcl_baseline.contracts import TaskDefinition
from cl_fcl_baseline.datasets import (
    RandomClassificationDataset,
    build_class_incremental_tasks,
    build_dataloader,
    build_torchvision_dataset,
    dataset_info,
)
from cl_fcl_baseline.models import build_model_from_args
from cl_fcl_baseline.trainers.utils import set_seed

try:
    from ..args import parse_lwf_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import parse_lwf_args


def _build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )


def _build_task_stream(
    args: argparse.Namespace,
) -> tuple[list[TaskDefinition], dict[str, Dataset], dict[str, Dataset], tuple[int, int, int], int]:
    if args.dataset == "random_classification":
        input_shape = tuple(int(value) for value in args.input_shape)
        total_num_classes = int(args.num_classes)
        train_dataset: Dataset = RandomClassificationDataset(
            num_samples=args.num_samples if args.num_samples > 0 else 512,
            input_shape=input_shape,
            num_classes=total_num_classes,
            seed=args.seed,
        )
        test_dataset: Dataset = RandomClassificationDataset(
            num_samples=args.num_samples if args.num_samples > 0 else 512,
            input_shape=input_shape,
            num_classes=total_num_classes,
            seed=args.seed + 10_000,
        )
    else:
        input_shape, total_num_classes = dataset_info(args.dataset)
        train_dataset = build_torchvision_dataset(
            args.dataset, True, args.data_dir, args.num_samples, args.seed
        )
        test_dataset = build_torchvision_dataset(
            args.dataset, False, args.data_dir, args.num_samples, args.seed + 1
        )
    class_order = list(range(total_num_classes))
    if args.task_order_shuffle:
        permutation = torch.randperm(
            total_num_classes, generator=torch.Generator().manual_seed(int(args.seed))
        ).tolist()
        class_order = [class_order[index] for index in permutation]
    num_tasks = int(args.num_tasks) if args.num_tasks > 0 else total_num_classes // args.classes_per_task
    train_splits = build_class_incremental_tasks(
        train_dataset, args.classes_per_task, num_tasks, class_order=class_order, remap_labels=False
    )
    test_splits = build_class_incremental_tasks(
        test_dataset, args.classes_per_task, num_tasks, class_order=class_order, remap_labels=False
    )
    tasks: list[TaskDefinition] = []
    train_datasets: dict[str, Dataset] = {}
    test_datasets: dict[str, Dataset] = {}
    for task_index, (train_split, test_split) in enumerate(zip(train_splits, test_splits)):
        task_id = f"task_{task_index}"
        tasks.append(
            TaskDefinition(
                task_id=task_id,
                name=task_id,
                num_classes=len(train_split.class_ids),
                metadata={"classes": list(train_split.class_ids)},
            )
        )
        train_datasets[task_id] = train_split
        test_datasets[task_id] = test_split
    return tasks, train_datasets, test_datasets, input_shape, total_num_classes


def _continual_metrics(
    accuracy_matrix: list[dict[str, float]], current: dict[str, float]
) -> dict[str, float]:
    forgetting: list[float] = []
    backward_transfer: list[float] = []
    for task_index, task_id in enumerate(list(current)[:-1]):
        forgetting.append(
            max(row[task_id] for row in accuracy_matrix if task_id in row) - current[task_id]
        )
        backward_transfer.append(current[task_id] - accuracy_matrix[task_index][task_id])
    return {
        "average_accuracy": sum(current.values()) / max(1, len(current)),
        "average_forgetting": sum(forgetting) / len(forgetting) if forgetting else 0.0,
        "backward_transfer": (
            sum(backward_transfer) / len(backward_transfer) if backward_transfer else 0.0
        ),
    }


def main() -> None:
    args = parse_lwf_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    tasks, train_datasets, test_datasets, input_shape, total_num_classes = _build_task_stream(args)
    train_loaders = {
        task_id: build_dataloader(dataset, args.batch_size, True, args.num_workers)
        for task_id, dataset in train_datasets.items()
    }
    test_loaders = {
        task_id: build_dataloader(dataset, args.batch_size, False, args.num_workers)
        for task_id, dataset in test_datasets.items()
    }
    model = build_model_from_args(args, input_shape=input_shape, num_classes=total_num_classes)
    learner = LwFLearner(
        model=model,
        optimizer=_build_optimizer(args, model),
        device=device,
        scenario=args.scenario,
        temperature=args.temperature,
        distillation_weight=args.distillation_weight,
        warmup_epochs=args.warmup_epochs,
    )
    log_path = args.log_file.strip()
    if not log_path:
        Path("logs").mkdir(parents=True, exist_ok=True)
        log_path = str(Path("logs") / f"cl_lwf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"type": "config", "args": vars(args), "resolved_device": str(device)},
                ensure_ascii=False,
            )
            + "\n"
        )

    accuracy_matrix: list[dict[str, float]] = []
    for task_index, task in enumerate(tasks):
        learner.begin_task(task)
        completed_epochs = 0
        last_eval_epoch = 0
        last_eval_row: dict[str, float] | None = None

        def _evaluate(epoch: int) -> dict[str, float]:
            row: dict[str, float] = {}
            task_metrics: dict[str, dict[str, float]] = {}
            for seen_task in tasks[: task_index + 1]:
                metrics = learner.evaluate(test_loaders[seen_task.task_id], seen_task.task_id)
                task_metrics[seen_task.task_id] = metrics
                row[seen_task.task_id] = float(metrics["accuracy"])
            summary = _continual_metrics(accuracy_matrix, row)
            record = {
                "type": "eval", "task_id": task.task_id, "task_index": task_index,
                "epoch": epoch, "metrics": summary, "task_metrics": task_metrics,
            }
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            per_task_accuracy = " ".join(
                f"{task_id}={metrics['accuracy']:.4f}"
                for task_id, metrics in task_metrics.items()
            )
            print(
                f"[eval] task={task.task_id} epoch={epoch} "
                f"avg_acc={summary['average_accuracy']:.4f} "
                f"forgetting={summary['average_forgetting']:.4f} {per_task_accuracy}"
            )
            return row

        def _on_epoch(metrics: dict[str, float]) -> None:
            nonlocal completed_epochs, last_eval_epoch, last_eval_row
            completed_epochs += 1
            record = {"type": "train", "task_id": task.task_id, "task_index": task_index, "metrics": metrics}
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.log_each_epoch:
                print(f"task={task.task_id} metrics={metrics}")
            if args.eval_every > 0 and completed_epochs % int(args.eval_every) == 0:
                last_eval_row = _evaluate(completed_epochs - 1)
                last_eval_epoch = completed_epochs

        learner.train_task(
            task, train_loaders[task.task_id], args.epochs, epoch_callback=_on_epoch
        )
        learner.end_task(task, train_loaders[task.task_id])
        if last_eval_row is None or last_eval_epoch != completed_epochs:
            last_eval_row = _evaluate(max(0, completed_epochs - 1))
        accuracy_matrix.append(last_eval_row)
    print(f"LwF finished. log={log_path}")


if __name__ == "__main__":
    main()
