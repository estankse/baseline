from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset

from cl_fcl_baseline.algorithms.CL_robust.TABA import TABALearner
from cl_fcl_baseline.algorithms.FL_robust.PGD import PGDConfig
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
    from ..args import parse_taba_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import parse_taba_args


NORMALIZATION_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _build_optimizer(
    args: argparse.Namespace, model: torch.nn.Module
) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def _build_pgd_config(args: argparse.Namespace, *, evaluation: bool) -> PGDConfig:
    steps = int(args.eval_pgd_steps if evaluation else args.pgd_steps)
    key = str(args.dataset).lower()
    if key not in NORMALIZATION_STATS:
        return PGDConfig(
            epsilon=float(args.pgd_epsilon),
            step_size=float(args.pgd_step_size),
            steps=steps,
            random_start=bool(args.pgd_random_start),
        )
    mean, std = NORMALIZATION_STATS[key]
    clip_min = [(0.0 - value) / scale for value, scale in zip(mean, std)]
    clip_max = [(1.0 - value) / scale for value, scale in zip(mean, std)]
    if args.pgd_normalized_space:
        epsilon: float | list[float] = float(args.pgd_epsilon)
        step_size: float | list[float] = float(args.pgd_step_size)
    else:
        epsilon = [float(args.pgd_epsilon) / scale for scale in std]
        step_size = [float(args.pgd_step_size) / scale for scale in std]
    return PGDConfig(
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        random_start=bool(args.pgd_random_start),
        clip_min=clip_min,
        clip_max=clip_max,
    )


def _build_task_stream(
    args: argparse.Namespace,
) -> tuple[
    list[TaskDefinition],
    dict[str, Dataset],
    dict[str, Dataset],
    tuple[int, int, int],
    int,
]:
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
            total_num_classes,
            generator=torch.Generator().manual_seed(int(args.seed)),
        ).tolist()
        class_order = [class_order[index] for index in permutation]
    num_tasks = int(args.num_tasks) if args.num_tasks > 0 else 0
    classes_per_task = int(args.classes_per_task)
    if classes_per_task <= 0:
        if num_tasks <= 0:
            raise ValueError(
                "At least one of --num-tasks or --classes-per-task must be positive."
            )
        if total_num_classes % num_tasks:
            raise ValueError(
                "The dataset class count must be divisible by --num-tasks when "
                "--classes-per-task is automatic."
            )
        classes_per_task = total_num_classes // num_tasks
    if num_tasks <= 0:
        num_tasks = total_num_classes // classes_per_task
    # Persist resolved values so the JSONL config records the actual stream
    # (2 classes/task for CIFAR-10, 20 for CIFAR-100), not the auto sentinel.
    args.num_tasks = num_tasks
    args.classes_per_task = classes_per_task
    train_splits = build_class_incremental_tasks(
        train_dataset,
        classes_per_task,
        num_tasks,
        class_order=class_order,
        remap_labels=False,
    )
    test_splits = build_class_incremental_tasks(
        test_dataset,
        classes_per_task,
        num_tasks,
        class_order=class_order,
        remap_labels=False,
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
    accuracy_matrix: list[dict[str, float]],
    current: dict[str, float],
    learned_task_ids: list[str],
) -> dict[str, float]:
    if set(current) != set(learned_task_ids):
        raise ValueError("current must contain exactly the learned tasks.")
    if len(accuracy_matrix) < max(0, len(learned_task_ids) - 1):
        raise ValueError("accuracy_matrix is missing a completed task stage.")
    average_accuracy = sum(current[task_id] for task_id in learned_task_ids) / max(
        1, len(learned_task_ids)
    )
    forgetting: list[float] = []
    backward_transfer: list[float] = []
    for task_index, task_id in enumerate(learned_task_ids[:-1]):
        history = [row[task_id] for row in accuracy_matrix if task_id in row]
        if not history or task_id not in accuracy_matrix[task_index]:
            raise ValueError(f"Missing accuracy history for {task_id}.")
        forgetting.append(max(0.0, max(history) - current[task_id]))
        backward_transfer.append(current[task_id] - accuracy_matrix[task_index][task_id])
    return {
        "average_accuracy": average_accuracy,
        "average_forgetting": sum(forgetting) / len(forgetting) if forgetting else 0.0,
        "backward_transfer": (
            sum(backward_transfer) / len(backward_transfer) if backward_transfer else 0.0
        ),
    }


def main() -> None:
    args = parse_taba_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    tasks, train_datasets, test_datasets, input_shape, total_classes = _build_task_stream(args)
    train_loaders = {
        task_id: build_dataloader(dataset, args.batch_size, True, args.num_workers)
        for task_id, dataset in train_datasets.items()
    }
    test_loaders = {
        task_id: build_dataloader(dataset, args.batch_size, False, args.num_workers)
        for task_id, dataset in test_datasets.items()
    }
    model = build_model_from_args(args, input_shape=input_shape, num_classes=total_classes)
    learner = TABALearner(
        model=model,
        optimizer=_build_optimizer(args, model),
        device=device,
        scenario=args.scenario,
        memory_budget=args.memory_budget,
        replay_batch_size=args.replay_batch_size,
        attack_config=_build_pgd_config(args, evaluation=False),
        eval_attack_config=_build_pgd_config(args, evaluation=True),
        mix_batch_size=args.taba_mix_batch_size,
        mix_lambda_min=args.taba_lambda_min,
        mix_lambda_max=args.taba_lambda_max,
    )

    log_path = args.log_file.strip()
    if not log_path:
        Path("logs").mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(Path("logs") / f"cl_robust_taba_{stamp}.jsonl")
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

    clean_matrix: list[dict[str, float]] = []
    robust_matrix: list[dict[str, float]] = []
    for task_index, task in enumerate(tasks):
        learner.begin_task(task)
        completed_epochs = 0
        last_eval_epoch = -1
        last_clean_row: dict[str, float] | None = None
        last_robust_row: dict[str, float] | None = None

        def _evaluate(
            epoch: int, *, phase: str = "epoch"
        ) -> tuple[dict[str, float], dict[str, float]]:
            clean_row: dict[str, float] = {}
            robust_row: dict[str, float] = {}
            task_metrics: dict[str, dict[str, float]] = {}
            learned_tasks = tasks[: task_index + 1]
            learned_task_ids = [seen_task.task_id for seen_task in learned_tasks]
            for seen_task in learned_tasks:
                clean = learner.evaluate(test_loaders[seen_task.task_id], seen_task.task_id)
                robust = learner.evaluate_robust(
                    test_loaders[seen_task.task_id],
                    seen_task.task_id,
                    max_batches=args.pgd_max_batches,
                )
                task_metrics[seen_task.task_id] = {**clean, **robust}
                clean_row[seen_task.task_id] = float(clean["accuracy"])
                robust_row[seen_task.task_id] = float(robust["robust_accuracy"])
            clean_summary = _continual_metrics(
                clean_matrix, clean_row, learned_task_ids
            )
            robust_summary = _continual_metrics(
                robust_matrix, robust_row, learned_task_ids
            )
            record = {
                "type": "eval",
                "task_id": task.task_id,
                "task_index": task_index,
                "epoch": epoch,
                "phase": phase,
                "clean_metrics": clean_summary,
                "robust_metrics": robust_summary,
                "task_metrics": task_metrics,
            }
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            per_task_accuracy = " ".join(
                f"{task_id}:clean={task_metrics[task_id]['accuracy']:.4f},"
                f"robust={task_metrics[task_id]['robust_accuracy']:.4f}"
                for task_id in learned_task_ids
            )
            print(
                f"[eval] task={task.task_id} epoch={epoch} "
                f"clean={clean_summary['average_accuracy']:.4f} "
                f"robust={robust_summary['average_accuracy']:.4f} "
                f"clean_fgt={clean_summary['average_forgetting']:.4f} "
                f"robust_fgt={robust_summary['average_forgetting']:.4f} "
                f"{per_task_accuracy}"
            )
            return clean_row, robust_row

        def _on_epoch(metrics: dict[str, float]) -> None:
            nonlocal completed_epochs, last_eval_epoch, last_clean_row, last_robust_row
            completed_epochs += 1
            record = {
                "type": "train",
                "task_id": task.task_id,
                "task_index": task_index,
                "metrics": metrics,
            }
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.log_each_epoch:
                print(f"task={task.task_id} metrics={metrics}")
            if args.eval_every > 0 and completed_epochs % int(args.eval_every) == 0:
                # As in the iCaRL runner, evaluation first refreshes the
                # current task's herding exemplars and nearest-mean
                # prototypes. Otherwise BCE-head calibration can make a
                # correctly learned new task appear to have zero accuracy.
                learner.refresh_nme(train_loaders[task.task_id])
                last_clean_row, last_robust_row = _evaluate(completed_epochs - 1)
                last_eval_epoch = completed_epochs

        learner.train_task(
            task, train_loaders[task.task_id], args.epochs, epoch_callback=_on_epoch
        )
        learner.end_task(task, train_loaders[task.task_id])
        # TABA inherits iCaRL's herding/NME inference. The current task's
        # exemplars and class means exist only after end_task, so the row used
        # in the continual matrix must always be this post-task evaluation,
        # not the classifier-based diagnostic from the final epoch callback.
        last_clean_row, last_robust_row = _evaluate(
            max(0, completed_epochs - 1), phase="post_task"
        )
        clean_matrix.append(last_clean_row)
        if last_robust_row is None:
            raise RuntimeError("Robust evaluation did not produce a result row.")
        robust_matrix.append(last_robust_row)
    print(f"TABA finished. log={log_path}")


if __name__ == "__main__":
    main()
