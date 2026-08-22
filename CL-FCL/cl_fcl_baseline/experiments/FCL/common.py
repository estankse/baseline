from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from cl_fcl_baseline.algorithms.fcl import FCLExperiment, NaiveContinualStrategy
from cl_fcl_baseline.contracts import TaskDefinition
from cl_fcl_baseline.datasets import (
    RandomClassificationDataset,
    build_class_incremental_tasks,
    build_dataloader,
    build_torchvision_dataset,
    dataset_info,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.models import PromptedVisionTransformer
from cl_fcl_baseline.trainers.utils import set_seed
from cl_fcl_baseline.trainers.utils import move_to_device


EvaluationMetrics = dict[str, float]
EvaluationUnit = Callable[[str | None, str], EvaluationMetrics | None]


def evaluate_classification(
    loader: DataLoader,
    *,
    device: torch.device,
    predict: Callable[[torch.Tensor], torch.Tensor],
    class_ids: list[int] | None = None,
    seen_class_ids: list[int] | None = None,
    num_classes: int | None = None,
) -> EvaluationMetrics:
    """Evaluate one task under task-aware, seen-class, and unmasked protocols."""

    task_classes = [int(class_id) for class_id in (class_ids or [])]
    class_tensor = None
    target_lookup = None
    if task_classes and num_classes is not None and len(task_classes) < int(num_classes):
        class_tensor = torch.tensor(task_classes, device=device, dtype=torch.long)
        target_lookup = torch.full(
            (int(num_classes),), -1, device=device, dtype=torch.long
        )
        target_lookup[class_tensor] = torch.arange(
            len(task_classes), device=device, dtype=torch.long
        )

    seen_classes = [int(class_id) for class_id in (seen_class_ids or [])]
    report_seen_accuracy = seen_class_ids is not None
    seen_class_tensor = None
    if seen_classes and num_classes is not None and len(seen_classes) < int(num_classes):
        seen_class_tensor = torch.tensor(seen_classes, device=device, dtype=torch.long)

    total_loss = 0.0
    task_correct = 0
    seen_correct = 0
    unmasked_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = move_to_device(inputs, device)
            targets = move_to_device(targets, device)
            logits = predict(inputs)
            if class_tensor is None or target_lookup is None:
                loss = torch.nn.functional.cross_entropy(logits, targets)
                task_predictions = logits.argmax(dim=1)
            else:
                local_targets = target_lookup[targets]
                if bool((local_targets < 0).any()):
                    raise ValueError("Evaluation loader contains labels outside task_classes.")
                task_logits = logits.index_select(dim=1, index=class_tensor)
                loss = torch.nn.functional.cross_entropy(task_logits, local_targets)
                task_predictions = class_tensor[task_logits.argmax(dim=1)]
            count = int(targets.shape[0])
            total_examples += count
            total_loss += float(loss.detach().item()) * count
            task_correct += int((task_predictions == targets).sum().item())
            if seen_class_tensor is None:
                seen_predictions = logits.argmax(dim=1)
            else:
                seen_predictions = seen_class_tensor[
                    logits.index_select(dim=1, index=seen_class_tensor).argmax(dim=1)
                ]
            seen_correct += int((seen_predictions == targets).sum().item())
            unmasked_correct += int((logits.argmax(dim=1) == targets).sum().item())
    metrics = {
        "loss": total_loss / max(1, total_examples),
        "accuracy": task_correct / max(1, total_examples),
        "unmasked_accuracy": unmasked_correct / max(1, total_examples),
        "num_samples": float(total_examples),
    }
    if report_seen_accuracy:
        metrics["seen_accuracy"] = seen_correct / max(1, total_examples)
    return metrics


class HistoricalTaskEvaluator:
    """Shared historical-task evaluation, aggregation, forgetting, and JSONL logging."""

    def __init__(
        self,
        *,
        algorithm: str,
        evaluation_scope: str,
        evaluate_unit: EvaluationUnit,
        task_classes: Mapping[str, list[int]] | None = None,
        model_source: str | None = None,
        eval_mode: str = "task_aware",
    ) -> None:
        if evaluation_scope not in {"server", "client_average"}:
            raise ValueError("evaluation_scope must be 'server' or 'client_average'.")
        self.algorithm = algorithm
        self.evaluation_scope = evaluation_scope
        self.evaluate_unit = evaluate_unit
        self.task_classes = dict(task_classes or {})
        self.model_source = model_source or evaluation_scope
        self.eval_mode = eval_mode
        self.experiment: FCLExperiment | None = None
        self.log_path: str | None = None
        self.best_accuracy: dict[str, float] = {}

    def attach_experiment(self, experiment: FCLExperiment, log_path: str) -> None:
        self.experiment = experiment
        self.log_path = log_path

    @staticmethod
    def _mean_metrics(items: list[EvaluationMetrics]) -> EvaluationMetrics:
        names = sorted(
            {
                name
                for metrics in items
                for name in metrics
                if name != "num_samples"
            }
        )
        return {
            name: sum(metrics[name] for metrics in items if name in metrics)
            / max(1, sum(name in metrics for metrics in items))
            for name in names
        }

    def __call__(self, task_id: str, round_idx: int) -> None:
        if self.experiment is None or self.log_path is None:
            raise RuntimeError("HistoricalTaskEvaluator must be attached before use.")

        evaluation_groups = self.experiment.evaluation_task_groups
        cache: dict[object, EvaluationMetrics | None] = {}
        task_metrics: dict[str, EvaluationMetrics] = {}
        for eval_task_id, client_task_ids in evaluation_groups:
            evaluated: list[EvaluationMetrics] = []
            for client_id, client_task_id in client_task_ids.items():
                if self.evaluation_scope == "server":
                    cache_key: object = client_task_id
                    unit_client_id = None
                else:
                    cache_key = (client_id, client_task_id)
                    unit_client_id = client_id
                if cache_key not in cache:
                    cache[cache_key] = self.evaluate_unit(
                        unit_client_id, client_task_id
                    )
                metrics = cache[cache_key]
                if metrics is not None:
                    evaluated.append(metrics)

            metrics = self._mean_metrics(evaluated)
            accuracy = float(metrics.get("accuracy", 0.0))
            previous_best = self.best_accuracy.get(eval_task_id)
            forgetting = (
                max(0.0, previous_best - accuracy)
                if previous_best is not None and evaluated
                else 0.0
            )
            if evaluated:
                self.best_accuracy[eval_task_id] = max(
                    accuracy, self.best_accuracy.get(eval_task_id, accuracy)
                )
            metrics.update(
                {
                    "accuracy": accuracy,
                    "forgetting": forgetting,
                    "num_eval_clients": float(
                        len(evaluated)
                        if self.evaluation_scope == "client_average"
                        else 0
                    ),
                    "num_eval_assignments": float(len(evaluated)),
                    "num_eval_samples": sum(
                        item.get("num_samples", 0.0) for item in evaluated
                    )
                    / max(1, len(evaluated)),
                }
            )
            task_metrics[eval_task_id] = metrics

        metric_names = sorted(
            {
                name
                for metrics in task_metrics.values()
                for name in metrics
                if name
                not in {
                    "num_eval_clients",
                    "num_eval_assignments",
                    "num_eval_samples",
                }
            }
        )
        avg_metrics = {
            name: sum(metrics.get(name, 0.0) for metrics in task_metrics.values())
            / max(1, len(task_metrics))
            for name in metric_names
        }
        per_task = " ".join(
            f"{seen_task_id}={metrics.get('accuracy', 0.0):.4f}"
            for seen_task_id, metrics in task_metrics.items()
        )
        print(
            f"[eval] algorithm={self.algorithm} scope={self.evaluation_scope} "
            f"task={task_id} round={round_idx}: "
            f"avg_acc={avg_metrics.get('accuracy', 0.0):.4f} "
            f"avg_fgt={avg_metrics.get('forgetting', 0.0):.4f} {per_task}"
        )
        record = {
            "type": "eval",
            "algorithm": self.algorithm,
            "task_id": task_id,
            "task_position": self.experiment.current_task_position,
            "round": round_idx,
            "avg_metrics": avg_metrics,
            "task_metrics": task_metrics,
            "evaluation_scope": self.evaluation_scope,
            "evaluation_model_source": self.model_source,
            "eval_mode": self.eval_mode,
            "task_classes": {
                seen_task_id: list(self.task_classes.get(seen_task_id, []))
                for seen_task_id in self.experiment.seen_task_ids
            },
            "heterogeneous_eval_mode": self.experiment.effective_eval_mode,
            "evaluation_task_groups": {
                eval_task_id: client_task_ids
                for eval_task_id, client_task_ids in evaluation_groups
            },
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {name} to CPU.")
        return torch.device("cpu")
    return device


def build_task_stream(
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
        num_classes = int(args.num_classes)
        num_tasks = int(args.num_tasks) if int(args.num_tasks) > 0 else 2
        tasks = []
        train_datasets = {}
        test_datasets = {}
        for index in range(num_tasks):
            task_id = f"task_{index}"
            tasks.append(
                TaskDefinition(
                    task_id=task_id,
                    name=task_id,
                    num_classes=num_classes,
                    metadata={"classes": list(range(num_classes))},
                )
            )
            sample_count = args.num_samples if args.num_samples > 0 else 256
            train_datasets[task_id] = RandomClassificationDataset(
                sample_count,
                input_shape,
                num_classes,
                seed=args.seed + index,
            )
            test_datasets[task_id] = RandomClassificationDataset(
                sample_count,
                input_shape,
                num_classes,
                seed=args.seed + 10_000 + index,
            )
        return tasks, train_datasets, test_datasets, input_shape, num_classes

    input_shape, num_classes = dataset_info(args.dataset)
    train_dataset = build_torchvision_dataset(
        args.dataset,
        train=True,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        image_size=int(getattr(args, "image_size", input_shape[-1])),
        download=bool(getattr(args, "download", True)),
        normalization=(
            "clip"
            if getattr(args, "backbone_source", "vit") == "clip"
            else "dataset"
        ),
    )
    test_dataset = build_torchvision_dataset(
        args.dataset,
        train=False,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed + 1,
        image_size=int(getattr(args, "image_size", input_shape[-1])),
        download=bool(getattr(args, "download", True)),
        normalization=(
            "clip"
            if getattr(args, "backbone_source", "vit") == "clip"
            else "dataset"
        ),
    )
    classes_per_task = int(args.classes_per_task)
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive.")
    num_tasks = int(args.num_tasks) if int(args.num_tasks) > 0 else num_classes // classes_per_task
    train_splits = build_class_incremental_tasks(
        train_dataset,
        classes_per_task=classes_per_task,
        num_tasks=num_tasks,
        seed=args.seed,
        shuffle_classes=args.task_order_shuffle,
        remap_labels=False,
    )
    test_splits = build_class_incremental_tasks(
        test_dataset,
        classes_per_task=classes_per_task,
        num_tasks=num_tasks,
        seed=args.seed,
        shuffle_classes=args.task_order_shuffle,
        remap_labels=False,
    )
    tasks = []
    train_datasets = {}
    test_datasets = {}
    for index, (train_split, test_split) in enumerate(zip(train_splits, test_splits)):
        task_id = f"task_{index}"
        classes = list(train_split.class_ids)
        tasks.append(
            TaskDefinition(
                task_id=task_id,
                name=task_id,
                num_classes=num_classes,
                metadata={"classes": classes},
            )
        )
        train_datasets[task_id] = train_split
        test_datasets[task_id] = test_split
    return tasks, train_datasets, test_datasets, input_shape, num_classes


def build_client_loaders(
    args: argparse.Namespace,
    tasks: list[TaskDefinition],
    datasets: dict[str, Dataset],
) -> list[dict[str, DataLoader]]:
    loaders = [dict() for _ in range(args.num_clients)]
    for task_index, task in enumerate(tasks):
        dataset = datasets[task.task_id]
        seed = int(args.seed) + task_index
        if args.partition == "iid":
            partitions = partition_dataset_iid(dataset, args.num_clients, seed=seed)
        elif args.noniid_method == "dirichlet":
            partitions = partition_dataset_dirichlet(
                dataset,
                args.num_clients,
                beta=args.dirichlet_beta,
                num_classes=task.num_classes,
                seed=seed,
            )
        else:
            partitions = partition_dataset_noniid(
                dataset,
                args.num_clients,
                num_shards=args.noniid_shards,
                seed=seed,
            )
        for client_index, partition in enumerate(partitions):
            loaders[client_index][task.task_id] = build_dataloader(
                partition,
                batch_size=args.batch_size,
                shuffle=True,
            )
    return loaders


def build_test_loaders(
    args: argparse.Namespace,
    datasets: dict[str, Dataset],
) -> dict[str, DataLoader]:
    return {
        task_id: build_dataloader(dataset, args.batch_size, shuffle=False)
        for task_id, dataset in datasets.items()
    }


def build_public_loaders(
    args: argparse.Namespace,
    datasets: dict[str, Dataset],
) -> dict[str, DataLoader]:
    sample_count = int(getattr(args, "server_public_samples", 64))
    loaders = {}
    for index, (task_id, dataset) in enumerate(datasets.items()):
        public: Dataset = dataset
        if 0 < sample_count < len(dataset):
            generator = torch.Generator().manual_seed(int(args.seed) + 20_000 + index)
            indices = torch.randperm(len(dataset), generator=generator)[:sample_count]
            public = Subset(dataset, indices.tolist())
        loaders[task_id] = build_dataloader(public, args.batch_size, shuffle=False)
    return loaders


def build_prompt_backbone(
    args: argparse.Namespace,
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> PromptedVisionTransformer:
    model = PromptedVisionTransformer(
        input_shape=input_shape,
        num_classes=num_classes,
        patch_size=args.vit_patch_size,
        embed_dim=args.fcl_embed_dim,
        depth=args.fcl_depth,
        num_heads=args.fcl_num_heads,
        mlp_ratio=args.vit_mlp_ratio,
        dropout=args.vit_dropout,
        backbone_source=getattr(args, "backbone_source", "vit"),
    )
    checkpoint = str(getattr(args, "backbone_checkpoint", "")).strip()
    if checkpoint:
        report = model.load_pretrained_checkpoint(checkpoint)
        print(
            "Loaded pretrained vision backbone: "
            f"{report['loaded_backbone_tensors']}/{report['total_backbone_tensors']} "
            f"tensors from {report['path']} ({report['format']}); "
            "classifier remains task-specific."
        )
    model.freeze_backbone()
    return model


def run_experiment(
    args: argparse.Namespace,
    algorithm: str,
    server: object,
    tasks: list[TaskDefinition],
    eval_fn: Callable[[str, int], None] | None = None,
) -> FCLExperiment:
    set_seed(args.seed)
    log_path = str(args.log_file).strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"{algorithm}_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"type": "config", "algorithm": algorithm, "args": vars(args)},
                ensure_ascii=False,
            )
            + "\n"
        )
    experiment = FCLExperiment(
        server=server,  # type: ignore[arg-type]
        strategy=NaiveContinualStrategy(),
        tasks=tasks,
        rounds_per_task=args.rounds_per_task,
        heterogeneous_task_order=args.heterogeneous_task_order,
        heterogeneous_eval_mode=args.heterogeneous_eval_mode,
        seed=args.seed,
        log_each_round=True,
        eval_every=args.eval_every,
        eval_fn=eval_fn,
        log_path=log_path,
    )
    attach_experiment = getattr(eval_fn, "attach_experiment", None)
    if callable(attach_experiment):
        attach_experiment(experiment, log_path)
    experiment.run()
    print(f"{algorithm} finished. log={log_path}")
    return experiment
