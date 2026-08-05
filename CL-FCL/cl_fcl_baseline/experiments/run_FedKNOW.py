from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset

from cl_fcl_baseline.algorithms.fcl import FCLExperiment, NaiveContinualStrategy
from cl_fcl_baseline.algorithms.fedknow import FedKNOWClient, FedKNOWServer
from cl_fcl_baseline.contracts import TaskDefinition
from cl_fcl_baseline.datasets import build_class_incremental_tasks, build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    RandomClassificationDataset,
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.models import build_model_from_args
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import move_to_device, set_seed

try:
    from .args import parse_fedknow_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import parse_fedknow_args


def _build_model(args: argparse.Namespace, input_shape: tuple[int, int, int], num_classes: int) -> torch.nn.Module:
    return build_model_from_args(args, input_shape=input_shape, num_classes=num_classes)


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
    for task_idx, (train_split, test_split) in enumerate(zip(train_task_datasets, test_task_datasets)):
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
    args = parse_fedknow_args()
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
    task_classes = {
        task.task_id: list(task.metadata.get("classes", range(task.num_classes)))
        for task in tasks
    }

    clients = []
    for idx in range(args.num_clients):
        model = _build_model(args, input_shape=input_shape, num_classes=task_num_classes)
        optimizer = _build_optimizer(args, model)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device)
        clients.append(
            FedKNOWClient(
                client_id=f"client_{idx}",
                trainer=trainer,
                task_loaders=client_task_loaders[idx],
                epochs=args.local_epochs,
                knowledge_ratio=args.knowledge_ratio,
                signature_k=args.signature_k,
                integrator_steps=args.integrator_steps,
                knowledge_finetune_epochs=args.knowledge_finetune_epochs,
                post_aggregation_epochs=args.post_aggregation_epochs,
                distillation_warmup_epochs=args.distillation_warmup_epochs,
                restorer_loss=args.restorer_loss,
                restorer_temperature=args.restorer_temperature,
                optimizer_name=args.optimizer,
                lr=args.lr,
                weight_decay=5e-4 if args.optimizer == "sgd" else 0.0,
                task_classes=task_classes,
            )
        )

    server_model = _build_model(args, input_shape=input_shape, num_classes=task_num_classes)
    if clients:
        server_model.load_state_dict(clients[0].trainer.model.state_dict(), strict=True)
    server = FedKNOWServer(
        model=server_model,
        clients=clients,
        client_sample_ratio=args.client_sample_ratio,
    )

    eval_model = _build_model(args, input_shape=input_shape, num_classes=task_num_classes)
    eval_optimizer = _build_optimizer(args, eval_model)
    eval_trainer = BaseTrainer(model=eval_model, optimizer=eval_optimizer, device=device)

    log_path = args.log_file.strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"fedknow_{timestamp}.jsonl")
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def _evaluate_task(task_id: str) -> dict[str, float]:
        task_loader = test_loaders[task_id]
        class_ids = task_classes.get(task_id, [])
        class_tensor = None
        if class_ids and len(class_ids) < task_num_classes:
            class_tensor = torch.tensor([int(class_id) for class_id in class_ids], device=device, dtype=torch.long)
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
        for eval_task_id, client_task_ids in evaluation_groups:
            evaluated_clients = 0
            client_eval_samples = 0
            client_accuracy = 0.0
            client_loss = 0.0
            for client in server.clients:
                client_task_id = client_task_ids.get(client.client_id)
                if client_task_id is None:
                    continue
                if client.current_state is None and client.personal_state is None:
                    continue
                task_loader = test_loaders[client_task_id]
                eval_state = server.build_eval_state(client_task_id, client_id=client.client_id)
                eval_trainer.model.load_state_dict(eval_state, strict=True)
                local_metrics = _evaluate_task(client_task_id)
                evaluated_clients += 1
                client_eval_samples += len(task_loader.dataset)
                client_accuracy += float(local_metrics.get("accuracy", 0.0))
                client_loss += float(local_metrics.get("loss", 0.0))
            metrics = {
                "accuracy": client_accuracy / max(1, evaluated_clients),
                "loss": client_loss / max(1, evaluated_clients),
                "num_eval_clients": float(evaluated_clients),
                "num_eval_samples": float(client_eval_samples / max(1, evaluated_clients)),
            }
            task_metrics[eval_task_id] = metrics
            avg_accuracy += float(metrics.get("accuracy", 0.0))
            avg_loss += float(metrics.get("loss", 0.0))
        avg_accuracy /= max(1, len(evaluation_groups))
        avg_loss /= max(1, len(evaluation_groups))
        per_task_accuracy = " ".join(
            f"{seen_task_id}={metrics.get('accuracy', 0.0):.4f}"
            for seen_task_id, metrics in task_metrics.items()
        )
        print(f"[eval] task={task_id} round={round_idx}: avg_acc={avg_accuracy:.4f} {per_task_accuracy}")
        record = {
            "type": "eval",
            "task_id": task_id,
            "round": round_idx,
            "avg_metrics": {"accuracy": avg_accuracy, "loss": avg_loss},
            "task_metrics": task_metrics,
            "eval_mode": "task_aware",
            "task_classes": {
                seen_task_id: list(task_classes.get(seen_task_id, []))
                for seen_task_id in experiment.seen_task_ids
            },
            "heterogeneous_eval_mode": experiment.effective_eval_mode,
            "evaluation_task_groups": {
                eval_task_id: client_task_ids
                for eval_task_id, client_task_ids in evaluation_groups
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
    print("FedKNOW finished.")


if __name__ == "__main__":
    main()
