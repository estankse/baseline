from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

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

import copy

from cl_fcl_baseline.algorithms.FCL import FedViTClient, FedViTServer
from cl_fcl_baseline.experiments.args import parse_fedvit_args
from cl_fcl_baseline.experiments.FCL.common import (
    HistoricalTaskEvaluator,
    evaluate_classification,
)

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {name} to CPU.")
        return torch.device("cpu")
    return device


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
    image_size = int(getattr(args, "image_size", input_shape[-1]))
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    input_shape = (int(input_shape[0]), image_size, image_size)
    train_dataset = build_torchvision_dataset(
        args.dataset,
        train=True,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        image_size=image_size,
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
        image_size=image_size,
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
    num_tasks = (
        int(args.num_tasks)
        if int(args.num_tasks) > 0
        else (num_classes + classes_per_task - 1) // classes_per_task
    )
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


def _build_client_loaders(
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


def _build_test_loaders(
    args: argparse.Namespace,
    datasets: dict[str, Dataset],
) -> dict[str, DataLoader]:
    return {
        task_id: build_dataloader(dataset, args.batch_size, shuffle=False)
        for task_id, dataset in datasets.items()
    }


def _build_public_loaders(
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


def _build_prompt_backbone(
    args: argparse.Namespace,
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> PromptedVisionTransformer:
    specs = {
        "ViTTiny": (192, 12, 3, int(args.vit_patch_size)),
        "ViTSmall": (384, 12, 6, int(args.vit_patch_size)),
        "ViTBase": (768, 12, 12, int(args.vit_patch_size)),
        "ViTBasePatch16": (768, 12, 12, 16),
    }
    model_name = str(args.model)
    if model_name not in specs:
        raise ValueError(
            f"{model_name} is not a prompt-compatible ViT. Choose one of: "
            + ", ".join(specs)
        )
    embed_dim, depth, num_heads, patch_size = specs[model_name]
    args.fcl_embed_dim = embed_dim
    args.fcl_depth = depth
    args.fcl_num_heads = num_heads
    args.vit_patch_size = patch_size
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


def _run_experiment(
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

def main() -> None:
    args = parse_fedvit_args()
    device = _resolve_device(args.device)
    tasks, train_sets, test_sets, input_shape, num_classes = _build_task_stream(args)
    loaders = _build_client_loaders(args, tasks, train_sets)
    test_loaders = _build_test_loaders(args, test_sets)
    task_classes = {
        task.task_id: [int(class_id) for class_id in task.metadata.get("classes", [])]
        for task in tasks
    }
    initial_model = _build_prompt_backbone(args, input_shape, num_classes)
    clients = [
        FedViTClient(
            client_id=f"client_{index}",
            model=copy.deepcopy(initial_model),
            task_loaders=loaders[index],
            device=device,
            lr=args.lr,
            weight_decay=args.fedvit_weight_decay,
            epochs=args.local_epochs,
            head_epochs=args.fedvit_head_epochs,
            post_aggregation_epochs=args.fedvit_post_aggregation_epochs,
            knowledge_ratio=args.fedvit_knowledge_ratio,
            signature_k=args.fedvit_signature_k,
            integrator_steps=args.fedvit_integrator_steps,
        )
        for index in range(args.num_clients)
    ]
    server = FedViTServer(
        model=copy.deepcopy(initial_model),
        clients=clients,
        client_sample_ratio=args.client_sample_ratio,
        seed=args.seed,
    )
    clients_by_id = {client.client_id: client for client in clients}
    eval_models = {
        client.client_id: copy.deepcopy(client.model).to(device).eval()
        for client in clients
    }

    def _evaluate_unit(client_id: str | None, task_id: str) -> dict[str, float] | None:
        if client_id is None:
            return None
        client = clients_by_id[client_id]
        if task_id not in client.task_heads:
            return None
        model = eval_models[client_id]
        model.load_state_dict(server.get_body_state(), strict=False)
        model.classifier.load_state_dict(client.task_heads[task_id], strict=True)
        model.eval()
        return evaluate_classification(
            test_loaders[task_id],
            device=device,
            predict=model,
            class_ids=task_classes.get(task_id),
            num_classes=num_classes,
        )

    evaluator = HistoricalTaskEvaluator(
        algorithm="fedvit",
        evaluation_scope="client_average",
        evaluate_unit=_evaluate_unit,
        task_classes=task_classes,
        model_source="server_body_plus_client_private_task_head",
    )
    _run_experiment(
        args,
        "fedvit",
        server,
        tasks,
        eval_fn=evaluator if args.eval_every and args.eval_every > 0 else None,
    )


if __name__ == "__main__":
    main()
