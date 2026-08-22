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
    build_domain_incremental_tasks,
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

from torch import nn

from cl_fcl_baseline.algorithms.FCL import FedDuetClient, FedDuetServer
from cl_fcl_baseline.experiments.args import parse_fedduet_args
from cl_fcl_baseline.experiments.FCL.common import (
    HistoricalTaskEvaluator,
    evaluate_classification,
)
from cl_fcl_baseline.models import (
    BottleneckAdapter,
    CLIPVisionLanguageModel,
    PromptPool,
    SparseAdapterGate,
    infer_class_names,
    semantic_prompt_repository,
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
    if getattr(args, "scenario", "class") == "domain":
        requested = int(args.num_tasks) if int(args.num_tasks) > 0 else None
        train_splits = build_domain_incremental_tasks(train_dataset, num_tasks=requested)
        domain_order = [split.domain for split in train_splits]
        test_splits = build_domain_incremental_tasks(test_dataset, domains=domain_order)
        tasks: list[TaskDefinition] = []
        train_datasets: dict[str, Dataset] = {}
        test_datasets: dict[str, Dataset] = {}
        for index, (train_split, test_split) in enumerate(zip(train_splits, test_splits)):
            task_id = f"task_{index}"
            tasks.append(TaskDefinition(task_id=task_id, name=train_split.domain, num_classes=num_classes, metadata={"classes": list(range(num_classes)), "domain": train_split.domain}))
            train_datasets[task_id] = train_split
            test_datasets[task_id] = test_split
        return tasks, train_datasets, test_datasets, input_shape, num_classes
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
    args = parse_fedduet_args()
    device = _resolve_device(args.device)
    tasks, train_sets, test_sets, input_shape, num_classes = _build_task_stream(args)
    loaders = _build_client_loaders(args, tasks, train_sets)
    test_loaders = _build_test_loaders(args, test_sets)
    task_classes = {
        task.task_id: [int(class_id) for class_id in task.metadata.get("classes", [])]
        for task in tasks
    }
    backbone = _build_prompt_backbone(args, input_shape, num_classes)
    clip_model: CLIPVisionLanguageModel | None = None
    if args.backbone_source == "clip":
        class_names = infer_class_names(next(iter(train_sets.values())), num_classes)
        clip_model = CLIPVisionLanguageModel.from_checkpoint(
            backbone,
            args.backbone_checkpoint,
            class_names,
            bpe_path=args.clip_bpe_path or None,
        )
        backbone = clip_model.visual
    shared_adapters = nn.ModuleList(
        [
            BottleneckAdapter(
                args.fcl_embed_dim,
                args.fcl_adapter_dim * args.fedduet_num_experts,
                dropout=args.fedduet_adapter_dropout,
                scale=args.fedduet_adapter_scale,
            )
            for _ in range(args.fcl_depth)
        ]
    )
    clients = []
    for index in range(args.num_clients):
        client_vlm = copy.deepcopy(clip_model) if clip_model is not None else None
        client_backbone = client_vlm.visual if client_vlm is not None else copy.deepcopy(backbone)
        prompt_width = client_vlm.text_width if client_vlm is not None else args.fcl_embed_dim
        clients.append(
            FedDuetClient(
                client_id=f"client_{index}",
                backbone=client_backbone,
                clip_model=client_vlm,
                shared_adapters=copy.deepcopy(shared_adapters),
                local_adapters=nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                BottleneckAdapter(
                                    args.fcl_embed_dim,
                                    args.fcl_adapter_dim,
                                    dropout=args.fedduet_adapter_dropout,
                                    scale=args.fedduet_adapter_scale,
                                )
                                for _ in range(args.fedduet_num_experts)
                            ]
                        )
                        for _ in range(args.fcl_depth)
                    ]
                ),
                router=nn.ModuleList(
                    [
                        SparseAdapterGate(args.fcl_embed_dim, args.fedduet_num_experts)
                        for _ in range(args.fcl_depth)
                    ]
                ),
                local_prompt=nn.Parameter(
                    torch.empty(args.fcl_prompt_length, prompt_width).normal_(std=0.02)
                ),
                text_router=(
                    nn.ModuleList(
                        [
                            SparseAdapterGate(client_vlm.output_dim, args.fedduet_num_experts)
                            for _ in range(args.fcl_depth)
                        ]
                    )
                    if client_vlm is not None else None
                ),
                fusion_gating=(
                    nn.MultiheadAttention(
                        client_vlm.output_dim,
                        args.fedduet_fusion_heads,
                        dropout=args.fedduet_fusion_dropout,
                        batch_first=True,
                    )
                    if client_vlm is not None else None
                ),
                task_loaders=loaders[index],
                device=device,
                epochs=args.local_epochs,
                lr=args.lr,
                phase_switch_round=args.fedduet_phase_switch_round,
                shared_logit_weight=args.fedduet_shared_logit_weight,
                moe_weight=args.fedduet_moe_weight,
                cross_modal_weight=args.fedduet_cross_modal_weight,
                stability_weight=args.fedduet_stability_weight,
                top_k_adapters=args.fedduet_top_k,
                dp_clip=args.fedduet_dp_clip,
                dp_noise_multiplier=args.fedduet_dp_noise_multiplier,
            )
        )
    repository_width = clip_model.text_width if clip_model is not None else args.fcl_embed_dim
    repository = PromptPool(
        args.fedduet_repository_size,
        args.fcl_prompt_length,
        repository_width,
        min(args.fedduet_dispatch_count, args.fedduet_repository_size),
    )
    if clip_model is not None:
        concept_file = (
            Path(__file__).resolve().parents[3]
            / "otherswork" / "FCL复现" / "Fed-Duet-main" / "cil"
            / "dataset_reqs" / "imagenet1000_classes.txt"
        )
        concepts = class_names
        if concept_file.is_file():
            concepts = [
                line.strip() for line in concept_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        repository.prompts.copy_(
            semantic_prompt_repository(
                clip_model,
                concepts,
                args.fedduet_repository_size,
                args.fcl_prompt_length,
            ).cpu()
        )
    server = FedDuetServer(
        prompt_repository=repository,
        dispatch_gate=nn.Sequential(
            nn.Linear(clip_model.output_dim if clip_model is not None else args.fcl_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, args.fedduet_repository_size),
        ),
        shared_adapters=copy.deepcopy(shared_adapters),
        head=copy.deepcopy(backbone.classifier),
        device=device,
        experts_per_client=args.fedduet_dispatch_count,
        gate_lr=args.fedduet_gate_lr,
        clients=clients,
        client_sample_ratio=args.client_sample_ratio,
        seed=args.seed,
    )
    clients_by_id = {client.client_id: client for client in clients}

    def _evaluate_unit(client_id: str | None, task_id: str) -> dict[str, float] | None:
        if client_id is None:
            return None
        client = clients_by_id[client_id]
        client.shared_adapters.load_state_dict(server.shared_adapters.state_dict(), strict=True)
        if client.clip_model is None:
            client.backbone.classifier.load_state_dict(server.head.state_dict(), strict=True)
        shared_prompts, _ = server._dispatch(client.summarize(task_id, apply_dp=False))
        return evaluate_classification(
            test_loaders[task_id],
            device=device,
            predict=lambda inputs: client.predict(inputs, shared_prompts),
            class_ids=task_classes.get(task_id),
            num_classes=num_classes,
        )

    evaluator = HistoricalTaskEvaluator(
        algorithm="fedduet",
        evaluation_scope="client_average",
        evaluate_unit=_evaluate_unit,
        task_classes=task_classes,
        model_source="server_shared_state_plus_client_private_state",
    )
    _run_experiment(
        args,
        "fedduet",
        server,
        tasks,
        eval_fn=evaluator if args.eval_every and args.eval_every > 0 else None,
    )


if __name__ == "__main__":
    main()
