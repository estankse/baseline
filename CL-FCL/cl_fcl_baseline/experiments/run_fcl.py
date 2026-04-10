from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset

from cl_fcl_baseline.algorithms.fcl import ContinualClient, FCLExperiment, FCLServer, NaiveContinualStrategy
from cl_fcl_baseline.contracts import TaskDefinition
from cl_fcl_baseline.datasets import build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    RandomClassificationDataset,
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.models.simple_model import MLPClassifier
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import set_seed

try:
    from .args import parse_fcl_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import parse_fcl_args


def _make_task_dataset(task_seed: int, args: argparse.Namespace, train: bool) -> RandomClassificationDataset | Dataset:
    if args.dataset == "random_classification":
        return RandomClassificationDataset(
            num_samples=args.num_samples,
            input_shape=tuple(args.input_shape),
            num_classes=args.num_classes,
            seed=task_seed,
        )
    return build_torchvision_dataset(
        name=args.dataset,
        train=train,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=task_seed,
    )


def main() -> None:
    args = parse_fcl_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dataset == "random_classification":
        input_shape = tuple(args.input_shape)
        num_classes = args.num_classes
    else:
        input_shape, num_classes = dataset_info(args.dataset)
    tasks = [
        TaskDefinition(task_id=task_id, name=task_id, num_classes=num_classes)
        for task_id in args.tasks
    ]

    task_datasets = {task.task_id: _make_task_dataset(seed, args, True) for seed, task in enumerate(tasks)}
    client_task_loaders = [dict() for _ in range(args.num_clients)]
    for task_id, dataset in task_datasets.items():
        if args.partition == "iid":
            partitions = partition_dataset_iid(dataset, num_clients=args.num_clients, seed=args.seed)
        elif args.noniid_method == "dirichlet":
            partitions = partition_dataset_dirichlet(
                dataset,
                num_clients=args.num_clients,
                beta=args.dirichlet_beta,
                num_classes=num_classes,
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
            client_task_loaders[client_idx][task_id] = build_dataloader(
                partition,
                batch_size=args.batch_size,
                shuffle=True,
            )
    test_datasets = {task.task_id: _make_task_dataset(seed + 1, args, False) for seed, task in enumerate(tasks)}
    test_loaders = {
        task_id: build_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
        for task_id, dataset in test_datasets.items()
    }

    clients = []
    for idx in range(args.num_clients):
        model = MLPClassifier(input_shape=input_shape, hidden_dim=args.hidden_dim, num_classes=num_classes)
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        trainer = BaseTrainer(model=model, optimizer=optimizer, device=device)
        clients.append(
            ContinualClient(
                client_id=f"client_{idx}",
                trainer=trainer,
                task_loaders=client_task_loaders[idx],
            )
        )

    server_model = MLPClassifier(input_shape=input_shape, hidden_dim=args.hidden_dim, num_classes=num_classes)
    server = FCLServer(model=server_model, clients=clients, client_sample_ratio=args.client_sample_ratio)
    if args.optimizer == "adam":
        eval_optimizer = torch.optim.Adam(server.model.parameters(), lr=args.lr)
    else:
        eval_optimizer = torch.optim.SGD(server.model.parameters(), lr=args.lr)
    eval_trainer = BaseTrainer(
        model=server.model,
        optimizer=eval_optimizer,
        device=device,
    )

    log_path = args.log_file.strip()
    if not log_path:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(log_dir / f"fcl_{timestamp}.jsonl")

    def _eval_round(task_id: str, round_idx: int) -> None:
        loader = test_loaders[task_id]
        test_metrics = eval_trainer.evaluate(loader)
        print(f"[eval] task={task_id} round={round_idx}: acc={test_metrics.get('accuracy', 0.0):.4f}")
        record = {"type": "eval", "task_id": task_id, "round": round_idx, "metrics": test_metrics}
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

    history = experiment.run()
    print("FCL finished.")


if __name__ == "__main__":
    main()
