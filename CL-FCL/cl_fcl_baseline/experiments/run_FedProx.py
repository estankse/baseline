from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch

from cl_fcl_baseline.algorithms.fl import FedAvgAggregator
from cl_fcl_baseline.algorithms.fedprox import FedProxClient, FedProxTrainer
from cl_fcl_baseline.datasets import build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    RandomClassificationDataset,
    build_dataloader,
    partition_dataset_dirichlet,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.models import build_model_from_args
from cl_fcl_baseline.trainers.server import FederatedExperiment, FederatedServer
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import set_seed

try:
    from .args import parse_fedprox_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import parse_fedprox_args


def main() -> None:
    global netmodel
    args = parse_fedprox_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)


    input_shape, num_classes = dataset_info(args.dataset)
    dataset = build_torchvision_dataset(
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
    loaders = [build_dataloader(part, batch_size=args.batch_size, shuffle=True) for part in partitions]
    test_loader = build_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)



    clients = []
    for idx, loader in enumerate(loaders):
        model = build_model_from_args(args, input_shape=input_shape, num_classes=num_classes)
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
        trainer = FedProxTrainer(model=model, optimizer=optimizer, device=device, proximal_mu=args.prox_mu)
        clients.append(
            FedProxClient(
                client_id=f"client_{idx}",
                trainer=trainer,
                train_loader=loader,
                epochs=args.local_epochs,
            )
        )

    server_model = build_model_from_args(args, input_shape=input_shape, num_classes=num_classes)

    server = FederatedServer(
        model=server_model,
        clients=clients,
        aggregator=FedAvgAggregator(),
        client_sample_ratio=args.client_sample_ratio,
    )

    if args.optimizer == "adam":
        eval_optimizer = torch.optim.Adam(server.model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        eval_optimizer = torch.optim.SGD(server.model.parameters(), lr=args.lr,momentum=0.9)
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
        log_path = str(log_dir / f"fedprox_{timestamp}.jsonl")

    def _eval_round(round_idx: int) -> None:
        test_metrics = eval_trainer.evaluate(test_loader)
        print(f"[eval] round {round_idx}: acc={test_metrics.get('accuracy', 0.0):.4f}")
        record = {"type": "eval", "round": round_idx, "metrics": test_metrics}
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    experiment = FederatedExperiment(
        server=server,
        num_rounds=args.num_rounds,
        show_progress=args.show_progress,
        log_each_round=True,
        eval_every=args.eval_every,
        eval_fn=_eval_round if args.eval_every and args.eval_every > 0 else None,
        log_path=log_path,
    )

    history = experiment.run()
    print("FedProx finished.")


if __name__ == "__main__":
    main()
