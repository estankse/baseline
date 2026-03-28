from __future__ import annotations

import torch

from cl_fcl_baseline.algorithms.fl import FedAvgAggregator
from cl_fcl_baseline.datasets import build_torchvision_dataset, dataset_info
from cl_fcl_baseline.datasets.build import (
    RandomClassificationDataset,
    build_dataloader,
    partition_dataset_iid,
    partition_dataset_noniid,
)
from cl_fcl_baseline.models.simple_model import MLPClassifier, SimpleCNN
from cl_fcl_baseline.trainers.client import FederatedClient
from cl_fcl_baseline.trainers.server import FederatedExperiment, FederatedServer
from cl_fcl_baseline.trainers.trainer import BaseTrainer
from cl_fcl_baseline.trainers.utils import set_seed

try:
    from .args import parse_fedavg_args
except ImportError:  # pragma: no cover
    from cl_fcl_baseline.experiments.args import parse_fedavg_args


def main() -> None:
    global netmodel
    args = parse_fedavg_args()
    set_seed(args.seed)

    if args.dataset == "random_classification":
        input_shape = tuple(args.input_shape)
        num_classes = args.num_classes
        dataset = RandomClassificationDataset(
            num_samples=args.num_samples,
            input_shape=input_shape,
            num_classes=num_classes,
            seed=args.seed,
        )
        test_dataset = RandomClassificationDataset(
            num_samples=args.num_samples,
            input_shape=input_shape,
            num_classes=num_classes,
            seed=args.seed + 1,
        )
    else:
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
        if args.model == "mlp":
            model = MLPClassifier(
                input_shape=input_shape,
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
            )
        elif args.model == "simplecnn":
            model = SimpleCNN(
                input_channels=1,
                num_classes=num_classes,
            )
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
        trainer = BaseTrainer(model=model, optimizer=optimizer)
        clients.append(
            FederatedClient(
                client_id=f"client_{idx}",
                trainer=trainer,
                train_loader=loader,
                epochs=args.local_epochs,
            )
        )

    if args.model == "mlp":
        server_model = MLPClassifier(
            input_shape=input_shape,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
        )
    elif args.model == "simplecnn":
        server_model = SimpleCNN(
            input_channels=1,
            num_classes=num_classes,
        )


    server = FederatedServer(model=server_model, clients=clients, aggregator=FedAvgAggregator())

    if args.optimizer == "adam":
        eval_optimizer = torch.optim.Adam(server.model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        eval_optimizer = torch.optim.SGD(server.model.parameters(), lr=args.lr,momentum=0.9)
    eval_trainer = BaseTrainer(
        model=server.model,
        optimizer=eval_optimizer,
    )
    def _eval_round(round_idx: int) -> None:
        test_metrics = eval_trainer.evaluate(test_loader)
        print(f"[eval] round {round_idx}: acc={test_metrics.get('accuracy', 0.0):.4f}")

    experiment = FederatedExperiment(
        server=server,
        num_rounds=args.num_rounds,
        show_progress=args.show_progress,
        log_each_round=True,
        eval_every=args.eval_every,
        eval_fn=_eval_round if args.eval_every and args.eval_every > 0 else None,
    )

    history = experiment.run()
    print("FedAvg finished.")


if __name__ == "__main__":
    main()
