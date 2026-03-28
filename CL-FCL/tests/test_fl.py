import torch

from cl_fcl_baseline.algorithms.fl import FedAvgAggregator
from cl_fcl_baseline.datasets.build import (
    RandomClassificationDataset,
    build_dataloader,
    partition_dataset_iid,
)
from cl_fcl_baseline.models.simple_model import MLPClassifier
from cl_fcl_baseline.trainers.client import FederatedClient
from cl_fcl_baseline.trainers.server import FederatedExperiment, FederatedServer
from cl_fcl_baseline.trainers.trainer import BaseTrainer


def test_fl_round_runs() -> None:
    dataset = RandomClassificationDataset(num_samples=32, input_shape=(1, 28, 28), num_classes=10, seed=0)
    partitions = partition_dataset_iid(dataset, num_clients=2, seed=0)
    loaders = [build_dataloader(part, batch_size=8, shuffle=False) for part in partitions]

    clients = []
    for idx, loader in enumerate(loaders):
        model = MLPClassifier(input_shape=(1, 28, 28), hidden_dim=32, num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = BaseTrainer(model=model, optimizer=optimizer)
        clients.append(FederatedClient(client_id=f"client_{idx}", trainer=trainer, train_loader=loader, epochs=1))

    server_model = MLPClassifier(input_shape=(1, 28, 28), hidden_dim=32, num_classes=10)
    server = FederatedServer(model=server_model, clients=clients, aggregator=FedAvgAggregator())
    experiment = FederatedExperiment(server=server, num_rounds=1, show_progress=False)

    history = experiment.run()
    assert len(history) == 1
    assert "num_clients" in history[0]
    assert "total_samples" in history[0]
