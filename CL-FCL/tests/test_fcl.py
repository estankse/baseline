import torch

from cl_fcl_baseline.contracts import TaskDefinition
from cl_fcl_baseline.algorithms.fcl import ContinualClient, FCLExperiment, FCLServer, NaiveContinualStrategy
from cl_fcl_baseline.datasets.build import RandomClassificationDataset, build_dataloader
from cl_fcl_baseline.models.simple_model import MLPClassifier
from cl_fcl_baseline.trainers.trainer import BaseTrainer


def test_fcl_runs_multiple_tasks() -> None:
    tasks = [
        TaskDefinition(task_id="task_1", name="Task 1", num_classes=10),
        TaskDefinition(task_id="task_2", name="Task 2", num_classes=10),
    ]

    task_loaders = {}
    for idx, task in enumerate(tasks):
        dataset = RandomClassificationDataset(num_samples=24, input_shape=(1, 28, 28), num_classes=10, seed=idx)
        task_loaders[task.task_id] = build_dataloader(dataset, batch_size=8, shuffle=False)

    clients = []
    for idx in range(2):
        model = MLPClassifier(input_shape=(1, 28, 28), hidden_dim=32, num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = BaseTrainer(model=model, optimizer=optimizer)
        clients.append(ContinualClient(client_id=f"client_{idx}", trainer=trainer, task_loaders=task_loaders))

    server_model = MLPClassifier(input_shape=(1, 28, 28), hidden_dim=32, num_classes=10)
    server = FCLServer(model=server_model, clients=clients)
    experiment = FCLExperiment(server=server, strategy=NaiveContinualStrategy(), tasks=tasks, rounds_per_task=1)

    history = experiment.run()
    assert len(history) == 2
    for result in history:
        assert "task_id" in result.metadata
        assert "round_idx" in result.metadata
