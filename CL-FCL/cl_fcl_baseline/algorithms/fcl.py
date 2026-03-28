from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..contracts import (
    AggregationResult,
    ClientContext,
    ContinualStrategy,
    MetricDict,
    TaskDefinition,
    TrainResult,
)
from ..trainers.trainer import BaseTrainer
from .fl import FedAvgAggregator
from ..trainers.utils import detach_state_dict


def _clone_tensor_state(state: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[name] = value.detach().clone()
    return cloned


@dataclass
class NaiveContinualStrategy(ContinualStrategy):
    current_task: Optional[TaskDefinition] = None
    completed_tasks: List[str] = field(default_factory=list)

    def on_task_start(self, task: TaskDefinition) -> None:
        self.current_task = task

    def on_task_end(self, task: TaskDefinition) -> None:
        self.completed_tasks.append(task.task_id)
        if self.current_task is not None and self.current_task.task_id == task.task_id:
            self.current_task = None

    def regularization_loss(self, model: object) -> torch.Tensor:
        del model
        return torch.tensor(0.0)


@dataclass
class ContinualClient:
    client_id: str
    trainer: BaseTrainer
    task_loaders: Mapping[str, DataLoader]
    epochs: int = 1

    def fit(self, global_state: Dict[str, torch.Tensor], context: ClientContext) -> TrainResult:
        self.trainer.model.load_state_dict(global_state, strict=True)
        task_id = context.task_id
        if task_id is None or task_id not in self.task_loaders:
            loader = next(iter(self.task_loaders.values()))
        else:
            loader = self.task_loaders[task_id]

        metrics: MetricDict = {}
        for _ in range(self.epochs):
            metrics = self.trainer.train_epoch(loader)
        payload = {"model_state": detach_state_dict(self.trainer.model.state_dict())}
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics=metrics,
            payload=payload,
        )


@dataclass
class FCLServer:
    model: torch.nn.Module
    clients: Sequence[ContinualClient]
    aggregator: FedAvgAggregator = field(default_factory=FedAvgAggregator)

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        return detach_state_dict(self.model.state_dict())

    def set_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        global_state = self.get_global_state()
        client_results: List[TrainResult] = []
        for client in self.clients:
            context = ClientContext(client_id=client.client_id, round_idx=round_idx, task_id=task_id)
            client_results.append(client.fit(global_state, context))
        aggregation_result = self.aggregator.aggregate(client_results)
        if aggregation_result.global_state:
            self.set_global_state(aggregation_result.global_state)
        metadata = dict(aggregation_result.metadata)
        metadata["round_idx"] = round_idx
        metadata["task_id"] = task_id
        return AggregationResult(
            global_state=aggregation_result.global_state,
            metrics=dict(aggregation_result.metrics),
            metadata=metadata,
        )


@dataclass
class FCLExperiment:
    server: FCLServer
    strategy: ContinualStrategy
    tasks: Sequence[TaskDefinition]
    rounds_per_task: int = 1
    history: List[AggregationResult] = field(default_factory=list)
    log_each_round: bool = False
    eval_every: int | None = None
    eval_fn: Callable[[str, int], None] | None = None

    def run(self) -> List[AggregationResult]:
        for task in self.tasks:
            self.strategy.on_task_start(task)
            for round_idx in range(self.rounds_per_task):
                result = self.server.run_round(round_idx, task.task_id)
                self.history.append(result)
                if self.log_each_round:
                    print(f"task={task.task_id} round={round_idx} metrics={result.metrics}")
                if self.eval_fn is not None and self.eval_every:
                    if round_idx % int(self.eval_every) == 0:
                        self.eval_fn(task.task_id, round_idx)
            self.strategy.on_task_end(task)
        return self.history
