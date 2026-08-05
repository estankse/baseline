from __future__ import annotations

from dataclasses import dataclass, field
import json
import random
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
    client_sample_ratio: float = 1.0
    client_task_ids: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < float(self.client_sample_ratio) <= 1.0):
            raise ValueError("client_sample_ratio must be in (0, 1].")

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        return detach_state_dict(self.model.state_dict())

    def set_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def set_client_task_ids(self, client_task_ids: Mapping[str, str]) -> None:
        self.client_task_ids = {
            str(client_id): str(task_id)
            for client_id, task_id in client_task_ids.items()
        }

    def task_id_for_client(self, client_id: str, default_task_id: str) -> str:
        return self.client_task_ids.get(client_id, default_task_id)

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        global_state = self.get_global_state()
        client_results: List[TrainResult] = []
        clients = list(self.clients)
        if clients and self.client_sample_ratio < 1.0:
            num_selected = max(1, int(len(clients) * self.client_sample_ratio))
            clients = random.sample(clients, k=num_selected)
        for client in clients:
            client_task_id = self.task_id_for_client(client.client_id, task_id)
            context = ClientContext(
                client_id=client.client_id,
                round_idx=round_idx,
                task_id=client_task_id,
            )
            client_results.append(client.fit(global_state, context))
        aggregation_result = self.aggregator.aggregate(client_results)
        if aggregation_result.global_state:
            self.set_global_state(aggregation_result.global_state)
        metadata = dict(aggregation_result.metadata)
        metadata["round_idx"] = round_idx
        metadata["task_id"] = task_id
        metadata["client_task_ids"] = dict(self.client_task_ids)
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
    log_path: str | None = None
    heterogeneous_task_order: bool = False
    heterogeneous_eval_mode: str = "position"
    seed: int = 0
    client_task_orders: Mapping[str, Sequence[str]] | None = None
    current_task_position: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        if self.heterogeneous_eval_mode not in {"position", "task"}:
            raise ValueError("heterogeneous_eval_mode must be either 'position' or 'task'.")
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("FCL task IDs must be unique.")

        client_ids = [str(client.client_id) for client in self.server.clients]
        if self.client_task_orders is None:
            orders: Dict[str, List[str]] = {
                client_id: list(task_ids) for client_id in client_ids
            }
            if self.heterogeneous_task_order and len(task_ids) > 1:
                rng = random.Random(int(self.seed))
                for client_id in client_ids:
                    rng.shuffle(orders[client_id])
                all_orders_match = len({tuple(order) for order in orders.values()}) == 1
                if len(client_ids) > 1 and all_orders_match:
                    last_client_id = client_ids[-1]
                    orders[last_client_id] = orders[last_client_id][1:] + orders[last_client_id][:1]
            self.client_task_orders = orders
        else:
            expected = sorted(task_ids)
            provided = {
                str(client_id): [str(task_id) for task_id in order]
                for client_id, order in self.client_task_orders.items()
            }
            if set(provided) != set(client_ids):
                raise ValueError(
                    "client_task_orders must define exactly one order for every client."
                )
            for client_id, order in provided.items():
                if sorted(order) != expected:
                    raise ValueError(
                        f"Task order for {client_id!r} must be a permutation of all FCL task IDs."
                    )
            self.client_task_orders = provided

    @property
    def active_client_task_ids(self) -> Dict[str, str]:
        if self.current_task_position < 0:
            return {}
        return {
            client_id: list(order)[self.current_task_position]
            for client_id, order in (self.client_task_orders or {}).items()
        }

    @property
    def seen_task_ids(self) -> List[str]:
        if self.current_task_position < 0:
            return []
        seen = {
            task_id
            for order in (self.client_task_orders or {}).values()
            for task_id in list(order)[: self.current_task_position + 1]
        }
        return [task.task_id for task in self.tasks if task.task_id in seen]

    def client_has_seen_task(self, client_id: str, task_id: str) -> bool:
        order = list((self.client_task_orders or {}).get(client_id, []))
        return task_id in order[: self.current_task_position + 1]

    @property
    def effective_eval_mode(self) -> str:
        if not self.heterogeneous_task_order:
            return "task"
        return self.heterogeneous_eval_mode

    @property
    def evaluation_task_groups(self) -> List[tuple[str, Dict[str, str]]]:
        if self.current_task_position < 0:
            return []
        if self.effective_eval_mode == "position":
            return [
                (
                    self.tasks[position].task_id,
                    {
                        client_id: list(order)[position]
                        for client_id, order in (self.client_task_orders or {}).items()
                    },
                )
                for position in range(self.current_task_position + 1)
            ]
        return [
            (
                task_id,
                {
                    client_id: task_id
                    for client_id in (self.client_task_orders or {})
                    if self.client_has_seen_task(client_id, task_id)
                },
            )
            for task_id in self.seen_task_ids
        ]

    def run(self) -> List[AggregationResult]:
        log_handle = open(self.log_path, "a", encoding="utf-8") if self.log_path else None
        try:
            task_by_id = {task.task_id: task for task in self.tasks}
            for task_position, task in enumerate(self.tasks):
                self.current_task_position = task_position
                client_task_ids = self.active_client_task_ids
                set_client_task_ids = getattr(self.server, "set_client_task_ids", None)
                if callable(set_client_task_ids):
                    set_client_task_ids(client_task_ids)
                self.strategy.on_task_start(task)
                server_client_tasks_start = getattr(self.server, "on_client_tasks_start", None)
                if callable(server_client_tasks_start):
                    server_client_tasks_start(
                        {
                            client_id: task_by_id[task_id]
                            for client_id, task_id in client_task_ids.items()
                        }
                    )
                else:
                    server_task_start = getattr(self.server, "on_task_start", None)
                    if callable(server_task_start):
                        server_task_start(task)
                for round_idx in range(self.rounds_per_task):
                    result = self.server.run_round(round_idx, task.task_id)
                    result.metadata.setdefault("task_position", task_position)
                    result.metadata.setdefault("client_task_ids", dict(client_task_ids))
                    self.history.append(result)
                    if self.log_each_round:
                        print(f"task={task.task_id} round={round_idx} metrics={result.metrics}")
                    if log_handle is not None:
                        record = {
                            "type": "train",
                            "task_id": task.task_id,
                            "task_position": task_position,
                            "client_task_ids": dict(client_task_ids),
                            "round": round_idx,
                            "metrics": dict(result.metrics),
                        }
                        log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log_handle.flush()
                    if self.eval_fn is not None and self.eval_every:
                        if round_idx % int(self.eval_every) == 0:
                            self.eval_fn(task.task_id, round_idx)
                self.strategy.on_task_end(task)
                server_client_tasks_end = getattr(self.server, "on_client_tasks_end", None)
                if callable(server_client_tasks_end):
                    server_client_tasks_end(
                        {
                            client_id: task_by_id[task_id]
                            for client_id, task_id in client_task_ids.items()
                        }
                    )
                else:
                    server_task_end = getattr(self.server, "on_task_end", None)
                    if callable(server_task_end):
                        server_task_end(task)
        finally:
            if log_handle is not None:
                log_handle.close()
        return self.history
