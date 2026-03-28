from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

ConfigDict = Dict[str, Any]
MetricDict = Dict[str, float]
StateDict = Dict[str, Any]


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    num_classes: int
    metadata: ConfigDict = field(default_factory=dict)


@dataclass
class ClientContext:
    client_id: str
    round_idx: int
    task_id: Optional[str] = None
    metadata: ConfigDict = field(default_factory=dict)


@dataclass
class TrainResult:
    client_id: str
    num_samples: int
    metrics: MetricDict = field(default_factory=dict)
    payload: StateDict = field(default_factory=dict)


@dataclass
class AggregationResult:
    global_state: StateDict
    metrics: MetricDict = field(default_factory=dict)
    metadata: ConfigDict = field(default_factory=dict)


@runtime_checkable
class ContinualStrategy(Protocol):
    def on_task_start(self, task: TaskDefinition) -> None: ...

    def on_task_end(self, task: TaskDefinition) -> None: ...

    def regularization_loss(self, model: Any) -> Any: ...


@runtime_checkable
class Aggregator(Protocol):
    def aggregate(self, client_results: List[TrainResult]) -> AggregationResult: ...


@runtime_checkable
class ClientAlgorithm(Protocol):
    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult: ...


@runtime_checkable
class ServerAlgorithm(Protocol):
    def run_round(self, round_idx: int) -> AggregationResult: ...
