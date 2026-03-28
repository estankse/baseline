from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Sequence

from tqdm import tqdm

from ..contracts import AggregationResult, ClientContext, MetricDict, StateDict, TrainResult
from .client import FederatedClient
from .utils import detach_state_dict


@dataclass
class FederatedServer:
    model: Any
    clients: Sequence[FederatedClient]
    aggregator: Any

    def get_global_state(self) -> StateDict:
        return detach_state_dict(self.model.state_dict())

    def set_global_state(self, state_dict: StateDict) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def run_round(self, round_idx: int) -> AggregationResult:
        global_state = self.get_global_state()
        client_results: List[TrainResult] = []
        for client in self.clients:
            context = ClientContext(client_id=client.client_id, round_idx=round_idx)
            client_results.append(client.fit(global_state, context))
        aggregation_result = self.aggregator.aggregate(client_results)
        self.set_global_state(aggregation_result.global_state)
        return aggregation_result


@dataclass
class FederatedExperiment:
    server: FederatedServer
    num_rounds: int = 1
    history: List[MetricDict] = field(default_factory=list)
    show_progress: bool = True
    log_each_round: bool = False
    eval_every: int | None = None
    eval_fn: Callable[[int], None] | None = None

    def run(self) -> List[MetricDict]:
        rounds: Iterable[int] = range(self.num_rounds)
        if self.show_progress:
            rounds = tqdm(rounds, desc="federated_rounds")
        for round_idx in rounds:
            result = self.server.run_round(round_idx)
            metrics = dict(result.metrics)
            self.history.append(metrics)
            if self.log_each_round:
                print(f"round {round_idx}: {metrics}")
            if self.eval_fn is not None and self.eval_every:
                if round_idx % int(self.eval_every) == 0:
                    self.eval_fn(round_idx)
        return self.history
