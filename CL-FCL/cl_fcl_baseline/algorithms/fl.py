from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import torch

from ..contracts import AggregationResult, Aggregator, TrainResult


def _clone_tensor_state(state: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[name] = value.detach().clone()
    return cloned


@dataclass
class FedAvgAggregator(Aggregator):
    metric_prefix: str = "client_"

    def aggregate(self, client_results: List[TrainResult]) -> AggregationResult:
        if not client_results:
            return AggregationResult(
                global_state={},
                metrics={"num_clients": 0.0, "total_samples": 0.0},
            )

        total_samples = 0
        accumulator: Dict[str, torch.Tensor] = {}
        metric_sums: Dict[str, float] = {}
        metric_weights: Dict[str, float] = {}

        for result in client_results:
            payload_state = result.payload.get("model_state", {})
            if not isinstance(payload_state, Mapping):
                continue

            weight = max(int(result.num_samples), 0)
            if weight == 0:
                continue

            state = _clone_tensor_state(payload_state)
            if not state:
                continue

            total_samples += weight
            for name, tensor in state.items():
                weighted_tensor = tensor * float(weight)
                if name not in accumulator:
                    accumulator[name] = weighted_tensor
                else:
                    accumulator[name] += weighted_tensor

            for metric_name, metric_value in result.metrics.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + float(metric_value) * float(weight)
                metric_weights[metric_name] = metric_weights.get(metric_name, 0.0) + float(weight)

        if total_samples == 0 or not accumulator:
            return AggregationResult(
                global_state={},
                metrics={"num_clients": float(len(client_results)), "total_samples": 0.0},
            )

        averaged_state = {
            name: tensor / float(total_samples)
            for name, tensor in accumulator.items()
        }

        averaged_metrics = {
            f"{self.metric_prefix}{name}": metric_sums[name] / metric_weights[name]
            for name in metric_sums
            if metric_weights.get(name, 0.0) > 0.0
        }
        averaged_metrics["num_clients"] = float(len(client_results))
        averaged_metrics["total_samples"] = float(total_samples)

        return AggregationResult(
            global_state=averaged_state,
            metrics=averaged_metrics,
            metadata={"aggregator": "fedavg"},
        )
