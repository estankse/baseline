from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence

import torch

from ..contracts import AggregationResult, StateDict, TrainResult
from .FAT import FedWeITFATClient
from .fedweit import FedWeITAggregator, FedWeITServer


def _safe_metric(result: TrainResult, metric_name: str) -> float:
    value = result.metrics.get(metric_name)
    if value is None and metric_name != "adv_ce_loss":
        value = result.metrics.get("adv_ce_loss")
    if value is None:
        value = result.metrics.get("ce_loss", result.metrics.get("loss", 0.0))
    try:
        metric = float(value)
    except (TypeError, ValueError):
        metric = 0.0
    if not math.isfinite(metric):
        return 0.0
    return metric


@dataclass
class FedWeITSFATClient(FedWeITFATClient):
    """FedWeIT client with the SFAT local adversarial-training step.

    SFAT keeps FAT-style local adversarial training and sends the local
    adversarial loss to the server, where the alpha-slack aggregation is
    applied.
    """


@dataclass
class FedWeITSFATAggregator(FedWeITAggregator):
    """FedWeIT aggregator with SFAT alpha-slack client reweighting.

    The server ranks clients by ``N_k / N * L_k`` in ascending order, upweights
    the lowest-loss ``enhanced_clients`` by ``(1 + alpha) / (1 - alpha)``, and
    uses those relative weights for FedWeIT's active-mask shared parameters.
    """

    alpha: float = 1.0 / 11.0
    enhanced_clients: int = 1
    slack_loss_metric: str = "adv_ce_loss"

    def _slack_weight_info(self, client_results: Sequence[TrainResult]) -> tuple[Dict[str, float], List[str]]:
        if not client_results:
            return {}, []

        alpha = min(max(float(self.alpha), 0.0), 1.0 - 1e-6)
        sample_counts = {
            result.client_id: float(max(int(result.num_samples), 0))
            for result in client_results
        }
        total_samples = sum(sample_counts.values())
        if total_samples <= 0.0:
            sample_counts = {result.client_id: 1.0 for result in client_results}
            total_samples = float(len(client_results))

        scored = []
        for result in client_results:
            sample_weight = sample_counts[result.client_id] / total_samples
            loss = _safe_metric(result, self.slack_loss_metric)
            scored.append((sample_weight * loss, result.client_id))

        num_clients = len(client_results)
        max_enhanced = max(1, num_clients // 2) if num_clients > 1 else 1
        enhanced = min(max(1, int(self.enhanced_clients)), max_enhanced, num_clients)
        selected = {
            client_id
            for _, client_id in sorted(scored, key=lambda item: (item[0], item[1]))[:enhanced]
        }

        boost = (1.0 + alpha) / max(1.0 - alpha, 1e-6)
        weights = {
            result.client_id: sample_counts[result.client_id] * (boost if result.client_id in selected else 1.0)
            for result in client_results
        }
        return weights, sorted(selected)

    def aggregate(self, client_results: List[TrainResult]) -> AggregationResult:
        if not client_results:
            return AggregationResult(global_state={}, metrics={"num_clients": 0.0, "total_samples": 0.0})

        slack_weights, selected_clients = self._slack_weight_info(client_results)
        shared_states: List[tuple[StateDict, float]] = []
        hard_mask_states: List[tuple[StateDict, float]] = []
        buffer_states: List[tuple[StateDict, float]] = []
        metric_sums: Dict[str, float] = {}
        metric_weights: Dict[str, float] = {}
        for result in client_results:
            weight = float(slack_weights.get(result.client_id, 1.0))
            shared_state = result.payload.get("shared_state", {})
            hard_mask_state = result.payload.get("hard_mask_state", {})
            if isinstance(shared_state, Mapping) and shared_state:
                shared_states.append((shared_state, weight))
                hard_mask_states.append((hard_mask_state if isinstance(hard_mask_state, Mapping) else {}, weight))
            buffer_state = result.payload.get("buffer_state", {})
            if isinstance(buffer_state, Mapping) and buffer_state:
                buffer_states.append((buffer_state, weight))
            for metric_name, metric_value in result.metrics.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + float(metric_value)
                metric_weights[metric_name] = metric_weights.get(metric_name, 0.0) + 1.0

        if not shared_states:
            return AggregationResult(
                global_state={},
                metrics={"num_clients": float(len(client_results)), "total_samples": 0.0},
            )

        averaged_state: Dict[str, torch.Tensor] = {}
        first_shared_state = shared_states[0][0]
        for name in first_shared_state.keys():
            weighted_values = [
                state[name].detach().float() * weight
                for state, weight in shared_states
                if name in state
            ]
            weighted_masks = [
                mask_state[name].detach().float() * weight
                for mask_state, weight in hard_mask_states
                if isinstance(mask_state, Mapping) and name in mask_state
            ]
            if not weighted_values:
                continue
            numerator = torch.stack(weighted_values, dim=0).sum(dim=0)
            if len(weighted_masks) == len(weighted_values):
                denominator = torch.stack(weighted_masks, dim=0).sum(dim=0)
                averaged_state[name] = torch.where(
                    denominator > 0,
                    numerator / torch.clamp(denominator, min=1e-12),
                    torch.zeros_like(numerator),
                )
            else:
                total_weight = sum(
                    weight
                    for state, weight in shared_states
                    if name in state
                )
                averaged_state[name] = numerator / max(total_weight, 1e-12)

        if buffer_states:
            first_buffer_state = buffer_states[0][0]
            for name, first_value in first_buffer_state.items():
                if not isinstance(first_value, torch.Tensor):
                    continue
                values = [
                    (state[name], weight)
                    for state, weight in buffer_states
                    if name in state and isinstance(state[name], torch.Tensor)
                ]
                if not values:
                    continue
                if first_value.is_floating_point():
                    total_weight = sum(weight for _, weight in values)
                    averaged_state[name] = (
                        torch.stack([value.detach().float() * weight for value, weight in values], dim=0).sum(dim=0)
                        / max(total_weight, 1e-12)
                    )
                else:
                    averaged_state[name] = values[0][0].detach().clone()

        averaged_metrics = {
            f"{self.metric_prefix}{metric_name}": metric_sums[metric_name] / metric_weights[metric_name]
            for metric_name in metric_sums
            if metric_weights.get(metric_name, 0.0) > 0.0
        }
        averaged_metrics["num_clients"] = float(len(shared_states))
        averaged_metrics["total_samples"] = float(sum(max(int(result.num_samples), 0) for result in client_results))
        averaged_metrics["sfat_alpha"] = float(min(max(float(self.alpha), 0.0), 1.0 - 1e-6))
        averaged_metrics["sfat_enhanced_clients"] = float(len(selected_clients))
        averaged_metrics["sfat_weight_min"] = float(min(slack_weights.values())) if slack_weights else 0.0
        averaged_metrics["sfat_weight_max"] = float(max(slack_weights.values())) if slack_weights else 0.0

        return AggregationResult(
            global_state=averaged_state,
            metrics=averaged_metrics,
            metadata={
                "aggregator": "fedweit_sfat",
                "aggregation": "sfat_slack_active_mask_mean_B_mask",
                "sfat_alpha": float(min(max(float(self.alpha), 0.0), 1.0 - 1e-6)),
                "sfat_enhanced_clients": float(len(selected_clients)),
                "sfat_selected_clients": selected_clients,
                "sfat_client_weights": dict(slack_weights),
                "sfat_loss_metric": self.slack_loss_metric,
            },
        )


@dataclass
class FedWeITSFATServer(FedWeITServer):
    alpha: float = 1.0 / 11.0
    enhanced_clients: int = 1
    slack_loss_metric: str = "adv_ce_loss"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.aggregator = FedWeITSFATAggregator(
            alpha=self.alpha,
            enhanced_clients=self.enhanced_clients,
            slack_loss_metric=self.slack_loss_metric,
        )
