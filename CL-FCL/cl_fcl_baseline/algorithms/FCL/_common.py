from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ...contracts import AggregationResult, ClientContext, TrainResult


TensorState = dict[str, torch.Tensor]


def clone_state(state: Mapping[str, object]) -> TensorState:
    return {
        name: value.detach().clone()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def weighted_average_state(
    states: Sequence[tuple[Mapping[str, object], int]],
) -> TensorState:
    valid = [(clone_state(state), max(int(weight), 0)) for state, weight in states if weight > 0]
    valid = [(state, weight) for state, weight in valid if state]
    if not valid:
        return {}
    common_names = set.intersection(*(set(state) for state, _ in valid))
    total_weight = sum(weight for _, weight in valid)
    averaged: TensorState = {}
    for name in sorted(common_names):
        reference = valid[0][0][name]
        if reference.is_floating_point() or reference.is_complex():
            value = sum(
                state[name].to(reference.device) * float(weight)
                for state, weight in valid
            ) / float(total_weight)
        else:
            value = reference.detach().clone()
        averaged[name] = value
    return averaged


def parameters_to_vector(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    vectors = [parameter.reshape(-1) for parameter in parameters]
    if not vectors:
        return torch.empty(0)
    return torch.cat(vectors)


def gradients_to_vector(parameters: Sequence[nn.Parameter]) -> torch.Tensor:
    vectors = [
        torch.zeros_like(parameter).reshape(-1)
        if parameter.grad is None
        else parameter.grad.detach().reshape(-1)
        for parameter in parameters
    ]
    if not vectors:
        return torch.empty(0)
    return torch.cat(vectors)


def vector_to_gradients(vector: torch.Tensor, parameters: Sequence[nn.Parameter]) -> None:
    offset = 0
    for parameter in parameters:
        numel = parameter.numel()
        gradient = vector[offset : offset + numel].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = gradient.detach().clone()
        else:
            parameter.grad.copy_(gradient)
        offset += numel
    if offset != vector.numel():
        raise ValueError("Gradient vector length does not match the parameters.")


def integrate_gradients(
    current: torch.Tensor,
    constraints: Sequence[torch.Tensor],
    steps: int = 100,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Solve the GEM/FedViT dual QP and return a feasible integrated gradient."""

    if current.numel() == 0 or not constraints:
        return current
    matrix = torch.stack([gradient.to(current) for gradient in constraints])
    if torch.all(matrix @ current >= -1e-10):
        return current
    gram = matrix @ matrix.t()
    gram = gram + float(eps) * torch.eye(gram.shape[0], device=gram.device)
    linear = matrix @ current
    eigen_bound = torch.linalg.eigvalsh(gram).max().clamp_min(float(eps))
    learning_rate = 1.0 / eigen_bound
    dual = torch.zeros(matrix.shape[0], device=current.device, dtype=current.dtype)
    for _ in range(max(int(steps), 1)):
        dual = (dual - learning_rate * (gram @ dual + linear)).clamp_min(0.0)
    integrated = current + matrix.t() @ dual
    # Numerical cleanup keeps every constraint acute after a finite QP solve.
    for gradient in matrix:
        dot = torch.dot(integrated, gradient)
        if dot < 0:
            integrated = integrated - dot * gradient / gradient.square().sum().clamp_min(1e-12)
    return integrated


def task_loader(
    task_loaders: Mapping[str, DataLoader],
    task_id: str | None,
) -> DataLoader:
    if task_id is not None and task_id in task_loaders:
        return task_loaders[task_id]
    return next(iter(task_loaders.values()))


def mean_metrics(results: Sequence[TrainResult]) -> dict[str, float]:
    total = sum(max(int(result.num_samples), 0) for result in results)
    metrics: dict[str, float] = {}
    if total <= 0:
        return metrics
    names = {name for result in results for name in result.metrics}
    for name in names:
        metrics[f"client_{name}"] = sum(
            float(result.metrics.get(name, 0.0)) * max(int(result.num_samples), 0)
            for result in results
        ) / float(total)
    metrics["num_clients"] = float(len(results))
    metrics["total_samples"] = float(total)
    return metrics


@dataclass
class PartialStateServer:
    """Shared round orchestration for FCL methods that exchange partial states."""

    clients: Sequence[object]
    client_sample_ratio: float = 1.0
    client_task_ids: dict[str, str] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.client_sample_ratio) <= 1.0:
            raise ValueError("client_sample_ratio must be in (0, 1].")

    def set_client_task_ids(self, client_task_ids: Mapping[str, str]) -> None:
        self.client_task_ids = {
            str(client_id): str(task_id) for client_id, task_id in client_task_ids.items()
        }

    def selected_clients(self, round_idx: int) -> list[object]:
        clients = list(self.clients)
        if not clients or self.client_sample_ratio >= 1.0:
            return clients
        count = max(1, int(len(clients) * self.client_sample_ratio))
        rng = random.Random(int(self.seed) + int(round_idx))
        return rng.sample(clients, count)

    def client_context(
        self,
        client: object,
        round_idx: int,
        default_task_id: str,
    ) -> ClientContext:
        client_id = str(getattr(client, "client_id"))
        return ClientContext(
            client_id=client_id,
            round_idx=round_idx,
            task_id=self.client_task_ids.get(client_id, default_task_id),
        )

    def empty_result(self, round_idx: int, task_id: str) -> AggregationResult:
        return AggregationResult(
            global_state={},
            metrics={"num_clients": 0.0, "total_samples": 0.0},
            metadata={"round_idx": round_idx, "task_id": task_id},
        )
