from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ...contracts import AggregationResult, ClientContext, TaskDefinition, TrainResult
from ...trainers.utils import move_to_device
from ._common import (
    PartialStateServer,
    clone_state,
    mean_metrics,
    task_loader,
    weighted_average_state,
)


@dataclass
class GradientSubspace:
    """Layer-wise orthonormal bases used by FedProTIP Eqs. (7)--(10)."""

    bases: dict[str, torch.Tensor] = field(default_factory=dict)

    @staticmethod
    def extract(
        activation: torch.Tensor,
        threshold: float,
        previous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        matrix = activation.float()
        if previous is not None and previous.numel() > 0:
            basis = previous.to(matrix)
            matrix = matrix - basis @ (basis.t() @ matrix)
        if matrix.numel() == 0 or torch.linalg.vector_norm(matrix) <= 1e-12:
            return matrix.new_empty((matrix.shape[0], 0))
        left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        mass = singular_values / singular_values.sum().clamp_min(1e-12)
        rank = int((mass.cumsum(0) < float(threshold)).sum().item()) + 1
        return left[:, : min(rank, left.shape[1])]

    def append(self, name: str, local_basis: torch.Tensor) -> None:
        if local_basis.numel() == 0:
            return
        previous = self.bases.get(name)
        if previous is None or previous.numel() == 0:
            merged = local_basis
        else:
            previous = previous.to(local_basis)
            residual = local_basis - previous @ (previous.t() @ local_basis)
            merged = torch.cat((previous, residual), dim=1)
        q, _ = torch.linalg.qr(merged, mode="reduced")
        self.bases[name] = q[:, : min(q.shape[0], q.shape[1])].detach().cpu()

    def project_gradient(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        basis = self.bases.get(name)
        if basis is None or basis.numel() == 0 or gradient.ndim < 2:
            return gradient
        flattened = gradient.reshape(gradient.shape[0], -1)
        if basis.shape[0] != flattened.shape[1]:
            return gradient
        basis = basis.to(flattened)
        projected = flattened - (flattened @ basis) @ basis.t()
        return projected.view_as(gradient)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return clone_state(self.bases)

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.bases = clone_state(state)


def _activation_modules(model: nn.Module) -> dict[str, nn.Module]:
    return {
        f"{name}.weight" if name else "weight": module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    }


@dataclass
class FedProTIPClient:
    client_id: str
    model: nn.Module
    optimizer: torch.optim.Optimizer
    task_loaders: Mapping[str, DataLoader]
    device: str | torch.device = "cpu"
    epochs: int = 1
    threshold: float = 0.7
    activation_batches: int = 5
    activation_columns: int = 512
    subspace: GradientSubspace = field(default_factory=GradientSubspace)
    task_bases: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    task_references: dict[str, torch.Tensor] = field(default_factory=dict)
    final_activations: dict[str, torch.Tensor] = field(default_factory=dict)

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.to(self.device)
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for inputs, targets in loader:
            inputs = move_to_device(inputs, self.device)
            targets = move_to_device(targets, self.device)
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    parameter.grad.copy_(self.subspace.project_gradient(name, parameter.grad))
            self.optimizer.step()
            count = int(targets.shape[0])
            total_examples += count
            total_loss += float(loss.detach()) * count
            total_correct += int((logits.argmax(1) == targets).sum())
        return {
            "loss": total_loss / max(1, total_examples),
            "accuracy": total_correct / max(1, total_examples),
        }

    def fit(self, global_state: Mapping[str, object], context: ClientContext) -> TrainResult:
        self.model.load_state_dict(clone_state(global_state), strict=True)
        loader = task_loader(self.task_loaders, context.task_id)
        metrics: dict[str, float] = {}
        for _ in range(max(int(self.epochs), 1)):
            metrics = self._train_epoch(loader)
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics=metrics,
            payload={"model_state": clone_state(self.model.state_dict())},
        )

    def collect_activations(self, task_id: str) -> dict[str, torch.Tensor]:
        loader = task_loader(self.task_loaders, task_id)
        modules = _activation_modules(self.model)
        collected: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
        handles = []

        def _hook(name: str, module: nn.Module):
            def capture(_module: nn.Module, args: tuple[object, ...]) -> None:
                if not args or not isinstance(args[0], torch.Tensor):
                    return
                inputs = args[0].detach()
                if isinstance(module, nn.Conv2d):
                    unfolded = F.unfold(
                        inputs,
                        kernel_size=module.kernel_size,
                        dilation=module.dilation,
                        padding=module.padding,
                        stride=module.stride,
                    )
                    matrix = unfolded.permute(1, 0, 2).flatten(1)
                else:
                    matrix = inputs.reshape(-1, inputs.shape[-1]).t()
                if matrix.shape[1] > self.activation_columns > 0:
                    indices = torch.linspace(
                        0,
                        matrix.shape[1] - 1,
                        self.activation_columns,
                        device=matrix.device,
                    ).long()
                    matrix = matrix[:, indices]
                collected[name].append(matrix.cpu())

            return capture

        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(_hook(name, module)))
        self.model.to(self.device).eval()
        try:
            with torch.no_grad():
                for batch_idx, (inputs, _) in enumerate(loader):
                    self.model(move_to_device(inputs, self.device))
                    if self.activation_batches > 0 and batch_idx + 1 >= self.activation_batches:
                        break
        finally:
            for handle in handles:
                handle.remove()
        return {
            name: torch.cat(matrices, dim=1)
            for name, matrices in collected.items()
            if matrices
        }

    def extract_task_basis(self, task_id: str, task_index: int) -> dict[str, torch.Tensor]:
        local: dict[str, torch.Tensor] = {}
        threshold = min(float(self.threshold) + 0.001 * int(task_index), 1.0)
        activations = self.collect_activations(task_id)
        for name, activation in activations.items():
            basis = GradientSubspace.extract(
                activation,
                threshold=threshold,
                previous=self.subspace.bases.get(name),
            )
            if basis.numel() > 0:
                local[name] = basis.cpu()
        self.task_bases[task_id] = clone_state(local)
        if local:
            final_name = next(reversed(local))
            activation = activations[final_name]
            self.final_activations[task_id] = activation.cpu()
        return local

    def update_references(
        self,
        task_bases: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> None:
        task_ids = list(task_bases)
        for source_task, activation in self.final_activations.items():
            relevance = []
            for target_task in task_ids:
                candidates = task_bases[target_task]
                basis = next(reversed(candidates.values()), None)
                if basis is None or basis.shape[0] != activation.shape[0]:
                    relevance.append(activation.new_zeros(()))
                else:
                    basis = basis.to(activation)
                    relevance.append(
                        torch.linalg.vector_norm(basis @ (basis.t() @ activation))
                    )
            self.task_references[source_task] = torch.stack(relevance).cpu()


@dataclass
class FedProTIPServer(PartialStateServer):
    model: nn.Module = field(default_factory=nn.Identity)
    subspace: GradientSubspace = field(default_factory=GradientSubspace)
    task_bases: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    current_task_index: int = 0

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        global_state = clone_state(self.model.state_dict())
        results: list[TrainResult] = []
        for client in self.selected_clients(round_idx):
            assert isinstance(client, FedProTIPClient)
            client.subspace.load_state_dict(self.subspace.state_dict())
            results.append(client.fit(global_state, self.client_context(client, round_idx, task_id)))
        if not results:
            return self.empty_result(round_idx, task_id)
        averaged = weighted_average_state(
            [(result.payload["model_state"], result.num_samples) for result in results]
        )
        self.model.load_state_dict(averaged, strict=True)
        return AggregationResult(
            global_state=clone_state(self.model.state_dict()),
            metrics=mean_metrics(results),
            metadata={"round_idx": round_idx, "task_id": task_id, "algorithm": "fedprotip"},
        )

    def on_client_tasks_end(self, tasks: Mapping[str, TaskDefinition]) -> None:
        task_local_bases: dict[str, list[dict[str, torch.Tensor]]] = {}
        for client in self.clients:
            assert isinstance(client, FedProTIPClient)
            task = tasks.get(client.client_id)
            if task is None:
                continue
            local = client.extract_task_basis(task.task_id, self.current_task_index)
            task_local_bases.setdefault(task.task_id, []).append(local)
        for task_id, client_bases in task_local_bases.items():
            aggregated = GradientSubspace()
            for local in client_bases:
                for name, basis in local.items():
                    aggregated.append(name, basis)
                    self.subspace.append(name, basis)
            self.task_bases[task_id] = aggregated.state_dict()
        for client in self.clients:
            assert isinstance(client, FedProTIPClient)
            client.update_references(self.task_bases)
        self.current_task_index += 1

    def predict_task(self, activation: torch.Tensor) -> str:
        """FedProTIP Eqs. (11)--(12), with majority vote over client references."""

        if not self.task_bases:
            raise RuntimeError("No FedProTIP task bases have been extracted.")
        relevance = []
        for bases in self.task_bases.values():
            basis = next(reversed(bases.values()))
            if basis.shape[0] != activation.shape[0]:
                relevance.append(activation.new_zeros(()))
            else:
                basis = basis.to(activation)
                relevance.append(torch.linalg.vector_norm(basis @ (basis.t() @ activation)))
        query = torch.stack(relevance).cpu()
        votes: list[str] = []
        for client in self.clients:
            assert isinstance(client, FedProTIPClient)
            compatible = {
                task_id: reference
                for task_id, reference in client.task_references.items()
                if reference.numel() == query.numel()
            }
            if not compatible:
                continue
            votes.append(
                max(
                    compatible,
                    key=lambda task_id: float(
                        F.cosine_similarity(
                            query.unsqueeze(0),
                            compatible[task_id].unsqueeze(0),
                        )
                    ),
                )
            )
        if not votes:
            return next(iter(self.task_bases))
        return max(set(votes), key=votes.count)
