from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ...contracts import AggregationResult, ClientContext, TrainResult
from ...trainers.utils import move_to_device
from ._common import (
    PartialStateServer,
    clone_state,
    gradients_to_vector,
    integrate_gradients,
    mean_metrics,
    task_loader,
    vector_to_gradients,
    weighted_average_state,
)


def _body_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.") and parameter.requires_grad
    ]


def _body_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("classifier.")
    }


@dataclass
class FedViTClient:
    """Paper FedViT client: private head, sample knowledge and gradient integration."""

    client_id: str
    model: nn.Module
    task_loaders: Mapping[str, DataLoader]
    device: str | torch.device = "cpu"
    lr: float = 5e-4
    weight_decay: float = 0.05
    epochs: int = 6
    head_epochs: int = 1
    post_aggregation_epochs: int = 1
    knowledge_ratio: float = 0.1
    signature_k: int = 10
    integrator_steps: int = 100
    memories: dict[str, TensorDataset] = field(default_factory=dict)
    task_heads: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    _last_pre_aggregation_gradient: torch.Tensor | None = None

    def _load_body(self, state: Mapping[str, object]) -> None:
        self.model.load_state_dict(clone_state(state), strict=False)

    def _set_task_head(self, task_id: str) -> None:
        classifier = getattr(self.model, "classifier", None)
        if not isinstance(classifier, nn.Module):
            raise TypeError("FedViT requires a model with a classifier module.")
        if task_id in self.task_heads:
            classifier.load_state_dict(self.task_heads[task_id], strict=True)

    def _save_task_head(self, task_id: str) -> None:
        classifier = getattr(self.model, "classifier")
        self.task_heads[task_id] = clone_state(classifier.state_dict())

    def _gradient_on_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        parameters: Sequence[nn.Parameter],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        return gradients_to_vector(parameters), loss.detach()

    def _memory_gradient(
        self,
        memory: TensorDataset,
        parameters: Sequence[nn.Parameter],
    ) -> torch.Tensor:
        inputs, targets = memory.tensors
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        gradient, _ = self._gradient_on_batch(inputs, targets, parameters)
        return gradient

    def _integrated_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, int]:
        body_parameters = _body_parameters(self.model)
        current, loss = self._gradient_on_batch(inputs, targets, body_parameters)
        old_gradients = [
            self._memory_gradient(memory, body_parameters) for memory in self.memories.values()
        ]
        if len(old_gradients) > self.signature_k > 0:
            distances = torch.stack(
                [torch.linalg.vector_norm(gradient - current) for gradient in old_gradients]
            )
            indices = distances.topk(self.signature_k).indices.tolist()
            old_gradients = [old_gradients[index] for index in indices]
        integrated = integrate_gradients(
            current,
            old_gradients,
            steps=self.integrator_steps,
        )
        self.model.zero_grad(set_to_none=True)
        vector_to_gradients(integrated, body_parameters)
        optimizer.step()
        self._last_pre_aggregation_gradient = integrated.detach().cpu()
        with torch.no_grad():
            correct = int((self.model(inputs).argmax(1) == targets).sum())
        return float(loss), correct

    def fit(self, global_state: Mapping[str, object], context: ClientContext) -> TrainResult:
        if context.task_id is None:
            raise ValueError("FedViT requires a task ID during training.")
        task_id = context.task_id
        self._load_body(global_state)
        self._set_task_head(task_id)
        loader = task_loader(self.task_loaders, task_id)
        self.model.to(self.device)

        classifier = getattr(self.model, "classifier")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in classifier.parameters():
            parameter.requires_grad = True
        head_optimizer = torch.optim.AdamW(
            classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        for _ in range(max(int(self.head_epochs), 0)):
            self.model.train()
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                head_optimizer.zero_grad()
                loss = F.cross_entropy(self.model(inputs), targets)
                loss.backward()
                head_optimizer.step()

        for parameter in self.model.parameters():
            parameter.requires_grad = True
        for parameter in classifier.parameters():
            parameter.requires_grad = False
        body_optimizer = torch.optim.AdamW(
            _body_parameters(self.model), lr=self.lr, weight_decay=self.weight_decay
        )
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for _ in range(max(int(self.epochs), 1)):
            self.model.train()
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                loss, correct = self._integrated_step(
                    inputs,
                    targets,
                    body_optimizer,
                )
                count = int(targets.shape[0])
                total_loss += loss * count
                total_correct += correct
                total_examples += count
        for parameter in classifier.parameters():
            parameter.requires_grad = True
        self._save_task_head(task_id)
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics={
                "loss": total_loss / max(1, total_examples),
                "accuracy": total_correct / max(1, total_examples),
                "memory_tasks": float(len(self.memories)),
            },
            payload={"body_state": _body_state(self.model)},
        )

    def post_aggregate(self, body_state: Mapping[str, object], context: ClientContext) -> None:
        if self.post_aggregation_epochs <= 0 or context.task_id is None:
            self._load_body(body_state)
            return
        before = self._last_pre_aggregation_gradient
        self._load_body(body_state)
        self._set_task_head(context.task_id)
        loader = task_loader(self.task_loaders, context.task_id)
        body_parameters = _body_parameters(self.model)
        optimizer = torch.optim.AdamW(body_parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.model.to(self.device).train()
        for _ in range(self.post_aggregation_epochs):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                after, _ = self._gradient_on_batch(inputs, targets, body_parameters)
                constraints = [] if before is None else [before.to(after)]
                integrated = integrate_gradients(after, constraints, steps=self.integrator_steps)
                self.model.zero_grad(set_to_none=True)
                vector_to_gradients(integrated, body_parameters)
                optimizer.step()

    def extract_knowledge(self, task_id: str) -> None:
        loader = task_loader(self.task_loaders, task_id)
        by_class: dict[int, list[tuple[float, torch.Tensor, torch.Tensor]]] = {}
        self.model.to(self.device).eval()
        with torch.no_grad():
            for inputs, targets in loader:
                device_inputs = move_to_device(inputs, self.device)
                device_targets = move_to_device(targets, self.device)
                losses = F.cross_entropy(
                    self.model(device_inputs),
                    device_targets,
                    reduction="none",
                )
                for input_item, target, loss in zip(inputs, targets, losses.cpu()):
                    by_class.setdefault(int(target), []).append(
                        (float(loss), input_item.detach().cpu(), target.detach().cpu())
                    )
        retained_inputs: list[torch.Tensor] = []
        retained_targets: list[torch.Tensor] = []
        for examples in by_class.values():
            examples.sort(key=lambda item: item[0])
            count = max(1, int(len(examples) * float(self.knowledge_ratio)))
            for _, inputs, target in examples[:count]:
                retained_inputs.append(inputs)
                retained_targets.append(target)
        if retained_inputs:
            self.memories[task_id] = TensorDataset(
                torch.stack(retained_inputs),
                torch.stack(retained_targets).long(),
            )


@dataclass
class FedViTServer(PartialStateServer):
    model: nn.Module = field(default_factory=nn.Identity)

    def get_body_state(self) -> dict[str, torch.Tensor]:
        return _body_state(self.model)

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        body_state = self.get_body_state()
        results: list[TrainResult] = []
        selected = self.selected_clients(round_idx)
        for client in selected:
            assert isinstance(client, FedViTClient)
            results.append(client.fit(body_state, self.client_context(client, round_idx, task_id)))
        if not results:
            return self.empty_result(round_idx, task_id)
        averaged = weighted_average_state(
            [(result.payload["body_state"], result.num_samples) for result in results]
        )
        self.model.load_state_dict(averaged, strict=False)
        for client in selected:
            assert isinstance(client, FedViTClient)
            client.post_aggregate(
                averaged,
                self.client_context(client, round_idx, task_id),
            )
        return AggregationResult(
            global_state=self.get_body_state(),
            metrics=mean_metrics(results),
            metadata={"round_idx": round_idx, "task_id": task_id, "algorithm": "fedvit"},
        )

    def on_client_tasks_end(self, tasks: Mapping[str, object]) -> None:
        for client in self.clients:
            assert isinstance(client, FedViTClient)
            task = tasks.get(client.client_id)
            task_id = getattr(task, "task_id", None)
            if task_id is not None:
                client.extract_knowledge(str(task_id))
