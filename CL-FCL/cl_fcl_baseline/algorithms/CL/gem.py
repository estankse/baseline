from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from ...contracts import MetricDict, TaskDefinition
from .base import (
    ContinualLearner,
    assign_flat_gradient,
    flatten_gradients,
    trainable_parameters,
)


def project_gradient(
    gradient: torch.Tensor,
    memory_gradients: torch.Tensor,
    margin: float = 0.5,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Solve GEM's dual QP and return the projected gradient.

    This follows the authors' ``project2cone2`` implementation: if rows of
    ``memory_gradients`` are old-task gradients M, solve
    ``min 1/2 v' M M' v + (M g)' v`` with ``v >= margin`` and recover
    ``g_tilde = g + v' M``.  The small convex dual is solved directly in
    PyTorch, avoiding the original project's optional ``quadprog`` binary.
    """
    if memory_gradients.ndim != 2:
        raise ValueError("memory_gradients must have shape [tasks, parameters].")
    if memory_gradients.shape[0] == 0:
        return gradient
    memories = memory_gradients.detach().cpu().double()
    current = gradient.detach().cpu().double().reshape(-1)
    quadratic = memories @ memories.T
    quadratic = 0.5 * (quadratic + quadratic.T)
    quadratic += torch.eye(quadratic.shape[0], dtype=torch.double) * float(eps)
    # quadprog's second argument in the reference code is ``-M g`` because
    # that solver defines the objective as 1/2 v'Pv - a'v.  SciPy uses the
    # usual additive linear term, hence it receives ``M g`` here.
    linear = memories @ current

    lower_bound = float(margin)
    dual = torch.full(
        (memories.shape[0],), max(lower_bound, 0.0), dtype=torch.double
    )
    accelerated = dual.clone()
    momentum = 1.0
    lipschitz = max(float(torch.linalg.eigvalsh(quadratic).max().item()), 1e-12)
    # FISTA with projection onto v >= gamma solves exactly the same bound-QP
    # as quadprog.  Epsilon makes P positive definite and convergence stable.
    for _ in range(10_000):
        updated = torch.clamp(
            accelerated - (quadratic @ accelerated + linear) / lipschitz,
            min=lower_bound,
        )
        if float(torch.max(torch.abs(updated - dual)).item()) < 1e-10:
            dual = updated
            break
        next_momentum = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5)
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - dual)
        dual = updated
        momentum = next_momentum

    projected = current + dual @ memories
    return projected.to(device=gradient.device, dtype=gradient.dtype).view_as(gradient)


@dataclass
class GEMLearner(ContinualLearner):
    """Gradient Episodic Memory (Lopez-Paz and Ranzato, 2017)."""

    memory_size: int = 256
    memory_strength: float = 0.5
    qp_eps: float = 1e-3
    memory: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    memory_positions: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.memory_size) <= 0:
            raise ValueError("memory_size must be positive.")
        if float(self.memory_strength) < 0.0:
            raise ValueError("memory_strength must be non-negative.")

    def _on_task_start(self, task: TaskDefinition) -> None:
        self.memory.setdefault(task.task_id, [])
        self.memory_positions.setdefault(task.task_id, 0)

    def _update_memory(self, task_id: str, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        task_memory = self.memory.setdefault(task_id, [])
        position = self.memory_positions.setdefault(task_id, 0)
        # The reference GEM code uses a fixed-size cyclic episodic buffer for
        # every task and updates it online before computing the constraints.
        for sample, target in zip(inputs.detach().cpu(), targets.detach().cpu()):
            item = (sample.clone(), target.clone())
            if len(task_memory) < int(self.memory_size):
                task_memory.append(item)
            else:
                task_memory[position] = item
            position = (position + 1) % int(self.memory_size)
        self.memory_positions[task_id] = position

    def _memory_batch(self, task_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        samples = self.memory[task_id]
        inputs = torch.stack([sample for sample, _target in samples]).to(self.device)
        targets = torch.stack([target for _sample, target in samples]).long().to(self.device)
        return inputs, targets

    def observe(self, inputs: torch.Tensor, targets: torch.Tensor, task_id: str) -> MetricDict:
        self._update_memory(task_id, inputs, targets)
        parameters = trainable_parameters(self.model)
        past_task_ids = [
            previous_task_id
            for previous_task_id in self.task_order
            if previous_task_id != task_id and self.memory.get(previous_task_id)
        ]
        old_gradients: List[torch.Tensor] = []
        for previous_task_id in past_task_ids:
            self.optimizer.zero_grad()
            memory_inputs, memory_targets = self._memory_batch(previous_task_id)
            memory_logits = self.model(memory_inputs)
            memory_loss = self.restricted_cross_entropy(
                memory_logits, memory_targets, previous_task_id
            )
            memory_loss.backward()
            old_gradients.append(flatten_gradients(parameters).clone())

        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = self.restricted_cross_entropy(logits, targets, task_id)
        loss.backward()
        gradient = flatten_gradients(parameters)
        violations = 0
        projected = False
        if old_gradients:
            memories = torch.stack(old_gradients)
            dot_products = memories @ gradient
            violations = int((dot_products < 0.0).sum().item())
            if violations:
                gradient = project_gradient(
                    gradient,
                    memories,
                    margin=float(self.memory_strength),
                    eps=float(self.qp_eps),
                )
                assign_flat_gradient(parameters, gradient)
                projected = True
        self.optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "constraint_violations": float(violations),
            "gradient_projected": float(projected),
        }
