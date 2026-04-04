from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ..contracts import ClientContext, MetricDict, StateDict, TrainResult
from ..trainers.trainer import BaseTrainer
from ..trainers.utils import detach_state_dict, move_to_device


class DistillMLP(nn.Module):
    """A tiny student network for client-side distillation.

    This intentionally stays small to keep edge training lightweight.
    """

    def __init__(self, input_shape: Iterable[int], hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        input_dim = 1
        for dim in input_shape:
            input_dim *= int(dim)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class DistillationConfig:
    """Hyper-parameters for local distillation."""

    epochs: int = 1
    temperature: float = 2.0
    alpha: float = 1.0  # weight for distillation loss


@dataclass
class FedKEMClient:
    """Client that optionally trains a local distillation network before FL updates.

    Typical flow:
    1) Load the global model as teacher.
    2) Train a small student on local data using KL divergence to teacher logits.
    3) Run standard local training on the main model and return updates.
    """

    client_id: str
    trainer: BaseTrainer
    train_loader: DataLoader
    distill_student: nn.Module | None = None
    distill_config: DistillationConfig | None = None
    distill_optimizer: torch.optim.Optimizer | None = None
    epochs: int = 1

    def _distill_step(self, teacher: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss between student and teacher."""
        assert self.distill_student is not None
        assert self.distill_config is not None
        temperature = float(self.distill_config.temperature)

        with torch.no_grad():
            teacher_logits = teacher(inputs)

        student_logits = self.distill_student(inputs)
        # KL divergence on softened probabilities
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
        return kd_loss

    def _run_distillation(self, global_state: StateDict) -> MetricDict:
        """Train the local student using the global model as teacher."""
        if self.distill_student is None or self.distill_config is None:
            return {}

        teacher = self.trainer.model
        teacher.load_state_dict(global_state, strict=True)
        teacher.eval()

        if self.distill_optimizer is None:
            self.distill_optimizer = torch.optim.SGD(self.distill_student.parameters(), lr=0.01)

        self.distill_student.to(self.trainer.device)
        total_loss = 0.0
        total_examples = 0

        for _ in range(int(self.distill_config.epochs)):
            for inputs, _targets in self.train_loader:
                inputs = move_to_device(inputs, self.trainer.device)
                self.distill_optimizer.zero_grad()
                loss = self._distill_step(teacher, inputs)
                loss.backward()
                self.distill_optimizer.step()
                batch_size = int(inputs.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size

        if total_examples == 0:
            return {"distill_loss": 0.0}
        return {"distill_loss": total_loss / total_examples}

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        """Run optional distillation, then normal local training."""
        del context
        distill_metrics = self._run_distillation(global_state)

        # Standard FL local update
        self.trainer.model.load_state_dict(global_state, strict=True)
        metrics: MetricDict = {}
        for _ in range(self.epochs):
            metrics = self.trainer.train_epoch(self.train_loader)
        metrics.update(distill_metrics)

        payload = {
            "model_state": detach_state_dict(self.trainer.model.state_dict()),
        }
        if self.distill_student is not None:
            payload["distill_state"] = detach_state_dict(self.distill_student.state_dict())

        return TrainResult(
            client_id=self.client_id,
            num_samples=len(self.train_loader.dataset),
            metrics=metrics,
            payload=payload,
        )
