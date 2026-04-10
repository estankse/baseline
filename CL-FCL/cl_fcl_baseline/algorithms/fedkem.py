from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Iterable, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ..contracts import AggregationResult, ClientContext, MetricDict, StateDict, TrainResult
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
    alpha: float = 0.5  # weight for distillation loss / mutual KL


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
    mutual_learning: bool = True

    def _distill_step(self, teacher: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss between student and teacher."""
        assert self.distill_student is not None
        assert self.distill_config is not None
        temperature = float(self.distill_config.temperature)
        teacher.to(inputs.device)
        with torch.no_grad():
            teacher_logits = teacher(inputs)

        student_logits = self.distill_student(inputs)
        # KL divergence on softened probabilities
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
        return kd_loss

    # def _run_distillation(self, global_state: StateDict) -> MetricDict:
    #     """Train the local student using the global model as teacher."""
    #     if self.distill_student is None or self.distill_config is None:
    #         return {}
    #
    #     # Global state corresponds to the student network in FedKEMF.
    #     self.distill_student.load_state_dict(global_state, strict=True)
    #     teacher = self.trainer.model
    #     teacher.eval()
    #
    #     if self.distill_optimizer is None:
    #         self.distill_optimizer = torch.optim.SGD(self.distill_student.parameters(), lr=0.001)
    #
    #     self.distill_student.to(self.trainer.device)
    #     total_loss = 0.0
    #     total_examples = 0
    #
    #     for _ in range(int(self.distill_config.epochs)):
    #         for inputs, _targets in self.train_loader:
    #             inputs = move_to_device(inputs, self.trainer.device)
    #             self.distill_optimizer.zero_grad()
    #             loss = self._distill_step(teacher, inputs)
    #             loss.backward()
    #             self.distill_optimizer.step()
    #             batch_size = int(inputs.shape[0])
    #             total_examples += batch_size
    #             total_loss += float(loss.detach().item()) * batch_size
    #
    #     if total_examples == 0:
    #         return {"distill_loss": 0.0}
    #     return {"distill_loss": total_loss / total_examples}

    def _run_mutual_learning(self, global_state: StateDict) -> MetricDict:
        """Run deep mutual learning between main model and student on local data.

        Each model is optimized with its own supervised loss plus a KL term
        to match the other model's softened predictions.
        """
        if self.distill_student is None or self.distill_config is None:
            return {}

        temperature = float(self.distill_config.temperature)
        alpha = float(self.distill_config.alpha)

        # Initialize the student from the global state (server exchanges student params).
        self.distill_student.load_state_dict(global_state, strict=True)
        self.trainer.model.to(self.trainer.device)
        self.distill_student.to(self.trainer.device)

        self.trainer.model.train()
        self.distill_student.train()

        # if self.distill_optimizer is None:
        self.distill_optimizer = torch.optim.SGD(self.distill_student.parameters(), lr=0.01)

        total_loss = 0.0
        total_examples = 0

        for _ in range(int(self.distill_config.epochs)):
            for inputs, targets in self.train_loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)

                # Forward pass for both models.
                logits_main = self.trainer.model(inputs)
                logits_student = self.distill_student(inputs)

                # Supervised losses.
                loss_main = F.cross_entropy(logits_main, targets)
                loss_student = F.cross_entropy(logits_student, targets)

                # Mutual KL losses (each model treats the other as fixed).
                kl_main = F.kl_div(
                    F.log_softmax(logits_main / temperature, dim=1),
                    F.softmax(logits_student.detach() / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                kl_student = F.kl_div(
                    F.log_softmax(logits_student / temperature, dim=1),
                    F.softmax(logits_main.detach() / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                total_main = (1-alpha) * loss_main + alpha * kl_main
                total_student = (1-alpha) * loss_student + alpha * kl_student

                self.trainer.optimizer.zero_grad()
                self.distill_optimizer.zero_grad()
                total_main.backward(retain_graph=True)
                total_student.backward()
                self.trainer.optimizer.step()
                self.distill_optimizer.step()

                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(total_main.detach().item()) * batch_size
        # 评估 student 在本地数据上的准确率
        self.distill_student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                preds = self.distill_student(inputs).argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        print(f"[{self.client_id}] student local acc: {correct / total:.4f}")
        if total_examples == 0:
            return {"mutual_loss": 0.0}
        return {"mutual_loss": total_loss / total_examples}

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        """Run optional distillation, then normal local training."""
        del context
        metrics: MetricDict = {}
        if self.distill_student is not None:
            if self.distill_config is not None:
                if self.mutual_learning:
                    metrics.update(self._run_mutual_learning(global_state))
                # else:
                #     metrics.update(self._run_distillation(global_state))
                #     # self.trainer.model.load_state_dict(global_state, strict=True)
                #     for _ in range(self.epochs):
                #         metrics = self.trainer.train_epoch(self.train_loader)

        else:
            self.trainer.model.load_state_dict(global_state, strict=True)
            for _ in range(self.epochs):
                metrics = self.trainer.train_epoch(self.train_loader)

        payload = {
            "model_state": detach_state_dict(self.trainer.model.state_dict()),
        }
        if self.distill_student is not None:
            # In FedKEMF we exchange student parameters with the server.
            payload["model_state"] = detach_state_dict(self.distill_student.state_dict())
            payload["distill_state"] = payload["model_state"]

        return TrainResult(
            client_id=self.client_id,
            num_samples=len(self.train_loader.dataset),
            metrics=metrics,
            payload=payload,
        )




@dataclass
class FedKEMServerAggregator:
    """Server update for FedKEM using ensemble distillation on public data."""

    model: nn.Module
    public_loader: DataLoader
    lr: float = 0.1
    temperature: float = 1.0
    epochs: int = 1
    device: str | torch.device = "cpu"
    ensemble: str = "max"
    _optimizer: torch.optim.Optimizer = None

    def _ensemble_logits(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        if not logits_list:
            raise ValueError("No client logits available for ensemble.")
        stacked = torch.stack(logits_list, dim=0)
        if self.ensemble == "max":
            return torch.max(stacked, dim=0).values
        if self.ensemble == "mean":
            return torch.mean(stacked, dim=0)
        raise ValueError(f"Unsupported ensemble strategy: {self.ensemble}")

    def aggregate(self, client_results: List[TrainResult]) -> AggregationResult:
        if not client_results:
            return AggregationResult(
                global_state={},
                metrics={"num_clients": 0.0, "total_samples": 0.0},
            )

        client_states: List[StateDict] = []
        total_samples = 0
        for result in client_results:
            payload_state = result.payload.get("model_state", {})
            if isinstance(payload_state, dict) and payload_state:
                client_states.append(payload_state)
                total_samples += max(int(result.num_samples), 0)

        if not client_states:
            return AggregationResult(
                global_state={},
                metrics={"num_clients": float(len(client_results)), "total_samples": 0.0},
            )

        client_models: List[nn.Module] = []
        for state in client_states:
            client_model = copy.deepcopy(self.model)
            client_model.load_state_dict(state, strict=True)
            client_model.to(self.device)
            client_model.eval()
            client_models.append(client_model)

        self.model.to(self.device)
        self.model.train()
        if self._optimizer is None:
            self._optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=float(self.lr),
                momentum=0.9,
                weight_decay=5e-4
            )
            # self._optimizer = torch.optim.Adam(self.model.parameters(),lr=float(self.lr))
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.lr), momentum=0.9, weight_decay=5e-4)

        total_loss = 0.0
        total_examples = 0
        temperature = float(self.temperature)

        for _ in range(int(self.epochs)):
            for inputs, _targets in self.public_loader:
                inputs = move_to_device(inputs, self.device)
                with torch.no_grad():
                    logits_list = [model(inputs) for model in client_models]
                    ensemble_logits = self._ensemble_logits(logits_list)


                student_logits = self.model(inputs)
                loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(ensemble_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                batch_size = int(inputs.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size

        avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
        return AggregationResult(
            global_state=detach_state_dict(self.model.state_dict()),
            metrics={
                "server_distill_loss": avg_loss,
                "num_clients": float(len(client_states)),
                "total_samples": float(total_samples),
            },
            metadata={"aggregator": "fedkem_server"},
        )
