from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..contracts import ClientContext, MetricDict, StateDict, TrainResult
from ..trainers.trainer import build_default_loss
from ..trainers.utils import detach_state_dict, move_to_device


def _extract_global_params(global_state: StateDict, device: torch.device | str) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    for name, value in global_state.items():
        if isinstance(value, torch.Tensor):
            params[name] = value.to(device)
    return params


@dataclass
class FedProxTrainer:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    proximal_mu: float = 0.0
    loss_fn: nn.Module | None = None
    device: str | torch.device = "cpu"

    def train_epoch(self, dataloader: DataLoader, global_state: StateDict) -> MetricDict:
        self.model.to(self.device)
        self.model.train()
        loss_fn = self.loss_fn or build_default_loss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        global_params = _extract_global_params(global_state, self.device)
        mu = float(self.proximal_mu)

        for inputs, targets in dataloader:
            inputs = move_to_device(inputs, self.device)
            targets = move_to_device(targets, self.device)
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = loss_fn(logits, targets)
            if mu > 0.0 and global_params:
                prox_loss = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_params:
                        prox_loss = prox_loss + torch.sum((param - global_params[name]) ** 2)
                loss = loss + 0.5 * mu * prox_loss
            loss.backward()
            self.optimizer.step()
            batch_size = int(targets.shape[0])
            total_examples += batch_size
            total_loss += float(loss.detach().item()) * batch_size
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == targets).sum().item())
        if total_examples == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }


@dataclass
class FedProxClient:
    client_id: str
    trainer: FedProxTrainer
    train_loader: DataLoader
    epochs: int = 1

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        del context
        self.trainer.model.load_state_dict(global_state, strict=True)
        metrics: MetricDict = {}
        for _ in range(self.epochs):
            metrics = self.trainer.train_epoch(self.train_loader, global_state)
        payload = {"model_state": detach_state_dict(self.trainer.model.state_dict())}
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(self.train_loader.dataset),
            metrics=metrics,
            payload=payload,
        )
