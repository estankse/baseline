from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..contracts import MetricDict
from .utils import move_to_device


def build_default_loss() -> nn.Module:
    return nn.CrossEntropyLoss()


@dataclass
class BaseTrainer:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module | None = None
    device: str | torch.device = "cpu"

    def train_epoch(self, dataloader: DataLoader) -> MetricDict:
        self.model.to(self.device)
        self.model.train()
        loss_fn = self.loss_fn or build_default_loss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for inputs, targets in dataloader:
            inputs = move_to_device(inputs, self.device)
            targets = move_to_device(targets, self.device)
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = loss_fn(logits, targets)
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

    def evaluate(self, dataloader: DataLoader) -> MetricDict:
        self.model.to(self.device)
        self.model.eval()
        loss_fn = self.loss_fn or build_default_loss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                logits = self.model(inputs)
                loss = loss_fn(logits, targets)
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
