from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from ..contracts import ClientContext, MetricDict, StateDict, TrainResult
from .trainer import BaseTrainer
from .utils import detach_state_dict


@dataclass
class FederatedClient:
    client_id: str
    trainer: BaseTrainer
    train_loader: DataLoader
    epochs: int = 1

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        del context
        self.trainer.model.load_state_dict(global_state, strict=True)
        metrics: MetricDict = {}
        for _ in range(self.epochs):
            metrics = self.trainer.train_epoch(self.train_loader)
        payload = {"model_state": detach_state_dict(self.trainer.model.state_dict())}
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(self.train_loader.dataset),
            metrics=metrics,
            payload=payload,
        )
