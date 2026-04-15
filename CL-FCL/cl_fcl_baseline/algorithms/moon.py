

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..contracts import ClientContext, MetricDict, StateDict, TrainResult
from ..trainers.trainer import build_default_loss
from ..trainers.utils import detach_state_dict, move_to_device


def _extract_representation(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    # Try to use a model-specific penultimate representation for MOON contrastive learning.
    if hasattr(model, "network"):
        network = getattr(model, "network")
        if isinstance(network, nn.Sequential) and len(network) >= 2:
            return network[:-1](inputs)

    if hasattr(model, "features") and hasattr(model, "classifier"):
        features = getattr(model, "features")(inputs)
        if hasattr(model, "avgpool"):
            features = getattr(model, "avgpool")(features)
        features = torch.flatten(features, start_dim=1)
        if hasattr(model, "fc_features"):
            features = getattr(model, "fc_features")(features)
        return features

    if all(hasattr(model, name) for name in ["conv1", "bn1", "relu", "layer1", "layer2", "layer3", "avgpool", "fc"]):
        out = model.conv1(inputs)
        out = model.bn1(out)
        out = model.relu(out)
        out = model.layer1(out)
        out = model.layer2(out)
        out = model.layer3(out)
        if hasattr(model, "layer4"):
            out = model.layer4(out)
        out = model.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    # Fallback: use logits as representation.
    return model(inputs)


@dataclass
class MoonTrainer:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    temperature: float = 0.5
    moon_mu: float = 1.0
    loss_fn: nn.Module | None = None
    device: str | torch.device = "cpu"

    def train_epoch(
        self,
        dataloader: DataLoader,
        global_model: nn.Module,
        previous_model: nn.Module,
    ) -> MetricDict:
        self.model.to(self.device)
        self.model.train()
        global_model.to(self.device)
        global_model.eval()
        previous_model.to(self.device)
        previous_model.eval()

        loss_fn = self.loss_fn or build_default_loss()
        tau = max(float(self.temperature), 1e-12)
        mu = float(self.moon_mu)

        total_loss = 0.0
        total_sup_loss = 0.0
        total_contrastive_loss = 0.0
        total_correct = 0
        total_examples = 0

        for inputs, targets in dataloader:
            inputs = move_to_device(inputs, self.device)
            targets = move_to_device(targets, self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            sup_loss = loss_fn(logits, targets)

            z_local = _extract_representation(self.model, inputs)
            with torch.no_grad():
                z_global = _extract_representation(global_model, inputs)
                z_prev = _extract_representation(previous_model, inputs)

            sim_pos = F.cosine_similarity(z_local, z_global, dim=1)
            sim_neg = F.cosine_similarity(z_local, z_prev, dim=1)
            contrastive_logits = torch.stack([sim_pos, sim_neg], dim=1) / tau
            contrastive_targets = torch.zeros(inputs.shape[0], dtype=torch.long, device=self.device)
            contrastive_loss = F.cross_entropy(contrastive_logits, contrastive_targets)

            loss = sup_loss + mu * contrastive_loss
            loss.backward()
            self.optimizer.step()

            batch_size = int(targets.shape[0])
            total_examples += batch_size
            total_loss += float(loss.detach().item()) * batch_size
            total_sup_loss += float(sup_loss.detach().item()) * batch_size
            total_contrastive_loss += float(contrastive_loss.detach().item()) * batch_size
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == targets).sum().item())

        if total_examples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "sup_loss": 0.0, "contrastive_loss": 0.0}

        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
            "sup_loss": total_sup_loss / total_examples,
            "contrastive_loss": total_contrastive_loss / total_examples,
        }


@dataclass
class MoonClient:
    client_id: str
    trainer: MoonTrainer
    train_loader: DataLoader
    epochs: int = 1
    previous_local_state: StateDict | None = field(default=None)

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        del context
        self.trainer.model.load_state_dict(global_state, strict=True)

        global_model = copy.deepcopy(self.trainer.model)
        global_model.load_state_dict(global_state, strict=True)

        if self.previous_local_state is None:
            self.previous_local_state = detach_state_dict(global_state)

        previous_model = copy.deepcopy(self.trainer.model)
        previous_model.load_state_dict(self.previous_local_state, strict=True)

        metrics: MetricDict = {}
        for _ in range(int(self.epochs)):
            metrics = self.trainer.train_epoch(
                self.train_loader,
                global_model=global_model,
                previous_model=previous_model,
            )

        local_state = detach_state_dict(self.trainer.model.state_dict())
        self.previous_local_state = local_state
        payload = {"model_state": local_state}

        return TrainResult(
            client_id=self.client_id,
            num_samples=len(self.train_loader.dataset),
            metrics=metrics,
            payload=payload,
        )
