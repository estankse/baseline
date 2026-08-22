from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ...contracts import AggregationResult, ClientContext, TrainResult
from ...models.clip_vlm import CLIPVisionLanguageModel, FeaturePromptAdapter
from ...models.fcl_models import PromptedVisionTransformer, SparseAdapterGate
from ...trainers.utils import move_to_device
from ._common import PartialStateServer, clone_state, mean_metrics, task_loader, weighted_average_state


def _kmeans(features: torch.Tensor, clusters: int, steps: int = 20) -> torch.Tensor:
    """Deterministic torch K-Means used for MoAFCL's first-round assignment."""
    if features.shape[0] < clusters:
        return torch.arange(features.shape[0], device=features.device) % clusters
    centers = features[torch.linspace(0, features.shape[0] - 1, clusters).long()].clone()
    labels = torch.full((features.shape[0],), -1, dtype=torch.long, device=features.device)
    for _ in range(max(int(steps), 1)):
        new_labels = torch.cdist(features, centers).argmin(dim=1)
        if torch.equal(labels, new_labels):
            break
        labels = new_labels
        for index in range(clusters):
            selected = features[labels == index]
            if selected.numel() > 0:
                centers[index] = selected.mean(dim=0)
    return labels


def _moafcl_clip_logits(
    model: CLIPVisionLanguageModel,
    adapter: nn.Module,
    images: torch.Tensor,
    extract_layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equations (3)-(5): intermediate feature -> context -> CLIP logits."""
    with torch.no_grad():
        intermediate = model.visual.extract_layer_features(images, extract_layer)
    contexts = adapter(intermediate)
    if contexts.ndim != 3:
        raise ValueError("MoAFCL adapter must return [batch, prompt_length, text_width].")
    context = contexts.mean(dim=0)
    return model.zero_shot_logits(images, context=context), intermediate


@dataclass
class MoAFCLClient:
    client_id: str
    backbone: PromptedVisionTransformer
    adapter: nn.Module
    task_loaders: Mapping[str, DataLoader]
    device: str | torch.device = "cpu"
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.02
    summary_batches: int = 10
    assigned_adapter: int = 0
    routing_target: torch.Tensor | None = None
    clip_model: CLIPVisionLanguageModel | None = None
    extract_layer: int = 5

    def feature_samples(self, task_id: str) -> torch.Tensor:
        loader = task_loader(self.task_loaders, task_id)
        samples: list[torch.Tensor] = []
        model: nn.Module = self.clip_model if self.clip_model is not None else self.backbone
        model.to(self.device).eval()
        with torch.no_grad():
            for batch_index, (inputs, _targets) in enumerate(loader):
                inputs = move_to_device(inputs, self.device)
                if self.clip_model is not None:
                    features = self.clip_model.visual.extract_layer_features(inputs, self.extract_layer)
                else:
                    encoded = self.backbone.encode(inputs)
                    assert isinstance(encoded, torch.Tensor)
                    features = encoded
                samples.append(features.detach().cpu())
                if self.summary_batches > 0 and batch_index + 1 >= self.summary_batches:
                    break
        return torch.cat(samples) if samples else torch.zeros(1, self.backbone.embed_dim)

    def summarize(self, task_id: str) -> torch.Tensor:
        return self.feature_samples(task_id).mean(dim=0)

    def _fit_clip(self, adapter_state: Mapping[str, object], context: ClientContext) -> TrainResult:
        assert self.clip_model is not None
        self.adapter.load_state_dict(clone_state(adapter_state), strict=True)
        loader = task_loader(self.task_loaders, context.task_id)
        self.clip_model.to(self.device).eval()
        self.adapter.to(self.device).train()
        self.clip_model.freeze_backbone()
        optimizer = torch.optim.Adam(
            self.adapter.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-6,
            weight_decay=self.weight_decay,
        )
        total_loss = 0.0
        total_correct = total_examples = 0
        for _ in range(max(int(self.epochs), 1)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                logits, _features = _moafcl_clip_logits(
                    self.clip_model, self.adapter, inputs, self.extract_layer
                )
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count = int(targets.shape[0])
                total_examples += count
                total_loss += float(loss.detach()) * count
                total_correct += int((logits.argmax(1) == targets).sum())
        features = self.feature_samples(context.task_id or next(iter(self.task_loaders)))
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics={"loss": total_loss / max(1, total_examples), "accuracy": total_correct / max(1, total_examples), "adapter": float(self.assigned_adapter)},
            payload={"adapter_state": clone_state(self.adapter.state_dict()), "feature_summary": features.mean(0), "feature_samples": features, "adapter_index": self.assigned_adapter, "routing_target": (self.routing_target.clone() if self.routing_target is not None else None)},
        )

    def fit(
        self,
        adapter_state: Mapping[str, object],
        head_state: Mapping[str, object] | None,
        context: ClientContext,
    ) -> TrainResult:
        if self.clip_model is not None:
            return self._fit_clip(adapter_state, context)
        if head_state is None:
            raise ValueError("The ViT compatibility path requires a classifier head state.")
        self.adapter.load_state_dict(clone_state(adapter_state), strict=True)
        self.backbone.classifier.load_state_dict(clone_state(head_state), strict=True)
        loader = task_loader(self.task_loaders, context.task_id)
        self.backbone.to(self.device)
        self.adapter.to(self.device)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.Adam(
            [*self.adapter.parameters(), *self.backbone.classifier.parameters()],
            lr=self.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=self.weight_decay,
        )
        adapters: list[nn.Module | None] = [None] * self.backbone.depth
        adapters[-1] = self.adapter
        total_loss = 0.0
        total_correct = total_examples = 0
        for _ in range(max(int(self.epochs), 1)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                features = self.backbone.encode(inputs, adapters=adapters)
                assert isinstance(features, torch.Tensor)
                logits = self.backbone.classifier(features)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count = int(targets.shape[0])
                total_examples += count
                total_loss += float(loss.detach()) * count
                total_correct += int((logits.argmax(1) == targets).sum())
        samples = self.feature_samples(context.task_id or next(iter(self.task_loaders)))
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics={"loss": total_loss / max(1, total_examples), "accuracy": total_correct / max(1, total_examples)},
            payload={"adapter_state": clone_state(self.adapter.state_dict()), "head_state": clone_state(self.backbone.classifier.state_dict()), "feature_summary": samples.mean(0), "feature_samples": samples, "adapter_index": self.assigned_adapter, "routing_target": (self.routing_target.clone() if self.routing_target is not None else None)},
        )


@dataclass
class MoAFCLServer(PartialStateServer):
    adapter_bank: nn.ModuleList = field(default_factory=nn.ModuleList)
    gate: SparseAdapterGate = field(default_factory=lambda: None)  # type: ignore[arg-type]
    head: nn.Module = field(default_factory=nn.Identity)
    device: str | torch.device = "cpu"
    gate_lr: float = 1.5
    gate_epochs: int = 500
    top_k: int = 1
    gate_temperature: float = 5.0
    dp_epsilon: float = 100.0
    initialized: bool = False

    def _assign(
        self,
        clients: Sequence[MoAFCLClient],
        task_id: str,
        round_idx: int | None = None,
    ) -> None:
        task_ids = [
            (
                self.client_context(client, int(round_idx), task_id).task_id
                if round_idx is not None else task_id
            )
            for client in clients
        ]
        client_features = [
            client.feature_samples(actual_task or task_id)
            for client, actual_task in zip(clients, task_ids)
        ]
        if not self.initialized:
            labels = _kmeans(torch.stack([features.mean(0) for features in client_features]), len(self.adapter_bank))
            distributions = [
                F.one_hot(label, num_classes=len(self.adapter_bank)).float()
                for label in labels
            ]
            self.initialized = True
        else:
            labels_list = []
            distributions = []
            self.gate.to(self.device).eval()
            with torch.no_grad():
                for features in client_features:
                    logits = self.gate(features.to(self.device))["logits"]
                    frequency = torch.bincount(logits.argmax(1), minlength=len(self.adapter_bank))
                    labels_list.append(frequency.argmax())
                    distributions.append(frequency.float() / frequency.sum().clamp_min(1))
            labels = torch.stack(labels_list).cpu()
        for client, label, distribution in zip(clients, labels, distributions):
            client.assigned_adapter = int(label)
            client.routing_target = distribution.detach().cpu()

    def _train_gate(self, results: Sequence[TrainResult]) -> None:
        if not results or self.gate_epochs <= 0:
            return
        feature_rows: list[torch.Tensor] = []
        target_rows: list[torch.Tensor] = []
        for result in results:
            features = result.payload["feature_samples"]
            assert isinstance(features, torch.Tensor)
            target = result.payload.get("routing_target")
            if not isinstance(target, torch.Tensor):
                target = torch.zeros(len(self.adapter_bank))
                target[int(result.payload["adapter_index"])] = 1.0
            feature_rows.append(features)
            target_rows.append(target.expand(features.shape[0], -1))
        features = torch.cat(feature_rows).to(self.device)
        targets = torch.cat(target_rows).to(self.device)
        if self.dp_epsilon > 0:
            scale = 1.0 / float(self.dp_epsilon)
            features = features + torch.distributions.Laplace(
                features.new_zeros(()), features.new_full((), scale)
            ).sample(features.shape)
        self.gate.to(self.device).train()
        optimizer = torch.optim.SGD(self.gate.parameters(), lr=self.gate_lr)
        for _ in range(int(self.gate_epochs)):
            output = self.gate(features, top_k=self.top_k, temperature=self.gate_temperature)
            loss = F.binary_cross_entropy_with_logits(output["logits"], targets) + 2.0 * output["load_balance_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict_clip(self, model: CLIPVisionLanguageModel, inputs: torch.Tensor, extract_layer: int) -> torch.Tensor:
        model.to(self.device).eval()
        self.adapter_bank.to(self.device).eval()
        self.gate.to(self.device).eval()
        intermediate = model.visual.extract_layer_features(inputs, extract_layer)
        routing = self.gate(intermediate, top_k=self.top_k, temperature=self.gate_temperature)["gates"]
        image_features = model.encode_image(inputs)
        logits = []
        for adapter in self.adapter_bank:
            context = adapter(intermediate).mean(0)
            text_features = model.encode_text(context=context)
            logits.append(model.logit_scale.exp().clamp(max=100.0) * image_features @ text_features.t())
        return (torch.stack(logits, dim=1) * routing.unsqueeze(-1)).sum(1)

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        clients = [client for client in self.selected_clients(round_idx) if isinstance(client, MoAFCLClient)]
        self._assign(clients, task_id, round_idx)
        clip_mode = any(client.clip_model is not None for client in clients)
        results = [
            client.fit(self.adapter_bank[client.assigned_adapter].state_dict(), None if clip_mode else self.head.state_dict(), self.client_context(client, round_idx, task_id))
            for client in clients
        ]
        if not results:
            return self.empty_result(round_idx, task_id)
        for index, adapter in enumerate(self.adapter_bank):
            grouped = [result for result in results if int(result.payload["adapter_index"]) == index]
            if grouped:
                averaged = weighted_average_state([(result.payload["adapter_state"], 1) for result in grouped])
                adapter.load_state_dict(averaged, strict=True)
        if not clip_mode:
            head_state = weighted_average_state([(result.payload["head_state"], result.num_samples) for result in results])
            self.head.load_state_dict(head_state, strict=True)
        self._train_gate(results)
        state: dict[str, torch.Tensor] = {}
        for index, adapter in enumerate(self.adapter_bank):
            state.update({f"adapters.{index}.{name}": value for name, value in clone_state(adapter.state_dict()).items()})
        state.update({f"gate.{name}": value for name, value in clone_state(self.gate.state_dict()).items()})
        if not clip_mode:
            state.update({f"head.{name}": value for name, value in clone_state(self.head.state_dict()).items()})
        return AggregationResult(
            global_state=state,
            metrics=mean_metrics(results),
            metadata={"round_idx": round_idx, "task_id": task_id, "algorithm": "moafcl", "clip_vlm": clip_mode},
        )


__all__ = ["FeaturePromptAdapter", "MoAFCLClient", "MoAFCLServer", "_kmeans"]
