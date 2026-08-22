from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ...contracts import AggregationResult, ClientContext, TaskDefinition, TrainResult
from ...models.fcl_models import BottleneckAdapter, PromptedVisionTransformer
from ...trainers.utils import move_to_device
from ._common import PartialStateServer, clone_state, mean_metrics, task_loader


def _module_state(prefix: str, module: nn.Module) -> dict[str, torch.Tensor]:
    return {f"{prefix}.{name}": value for name, value in clone_state(module.state_dict()).items()}


def _strip_state(state: Mapping[str, object], prefix: str) -> dict[str, torch.Tensor]:
    marker = f"{prefix}."
    return {
        name.removeprefix(marker): value
        for name, value in clone_state(state).items()
        if name.startswith(marker)
    }


@dataclass
class MultiFCLClient:
    client_id: str
    backbone: PromptedVisionTransformer
    adapters: nn.ModuleList
    expert_heads: nn.ModuleList
    expert_layers: Sequence[int]
    task_loaders: Mapping[str, DataLoader]
    task_classes: Mapping[str, Sequence[int]]
    device: str | torch.device = "cpu"
    epochs: int = 5
    lr_adapter: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    semantic_features: torch.Tensor | None = None
    visual_projection: torch.Tensor | None = None
    task_index: int = 0
    seen_tasks: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        client_id: str,
        backbone: PromptedVisionTransformer,
        task_loaders: Mapping[str, DataLoader],
        task_classes: Mapping[str, Sequence[int]],
        adapter_dim: int = 64,
        adapter_dropout: float = 0.1,
        num_experts: int = 4,
        **kwargs: object,
    ) -> "MultiFCLClient":
        # The paper uses prototype/cosine classifiers without a bias term.
        if backbone.classifier.bias is not None:
            replacement = nn.Linear(
                backbone.embed_dim,
                backbone.classifier.out_features,
                bias=False,
            )
            with torch.no_grad():
                replacement.weight.copy_(backbone.classifier.weight)
            backbone.classifier = replacement
        adapters = nn.ModuleList(
            [
                BottleneckAdapter(
                    backbone.embed_dim,
                    adapter_dim,
                    dropout=adapter_dropout,
                    scale=0.1,
                )
                for _ in range(backbone.depth)
            ]
        )
        count = min(max(int(num_experts), 1), backbone.depth)
        expert_layers = [
            max(0, min(backbone.depth - 1, round((index + 1) * backbone.depth / count) - 1))
            for index in range(count)
        ]
        expert_heads = nn.ModuleList(
            [
                nn.Linear(backbone.embed_dim, backbone.classifier.out_features, bias=False)
                for _ in expert_layers
            ]
        )
        return cls(
            client_id=client_id,
            backbone=backbone,
            adapters=adapters,
            expert_heads=expert_heads,
            expert_layers=expert_layers,
            task_loaders=task_loaders,
            task_classes=task_classes,
            **kwargs,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        state = _module_state("adapters", self.adapters)
        state.update(_module_state("head", self.backbone.classifier))
        state.update(_module_state("experts", self.expert_heads))
        return state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.adapters.load_state_dict(_strip_state(state, "adapters"), strict=True)
        self.backbone.classifier.load_state_dict(_strip_state(state, "head"), strict=True)
        self.expert_heads.load_state_dict(_strip_state(state, "experts"), strict=True)

    def _forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output = self.backbone.encode(
            inputs,
            adapters=list(self.adapters),
            return_intermediates=True,
        )
        assert isinstance(output, tuple)
        final_features, intermediates = output
        final_logits = self.backbone.classifier(final_features)
        expert_logits = [
            head(intermediates[layer])
            for head, layer in zip(self.expert_heads, self.expert_layers)
        ]
        return final_logits, expert_logits

    @staticmethod
    def multi_expert_loss(
        final_logits: torch.Tensor,
        expert_logits: Sequence[torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Equations (8), (10)-(13): every scale learns the labels while the
        # final expert distils the teachers using inverse-KL confidence.
        final_probabilities = F.softmax(final_logits, dim=1)
        inverse_kl = []
        for logits in expert_logits:
            probabilities = F.softmax(logits.detach(), dim=1)
            divergence = F.kl_div(
                probabilities.clamp_min(1e-8).log(),
                final_probabilities,
                reduction="none",
            ).sum(dim=1)
            inverse_kl.append(1.0 / divergence.clamp_min(1e-8))
        weights = torch.stack(inverse_kl, dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True)
        expert_loss = final_logits.new_zeros(())
        distillation_loss = final_logits.new_zeros(())
        for index, logits in enumerate(expert_logits):
            ce = F.cross_entropy(logits, targets, reduction="none")
            expert_loss = expert_loss + (weights[:, index] * ce).mean()
            teacher = F.softmax(logits.detach(), dim=1)
            kd = F.kl_div(
                F.log_softmax(final_logits, dim=1),
                teacher,
                reduction="none",
            ).sum(dim=1)
            distillation_loss = distillation_loss + (weights[:, index] * kd).mean()
        return F.cross_entropy(final_logits, targets) + expert_loss + distillation_loss

    def fit(self, global_state: Mapping[str, object], context: ClientContext) -> TrainResult:
        self.load_state_dict(global_state)
        loader = task_loader(self.task_loaders, context.task_id)
        self.backbone.to(self.device)
        self.adapters.to(self.device)
        self.expert_heads.to(self.device)
        # New-class prototypes are a task-boundary initialization.  Repeating
        # it in every communication round would overwrite the classifier rows
        # learned and aggregated in the preceding round.
        if (
            self.task_index > 0
            and context.round_idx == 0
            and context.task_id is not None
        ):
            self._initialize_new_prototypes(loader, context.task_id)
        adapter_trainable = self.task_index == 0
        for parameter in self.adapters.parameters():
            parameter.requires_grad = adapter_trainable
        groups = [
            {
                "params": list(self.backbone.classifier.parameters())
                + list(self.expert_heads.parameters()),
                "lr": self.lr_head,
            }
        ]
        if adapter_trainable:
            groups.append({"params": self.adapters.parameters(), "lr": self.lr_adapter})
        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        class_counts: dict[int, int] = {}
        for _ in range(max(int(self.epochs), 1)):
            self.backbone.train()
            self.adapters.train()
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                optimizer.zero_grad()
                final_logits, expert_logits = self._forward(inputs)
                loss = self.multi_expert_loss(final_logits, expert_logits, targets)
                loss.backward()
                optimizer.step()
                count = int(targets.shape[0])
                total_examples += count
                total_loss += float(loss.detach()) * count
                total_correct += int((final_logits.argmax(1) == targets).sum())
                for label, label_count in zip(*torch.unique(targets, return_counts=True)):
                    key = int(label)
                    class_counts[key] = class_counts.get(key, 0) + int(label_count)
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics={
                "loss": total_loss / max(1, total_examples),
                "accuracy": total_correct / max(1, total_examples),
            },
            payload={"trainable_state": self.state_dict(), "class_counts": class_counts},
        )

    @torch.no_grad()
    def _initialize_new_prototypes(self, loader: DataLoader, task_id: str) -> None:
        current_classes = [int(label) for label in self.task_classes.get(task_id, [])]
        previous_tasks = list(self.seen_tasks)
        old_classes = sorted(
            {
                int(label)
                for known_task in previous_tasks
                for label in self.task_classes[known_task]
            }
        )
        if not current_classes or not old_classes:
            return
        class_features: dict[int, list[torch.Tensor]] = {label: [] for label in current_classes}
        self.backbone.eval()
        self.adapters.eval()
        for inputs, targets in loader:
            inputs = move_to_device(inputs, self.device)
            features = self.backbone.encode(inputs, adapters=list(self.adapters))
            assert isinstance(features, torch.Tensor)
            for feature, target in zip(features, targets):
                label = int(target)
                if label in class_features:
                    class_features[label].append(feature)
        old_index = torch.tensor(old_classes, device=self.device)
        old_prototypes = self.backbone.classifier.weight.index_select(0, old_index)
        old_semantics = None
        if self.semantic_features is not None and self.visual_projection is not None:
            old_semantics = F.normalize(
                self.semantic_features.to(self.device).index_select(0, old_index), dim=1
            )
        for label, features in class_features.items():
            if not features:
                continue
            image_feature = torch.stack(features).mean(0)
            if old_semantics is not None:
                projected = image_feature @ self.visual_projection.to(self.device)
                attention = F.softmax(
                    old_semantics @ F.normalize(projected, dim=0), dim=0
                )
            else:
                attention = F.softmax(
                    F.normalize(old_prototypes, dim=1) @ F.normalize(image_feature, dim=0),
                    dim=0,
                )
            prototype = attention @ old_prototypes
            self.backbone.classifier.weight[label].copy_(prototype)
            for head in self.expert_heads:
                head.weight[label].copy_(prototype)

    def on_task_end(self, task_id: str) -> None:
        if task_id not in self.seen_tasks:
            self.seen_tasks.append(task_id)
        self.task_index += 1


@dataclass
class MultiFCLServer(PartialStateServer):
    global_state: dict[str, torch.Tensor] = field(default_factory=dict)
    task_index: int = 0

    def _aggregate(self, results: Sequence[TrainResult]) -> dict[str, torch.Tensor]:
        states = [clone_state(result.payload["trainable_state"]) for result in results]
        output = clone_state(self.global_state or states[0])
        total_samples = sum(result.num_samples for result in results)
        for name in output:
            if name.startswith("adapters."):
                if self.task_index == 0:
                    reference = output[name]
                    output[name] = sum(
                        state[name].to(reference)
                        * (result.num_samples / max(1, total_samples))
                        for state, result in zip(states, results)
                    )
                continue
            if not name.endswith("weight") or output[name].ndim != 2:
                continue
            rows = output[name].clone()
            for label in range(rows.shape[0]):
                denominator = sum(
                    int(result.payload.get("class_counts", {}).get(label, 0))
                    for result in results
                )
                if denominator <= 0:
                    continue
                rows[label] = sum(
                    state[name][label].to(rows)
                    * int(result.payload.get("class_counts", {}).get(label, 0))
                    / denominator
                    for state, result in zip(states, results)
                )
            output[name] = rows
        return output

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        results: list[TrainResult] = []
        for client in self.selected_clients(round_idx):
            assert isinstance(client, MultiFCLClient)
            if self.global_state:
                state = self.global_state
            else:
                state = client.state_dict()
            results.append(client.fit(state, self.client_context(client, round_idx, task_id)))
        if not results:
            return self.empty_result(round_idx, task_id)
        self.global_state = self._aggregate(results)
        return AggregationResult(
            global_state=clone_state(self.global_state),
            metrics=mean_metrics(results),
            metadata={"round_idx": round_idx, "task_id": task_id, "algorithm": "multifcl"},
        )

    def on_client_tasks_end(self, tasks: Mapping[str, TaskDefinition]) -> None:
        for client in self.clients:
            assert isinstance(client, MultiFCLClient)
            task = tasks.get(client.client_id)
            if task is not None:
                client.on_task_end(task.task_id)
        self.task_index += 1
