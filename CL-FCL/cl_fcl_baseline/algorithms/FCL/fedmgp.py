from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ...contracts import AggregationResult, ClientContext, TrainResult
from ...models.fcl_models import PrefixPromptPool, PromptPool, PromptedVisionTransformer
from ...trainers.utils import move_to_device
from ._common import (
    PartialStateServer,
    clone_state,
    mean_metrics,
    task_loader,
    weighted_average_state,
)


def _pack_global(prompt: PromptPool, head: nn.Module) -> dict[str, torch.Tensor]:
    state = {f"prompt.{name}": value for name, value in clone_state(prompt.state_dict()).items()}
    state.update({f"head.{name}": value for name, value in clone_state(head.state_dict()).items()})
    return state


def _unpack_global(
    state: Mapping[str, object],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prompt = {
        name.removeprefix("prompt."): value
        for name, value in clone_state(state).items()
        if name.startswith("prompt.")
    }
    head = {
        name.removeprefix("head."): value
        for name, value in clone_state(state).items()
        if name.startswith("head.")
    }
    return prompt, head


@dataclass
class FedMGPClient:
    client_id: str
    backbone: PromptedVisionTransformer
    global_prompt: PromptPool
    local_prompt: PrefixPromptPool
    local_head: nn.Module
    task_loaders: Mapping[str, DataLoader]
    device: str | torch.device = "cpu"
    lr: float = 1e-3
    epochs: int = 5
    pull_constraint: float = 0.1

    def _query(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            query = self.backbone.encode(inputs)
        assert isinstance(query, torch.Tensor)
        return query

    def _global_forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._query(inputs)
        selected = self.global_prompt(query)
        features = self.backbone.encode(inputs, input_prompts=selected["prompts"])
        assert isinstance(features, torch.Tensor)
        return self.backbone.classifier(features), selected["pull_similarity"]

    def _local_forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._query(inputs)
        global_selected = self.global_prompt(query)
        coarse = self.backbone.encode(inputs, input_prompts=global_selected["prompts"])
        assert isinstance(coarse, torch.Tensor)
        prompt_indices = labels.unsqueeze(1) if labels is not None else None
        local_selected = self.local_prompt(coarse, indices=prompt_indices)
        features = self.backbone.encode(
            inputs,
            input_prompts=global_selected["prompts"],
            prefixes=local_selected["prefixes"],
        )
        assert isinstance(features, torch.Tensor)
        return self.local_head(features), local_selected["pull_similarity"]

    def fit(self, global_state: Mapping[str, object], context: ClientContext) -> TrainResult:
        prompt_state, head_state = _unpack_global(global_state)
        self.global_prompt.load_state_dict(prompt_state, strict=True)
        self.backbone.classifier.load_state_dict(head_state, strict=True)
        loader = task_loader(self.task_loaders, context.task_id)
        self.backbone.to(self.device)
        self.global_prompt.to(self.device)
        self.local_prompt.to(self.device)
        self.local_head.to(self.device)

        global_optimizer = torch.optim.Adam(
            list(self.global_prompt.parameters()) + list(self.backbone.classifier.parameters()),
            lr=self.lr,
            weight_decay=1e-3,
        )
        total_global_loss = 0.0
        total_examples = 0
        for _ in range(max(int(self.epochs), 1)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                global_optimizer.zero_grad()
                logits, pull = self._global_forward(inputs)
                loss = F.cross_entropy(logits, targets) - float(self.pull_constraint) * pull
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.global_prompt.parameters())
                    + list(self.backbone.classifier.parameters()),
                    1.0,
                )
                global_optimizer.step()
                total_global_loss += float(loss.detach()) * int(targets.shape[0])
                total_examples += int(targets.shape[0])

        for parameter in self.global_prompt.parameters():
            parameter.requires_grad = False
        local_optimizer = torch.optim.Adam(
            list(self.local_prompt.parameters()) + list(self.local_head.parameters()),
            lr=self.lr,
            weight_decay=1e-3,
        )
        total_local_loss = 0.0
        total_correct = 0
        local_examples = 0
        for _ in range(max(int(self.epochs), 1)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                local_optimizer.zero_grad()
                logits, pull = self._local_forward(inputs, targets)
                loss = F.cross_entropy(logits, targets) - float(self.pull_constraint) * pull
                loss.backward()
                local_optimizer.step()
                count = int(targets.shape[0])
                total_local_loss += float(loss.detach()) * count
                total_correct += int((logits.argmax(1) == targets).sum())
                local_examples += count
        for parameter in self.global_prompt.parameters():
            parameter.requires_grad = True
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics={
                "global_loss": total_global_loss / max(1, total_examples),
                "local_loss": total_local_loss / max(1, local_examples),
                "accuracy": total_correct / max(1, local_examples),
            },
            payload={"global_state": _pack_global(self.global_prompt, self.backbone.classifier)},
        )

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        logits, _ = self._local_forward(inputs.to(self.device))
        return logits


@dataclass
class FedMGPServer(PartialStateServer):
    backbone: PromptedVisionTransformer = field(default_factory=lambda: None)  # type: ignore[arg-type]
    global_prompt: PromptPool = field(default_factory=lambda: None)  # type: ignore[arg-type]
    public_loaders: Mapping[str, DataLoader] = field(default_factory=dict)
    fusion_epochs: int = 1
    fusion_lr: float = 1e-3
    device: str | torch.device = "cpu"

    def get_global_state(self) -> dict[str, torch.Tensor]:
        return _pack_global(self.global_prompt, self.backbone.classifier)

    def _selective_prompt_fusion(
        self,
        prompt_states: Sequence[Mapping[str, object]],
        loader: DataLoader,
    ) -> None:
        if len(prompt_states) <= 1 or self.fusion_epochs <= 0:
            return
        teachers = [copy.deepcopy(self.global_prompt).to(self.device) for _ in prompt_states[1:]]
        for teacher, state in zip(teachers, prompt_states[1:]):
            teacher.load_state_dict(clone_state(state), strict=True)
            teacher.eval()
        self.backbone.to(self.device).eval()
        self.global_prompt.to(self.device).train()
        optimizer = torch.optim.Adam(self.global_prompt.parameters(), lr=self.fusion_lr)
        for _ in range(self.fusion_epochs):
            for inputs, _ in loader:
                inputs = move_to_device(inputs, self.device)
                with torch.no_grad():
                    query = self.backbone.encode(inputs)
                assert isinstance(query, torch.Tensor)
                student_prompt = self.global_prompt(query)["prompts"]
                student = self.backbone.encode(inputs, input_prompts=student_prompt)
                assert isinstance(student, torch.Tensor)
                targets = []
                with torch.no_grad():
                    for teacher in teachers:
                        prompt = teacher(query)["prompts"]
                        output = self.backbone.encode(inputs, input_prompts=prompt)
                        assert isinstance(output, torch.Tensor)
                        targets.append(output)
                loss = torch.stack([F.mse_loss(student, target) for target in targets]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        state = self.get_global_state()
        results: list[TrainResult] = []
        for client in self.selected_clients(round_idx):
            assert isinstance(client, FedMGPClient)
            results.append(client.fit(state, self.client_context(client, round_idx, task_id)))
        if not results:
            return self.empty_result(round_idx, task_id)
        averaged = weighted_average_state(
            [(result.payload["global_state"], result.num_samples) for result in results]
        )
        prompt_state, head_state = _unpack_global(averaged)
        self.global_prompt.load_state_dict(prompt_state, strict=True)
        self.backbone.classifier.load_state_dict(head_state, strict=True)
        loader = self.public_loaders.get(task_id)
        if loader is not None:
            client_prompt_states = [
                _unpack_global(result.payload["global_state"])[0] for result in results
            ]
            self._selective_prompt_fusion(client_prompt_states, loader)
        return AggregationResult(
            global_state=self.get_global_state(),
            metrics=mean_metrics(results),
            metadata={"round_idx": round_idx, "task_id": task_id, "algorithm": "fedmgp"},
        )
