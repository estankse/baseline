from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ...contracts import AggregationResult, ClientContext, TaskDefinition, TrainResult
from ...models.fcl_models import PromptPool, PromptedVisionTransformer
from ...trainers.utils import move_to_device
from ._common import PartialStateServer, clone_state, mean_metrics, task_loader, weighted_average_state


def dual_distillation_loss(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    class_correlation: torch.Tensor,
    temperature: float = 3.0,
) -> torch.Tensor:
    """Powder Eq. (5): class-weighted not-true distillation."""

    classes = logits.shape[1]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets.unsqueeze(1), False)
    student = logits[mask].view(-1, classes - 1) / float(temperature)
    teacher = teacher_logits[mask].view(-1, classes - 1) / float(temperature)
    teacher_probabilities = F.softmax(teacher, dim=1)
    per_sample = -(teacher_probabilities * F.log_softmax(student, dim=1)).sum(dim=1)
    row_scores = class_correlation.to(logits).sum(dim=1)
    beta = F.softmax(row_scores[targets], dim=0) * targets.shape[0]
    return (beta * per_sample).mean() * float(temperature) ** 2


def _powder_prompt_selection(
    pool: PromptPool,
    query: torch.Tensor,
    transferred_states: Sequence[Mapping[str, object]] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """CODA-style differentiable prompt generation used by Powder Eq. (1)-(2).

    ``PromptPool.forward`` performs L2P-style hard top-k selection.  Powder is
    based on CODAPrompt instead: every prompt in the communicated local pool is
    combined using its query-key cosine similarity.  Only ``pool`` is trainable;
    the additional first-step aggregated task pools are transferred knowledge.
    """

    similarities = [pool.similarity(query)]
    prompts = [pool.prompts]
    for raw_state in transferred_states:
        state = clone_state(raw_state)
        keys = state["keys"].to(query)
        attention = state["attention"].to(query)
        similarities.append(
            F.cosine_similarity(
                query.unsqueeze(1) * attention.unsqueeze(0),
                keys.unsqueeze(0),
                dim=-1,
                eps=1e-12,
            )
        )
        prompts.append(state["prompts"].to(query))
    similarity = torch.cat(similarities, dim=1)
    prompt_bank = torch.cat(prompts, dim=0)
    generated = torch.einsum("bm,mld->bld", similarity, prompt_bank)
    return generated, similarity


def _prompt_state_similarities(
    query: torch.Tensor,
    states: Sequence[Mapping[str, object]],
) -> torch.Tensor:
    """Eq. (6) weights over a canonical full global prompt pool."""

    similarities: list[torch.Tensor] = []
    for raw_state in states:
        state = clone_state(raw_state)
        keys = state["keys"].to(query)
        attention = state["attention"].to(query)
        similarities.append(
            F.cosine_similarity(
                query.unsqueeze(1) * attention.unsqueeze(0),
                keys.unsqueeze(0),
                dim=-1,
                eps=1e-12,
            )
        )
    if not similarities:
        raise ValueError("Powder correlation estimation requires a global prompt pool.")
    return torch.cat(similarities, dim=1)


def _task_view(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the task-specific classification-head view from Algorithm 1."""

    classes = torch.as_tensor(class_ids, device=logits.device, dtype=torch.long)
    if classes.numel() == 0:
        classes = torch.arange(logits.shape[1], device=logits.device)
    lookup = torch.full(
        (logits.shape[1],), -1, device=logits.device, dtype=torch.long
    )
    lookup[classes] = torch.arange(classes.numel(), device=logits.device)
    local_targets = lookup[targets]
    if bool((local_targets < 0).any()):
        raise ValueError("Powder loader contains a label outside the current task.")
    return logits.index_select(1, classes), local_targets, classes


def _average_prompt_states(
    prompt_states: Sequence[Mapping[str, object]],
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    normalized = weights / weights.sum().clamp_min(1e-12)
    states = [clone_state(state) for state in prompt_states]
    averaged: dict[str, torch.Tensor] = {}
    for name, reference in states[0].items():
        # Cached states may come from an untrained CPU template or a client
        # that has already trained on CUDA.  Move every contribution to one
        # reference device before adding them together.
        if reference.is_floating_point() or reference.is_complex():
            averaged[name] = sum(
                state[name].to(reference) * normalized[index].to(reference)
                for index, state in enumerate(states)
            )
        else:
            averaged[name] = reference.detach().clone()
    return averaged


@dataclass
class PowderClient:
    client_id: str
    backbone: PromptedVisionTransformer
    prompt: PromptPool
    head: nn.Module
    task_loaders: Mapping[str, DataLoader]
    device: str | torch.device = "cpu"
    epochs: int = 5
    lr: float = 5e-3
    dual_weight: float = 1.0
    temperature: float = 3.0
    task_classes: Mapping[str, Sequence[int]] = field(default_factory=dict)
    prompt_layers: tuple[int, ...] = (3, 4, 5)
    task_prompt_states: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    task_head_states: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)

    def _forward_with_pool(
        self,
        inputs: torch.Tensor,
        pool: PromptPool,
        head: nn.Module,
        transferred_states: Sequence[Mapping[str, object]] = (),
        correlation_states: Sequence[Mapping[str, object]] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.no_grad():
            query = self.backbone.encode(inputs)
        assert isinstance(query, torch.Tensor)
        prompts, similarity = _powder_prompt_selection(
            pool, query, transferred_states
        )
        layer_prompts = {
            layer: prompts
            for layer in self.prompt_layers
            if 0 <= int(layer) < int(self.backbone.depth)
        }
        features = self.backbone.encode(inputs, layer_prompts=layer_prompts)
        assert isinstance(features, torch.Tensor)
        correlation_similarity = (
            _prompt_state_similarities(query, correlation_states)
            if correlation_states
            else None
        )
        return head(features), similarity, correlation_similarity

    def fit(
        self,
        current_prompt_state: Mapping[str, object],
        local_prompt_states: Sequence[Mapping[str, object]],
        correlation_prompt_states: Sequence[Mapping[str, object]],
        head_state: Mapping[str, object],
        class_correlation: torch.Tensor,
        context: ClientContext,
    ) -> TrainResult:
        del current_prompt_state
        if not local_prompt_states:
            raise ValueError("Powder requires a non-empty local prompt pool.")
        self.prompt.load_state_dict(clone_state(local_prompt_states[0]), strict=True)
        self.head.load_state_dict(clone_state(head_state), strict=True)
        teacher_pool = copy.deepcopy(self.prompt)
        teacher_head = copy.deepcopy(self.head)
        transferred_states = local_prompt_states[1:]
        correlation_states = [
            {
                name: value.to(self.device)
                for name, value in clone_state(state).items()
            }
            for state in correlation_prompt_states
        ]
        loader = task_loader(self.task_loaders, context.task_id)
        self.backbone.to(self.device).eval()
        self.prompt.to(self.device)
        self.head.to(self.device)
        teacher_pool.to(self.device).eval()
        teacher_head.to(self.device).eval()
        optimizer = torch.optim.Adam(
            list(self.prompt.parameters()) + list(self.head.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
        )
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        selection_sum = torch.zeros(
            len(correlation_states) * self.prompt.pool_size, device=self.device
        )
        class_selection_sum: dict[int, torch.Tensor] = {}
        class_counts: dict[int, int] = {}
        task_id = str(context.task_id or "")
        class_ids = list(self.task_classes.get(task_id, ()))
        for _ in range(max(int(self.epochs), 1)):
            self.prompt.train()
            self.head.train()
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                with torch.no_grad():
                    teacher_logits, _, _ = self._forward_with_pool(
                        inputs,
                        teacher_pool,
                        teacher_head,
                        transferred_states,
                    )
                logits, _, correlation_similarity = self._forward_with_pool(
                    inputs,
                    self.prompt,
                    self.head,
                    transferred_states,
                    correlation_states,
                )
                assert correlation_similarity is not None
                task_logits, local_targets, classes = _task_view(
                    logits, targets, class_ids
                )
                teacher_task_logits = teacher_logits.index_select(1, classes)
                loss = F.cross_entropy(task_logits, local_targets)
                if float(self.dual_weight) != 0.0 and task_logits.shape[1] > 1:
                    task_correlation = class_correlation.to(task_logits).index_select(
                        0, classes
                    ).index_select(1, classes)
                    loss = loss + float(self.dual_weight) * dual_distillation_loss(
                        task_logits,
                        teacher_task_logits,
                        local_targets,
                        task_correlation,
                        temperature=self.temperature,
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                probabilities = F.softmax(correlation_similarity.detach(), dim=1)
                selection_sum += probabilities.sum(dim=0)
                for class_id in targets.unique().tolist():
                    class_id = int(class_id)
                    class_mask = targets == class_id
                    contribution = probabilities[class_mask].sum(dim=0)
                    class_selection_sum[class_id] = (
                        class_selection_sum.get(
                            class_id, torch.zeros_like(selection_sum)
                        )
                        + contribution
                    )
                    class_counts[class_id] = class_counts.get(class_id, 0) + int(
                        class_mask.sum().item()
                    )
                count = int(targets.shape[0])
                total_examples += count
                total_loss += float(loss.detach()) * count
                total_correct += int(
                    (task_logits.argmax(1) == local_targets).sum().item()
                )
        selection = selection_sum / selection_sum.sum().clamp_min(1e-12)
        if context.task_id is not None:
            self.task_prompt_states[context.task_id] = clone_state(self.prompt.state_dict())
            self.task_head_states[context.task_id] = clone_state(self.head.state_dict())
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics={
                "loss": total_loss / max(1, total_examples),
                "accuracy": total_correct / max(1, total_examples),
            },
            payload={
                "prompt_state": clone_state(self.prompt.state_dict()),
                "head_state": clone_state(self.head.state_dict()),
                "selection": selection.cpu(),
                "class_selections": {
                    class_id: (
                        value / max(1, class_counts[class_id])
                    ).cpu()
                    for class_id, value in class_selection_sum.items()
                },
                "class_counts": dict(class_counts),
                "task_id": context.task_id,
            },
        )


@dataclass
class PowderServer(PartialStateServer):
    prompt_template: PromptPool = field(default_factory=lambda: None)  # type: ignore[arg-type]
    head_template: nn.Module = field(default_factory=nn.Identity)
    top_k_tasks: int = 3
    correlation_power: float = 30.0
    prompt_states: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    head_states: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    task_selections: dict[str, torch.Tensor] = field(default_factory=dict)
    class_selections: dict[int, torch.Tensor] = field(default_factory=dict)
    task_ids: list[str] = field(default_factory=list)
    task_correlation: torch.Tensor = field(default_factory=lambda: torch.empty(0, 0))
    class_correlation: torch.Tensor = field(default_factory=lambda: torch.empty(0, 0))
    last_selected_client_ids: set[str] = field(default_factory=set)

    def _ensure_task(self, task_id: str) -> None:
        if task_id in self.prompt_states:
            return
        self.task_ids.append(task_id)
        self.prompt_states[task_id] = clone_state(self.prompt_template.state_dict())
        self.head_states[task_id] = clone_state(self.head_template.state_dict())
        count = len(self.task_ids)
        expanded = torch.zeros(count, count)
        if self.task_correlation.numel() > 0:
            expanded[:-1, :-1] = self.task_correlation
        expanded[-1, -1] = 1.0
        self.task_correlation = expanded
        if isinstance(self.head_template, nn.Linear):
            classes = self.head_template.out_features
            if self.class_correlation.numel() == 0:
                self.class_correlation = torch.eye(classes)

    def _update_correlations(self) -> None:
        available = [task_id for task_id in self.task_ids if task_id in self.task_selections]
        if available:
            signature_size = max(self.task_selections[item].numel() for item in available)
            for item in available:
                selection = self.task_selections[item]
                if selection.numel() < signature_size:
                    self.task_selections[item] = F.pad(
                        selection, (0, signature_size - selection.numel())
                    )
        for left in available:
            left_index = self.task_ids.index(left)
            for right in available:
                right_index = self.task_ids.index(right)
                similarity = F.cosine_similarity(
                    self.task_selections[left].unsqueeze(0),
                    self.task_selections[right].unsqueeze(0),
                ).clamp_min(0.0)
                self.task_correlation[left_index, right_index] = similarity.pow(
                    float(self.correlation_power)
                )
        row_sum = self.task_correlation.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.task_correlation = self.task_correlation / row_sum

        if self.class_correlation.numel() > 0:
            available_classes = sorted(self.class_selections)
            if available_classes:
                signature_size = max(
                    self.class_selections[item].numel() for item in available_classes
                )
                for item in available_classes:
                    selection = self.class_selections[item]
                    if selection.numel() < signature_size:
                        self.class_selections[item] = F.pad(
                            selection, (0, signature_size - selection.numel())
                        )
            for left in available_classes:
                for right in available_classes:
                    similarity = F.cosine_similarity(
                        self.class_selections[left].unsqueeze(0),
                        self.class_selections[right].unsqueeze(0),
                    ).clamp_min(0.0)
                    self.class_correlation[left, right] = similarity.pow(
                        float(self.correlation_power)
                    )

    def _first_step_prompt_state(self, index: int) -> dict[str, torch.Tensor]:
        raw_states = [self.prompt_states[item] for item in self.task_ids]
        return _average_prompt_states(raw_states, self.task_correlation[index])

    def correlation_prompt_states(self) -> list[dict[str, torch.Tensor]]:
        """Canonical Eq. (3) global pool used to compute Eq. (6) signatures."""

        return [self._first_step_prompt_state(index) for index in range(len(self.task_ids))]

    def local_prompt_states(self, task_id: str) -> list[dict[str, torch.Tensor]]:
        self._ensure_task(task_id)
        index = self.task_ids.index(task_id)
        correlations = self.task_correlation[index]
        count = min(max(int(self.top_k_tasks), 1), len(self.task_ids))
        ranked = correlations.topk(count).indices.tolist()
        selected = [index, *(item for item in ranked if item != index)][:count]
        aggregated: list[dict[str, torch.Tensor]] = []
        # Eq. (3) produces one correlated prompt pool for every selected task;
        # Eq. (4) communicates the top-k resulting pools without collapsing
        # them back into a single M-slot pool.
        for selected_index in selected:
            aggregated.append(self._first_step_prompt_state(selected_index))
        return aggregated

    def local_prompt_state(self, task_id: str) -> dict[str, torch.Tensor]:
        """Backward-compatible access to the current task's Eq. (3) state."""

        return self.local_prompt_states(task_id)[0]

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        results: list[TrainResult] = []
        selected = self.selected_clients(round_idx)
        self.last_selected_client_ids = {str(client.client_id) for client in selected}
        assignments: list[tuple[PowderClient, ClientContext, str]] = []
        for client in selected:
            assert isinstance(client, PowderClient)
            context = self.client_context(client, round_idx, task_id)
            client_task_id = context.task_id or task_id
            self._ensure_task(client_task_id)
            assignments.append((client, context, client_task_id))
        correlation_prompt_states = self.correlation_prompt_states()
        for client, context, client_task_id in assignments:
            results.append(
                client.fit(
                    self.prompt_states[client_task_id],
                    self.local_prompt_states(client_task_id),
                    correlation_prompt_states,
                    self.head_states[client_task_id],
                    self.class_correlation,
                    context,
                )
            )
        if not results:
            return self.empty_result(round_idx, task_id)
        updated_task_ids = sorted(
            {str(result.payload.get("task_id") or task_id) for result in results}
        )
        for updated_task_id in updated_task_ids:
            grouped = [
                result
                for result in results
                if str(result.payload.get("task_id") or task_id) == updated_task_id
            ]
            self.prompt_states[updated_task_id] = weighted_average_state(
                [(result.payload["prompt_state"], result.num_samples) for result in grouped]
            )
            self.head_states[updated_task_id] = weighted_average_state(
                [(result.payload["head_state"], result.num_samples) for result in grouped]
            )
            self.task_selections[updated_task_id] = sum(
                result.payload["selection"] * result.num_samples for result in grouped
            ) / max(1, sum(result.num_samples for result in grouped))
        class_updates: dict[int, list[tuple[torch.Tensor, int]]] = {}
        for result in results:
            selections = result.payload.get("class_selections", {})
            counts = result.payload.get("class_counts", {})
            if not isinstance(selections, Mapping) or not isinstance(counts, Mapping):
                continue
            for raw_class_id, selection in selections.items():
                class_id = int(raw_class_id)
                if not isinstance(selection, torch.Tensor):
                    continue
                class_updates.setdefault(class_id, []).append(
                    (selection, int(counts.get(raw_class_id, 0)))
                )
        for class_id, updates in class_updates.items():
            total = max(1, sum(count for _, count in updates))
            self.class_selections[class_id] = sum(
                selection * count for selection, count in updates
            ) / total
        self._update_correlations()
        state: dict[str, torch.Tensor] = {}
        for updated_task_id in updated_task_ids:
            state.update(
                {
                    f"tasks.{updated_task_id}.prompt.{name}": value
                    for name, value in self.prompt_states[updated_task_id].items()
                }
            )
            state.update(
                {
                    f"tasks.{updated_task_id}.head.{name}": value
                    for name, value in self.head_states[updated_task_id].items()
                }
            )
        return AggregationResult(
            global_state=state,
            metrics=mean_metrics(results),
            metadata={
                "round_idx": round_idx,
                "task_id": task_id,
                "algorithm": "powder",
                "task_correlation": self.task_correlation.tolist(),
                "class_correlation": self.class_correlation.tolist(),
            },
        )
