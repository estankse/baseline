from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...contracts import MetricDict, TaskDefinition
from ...trainers.utils import move_to_device
from ..CL.base import extract_normalized_features
from .base import RobustReplayLearner, _local_targets


@dataclass
class DAMLLearner(RobustReplayLearner):
    """Distillation and Additional Memory-data Loss (Mukai et al., 2024).

    DAML combines iCaRL's unified BCE target on adversarial stream/replay
    data with a CE loss on a *different* replay batch.  Keeping the second
    memory batch disjoint is essential to Eq. (10) and the paper's ablation.
    """

    additional_memory_weight: float = 0.2
    _prototype_classes: List[int] = field(default_factory=list, init=False, repr=False)
    _prototypes: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.additional_memory_weight) < 0.0:
            raise ValueError("additional_memory_weight must be non-negative.")

    def _on_task_start(self, task: TaskDefinition) -> None:
        super()._on_task_start(task)
        # The current classes do not have herding exemplars until a diagnostic
        # refresh or task end. Avoid using stale, incomplete class means.
        self._prototype_classes = []
        self._prototypes = None

    def _unified_targets(
        self,
        adversarial: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = list(self.seen_classes)
        logits = self._select_logits(self.model, adversarial, active)
        binary_targets = torch.zeros_like(logits)
        positions = {class_id: offset for offset, class_id in enumerate(active)}

        if self.old_classes:
            if self.teacher is None:
                raise RuntimeError("DAML teacher is missing after the first task.")
            with torch.no_grad():
                teacher_logits = self._select_logits(
                    self.teacher, adversarial, self.old_classes
                )
            old_positions = torch.tensor(
                [positions[class_id] for class_id in self.old_classes],
                device=logits.device,
                dtype=torch.long,
            )
            binary_targets.index_copy_(1, old_positions, torch.sigmoid(teacher_logits))

        for class_id in self.new_classes:
            binary_targets[:, positions[class_id]] = targets.eq(class_id).to(logits.dtype)
        return logits, binary_targets

    def observe(self, inputs: torch.Tensor, targets: torch.Tensor, task_id: str) -> MetricDict:
        del task_id
        replay_count = self.replay_count(int(targets.shape[0]))
        replay = self.sample_memory(replay_count)
        excluded: list[int] = []
        if replay is not None:
            replay_inputs, replay_targets, excluded = replay
            unified_inputs = torch.cat((inputs, replay_inputs), dim=0)
            unified_labels = torch.cat((targets, replay_targets), dim=0)
        else:
            unified_inputs, unified_labels = inputs, targets

        adversarial = self.generate_adversarial(
            unified_inputs, unified_labels, class_ids=self.seen_classes
        )
        logits, binary_targets = self._unified_targets(adversarial, unified_labels)
        # iCaRL's unified objective sums the binary class terms for each
        # sample, then averages samples.  PyTorch's default BCE reduction also
        # averages over classes, which weakens distillation/new-class learning
        # by 1 / len(seen_classes) relative to DAML's additional CE term.
        unified_loss = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none"
        ).sum(dim=1).mean()

        additional_loss = unified_loss.new_zeros(())
        additional = self.sample_memory(replay_count, exclude=excluded)
        if additional is not None:
            memory_inputs, memory_targets, _indices = additional
            memory_adversarial = self.generate_adversarial(
                memory_inputs, memory_targets, class_ids=self.seen_classes
            )
            memory_logits = self._select_logits(
                self.model, memory_adversarial, self.seen_classes
            )
            additional_loss = F.cross_entropy(
                memory_logits, _local_targets(memory_targets, self.seen_classes)
            )

        loss = unified_loss + float(self.additional_memory_weight) * additional_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "unified_bce_loss": float(unified_loss.detach().item()),
            "additional_memory_loss": float(additional_loss.detach().item()),
        }

    def train_epoch(self, dataloader: DataLoader, task_id: str) -> MetricDict:
        self.model.to(self.device).train()
        totals: Dict[str, float] = {}
        total_examples = 0
        for batch in dataloader:
            inputs = move_to_device(batch[0], self.device)
            targets = move_to_device(batch[1], self.device).long()
            metrics = self.observe(inputs, targets, task_id)
            batch_size = int(targets.shape[0])
            total_examples += batch_size
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        if total_examples == 0:
            return {"loss": 0.0}
        return {name: value / total_examples for name, value in totals.items()}

    def _build_prototypes(self) -> None:
        classes: List[int] = []
        prototypes: List[torch.Tensor] = []
        for class_id in self.seen_classes:
            exemplars = self.exemplar_sets.get(class_id, [])
            if not exemplars:
                continue
            features = self._features_for_samples(exemplars)
            prototype = F.normalize(
                features.mean(dim=0, keepdim=True), p=2, dim=1
            ).squeeze(0)
            classes.append(class_id)
            prototypes.append(prototype)
        self._prototype_classes = classes
        self._prototypes = (
            torch.stack(prototypes).to(self.device) if prototypes else None
        )

    def refresh_nme(self, train_loader: DataLoader) -> None:
        """Refresh herding exemplars and iCaRL nearest-mean prototypes."""

        self.update_herding_memory(train_loader)
        self._build_prototypes()

    def _on_task_end(self, task: TaskDefinition, train_loader: DataLoader) -> None:
        del task
        self.refresh_nme(train_loader)

    def predict(self, inputs: torch.Tensor, task_id: str | None = None) -> torch.Tensor:
        active = self.active_classes(task_id)
        prototype_positions = {
            class_id: offset for offset, class_id in enumerate(self._prototype_classes)
        }
        # During training, before current-class exemplars have been selected,
        # retain the classifier head as a diagnostic fallback.
        if self._prototypes is None or any(
            class_id not in prototype_positions for class_id in active
        ):
            return super().predict(inputs, task_id=task_id)
        positions = torch.tensor(
            [prototype_positions[class_id] for class_id in active],
            device=self.device,
            dtype=torch.long,
        )
        prototypes = self._prototypes.index_select(0, positions)
        features = extract_normalized_features(self.model, inputs)
        predictions = torch.matmul(features, prototypes.T).argmax(dim=1)
        return torch.tensor(active, device=self.device, dtype=torch.long)[predictions]


# Backward-compatible paper acronym.
DAML = DAMLLearner

__all__ = ["DAML", "DAMLLearner"]
