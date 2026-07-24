from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...contracts import MetricDict
from ...trainers.utils import move_to_device
from .base import RobustReplayLearner


@dataclass
class FLAIRLearner(RobustReplayLearner):
    """FLatness-preserving Adversarial Incremental learning for Robustness.

    Implements ADSL (paper Eq. 3) and FPD (Eq. 8).  The old/new BCE terms
    use the active-class fractions from the authors' released implementation;
    this keeps their relative scale stable as the classifier expands.
    """

    distillation_weight: float = 1.0
    flatness_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.distillation_weight) < 0.0:
            raise ValueError("distillation_weight must be non-negative.")
        if float(self.flatness_weight) < 0.0:
            raise ValueError("flatness_weight must be non-negative.")

    def observe(self, inputs: torch.Tensor, targets: torch.Tensor, task_id: str) -> MetricDict:
        del task_id
        replay = self.sample_memory(self.replay_count(int(targets.shape[0])))
        if replay is not None:
            replay_inputs, replay_targets, _indices = replay
            clean = torch.cat((inputs, replay_inputs), dim=0)
            labels = torch.cat((targets, replay_targets), dim=0)
        else:
            clean, labels = inputs, targets

        active = list(self.seen_classes)
        adversarial = self.generate_adversarial(clean, labels, class_ids=active)
        adversarial_logits = self.model(adversarial)
        clean_logits = self.model(clean)
        new_index = torch.tensor(self.new_classes, device=self.device, dtype=torch.long)
        new_logits = adversarial_logits.index_select(1, new_index)
        new_targets = torch.zeros_like(new_logits)
        for offset, class_id in enumerate(self.new_classes):
            new_targets[:, offset] = labels.eq(class_id).to(new_logits.dtype)

        active_count = len(active)
        old_count = len(self.old_classes)
        new_scale = float(len(self.new_classes)) / float(active_count)
        classification_loss = new_scale * F.binary_cross_entropy_with_logits(
            new_logits, new_targets
        )
        distillation_loss = classification_loss.new_zeros(())
        flatness_loss = classification_loss.new_zeros(())

        if old_count:
            if self.teacher is None:
                raise RuntimeError("FLAIR teacher is missing after the first task.")
            old_index = torch.tensor(self.old_classes, device=self.device, dtype=torch.long)
            with torch.no_grad():
                teacher_adversarial = self.teacher(adversarial).index_select(1, old_index)
                teacher_clean = self.teacher(clean).index_select(1, old_index)
            current_adversarial = adversarial_logits.index_select(1, old_index)
            current_clean = clean_logits.index_select(1, old_index)
            old_scale = float(old_count) / float(active_count)
            distillation_loss = (
                float(self.distillation_weight)
                * old_scale
                * F.binary_cross_entropy_with_logits(
                    current_adversarial, torch.sigmoid(teacher_adversarial)
                )
            )

            current_difference = current_adversarial - current_clean
            teacher_difference = teacher_adversarial - teacher_clean
            flatness_loss = float(self.flatness_weight) * F.kl_div(
                F.log_softmax(current_difference, dim=1),
                F.softmax(teacher_difference, dim=1),
                reduction="batchmean",
            )

        loss = classification_loss + distillation_loss + flatness_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "classification_loss": float(classification_loss.detach().item()),
            "adsl_distillation_loss": float(distillation_loss.detach().item()),
            "fpd_loss": float(flatness_loss.detach().item()),
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


FLAIR = FLAIRLearner

__all__ = ["FLAIR", "FLAIRLearner"]
