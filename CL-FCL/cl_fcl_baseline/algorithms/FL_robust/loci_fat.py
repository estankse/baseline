from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ...contracts import MetricDict
from ...trainers.utils import move_to_device
from ..CL.base import assign_flat_gradient, flatten_gradients, trainable_parameters
from ..CL.gem import project_gradient
from ..loci import LociClient, _task_view
from .own import task_aware_pgd_linf_attack
from .PGD import PGDConfig


@dataclass
class AdversarialLociClient(LociClient):
    """LOCI client with only local PGD adversarial training added."""

    pgd_config: PGDConfig = field(default_factory=PGDConfig)
    adversarial_ratio: float = 0.5
    warmup_rounds: int = 0
    warmup_adversarial_ratio: float = 0.1
    _active_round_idx: int | None = field(default=None, init=False, repr=False)

    def _round_adversarial_ratio(self, round_idx: int | None) -> float:
        if (
            int(self.warmup_rounds) > 0
            and round_idx is not None
            and int(round_idx) < int(self.warmup_rounds)
        ):
            ratio = float(self.warmup_adversarial_ratio)
        else:
            ratio = float(self.adversarial_ratio)
        return min(max(ratio, 0.0), 1.0)

    def _adversarial_count(self, batch_size: int, ratio: float) -> int:
        if batch_size <= 0 or ratio <= 0.0:
            return 0
        return min(batch_size, max(1, int(round(batch_size * ratio))))

    def _mixed_adversarial_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: str,
        adversarial_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(targets.shape[0])
        adv_count = self._adversarial_count(batch_size, adversarial_ratio)
        if adv_count <= 0:
            empty = torch.empty(0, dtype=torch.long, device=targets.device)
            clean = torch.arange(batch_size, dtype=torch.long, device=targets.device)
            return inputs, empty, clean

        permutation = torch.randperm(batch_size, device=targets.device)
        adv_indices = permutation[:adv_count]
        clean_indices = permutation[adv_count:]
        was_training = self.trainer.model.training
        self.trainer.model.eval()
        try:
            adversarial_inputs = task_aware_pgd_linf_attack(
                self.trainer.model,
                inputs.index_select(0, adv_indices),
                targets.index_select(0, adv_indices),
                self.pgd_config,
                class_ids=self._classes(task_id),
            )
        finally:
            self.trainer.model.train(was_training)

        mixed_inputs = inputs.detach().clone()
        mixed_inputs[adv_indices] = adversarial_inputs
        return mixed_inputs, adv_indices, clean_indices

    def _train_main_model(
        self,
        loader: DataLoader,
        task_id: str,
        teacher_state: Mapping[str, torch.Tensor],
        teacher_accuracy: float,
    ) -> MetricDict:
        teacher = copy.deepcopy(self.kd_model)
        teacher.load_state_dict(teacher_state, strict=True)
        teacher.to(self.trainer.device)
        teacher.eval()
        self.trainer.model.to(self.trainer.device)
        self.trainer.optimizer = self._build_main_optimizer()
        temperature = float(self.temperature)
        adversarial_ratio = self._round_adversarial_ratio(self._active_round_idx)

        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_clean_ce = 0.0
        total_adv_ce = 0.0
        total_clean_examples = 0
        total_adv_examples = 0
        total_correct = 0
        total_examples = 0
        total_batches = 0
        total_constraint_violations = 0
        total_gradient_projections = 0

        for _ in range(max(0, int(self.epochs))):
            self.trainer.model.train()
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                parameters = trainable_parameters(self.trainer.model)
                old_gradients: list[torch.Tensor] = []
                if self.continual_method == "gem":
                    self._update_episodic_memory(task_id, inputs, targets)
                    old_gradients = self._gem_memory_gradients(task_id, parameters)

                mixed_inputs, adv_indices, clean_indices = self._mixed_adversarial_batch(
                    inputs=inputs,
                    targets=targets,
                    task_id=task_id,
                    adversarial_ratio=adversarial_ratio,
                )
                logits, local_targets = _task_view(
                    self.trainer.model(mixed_inputs), targets, self._classes(task_id)
                )
                assert local_targets is not None
                with torch.no_grad():
                    teacher_logits, _ = _task_view(
                        teacher(mixed_inputs), None, self._classes(task_id)
                    )
                classification_loss = F.cross_entropy(logits, local_targets)
                distillation_loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = (
                    classification_loss
                    + float(self.integrator_weight)
                    * float(teacher_accuracy)
                    * distillation_loss
                )
                if self.continual_method == "ewc":
                    loss = loss + float(self.ewc_lambda) * self._ewc_penalty()

                self.trainer.optimizer.zero_grad()
                loss.backward()
                if old_gradients:
                    gradient = flatten_gradients(parameters)
                    memory_gradients = torch.stack(old_gradients)
                    dot_products = memory_gradients @ gradient
                    violations = int((dot_products < 0.0).sum().item())
                    total_constraint_violations += violations
                    if violations:
                        gradient = project_gradient(
                            gradient,
                            memory_gradients,
                            margin=float(self.gem_memory_strength),
                            eps=float(self.gem_qp_eps),
                        )
                        assign_flat_gradient(parameters, gradient)
                        total_gradient_projections += 1
                self.trainer.optimizer.step()

                batch_size = int(targets.shape[0])
                adv_count = int(adv_indices.numel())
                clean_count = int(clean_indices.numel())
                total_batches += 1
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_ce += float(classification_loss.detach().item()) * batch_size
                total_kd += float(distillation_loss.detach().item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == local_targets).sum().item())
                if clean_count > 0:
                    clean_loss = F.cross_entropy(
                        logits.index_select(0, clean_indices),
                        local_targets.index_select(0, clean_indices),
                    )
                    total_clean_ce += float(clean_loss.detach().item()) * clean_count
                    total_clean_examples += clean_count
                if adv_count > 0:
                    adv_loss = F.cross_entropy(
                        logits.index_select(0, adv_indices),
                        local_targets.index_select(0, adv_indices),
                    )
                    total_adv_ce += float(adv_loss.detach().item()) * adv_count
                    total_adv_examples += adv_count

        denominator = max(1, total_examples)
        return {
            "loss": total_loss / denominator,
            "accuracy": total_correct / denominator,
            "classification_loss": total_ce / denominator,
            "integrator_kd_loss": total_kd / denominator,
            "clean_ce_loss": total_clean_ce / max(1, total_clean_examples),
            "adv_ce_loss": total_adv_ce / max(1, total_adv_examples),
            "adversarial_ratio": float(adversarial_ratio),
            "num_adversarial_samples": float(total_adv_examples),
            "num_clean_samples": float(total_clean_examples),
            "gem_constraint_violations": total_constraint_violations
            / max(1, total_batches),
            "gem_gradient_projections": total_gradient_projections
            / max(1, total_batches),
        }

    def fit(self, global_state, context):
        self._active_round_idx = context.round_idx
        try:
            result = super().fit(global_state, context)
        finally:
            self._active_round_idx = None
        result.payload["robust_variant"] = "loci_at"
        return result


__all__ = ["AdversarialLociClient"]
