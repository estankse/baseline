from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ...contracts import (
    AggregationResult,
    ClientContext,
    MetricDict,
    StateDict,
    TrainResult,
)
from ...trainers.utils import detach_state_dict, move_to_device
from ..CL.base import assign_flat_gradient, flatten_gradients, trainable_parameters
from ..CL.gem import project_gradient
from ..loci import (
    LociClient,
    LociServer,
    LociTaskKnowledge,
    _Candidate,
    _clone_tensor_state,
    _task_view,
)
from .PGD import PGDConfig


def _channel_tensor(
    value: float | Sequence[float] | torch.Tensor | None,
    inputs: torch.Tensor,
) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=inputs.device, dtype=inputs.dtype)
    else:
        tensor = torch.as_tensor(value, device=inputs.device, dtype=inputs.dtype)
    if tensor.ndim == 0:
        return tensor
    shape = [1] * inputs.ndim
    shape[1] = int(tensor.numel())
    return tensor.reshape(shape)


def _project_linf(
    adversarial: torch.Tensor,
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    clip_min: torch.Tensor | None,
    clip_max: torch.Tensor | None,
) -> torch.Tensor:
    delta = torch.clamp(adversarial - clean, min=-epsilon, max=epsilon)
    adversarial = clean + delta
    if clip_min is not None:
        adversarial = torch.maximum(adversarial, clip_min)
    if clip_max is not None:
        adversarial = torch.minimum(adversarial, clip_max)
    return adversarial.detach()


def task_aware_pgd_linf_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: PGDConfig,
    *,
    class_ids: Sequence[int] | None = None,
    objective: str = "ce",
    reference_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a PGD attack in the active LOCI task's output space.

    LOCI keeps global classifier rows and may give a client a non-contiguous
    class subset.  The repository's generic PGD helper assumes targets index
    the complete output directly, so using it here would attack the wrong
    rows for shuffled class-incremental tasks.  This helper applies the same
    task projection as LOCI's training loss before computing CE or TRADES KL.
    """
    if objective not in {"ce", "kl", "hybrid"}:
        raise ValueError("objective must be 'ce', 'kl', or 'hybrid'.")
    if int(config.steps) <= 0:
        return inputs.detach()

    epsilon = _channel_tensor(config.epsilon, inputs)
    step_size = _channel_tensor(config.step_size, inputs)
    clip_min = _channel_tensor(config.clip_min, inputs)
    clip_max = _channel_tensor(config.clip_max, inputs)
    if epsilon is None or step_size is None:
        raise ValueError("PGD epsilon and step_size must not be None.")

    clean = inputs.detach()
    adversarial = clean.clone()
    if config.random_start:
        noise = torch.empty_like(adversarial).uniform_(-1.0, 1.0) * epsilon
        adversarial = _project_linf(
            clean + noise, clean, epsilon, clip_min, clip_max
        )

    if objective in {"kl", "hybrid"} and reference_logits is None:
        with torch.no_grad():
            reference_logits, _ = _task_view(model(clean), None, class_ids)
    reference_probabilities = (
        F.softmax(reference_logits.detach(), dim=1)
        if reference_logits is not None
        else None
    )

    for _ in range(int(config.steps)):
        adversarial.requires_grad_(True)
        task_logits, local_targets = _task_view(
            model(adversarial), targets, class_ids
        )
        if objective == "kl":
            assert reference_probabilities is not None
            attack_loss = F.kl_div(
                F.log_softmax(task_logits, dim=1),
                reference_probabilities,
                reduction="batchmean",
            )
        elif objective == "hybrid":
            assert reference_probabilities is not None
            assert local_targets is not None
            attack_loss = F.cross_entropy(task_logits, local_targets) + F.kl_div(
                F.log_softmax(task_logits, dim=1),
                reference_probabilities,
                reduction="batchmean",
            )
        else:
            assert local_targets is not None
            attack_loss = F.cross_entropy(task_logits, local_targets)
        gradient = torch.autograd.grad(
            attack_loss, adversarial, only_inputs=True
        )[0]
        adversarial = adversarial.detach() + step_size * gradient.sign()
        adversarial = _project_linf(
            adversarial, clean, epsilon, clip_min, clip_max
        )
    return adversarial.detach()


def _kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def _soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    teacher_weight: float,
    temperature: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine hard labels and LOCI knowledge in one non-conflicting target.

    Adding CE and KD as independent losses uses every output twice and can
    produce opposing gradients in robust continual learning.  RAMP instead
    forms one probability target for each clean/adversarial view.  This keeps
    LOCI's accuracy-gated knowledge integration while avoiding a frozen model
    snapshot and a second loss on the same logits.
    """
    weight = min(1.0, max(0.0, float(teacher_weight)))
    hard = F.one_hot(targets, num_classes=logits.shape[1]).to(logits)
    if weight <= 0.0:
        probabilities = hard
    else:
        soft = F.softmax(teacher_logits.detach() / float(temperature), dim=1)
        probabilities = (1.0 - weight) * hard + weight * soft
    losses = -torch.sum(probabilities * F.log_softmax(logits, dim=1), dim=1)
    if class_weights is None:
        return losses.mean()
    sample_weights = class_weights.to(logits).index_select(0, targets)
    return torch.sum(losses * sample_weights) / sample_weights.sum().clamp_min(1e-12)


def _boundary_alignment_loss(
    clean_logits: torch.Tensor,
    adversarial_logits: torch.Tensor,
    teacher_clean_logits: torch.Tensor,
    teacher_adversarial_logits: torch.Tensor,
) -> torch.Tensor:
    """Transfer the clean-to-adversarial boundary displacement.

    Centering removes the softmax-invariant logit offset.  Normalizing the
    displacement makes the signal portable between heterogeneous private and
    compact KD models whose logit scales need not match.
    """
    student = adversarial_logits - clean_logits
    teacher = teacher_adversarial_logits.detach() - teacher_clean_logits.detach()
    student = student - student.mean(dim=1, keepdim=True)
    teacher = teacher - teacher.mean(dim=1, keepdim=True)
    student = F.normalize(student, dim=1, eps=1e-12)
    teacher = F.normalize(teacher, dim=1, eps=1e-12)
    return (1.0 - F.cosine_similarity(student, teacher, dim=1)).mean()


def _flatten_explicit_gradients(
    parameters: Sequence[nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient)
            .detach()
            .reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def evaluate_task_aware_pgd(
    model: nn.Module,
    dataloader: DataLoader,
    config: PGDConfig,
    *,
    device: str | torch.device = "cpu",
    class_ids: Sequence[int] | None = None,
    max_batches: int | None = None,
) -> MetricDict:
    """Evaluate clean-label PGD robustness for one task-aware LOCI head."""
    model.to(device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    evaluated_batches = 0
    try:
        for batch_index, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and max_batches > 0 and batch_index >= max_batches:
                break
            inputs = move_to_device(inputs, device)
            targets = move_to_device(targets, device)
            adversarial = task_aware_pgd_linf_attack(
                model,
                inputs,
                targets,
                config,
                class_ids=class_ids,
            )
            with torch.no_grad():
                logits, local_targets = _task_view(
                    model(adversarial), targets, class_ids
                )
                assert local_targets is not None
                loss = F.cross_entropy(logits, local_targets)
                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.item()) * batch_size
                total_correct += int(
                    (logits.argmax(dim=1) == local_targets).sum().item()
                )
            evaluated_batches += 1
    finally:
        model.train(was_training)
    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
        "num_batches": float(evaluated_batches),
        "num_samples": float(total_examples),
    }


@dataclass
class RobustLociClient(LociClient):
    """LOCI with adversarially robust task knowledge (RobustLoci).

    Four experiment variants are kept in one implementation so their LOCI
    lifecycle is identical and the comparison changes only the robust loss:

    ``ramp`` (recommended)
        Robustness-Aware Memory Palace.  Clean labels and incoming LOCI
        knowledge are combined into one target, robustness is communicated as
        clean/adversarial boundary displacement, and GEM constrains the robust
        loss of old tasks directly.  It stores no historical model snapshot.

    ``radt`` (legacy ablation)
        Robust Adversarial Distillation for Tasks.  It jointly uses clean CE,
        adversarial CE, TRADES consistency, clean integrator KD and adversarial
        KD.  Robustness is therefore learned by the private heterogeneous main
        model and transferred into LOCI's common communicable KD model.
    ``trades``
        Ablation retaining clean KD and TRADES consistency, without adversarial
        hard-label CE or adversarial teacher matching.
    ``ard``
        Adversarially Robust Distillation ablation retaining adversarial CE and
        teacher matching, without student clean/adversarial consistency.

    The main-model continual learner remains LOCI's GEM. Robust Fisher is used
    only to rank KD-model weights for memory-palace pruning (and by the optional
    EWC ablation), before adversarial sparse fine-tuning.
    """

    continual_method: str = "gem"
    pgd_config: PGDConfig = field(default_factory=PGDConfig)
    variant: str = "ramp"
    clean_weight: float = 1.0
    adversarial_weight: float = 0.5
    trades_weight: float = 4.0
    robust_kd_weight: float = 1.0
    boundary_weight: float = 1.0
    robust_gradient_ratio: float = 1.0
    teacher_clean_weight: float = 0.5
    teacher_weight_floor: float = 0.0
    teacher_eval_batches: int = 2
    robust_warmup_rounds: int = 0
    fisher_adversarial_weight: float = 1.0
    importance_weight: float = 1.0
    knowledge_robust_weight: float = 1.0
    class_balance_power: float = 0.5
    class_balance_smoothing: float = 1e-3
    class_weight_max: float = 3.0
    replay_budget: int = 0
    replay_batch_size: int = 8
    replay_selection_batches: int = 5
    replay_weight: float = 1.0
    robust_memory_batch_size: int = 32
    public_refine_epochs: int = 0
    public_refine_lr_scale: float = 0.1
    kd_fisher_state: dict[str, torch.Tensor] = field(default_factory=dict)
    replay_memory: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    task_class_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    _robust_scale: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.variant = str(self.variant).lower()
        if self.variant not in {"ramp", "radt", "trades", "ard"}:
            raise ValueError("variant must be one of: ramp, radt, trades, ard.")
        nonnegative = {
            "adversarial_weight": self.adversarial_weight,
            "clean_weight": self.clean_weight,
            "trades_weight": self.trades_weight,
            "robust_kd_weight": self.robust_kd_weight,
            "boundary_weight": self.boundary_weight,
            "robust_gradient_ratio": self.robust_gradient_ratio,
            "fisher_adversarial_weight": self.fisher_adversarial_weight,
            "importance_weight": self.importance_weight,
            "knowledge_robust_weight": self.knowledge_robust_weight,
            "class_balance_power": self.class_balance_power,
            "replay_weight": self.replay_weight,
            "public_refine_lr_scale": self.public_refine_lr_scale,
        }
        for name, value in nonnegative.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0.0 <= float(self.teacher_clean_weight) <= 1.0:
            raise ValueError("teacher_clean_weight must be in [0, 1].")
        if not 0.0 <= float(self.teacher_weight_floor) <= 1.0:
            raise ValueError("teacher_weight_floor must be in [0, 1].")
        if float(self.class_balance_smoothing) <= 0.0:
            raise ValueError("class_balance_smoothing must be positive.")
        if float(self.class_weight_max) < 1.0:
            raise ValueError("class_weight_max must be at least 1.")
        if int(self.replay_budget) < 0 or int(self.replay_batch_size) < 0:
            raise ValueError("replay sizes must be non-negative.")
        if int(self.robust_memory_batch_size) < 0:
            raise ValueError("robust_memory_batch_size must be non-negative.")
        if int(self.public_refine_epochs) < 0:
            raise ValueError("public_refine_epochs must be non-negative.")

    def _variant_weights(self) -> tuple[float, float, float]:
        adversarial = float(self.adversarial_weight)
        trades = float(self.trades_weight)
        robust_kd = float(self.robust_kd_weight)
        if self.variant == "ramp":
            # RAMP uses a clean-anchored TRADES term but replaces adversarial
            # teacher KL with boundary displacement alignment.
            robust_kd = 0.0
        elif self.variant == "trades":
            adversarial = 0.0
            robust_kd = 0.0
        elif self.variant == "ard":
            trades = 0.0
        scale = float(self._robust_scale)
        return adversarial * scale, trades * scale, robust_kd * scale

    def _gem_memory_gradients(
        self,
        current_task_id: str,
        parameters: Sequence[nn.Parameter],
    ) -> list[torch.Tensor]:
        """Constrain past-task *robust* risk without a model snapshot.

        Vanilla LOCI/GEM only prevents an increase in clean memory loss.  The
        logs show that old PGD accuracy decays much faster than clean accuracy,
        so RAMP constructs each constraint from clean and adversarial views of
        the existing GEM memory at the current parameters.  No old network is
        retained or consulted.
        """
        if self.variant != "ramp":
            return super()._gem_memory_gradients(current_task_id, parameters)
        gradients: list[torch.Tensor] = []
        limit = int(self.robust_memory_batch_size)
        for task_id, samples in self.episodic_memory.items():
            if task_id == current_task_id or not samples:
                continue
            if limit <= 0 or len(samples) <= limit:
                selected = samples
            else:
                by_class: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
                for sample, target in samples:
                    by_class.setdefault(int(target.item()), []).append((sample, target))
                quota = max(1, limit // max(1, len(by_class)))
                selected = []
                leftovers: list[tuple[torch.Tensor, torch.Tensor]] = []
                for values in by_class.values():
                    shuffled = random.sample(values, k=len(values))
                    selected.extend(shuffled[:quota])
                    leftovers.extend(shuffled[quota:])
                if len(selected) < limit and leftovers:
                    selected.extend(
                        random.sample(
                            leftovers,
                            k=min(limit - len(selected), len(leftovers)),
                        )
                    )
                selected = selected[:limit]
            memory_inputs = torch.stack([sample for sample, _ in selected]).to(
                self.trainer.device
            )
            memory_targets = torch.stack([target for _, target in selected]).long().to(
                self.trainer.device
            )
            self.trainer.optimizer.zero_grad()
            clean_logits, local_targets = _task_view(
                self.trainer.model(memory_inputs),
                memory_targets,
                self._classes(task_id),
            )
            assert local_targets is not None
            adversarial = self._attack(
                self.trainer.model,
                memory_inputs,
                memory_targets,
                task_id,
                clean_logits=clean_logits.detach(),
            )
            self.trainer.model.train()
            adversarial_logits, _ = _task_view(
                self.trainer.model(adversarial), None, self._classes(task_id)
            )
            adversarial_weight, trades_weight, _robust_kd_weight = (
                self._variant_weights()
            )
            memory_loss = (
                float(self.clean_weight)
                * F.cross_entropy(clean_logits, local_targets)
                + adversarial_weight
                * F.cross_entropy(adversarial_logits, local_targets)
                + trades_weight
                * _kl_loss(
                    adversarial_logits,
                    clean_logits.detach(),
                    temperature=1.0,
                )
            )
            memory_loss.backward()
            gradients.append(flatten_gradients(parameters).clone())
        return gradients

    def _attack(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: str,
        *,
        clean_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # RADT needs a genuinely hard supervised adversary.  The previous
        # implementation used a three-step KL attack for RADT; its training
        # robust accuracy tracked clean accuracy within 0.6 points while
        # PGD-10 accuracy collapsed, a classic sign of an underpowered inner
        # maximization.  Only the explicit TRADES ablation uses KL-PGD now.
        objective = (
            "hybrid"
            if self.variant == "ramp"
            else "kl"
            if self.variant == "trades"
            else "ce"
        )
        was_training = model.training
        model.eval()
        try:
            return task_aware_pgd_linf_attack(
                model,
                inputs,
                targets,
                self.pgd_config,
                class_ids=self._classes(task_id),
                objective=objective,
                reference_logits=clean_logits,
            )
        finally:
            model.train(was_training)

    def _class_weights(
        self,
        task_id: str,
        loader: DataLoader,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        """Return capped inverse-frequency weights in the task-local head.

        This is the useful part of Sylva/CalFAT for LOCI's Dirichlet clients:
        rare local classes receive stronger gradients, but absent classes are
        not assigned infinite weights.  Missing-class knowledge is supplied by
        robust public refinement after server fusion instead.
        """
        cached = self.task_class_weights.get(task_id)
        if cached is not None:
            return cached.to(device)
        class_ids = self._classes(task_id)
        if not class_ids:
            return None
        positions = {int(class_id): index for index, class_id in enumerate(class_ids)}
        counts = torch.zeros(len(class_ids), dtype=torch.float32)
        for index in range(len(loader.dataset)):
            _sample, target = loader.dataset[index]
            target_value = int(target.item()) if isinstance(target, torch.Tensor) else int(target)
            position = positions.get(target_value)
            if position is not None:
                counts[position] += 1.0
        observed = counts > 0
        weights = torch.ones_like(counts)
        if bool(observed.any()):
            observed_counts = counts[observed].clamp_min(float(self.class_balance_smoothing))
            inverse = (observed_counts.sum() / observed_counts).pow(
                float(self.class_balance_power)
            )
            inverse = inverse / inverse.mean().clamp_min(1e-12)
            weights[observed] = inverse.clamp(max=float(self.class_weight_max))
        self.task_class_weights[task_id] = weights.detach().cpu()
        return weights.to(device)

    def _sample_replay(
        self,
        device: torch.device | str,
    ) -> tuple[str, torch.Tensor, torch.Tensor] | None:
        available = [
            task_id
            for task_id, (inputs, _targets) in self.replay_memory.items()
            if inputs.numel() > 0
        ]
        if not available or int(self.replay_batch_size) <= 0:
            return None
        task_id = random.choice(available)
        inputs, targets = self.replay_memory[task_id]
        count = min(int(self.replay_batch_size), int(targets.shape[0]))
        indices = torch.randperm(int(targets.shape[0]))[:count]
        return task_id, inputs[indices].to(device), targets[indices].to(device)

    def _replay_loss(
        self,
        model: nn.Module,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, int]:
        if self.variant == "ramp":
            # RAMP uses the same bounded GEM memory as robust inequality
            # constraints; sampling it again as an unconstrained additive loss
            # would double-count arbitrary old tasks.
            return torch.zeros((), device=device), 0
        replay = self._sample_replay(device)
        if replay is None:
            return torch.zeros((), device=device), 0
        replay_task_id, replay_inputs, replay_targets = replay
        clean_logits, local_targets = _task_view(
            model(replay_inputs), replay_targets, self._classes(replay_task_id)
        )
        assert local_targets is not None
        adversarial = self._attack(
            model,
            replay_inputs,
            replay_targets,
            replay_task_id,
            clean_logits=clean_logits.detach(),
        )
        model.train()
        adversarial_logits, _ = _task_view(
            model(adversarial), None, self._classes(replay_task_id)
        )
        replay_weights = self._class_weights(
            replay_task_id,
            self.task_loaders[replay_task_id],
            device,
        )
        clean_loss = F.cross_entropy(
            clean_logits, local_targets, weight=replay_weights
        )
        adversarial_loss = F.cross_entropy(
            adversarial_logits, local_targets, weight=replay_weights
        )
        consistency = _kl_loss(
            adversarial_logits, clean_logits.detach(), temperature=1.0
        )
        adversarial_weight, trades_weight, _robust_kd_weight = (
            self._variant_weights()
        )
        return (
            float(self.clean_weight) * clean_loss
            + adversarial_weight * adversarial_loss
            + trades_weight * consistency,
            int(replay_targets.shape[0]),
        )

    def _prepare_continual_gradients(
        self,
        task_id: str,
        *,
        inputs: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        remember: bool,
    ) -> tuple[list[nn.Parameter], list[torch.Tensor]]:
        """Prepare LOCI's GEM constraints before a main-model forward pass.

        Private stream examples populate GEM's episodic memory. Public repair
        examples must obey existing constraints but must not enter that private
        memory, so callers select the behavior with ``remember``.
        """
        parameters = trainable_parameters(self.trainer.model)
        if self.continual_method != "gem":
            return parameters, []
        if remember:
            if inputs is None or targets is None:
                raise ValueError("GEM memory updates require inputs and targets.")
            self._update_episodic_memory(task_id, inputs, targets)
        return parameters, self._gem_memory_gradients(task_id, parameters)

    def _apply_continual_step(
        self,
        loss: torch.Tensor,
        parameters: Sequence[nn.Parameter],
        memory_gradients: Sequence[torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> tuple[int, bool, torch.Tensor]:
        """Apply EWC only for its ablation, or project the full robust GEM step."""
        if self.continual_method == "ewc":
            loss = loss + float(self.ewc_lambda) * self._ewc_penalty()
        optimizer.zero_grad()
        loss.backward()
        violations = 0
        projected = False
        if memory_gradients:
            gradient = flatten_gradients(parameters)
            memories = torch.stack(list(memory_gradients))
            violations = int(((memories @ gradient) < 0.0).sum().item())
            if violations:
                gradient = project_gradient(
                    gradient,
                    memories,
                    margin=float(self.gem_memory_strength),
                    eps=float(self.gem_qp_eps),
                )
                assign_flat_gradient(parameters, gradient)
                projected = True
        optimizer.step()
        return violations, projected, loss

    def _apply_clean_preserving_step(
        self,
        clean_loss: torch.Tensor,
        robust_loss: torch.Tensor,
        parameters: Sequence[nn.Parameter],
        memory_gradients: Sequence[torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> tuple[int, bool, torch.Tensor, bool, float, float]:
        """Merge robust gradients without opposing the clean objective.

        This is a one-sided PCGrad-style update: clean learning is the primary
        objective.  Only the component of the robust gradient that conflicts
        with it is removed, then its norm is capped relative to the clean
        gradient.  Robust GEM constraints are applied to that merged direction
        afterwards, so past robust risk remains protected as well.
        """
        if self.continual_method == "ewc":
            clean_loss = clean_loss + float(self.ewc_lambda) * self._ewc_penalty()
        optimizer.zero_grad()
        clean_parts = torch.autograd.grad(
            clean_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        robust_parts = torch.autograd.grad(
            robust_loss,
            parameters,
            allow_unused=True,
        )
        clean_gradient = _flatten_explicit_gradients(parameters, clean_parts)
        robust_gradient = _flatten_explicit_gradients(parameters, robust_parts)
        clean_norm = torch.linalg.vector_norm(clean_gradient)
        robust_norm = torch.linalg.vector_norm(robust_gradient)
        dot_product = torch.dot(clean_gradient, robust_gradient)
        denominator = (clean_norm * robust_norm).clamp_min(1e-12)
        cosine = float((dot_product / denominator).item())
        conflict = bool(dot_product.item() < 0.0)
        if conflict and float(clean_norm.item()) > 0.0:
            robust_gradient = robust_gradient - (
                dot_product / clean_norm.square().clamp_min(1e-12)
            ) * clean_gradient
            robust_norm = torch.linalg.vector_norm(robust_gradient)

        ratio = float(self.robust_gradient_ratio)
        scale = 1.0
        if float(robust_norm.item()) > 0.0:
            maximum = ratio * float(clean_norm.item())
            scale = min(1.0, maximum / float(robust_norm.item()))
        merged_gradient = clean_gradient + scale * robust_gradient
        violations = 0
        projected = False
        if memory_gradients:
            memories = torch.stack(list(memory_gradients))
            violations = int(((memories @ merged_gradient) < 0.0).sum().item())
            if violations:
                merged_gradient = project_gradient(
                    merged_gradient,
                    memories,
                    margin=float(self.gem_memory_strength),
                    eps=float(self.gem_qp_eps),
                )
                projected = True
        assign_flat_gradient(parameters, merged_gradient)
        optimizer.step()
        return (
            violations,
            projected,
            clean_loss + robust_loss,
            conflict,
            cosine,
            scale,
        )

    def _evaluate_kd_state_robust(
        self,
        state: Mapping[str, torch.Tensor],
        loader: DataLoader,
        task_id: str,
    ) -> tuple[float, float, float]:
        self.kd_model.load_state_dict(state, strict=True)
        self.kd_model.to(self.trainer.device)
        self.kd_model.eval()
        clean_correct = 0
        robust_correct = 0
        total = 0
        max_batches = int(self.teacher_eval_batches)
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            inputs = move_to_device(inputs, self.trainer.device)
            targets = move_to_device(targets, self.trainer.device)
            with torch.no_grad():
                clean_logits, local_targets = _task_view(
                    self.kd_model(inputs), targets, self._classes(task_id)
                )
            assert local_targets is not None
            adversarial = self._attack(
                self.kd_model,
                inputs,
                targets,
                task_id,
                clean_logits=clean_logits,
            )
            with torch.no_grad():
                adversarial_logits, _ = _task_view(
                    self.kd_model(adversarial), None, self._classes(task_id)
                )
            clean_correct += int(
                (clean_logits.argmax(dim=1) == local_targets).sum().item()
            )
            robust_correct += int(
                (adversarial_logits.argmax(dim=1) == local_targets).sum().item()
            )
            total += int(targets.shape[0])
        clean_accuracy = clean_correct / max(1, total)
        robust_accuracy = robust_correct / max(1, total)
        clean_weight = float(self.teacher_clean_weight)
        score = clean_weight * clean_accuracy + (1.0 - clean_weight) * robust_accuracy
        return clean_accuracy, robust_accuracy, score

    def _train_main_model(
        self,
        loader: DataLoader,
        task_id: str,
        teacher_state: Mapping[str, torch.Tensor],
        teacher_accuracy: float,
    ) -> MetricDict:
        # RAMP consumes the live communicable KD model selected by LOCI.  The
        # legacy ablations retain an isolated copy for reproducibility, but the
        # recommended method never creates or stores a historical snapshot.
        teacher = self.kd_model if self.variant == "ramp" else copy.deepcopy(self.kd_model)
        teacher.load_state_dict(teacher_state, strict=True)
        teacher.to(self.trainer.device)
        teacher.eval()
        model = self.trainer.model.to(self.trainer.device)
        self.trainer.optimizer = self._build_main_optimizer()
        temperature = float(self.temperature)
        adversarial_weight, trades_weight, robust_kd_weight = self._variant_weights()
        teacher_weight = max(
            float(teacher_accuracy), float(self.teacher_weight_floor)
        )
        class_weights = self._class_weights(task_id, loader, self.trainer.device)
        totals = {
            "loss": 0.0,
            "classification_loss": 0.0,
            "adversarial_loss": 0.0,
            "trades_loss": 0.0,
            "integrator_kd_loss": 0.0,
            "robust_kd_loss": 0.0,
            "boundary_alignment_loss": 0.0,
            "replay_loss": 0.0,
            "accuracy": 0.0,
            "robust_accuracy": 0.0,
        }
        examples = 0
        batches = 0
        constraint_violations = 0
        gradient_projections = 0
        gradient_conflicts = 0
        gradient_cosine_sum = 0.0
        robust_gradient_scale_sum = 0.0
        for _ in range(max(0, int(self.epochs))):
            model.train()
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                parameters, memory_gradients = self._prepare_continual_gradients(
                    task_id,
                    inputs=inputs,
                    targets=targets,
                    remember=True,
                )
                clean_logits, local_targets = _task_view(
                    model(inputs), targets, self._classes(task_id)
                )
                assert local_targets is not None
                with torch.no_grad():
                    teacher_clean, _ = _task_view(
                        teacher(inputs), None, self._classes(task_id)
                    )
                adversarial = self._attack(
                    model,
                    inputs,
                    targets,
                    task_id,
                    clean_logits=clean_logits.detach(),
                )
                model.train()
                adversarial_logits, _ = _task_view(
                    model(adversarial), None, self._classes(task_id)
                )
                with torch.no_grad():
                    teacher_adversarial, _ = _task_view(
                        teacher(adversarial), None, self._classes(task_id)
                    )

                clean_ce = F.cross_entropy(
                    clean_logits, local_targets, weight=class_weights
                )
                adversarial_ce = F.cross_entropy(
                    adversarial_logits, local_targets, weight=class_weights
                )
                consistency = _kl_loss(
                    adversarial_logits, clean_logits.detach(), temperature=1.0
                )
                clean_kd = _kl_loss(clean_logits, teacher_clean, temperature)
                adversarial_kd = _kl_loss(
                    adversarial_logits, teacher_adversarial, temperature
                )
                replay_loss, _replay_examples = self._replay_loss(
                    model, self.trainer.device
                )
                if self.variant == "ramp":
                    chance = 1.0 / max(1, int(clean_logits.shape[1]))
                    reliability = max(
                        0.0,
                        (float(teacher_weight) - chance) / max(1e-12, 1.0 - chance),
                    )
                    integration = min(
                        1.0,
                        float(self.integrator_weight)
                        * float(self.kd_alpha)
                        * reliability,
                    )
                    clean_objective = _soft_target_cross_entropy(
                        clean_logits,
                        local_targets,
                        teacher_clean,
                        teacher_weight=integration,
                        temperature=temperature,
                        class_weights=class_weights,
                    )
                    adversarial_objective = _soft_target_cross_entropy(
                        adversarial_logits,
                        local_targets,
                        teacher_adversarial,
                        teacher_weight=integration,
                        temperature=temperature,
                        class_weights=class_weights,
                    )
                    boundary_loss = _boundary_alignment_loss(
                        clean_logits,
                        adversarial_logits,
                        teacher_clean,
                        teacher_adversarial,
                    )
                    clean_loss = float(self.clean_weight) * clean_objective
                    robust_loss = (
                        adversarial_weight * adversarial_objective
                        + trades_weight * consistency
                        + float(self.boundary_weight)
                        * float(self._robust_scale)
                        * integration
                        * boundary_loss
                    )
                    (
                        violations,
                        projected,
                        loss,
                        gradient_conflict,
                        gradient_cosine,
                        robust_gradient_scale,
                    ) = self._apply_clean_preserving_step(
                        clean_loss,
                        robust_loss,
                        parameters,
                        memory_gradients,
                        self.trainer.optimizer,
                    )
                else:
                    boundary_loss = torch.zeros_like(clean_ce)
                    # Accuracy-weighted KD is LOCI Eq. (7).  Here the accuracy
                    # is the clean/robust teacher score, preventing a brittle
                    # fused teacher from dominating local integration.
                    loss = (
                        float(self.clean_weight) * clean_ce
                        + adversarial_weight * adversarial_ce
                        + trades_weight * consistency
                        + float(self.integrator_weight) * teacher_weight * clean_kd
                        + robust_kd_weight * teacher_weight * adversarial_kd
                        + float(self.replay_weight) * replay_loss
                    )
                    violations, projected, loss = self._apply_continual_step(
                        loss,
                        parameters,
                        memory_gradients,
                        self.trainer.optimizer,
                    )
                    gradient_conflict = False
                    gradient_cosine = 0.0
                    robust_gradient_scale = 1.0
                batches += 1
                constraint_violations += violations
                gradient_projections += int(projected)
                gradient_conflicts += int(gradient_conflict)
                gradient_cosine_sum += gradient_cosine
                robust_gradient_scale_sum += robust_gradient_scale

                batch_size = int(targets.shape[0])
                examples += batch_size
                values = {
                    "loss": loss,
                    "classification_loss": clean_ce,
                    "adversarial_loss": adversarial_ce,
                    "trades_loss": consistency,
                    "integrator_kd_loss": clean_kd,
                    "robust_kd_loss": adversarial_kd,
                    "boundary_alignment_loss": boundary_loss,
                    "replay_loss": replay_loss,
                }
                for name, value in values.items():
                    totals[name] += float(value.detach().item()) * batch_size
                totals["accuracy"] += float(
                    (clean_logits.argmax(dim=1) == local_targets).sum().item()
                )
                totals["robust_accuracy"] += float(
                    (adversarial_logits.argmax(dim=1) == local_targets).sum().item()
                )
        denominator = max(1, examples)
        metrics = {name: value / denominator for name, value in totals.items()}
        metrics["robust_scale"] = float(self._robust_scale)
        metrics["teacher_weight"] = teacher_weight
        metrics["gem_constraint_violations"] = constraint_violations / max(1, batches)
        metrics["gem_gradient_projections"] = gradient_projections / max(1, batches)
        metrics["clean_robust_gradient_conflicts"] = gradient_conflicts / max(1, batches)
        metrics["clean_robust_gradient_cosine"] = gradient_cosine_sum / max(1, batches)
        metrics["robust_gradient_scale"] = robust_gradient_scale_sum / max(1, batches)
        if self.variant == "ramp":
            metrics["teacher_integration_weight"] = integration if batches else 0.0
        if class_weights is not None and class_weights.numel() > 0:
            metrics["class_weight_min"] = float(class_weights.min().item())
            metrics["class_weight_max"] = float(class_weights.max().item())
        return metrics

    def _distill_kd_model(self, loader: DataLoader, task_id: str) -> float:
        kd_model = self.kd_model.to(self.trainer.device)
        main_model = self.trainer.model.to(self.trainer.device)
        main_model.eval()
        optimizer = torch.optim.Adam(kd_model.parameters(), lr=float(self.kd_lr))
        temperature = float(self.temperature)
        alpha = float(self.kd_alpha)
        adversarial_weight, trades_weight, robust_kd_weight = self._variant_weights()
        class_weights = self._class_weights(task_id, loader, self.trainer.device)
        total_loss = 0.0
        total_examples = 0
        for _ in range(max(0, int(self.kd_epochs))):
            kd_model.train()
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                clean_logits, local_targets = _task_view(
                    kd_model(inputs), targets, self._classes(task_id)
                )
                assert local_targets is not None
                with torch.no_grad():
                    teacher_clean, _ = _task_view(
                        main_model(inputs), None, self._classes(task_id)
                    )
                adversarial = self._attack(
                    kd_model,
                    inputs,
                    targets,
                    task_id,
                    clean_logits=clean_logits.detach(),
                )
                kd_model.train()
                adversarial_logits, _ = _task_view(
                    kd_model(adversarial), None, self._classes(task_id)
                )
                with torch.no_grad():
                    teacher_adversarial, _ = _task_view(
                        main_model(adversarial), None, self._classes(task_id)
                    )

                clean_ce = F.cross_entropy(
                    clean_logits, local_targets, weight=class_weights
                )
                adversarial_ce = F.cross_entropy(
                    adversarial_logits, local_targets, weight=class_weights
                )
                clean_kd = _kl_loss(clean_logits, teacher_clean, temperature)
                adversarial_kd = _kl_loss(
                    adversarial_logits, teacher_adversarial, temperature
                )
                consistency = _kl_loss(
                    adversarial_logits, clean_logits.detach(), temperature=1.0
                )
                if self.variant == "ramp":
                    # One target per view avoids CE/KD gradient conflict.  The
                    # boundary term is the architecture-independent robust
                    # knowledge later stored by LOCI's sparse memory palace.
                    clean_objective = _soft_target_cross_entropy(
                        clean_logits,
                        local_targets,
                        teacher_clean,
                        teacher_weight=alpha,
                        temperature=temperature,
                        class_weights=class_weights,
                    )
                    adversarial_objective = _soft_target_cross_entropy(
                        adversarial_logits,
                        local_targets,
                        teacher_adversarial,
                        teacher_weight=alpha,
                        temperature=temperature,
                        class_weights=class_weights,
                    )
                    boundary_loss = _boundary_alignment_loss(
                        clean_logits,
                        adversarial_logits,
                        teacher_clean,
                        teacher_adversarial,
                    )
                    loss = (
                        float(self.clean_weight) * clean_objective
                        + adversarial_weight * adversarial_objective
                        + float(self.boundary_weight)
                        * float(self._robust_scale)
                        * boundary_loss
                    )
                else:
                    # Legacy RADT/TRADES/ARD ablations.
                    loss = (
                        (1.0 - alpha) * clean_ce
                        + alpha * clean_kd
                        + adversarial_weight * (1.0 - alpha) * adversarial_ce
                        + robust_kd_weight * alpha * adversarial_kd
                        + trades_weight * consistency
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
        return total_loss / max(1, total_examples)

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        task_id = context.task_id or self.current_task_id
        if task_id is None:
            task_id = next(iter(self.task_loaders))
        if self.current_task_id != task_id:
            self.on_task_start(task_id)
        if int(self.robust_warmup_rounds) > 0 and context.round_idx is not None:
            self._robust_scale = min(
                1.0,
                (int(context.round_idx) + 1.0) / float(self.robust_warmup_rounds),
            )
        else:
            self._robust_scale = 1.0

        loader = self.task_loaders[task_id]
        base_state = _clone_tensor_state(self.kd_state or self.initial_kd_state)
        aggregated_state = _clone_tensor_state(global_state) if global_state else base_state
        base_clean, base_robust, base_score = self._evaluate_kd_state_robust(
            base_state, loader, task_id
        )
        agg_clean, agg_robust, agg_score = self._evaluate_kd_state_robust(
            aggregated_state, loader, task_id
        )
        use_aggregated = bool(global_state) and agg_score >= base_score
        teacher_state = aggregated_state if use_aggregated else base_state
        teacher_score = agg_score if use_aggregated else base_score
        self.kd_model.load_state_dict(teacher_state, strict=True)

        metrics = self._train_main_model(
            loader, task_id, teacher_state, teacher_score
        )
        metrics["kd_distill_loss"] = self._distill_kd_model(loader, task_id)
        metrics.update(
            {
                "kd_accuracy_before": base_clean,
                "kd_robust_accuracy_before": base_robust,
                "kd_score_before": base_score,
                "kd_accuracy_aggregated": agg_clean,
                "kd_robust_accuracy_aggregated": agg_robust,
                "kd_score_aggregated": agg_score,
                "used_aggregated_kd": float(use_aggregated),
            }
        )
        self.current_state = detach_state_dict(self.trainer.model.state_dict())
        self.kd_state = detach_state_dict(self.kd_model.state_dict())
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics=metrics,
            payload={
                "model_state": _clone_tensor_state(self.kd_state),
                "kd_state": _clone_tensor_state(self.kd_state),
                "local_state": _clone_tensor_state(self.current_state),
                "task_id": task_id,
                "robust_variant": self.variant,
            },
        )

    def _estimate_model_fisher(
        self,
        model: nn.Module,
        loader: DataLoader,
        task_id: str,
    ) -> dict[str, torch.Tensor]:
        model.to(self.trainer.device)
        model.eval()
        fisher = {
            name: torch.zeros_like(parameter, device=self.trainer.device)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        examples = 0
        for batch_index, (inputs, targets) in enumerate(loader):
            if int(self.fisher_batches) > 0 and batch_index >= int(self.fisher_batches):
                break
            inputs = move_to_device(inputs, self.trainer.device)
            targets = move_to_device(targets, self.trainer.device)
            with torch.no_grad():
                clean_reference, _ = _task_view(
                    model(inputs), None, self._classes(task_id)
                )
            adversarial = self._attack(
                model,
                inputs,
                targets,
                task_id,
                clean_logits=clean_reference,
            )
            clean_logits, local_targets = _task_view(
                model(inputs), targets, self._classes(task_id)
            )
            adversarial_logits, _ = _task_view(
                model(adversarial), None, self._classes(task_id)
            )
            assert local_targets is not None
            loss = F.cross_entropy(clean_logits, local_targets) + float(
                self.fisher_adversarial_weight
            ) * F.cross_entropy(adversarial_logits, local_targets)
            model.zero_grad()
            loss.backward()
            batch_size = int(targets.shape[0])
            examples += batch_size
            for name, parameter in model.named_parameters():
                if name in fisher and parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().square() * batch_size
        return {
            name: (value / max(1, examples)).detach().cpu()
            for name, value in fisher.items()
        }

    def _estimate_fisher(
        self, loader: DataLoader, task_id: str
    ) -> dict[str, torch.Tensor]:
        return self._estimate_model_fisher(self.trainer.model, loader, task_id)

    def _pruning_masks(
        self, state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        parameter_names = [name for name, _ in self.kd_model.named_parameters()]
        scores: list[torch.Tensor] = []
        score_by_name: dict[str, torch.Tensor] = {}
        for name in parameter_names:
            value = state.get(name)
            if value is None or not value.is_floating_point():
                continue
            score = value.detach().abs().float()
            importance = self.kd_fisher_state.get(name)
            if importance is not None and importance.shape == score.shape:
                normalized = importance.float() / importance.float().mean().clamp_min(1e-12)
                score = score * (1.0 + float(self.importance_weight) * normalized)
            score_by_name[name] = score
            scores.append(score.flatten())
        total = sum(score.numel() for score in scores)
        keep = min(total, max(1, int(math.ceil(total * float(self.knowledge_ratio)))))
        masks = {
            name: torch.zeros_like(state[name], dtype=torch.bool)
            for name in score_by_name
        }
        if total == 0:
            return masks
        selected = torch.topk(
            torch.cat(scores), k=keep, largest=True, sorted=False
        ).indices
        offset = 0
        for name in parameter_names:
            if name not in masks:
                continue
            length = masks[name].numel()
            local = selected[(selected >= offset) & (selected < offset + length)] - offset
            masks[name].flatten()[local] = True
            offset += length
        return masks

    def extract_task_knowledge(self, task_id: str) -> LociTaskKnowledge:
        """Extract sparse KD knowledge and adversarially repair retained weights."""
        dense_state = _clone_tensor_state(self.kd_state or self.kd_model.state_dict())
        masks = self._pruning_masks(dense_state)
        sparse_model = copy.deepcopy(self.kd_model).to(self.trainer.device)
        sparse_state = _clone_tensor_state(dense_state)
        for name, mask in masks.items():
            sparse_state[name] = sparse_state[name] * mask.to(sparse_state[name])
        sparse_model.load_state_dict(sparse_state, strict=True)
        optimizer = torch.optim.Adam(sparse_model.parameters(), lr=float(self.kd_lr))
        named_parameters = dict(sparse_model.named_parameters())
        loader = self.task_loaders[task_id]
        for _ in range(max(0, int(self.knowledge_finetune_epochs))):
            sparse_model.train()
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                with torch.no_grad():
                    reference, _ = _task_view(
                        sparse_model(inputs), None, self._classes(task_id)
                    )
                adversarial = self._attack(
                    sparse_model,
                    inputs,
                    targets,
                    task_id,
                    clean_logits=reference,
                )
                sparse_model.train()
                clean_logits, local_targets = _task_view(
                    sparse_model(inputs), targets, self._classes(task_id)
                )
                adversarial_logits, _ = _task_view(
                    sparse_model(adversarial), None, self._classes(task_id)
                )
                assert local_targets is not None
                loss = F.cross_entropy(clean_logits, local_targets) + float(
                    self.knowledge_robust_weight
                ) * F.cross_entropy(adversarial_logits, local_targets)
                optimizer.zero_grad()
                loss.backward()
                for name, mask in masks.items():
                    parameter = named_parameters.get(name)
                    if parameter is not None and parameter.grad is not None:
                        parameter.grad.mul_(mask.to(parameter.grad))
                optimizer.step()
                with torch.no_grad():
                    for name, mask in masks.items():
                        parameter = named_parameters.get(name)
                        if parameter is not None:
                            parameter.mul_(mask.to(parameter))
        sparse_state = detach_state_dict(sparse_model.state_dict())
        return LociTaskKnowledge(
            knowledge_id=f"{self.client_id}:{task_id}",
            client_id=self.client_id,
            task_id=task_id,
            state=_clone_tensor_state(sparse_state),
            mask_state={
                name: mask.detach().cpu().clone() for name, mask in masks.items()
            },
            num_samples=len(loader.dataset),
        )

    def _update_replay_memory(self, task_id: str) -> None:
        """Keep class-balanced adversarial boundary samples for later tasks.

        TABA keeps decision-boundary samples and RAER prioritizes examples
        that expose weak robustness.  Here the per-sample adversarial CE is a
        common score that works with LOCI's arbitrary heterogeneous model.
        Memory never leaves the client and is globally capped.
        """
        if self.variant == "ramp":
            self.replay_memory.clear()
            return
        budget = int(self.replay_budget)
        if budget <= 0:
            self.replay_memory.clear()
            return
        task_count = len(set(self.replay_memory) | {task_id})
        per_task = max(1, budget // max(1, task_count))
        for old_task_id, (inputs, targets) in list(self.replay_memory.items()):
            self.replay_memory[old_task_id] = (
                inputs[:per_task].clone(),
                targets[:per_task].clone(),
            )

        model = self.trainer.model.to(self.trainer.device)
        model.eval()
        candidates: dict[int, list[tuple[float, torch.Tensor, int]]] = {}
        loader = self.task_loaders[task_id]
        for batch_index, (inputs, targets) in enumerate(loader):
            if (
                int(self.replay_selection_batches) > 0
                and batch_index >= int(self.replay_selection_batches)
            ):
                break
            inputs = move_to_device(inputs, self.trainer.device)
            targets = move_to_device(targets, self.trainer.device)
            with torch.no_grad():
                clean_logits, local_targets = _task_view(
                    model(inputs), targets, self._classes(task_id)
                )
            assert local_targets is not None
            adversarial = self._attack(
                model,
                inputs,
                targets,
                task_id,
                clean_logits=clean_logits,
            )
            with torch.no_grad():
                adversarial_logits, _ = _task_view(
                    model(adversarial), None, self._classes(task_id)
                )
                scores = F.cross_entropy(
                    adversarial_logits, local_targets, reduction="none"
                )
            for index in range(int(targets.shape[0])):
                target = int(targets[index].item())
                candidates.setdefault(target, []).append(
                    (
                        float(scores[index].item()),
                        inputs[index].detach().cpu().clone(),
                        target,
                    )
                )

        selected: list[tuple[float, torch.Tensor, int]] = []
        if candidates:
            class_quota = max(1, per_task // len(candidates))
            leftovers: list[tuple[float, torch.Tensor, int]] = []
            for values in candidates.values():
                ordered = sorted(values, key=lambda item: item[0], reverse=True)
                selected.extend(ordered[:class_quota])
                leftovers.extend(ordered[class_quota:])
            if len(selected) < per_task:
                leftovers.sort(key=lambda item: item[0], reverse=True)
                selected.extend(leftovers[: per_task - len(selected)])
            selected.sort(key=lambda item: item[0], reverse=True)
            selected = selected[:per_task]
        if selected:
            self.replay_memory[task_id] = (
                torch.stack([item[1] for item in selected]),
                torch.tensor([item[2] for item in selected], dtype=torch.long),
            )

    def refine_with_public_data(
        self,
        fused_kd_state: Mapping[str, torch.Tensor],
        loader: DataLoader,
        task_id: str,
    ) -> MetricDict:
        """Repair a fused KD model and inject missing-class knowledge locally.

        LOCI already allocates labeled public task data for activation-based
        selection.  Using one small robust refinement epoch after OT fusion
        prevents model-heterogeneous, Non-IID clients from evaluating classes
        that never appeared in their private partition with random head rows.
        """
        if int(self.public_refine_epochs) <= 0:
            return {}
        device = self.trainer.device
        kd_model = self.kd_model.to(device)
        main_model = self.trainer.model.to(device)
        kd_model.load_state_dict(fused_kd_state, strict=True)
        kd_optimizer = torch.optim.Adam(
            kd_model.parameters(),
            lr=float(self.kd_lr) * float(self.public_refine_lr_scale),
        )
        main_optimizer = self._build_main_optimizer()
        for group in main_optimizer.param_groups:
            group["lr"] = float(group["lr"]) * float(self.public_refine_lr_scale)
        # _gem_memory_gradients operates on trainer.model and clears its
        # registered optimizer. Keep that registration aligned with this phase.
        self.trainer.optimizer = main_optimizer
        temperature = float(self.temperature)
        adversarial_weight, trades_weight, robust_kd_weight = self._variant_weights()
        total_loss = 0.0
        total_examples = 0
        total_correct = 0
        total_robust_correct = 0
        total_constraint_violations = 0
        total_gradient_projections = 0
        total_batches = 0

        for _ in range(int(self.public_refine_epochs)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, device)
                targets = move_to_device(targets, device)

                kd_model.train()
                kd_clean, local_targets = _task_view(
                    kd_model(inputs), targets, self._classes(task_id)
                )
                assert local_targets is not None
                kd_adversarial_inputs = self._attack(
                    kd_model,
                    inputs,
                    targets,
                    task_id,
                    clean_logits=kd_clean.detach(),
                )
                kd_model.train()
                kd_adversarial, _ = _task_view(
                    kd_model(kd_adversarial_inputs), None, self._classes(task_id)
                )
                kd_loss = (
                    float(self.clean_weight)
                    * F.cross_entropy(kd_clean, local_targets)
                    + adversarial_weight
                    * F.cross_entropy(kd_adversarial, local_targets)
                    + trades_weight
                    * _kl_loss(
                        kd_adversarial, kd_clean.detach(), temperature=1.0
                    )
                )
                kd_optimizer.zero_grad()
                kd_loss.backward()
                kd_optimizer.step()

                kd_model.eval()
                main_model.train()
                parameters, memory_gradients = self._prepare_continual_gradients(
                    task_id,
                    remember=False,
                )
                main_clean, _ = _task_view(
                    main_model(inputs), targets, self._classes(task_id)
                )
                main_adversarial_inputs = self._attack(
                    main_model,
                    inputs,
                    targets,
                    task_id,
                    clean_logits=main_clean.detach(),
                )
                main_model.train()
                main_adversarial, _ = _task_view(
                    main_model(main_adversarial_inputs),
                    None,
                    self._classes(task_id),
                )
                with torch.no_grad():
                    teacher_clean, _ = _task_view(
                        kd_model(inputs), None, self._classes(task_id)
                    )
                    teacher_adversarial, _ = _task_view(
                        kd_model(main_adversarial_inputs),
                        None,
                        self._classes(task_id),
                    )
                consistency = _kl_loss(
                    main_adversarial, main_clean.detach(), temperature=1.0
                )
                main_loss = (
                    float(self.clean_weight)
                    * F.cross_entropy(main_clean, local_targets)
                    + adversarial_weight
                    * F.cross_entropy(main_adversarial, local_targets)
                    + trades_weight * consistency
                    + 0.5
                    * float(self.integrator_weight)
                    * _kl_loss(main_clean, teacher_clean, temperature)
                    + 0.5
                    * robust_kd_weight
                    * _kl_loss(main_adversarial, teacher_adversarial, temperature)
                )
                violations, projected, main_loss = self._apply_continual_step(
                    main_loss,
                    parameters,
                    memory_gradients,
                    main_optimizer,
                )
                total_batches += 1
                total_constraint_violations += violations
                total_gradient_projections += int(projected)

                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float((kd_loss + main_loss).detach().item()) * batch_size
                total_correct += int(
                    (main_clean.argmax(dim=1) == local_targets).sum().item()
                )
                total_robust_correct += int(
                    (main_adversarial.argmax(dim=1) == local_targets).sum().item()
                )

        self.current_state = detach_state_dict(main_model.state_dict())
        self.kd_state = detach_state_dict(kd_model.state_dict())
        return {
            "public_refine_loss": total_loss / max(1, total_examples),
            "public_refine_accuracy": total_correct / max(1, total_examples),
            "public_refine_robust_accuracy": total_robust_correct
            / max(1, total_examples),
            "public_refine_samples": float(total_examples),
            "public_refine_gem_constraint_violations": total_constraint_violations
            / max(1, total_batches),
            "public_refine_gem_gradient_projections": total_gradient_projections
            / max(1, total_batches),
        }

    def on_task_end(self, task_id: str) -> LociTaskKnowledge:
        # KD Fisher is a pruning-importance signal, independent of the GEM
        # memory used by the main model. The optional EWC ablation additionally
        # consolidates main-model Fisher inside super().on_task_end.
        self.kd_fisher_state = self._estimate_model_fisher(
            self.kd_model, self.task_loaders[task_id], task_id
        )
        knowledge = super().on_task_end(task_id)
        self._update_replay_memory(task_id)
        return knowledge


@dataclass
class RobustLociServer(LociServer):
    """LOCI server with clean/adversarial dual-view task selection.

    OT alignment and the memory-palace protocol remain exactly LOCI's.  Only
    activation similarity is enriched: every candidate is represented by its
    logits on both clean public inputs and its own task-aware PGD inputs.  A
    neighbor must therefore be close in semantics and local decision-boundary
    behavior before its sparse task knowledge is fused into a client center.
    """

    selector_pgd_config: PGDConfig = field(default_factory=PGDConfig)
    robust_similarity: bool = True
    fusion_clean_tolerance: float = 0.0
    fusion_clean_loss_tolerance: float = 0.02
    fusion_min_robust_gain: float = 0.0
    _robust_probe_cache: dict[
        tuple[str, str], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = field(default_factory=dict, init=False, repr=False)
    _robust_signature_cache: dict[
        tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.fusion_clean_tolerance) < 0.0:
            raise ValueError("fusion_clean_tolerance must be non-negative.")
        if float(self.fusion_clean_loss_tolerance) < 0.0:
            raise ValueError("fusion_clean_loss_tolerance must be non-negative.")
        if float(self.fusion_min_robust_gain) < 0.0:
            raise ValueError("fusion_min_robust_gain must be non-negative.")

    def _robust_probes(
        self, center: _Candidate, task_id: str
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate one shared set of boundary probes from the center model."""
        cache_key = (center.candidate_id, task_id)
        cached = self._robust_probe_cache.get(cache_key)
        if cached is not None:
            return cached
        loader = self.public_loaders[task_id]
        device = next(self.kd_model.parameters()).device
        model = copy.deepcopy(self.kd_model).to(device)
        model.load_state_dict(center.state, strict=True)
        model.eval()
        class_ids = None if self.task_classes is None else self.task_classes.get(task_id)
        probes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for batch_index, (inputs, targets) in enumerate(loader):
            if int(self.selector_batches) > 0 and batch_index >= int(self.selector_batches):
                break
            inputs = move_to_device(inputs, device)
            targets = move_to_device(targets, device)
            adversarial = task_aware_pgd_linf_attack(
                model,
                inputs,
                targets,
                self.selector_pgd_config,
                class_ids=class_ids,
                objective="ce",
            )
            probes.append(
                (
                    inputs.detach().cpu(),
                    adversarial.detach().cpu(),
                    targets.detach().cpu(),
                )
            )
        self._robust_probe_cache[cache_key] = probes
        return probes

    def _signature_on_shared_probes(
        self,
        candidate: _Candidate,
        center: _Candidate,
        task_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (candidate.candidate_id, center.candidate_id, task_id)
        cached = self._robust_signature_cache.get(cache_key)
        if cached is not None:
            return cached
        device = next(self.kd_model.parameters()).device
        model = copy.deepcopy(self.kd_model).to(device)
        model.load_state_dict(candidate.state, strict=True)
        model.eval()
        class_ids = None if self.task_classes is None else self.task_classes.get(task_id)
        clean_outputs: list[torch.Tensor] = []
        adversarial_outputs: list[torch.Tensor] = []
        # Materialize attacks before entering no_grad; otherwise Python invokes
        # _robust_probes from inside the context and disables PGD autograd.
        probes = self._robust_probes(center, task_id)
        with torch.no_grad():
            for clean, adversarial, _targets in probes:
                clean_logits, _ = _task_view(
                    model(clean.to(device)), None, class_ids
                )
                adversarial_logits, _ = _task_view(
                    model(adversarial.to(device)), None, class_ids
                )
                clean_outputs.append(clean_logits.detach().cpu())
                adversarial_outputs.append(adversarial_logits.detach().cpu())
        clean_signature = (
            torch.cat(clean_outputs, dim=0) if clean_outputs else torch.empty(0, 0)
        )
        adversarial_signature = (
            torch.cat(adversarial_outputs, dim=0)
            if adversarial_outputs
            else torch.empty(0, 0)
        )
        signature = (clean_signature, adversarial_signature)
        self._robust_signature_cache[cache_key] = signature
        return signature

    def _probe_metrics(
        self,
        candidate: _Candidate,
        center: _Candidate,
        task_id: str,
    ) -> dict[str, float]:
        clean_logits, adversarial_logits = self._signature_on_shared_probes(
            candidate, center, task_id
        )
        probes = self._robust_probes(center, task_id)
        if clean_logits.numel() == 0 or not probes:
            return {
                "clean_accuracy": 0.0,
                "robust_accuracy": 0.0,
                "clean_loss": float("inf"),
                "robust_loss": float("inf"),
            }
        targets = torch.cat([values[2] for values in probes], dim=0).long()
        class_ids = None if self.task_classes is None else self.task_classes.get(task_id)
        if class_ids:
            positions = {int(class_id): index for index, class_id in enumerate(class_ids)}
            local_targets = torch.tensor(
                [positions[int(target.item())] for target in targets],
                dtype=torch.long,
            )
        else:
            local_targets = targets
        return {
            "clean_accuracy": float(
                (clean_logits.argmax(dim=1) == local_targets).float().mean().item()
            ),
            "robust_accuracy": float(
                (adversarial_logits.argmax(dim=1) == local_targets)
                .float()
                .mean()
                .item()
            ),
            "clean_loss": float(F.cross_entropy(clean_logits, local_targets).item()),
            "robust_loss": float(
                F.cross_entropy(adversarial_logits, local_targets).item()
            ),
        }

    @staticmethod
    def _interpolate_state(
        center: Mapping[str, torch.Tensor],
        fused: Mapping[str, torch.Tensor],
        coefficient: float,
    ) -> dict[str, torch.Tensor]:
        state = _clone_tensor_state(center)
        for name, center_value in state.items():
            fused_value = fused.get(name)
            if (
                fused_value is not None
                and center_value.is_floating_point()
                and fused_value.shape == center_value.shape
            ):
                state[name] = center_value + float(coefficient) * (
                    fused_value.to(center_value) - center_value
                )
        return state

    def _guarded_fusion(
        self,
        center: _Candidate,
        fused_state: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], float, float, float]:
        """Line-search OT fusion for robust gain under a clean constraint."""
        baseline = self._probe_metrics(center, center, center.task_id)
        best_state = _clone_tensor_state(center.state)
        best_coefficient = 0.0
        best_metrics = baseline
        # With a global class head, a task knowledge model usually has random
        # rows for classes it never observed.  Averaging those rows was the
        # main reason incoming LOCI states lost both clean and robust accuracy.
        # Transfer the aligned representation while keeping the center's
        # active semantic rows intact.
        safe_fused_state = _clone_tensor_state(fused_state)
        alignable = [
            name
            for name, module in self.kd_model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        if alignable:
            prefix = f"{alignable[-1]}." if alignable[-1] else ""
            class_ids = (
                None
                if self.task_classes is None
                else self.task_classes.get(center.task_id)
            )
            for suffix in ("weight", "bias"):
                name = prefix + suffix
                center_value = center.state.get(name)
                fused_value = safe_fused_state.get(name)
                if (
                    center_value is None
                    or fused_value is None
                    or center_value.shape != fused_value.shape
                ):
                    continue
                if class_ids and center_value.ndim >= 1:
                    indices = torch.tensor(class_ids, dtype=torch.long)
                    if int(indices.max().item()) < int(center_value.shape[0]):
                        fused_value[indices] = center_value[indices].to(fused_value)
                else:
                    safe_fused_state[name] = center_value.detach().clone()
        for coefficient in (0.25, 0.5, 0.75, 1.0):
            trial = _Candidate(
                candidate_id=f"guard:{center.candidate_id}:{coefficient}",
                client_id=center.client_id,
                task_id=center.task_id,
                state=self._interpolate_state(
                    center.state, safe_fused_state, coefficient
                ),
            )
            metrics = self._probe_metrics(trial, center, center.task_id)
            clean_feasible = metrics["clean_accuracy"] >= (
                baseline["clean_accuracy"] - float(self.fusion_clean_tolerance)
            )
            loss_feasible = metrics["clean_loss"] <= baseline["clean_loss"] * (
                1.0 + float(self.fusion_clean_loss_tolerance)
            )
            robust_gain = baseline["robust_loss"] - metrics["robust_loss"]
            robust_feasible = robust_gain >= float(self.fusion_min_robust_gain)
            if (
                clean_feasible
                and loss_feasible
                and robust_feasible
                and metrics["robust_loss"] < best_metrics["robust_loss"]
            ):
                best_state = trial.state
                best_coefficient = coefficient
                best_metrics = metrics
        return (
            best_state,
            best_coefficient,
            best_metrics["clean_accuracy"] - baseline["clean_accuracy"],
            baseline["robust_loss"] - best_metrics["robust_loss"],
        )

    def _distance(self, left: _Candidate, right: _Candidate) -> float:
        if self.similarity == "weight" or not self.robust_similarity:
            return super()._distance(left, right)
        left_clean, left_adversarial = self._signature_on_shared_probes(
            left, left, left.task_id
        )
        right_clean, right_adversarial = self._signature_on_shared_probes(
            right, left, left.task_id
        )
        if (
            left_clean.numel() == 0
            or right_clean.shape != left_clean.shape
            or right_adversarial.shape != left_adversarial.shape
        ):
            return float("inf")
        temperature = float(self.temperature)
        left_probabilities = F.softmax(left_clean / temperature, dim=1)
        right_probabilities = F.softmax(right_clean / temperature, dim=1)
        mixture = 0.5 * (left_probabilities + right_probabilities)
        left_kl = torch.sum(
            left_probabilities
            * (
                torch.log(left_probabilities.clamp_min(1e-12))
                - torch.log(mixture.clamp_min(1e-12))
            ),
            dim=1,
        )
        right_kl = torch.sum(
            right_probabilities
            * (
                torch.log(right_probabilities.clamp_min(1e-12))
                - torch.log(mixture.clamp_min(1e-12))
            ),
            dim=1,
        )
        semantic_distance = (0.5 * (left_kl + right_kl)).mean()
        left_boundary = left_adversarial - left_clean
        right_boundary = right_adversarial - right_clean
        left_boundary = left_boundary - left_boundary.mean(dim=1, keepdim=True)
        right_boundary = right_boundary - right_boundary.mean(dim=1, keepdim=True)
        boundary_distance = 0.5 * (
            1.0
            - F.cosine_similarity(
                left_boundary,
                right_boundary,
                dim=1,
                eps=1e-12,
            ).mean()
        )
        return float((0.5 * semantic_distance + 0.5 * boundary_distance).item())

    def _activation_signature(
        self, candidate: _Candidate, task_id: str
    ) -> torch.Tensor:
        if not self.robust_similarity:
            return super()._activation_signature(candidate, task_id)
        cache_key = (candidate.candidate_id, task_id)
        cached = self._activation_cache.get(cache_key)
        if cached is not None:
            return cached
        loader = self.public_loaders[task_id]
        device = next(self.kd_model.parameters()).device
        model = copy.deepcopy(self.kd_model).to(device)
        model.load_state_dict(candidate.state, strict=True)
        model.eval()
        class_ids = None if self.task_classes is None else self.task_classes.get(task_id)
        clean_outputs: list[torch.Tensor] = []
        adversarial_outputs: list[torch.Tensor] = []
        for batch_index, (inputs, targets) in enumerate(loader):
            if int(self.selector_batches) > 0 and batch_index >= int(self.selector_batches):
                break
            inputs = move_to_device(inputs, device)
            targets = move_to_device(targets, device)
            with torch.no_grad():
                clean_logits, _ = _task_view(model(inputs), None, class_ids)
            adversarial = task_aware_pgd_linf_attack(
                model,
                inputs,
                targets,
                self.selector_pgd_config,
                class_ids=class_ids,
                objective="kl",
                reference_logits=clean_logits,
            )
            with torch.no_grad():
                adversarial_logits, _ = _task_view(
                    model(adversarial), None, class_ids
                )
            clean_outputs.append(clean_logits.detach().cpu())
            adversarial_outputs.append(adversarial_logits.detach().cpu())
        outputs = clean_outputs + adversarial_outputs
        signature = torch.cat(outputs, dim=0) if outputs else torch.empty(0, 0)
        self._activation_cache[cache_key] = signature
        return signature

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        self._robust_probe_cache.clear()
        self._robust_signature_cache.clear()
        result = super().run_round(round_idx, task_id)
        selected = result.metadata.get("selected_knowledge", {})
        active_client_ids = list(selected) if isinstance(selected, Mapping) else []
        clients_by_id = {client.client_id: client for client in self.clients}
        fusion_coefficients: dict[str, float] = {}
        fusion_clean_deltas: list[float] = []
        fusion_robust_gains: list[float] = []
        for client_id in active_client_ids:
            client = clients_by_id.get(client_id)
            fused_state = self.client_kd_states.get(client_id)
            if not isinstance(client, RobustLociClient) or not fused_state:
                continue
            client_task_id = self.task_id_for_client(client_id, task_id)
            center_state = _clone_tensor_state(
                client.kd_state or client.kd_model.state_dict()
            )
            center = _Candidate(
                candidate_id=f"round:{round_idx}:{client_id}",
                client_id=client_id,
                task_id=client_task_id,
                state=center_state,
            )
            guarded_state, coefficient, clean_delta, robust_gain = (
                self._guarded_fusion(center, fused_state)
            )
            self.client_kd_states[client_id] = guarded_state
            fusion_coefficients[client_id] = coefficient
            fusion_clean_deltas.append(clean_delta)
            fusion_robust_gains.append(robust_gain)

        refinement_sums: dict[str, float] = {}
        refined_clients = 0
        for client_id in active_client_ids:
            client = clients_by_id.get(client_id)
            if not isinstance(client, RobustLociClient):
                continue
            client_task_id = self.task_id_for_client(client_id, task_id)
            fused_state = self.client_kd_states.get(client_id)
            loader = self.public_loaders.get(client_task_id)
            if not fused_state or loader is None:
                continue
            refine_metrics = client.refine_with_public_data(
                fused_state, loader, client_task_id
            )
            if not refine_metrics:
                continue
            self.client_kd_states[client_id] = _clone_tensor_state(
                client.kd_state or fused_state
            )
            refined_clients += 1
            for name, value in refine_metrics.items():
                refinement_sums[name] = refinement_sums.get(name, 0.0) + float(value)

        metrics = dict(result.metrics)
        metrics["fusion_coefficient"] = sum(fusion_coefficients.values()) / max(
            1, len(fusion_coefficients)
        )
        metrics["fusion_clean_accuracy_delta"] = sum(fusion_clean_deltas) / max(
            1, len(fusion_clean_deltas)
        )
        metrics["fusion_robust_loss_gain"] = sum(fusion_robust_gains) / max(
            1, len(fusion_robust_gains)
        )
        if refined_clients > 0:
            for name, value in refinement_sums.items():
                metrics[name] = value / refined_clients
        metrics["public_refined_clients"] = float(refined_clients)
        metadata = dict(result.metadata)
        metadata["robust_similarity"] = bool(self.robust_similarity)
        metadata["fusion_coefficients"] = fusion_coefficients
        metadata["public_refined_clients"] = refined_clients
        first_client_id = next(iter(active_client_ids), None)
        global_state = (
            self.client_kd_states.get(first_client_id, result.global_state)
            if first_client_id is not None
            else result.global_state
        )
        return AggregationResult(
            global_state=_clone_tensor_state(global_state),
            metrics=metrics,
            metadata=metadata,
        )


# Short aliases make experiment scripts and downstream ablations easier to read.
OwnClient = RobustLociClient
OwnServer = RobustLociServer


__all__ = [
    "OwnClient",
    "OwnServer",
    "RobustLociClient",
    "RobustLociServer",
    "evaluate_task_aware_pgd",
    "task_aware_pgd_linf_attack",
]
