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

from ..contracts import (
    AggregationResult,
    ClientContext,
    MetricDict,
    StateDict,
    TaskDefinition,
    TrainResult,
)
from ..trainers.trainer import BaseTrainer
from ..trainers.utils import detach_state_dict, move_to_device
from .CL.base import assign_flat_gradient, flatten_gradients, trainable_parameters
from .CL.gem import project_gradient


def _clone_tensor_state(state: Mapping[str, object]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def _task_view(
    logits: torch.Tensor,
    targets: torch.Tensor | None,
    class_ids: Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not class_ids:
        return logits, targets
    ids = torch.tensor([int(class_id) for class_id in class_ids], device=logits.device)
    if ids.numel() == logits.shape[1] and torch.equal(
        ids, torch.arange(logits.shape[1], device=logits.device)
    ):
        return logits, targets
    task_logits = logits.index_select(1, ids)
    if targets is None:
        return task_logits, None
    local_targets = torch.zeros_like(targets)
    for local_index, class_id in enumerate(ids.tolist()):
        local_targets = torch.where(
            targets == int(class_id),
            torch.full_like(targets, local_index),
            local_targets,
        )
    return task_logits, local_targets


def _weight_distance(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    """Layer-averaged relative parameter distance from Loci Eq. (1)."""
    distances: list[float] = []
    for name, left_value in left.items():
        right_value = right.get(name)
        if (
            right_value is None
            or left_value.shape != right_value.shape
            or not left_value.is_floating_point()
        ):
            continue
        denominator = torch.linalg.vector_norm(left_value.float()).clamp_min(1e-12)
        numerator = torch.linalg.vector_norm(left_value.float() - right_value.float())
        distances.append(float((numerator / denominator).item()))
    return sum(distances) / max(1, len(distances))


def _sinkhorn_transport(
    reference_features: torch.Tensor,
    source_features: torch.Tensor,
    regularization: float,
    iterations: int,
) -> torch.Tensor:
    """Return a row-stochastic OT map from source neurons to reference neurons."""
    reference = F.normalize(reference_features.float(), dim=1, eps=1e-12)
    source = F.normalize(source_features.float(), dim=1, eps=1e-12)
    cost = torch.cdist(reference, source, p=2).square()
    scale = torch.median(cost.detach())
    if not torch.isfinite(scale) or float(scale.item()) <= 1e-12:
        scale = torch.tensor(1.0, device=cost.device)
    kernel = torch.exp(-cost / (max(float(regularization), 1e-6) * scale)).clamp_min(1e-30)
    rows, columns = kernel.shape
    row_mass = torch.full((rows,), 1.0 / rows, device=kernel.device)
    column_mass = torch.full((columns,), 1.0 / columns, device=kernel.device)
    left_scale = torch.ones_like(row_mass)
    right_scale = torch.ones_like(column_mass)
    for _ in range(max(1, int(iterations))):
        left_scale = row_mass / (kernel @ right_scale).clamp_min(1e-30)
        right_scale = column_mass / (kernel.transpose(0, 1) @ left_scale).clamp_min(1e-30)
    transport = left_scale[:, None] * kernel * right_scale[None, :]
    return transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-30)


def _align_input_channels(weight: torch.Tensor, transport: torch.Tensor) -> torch.Tensor:
    source_channels = int(transport.shape[1])
    if weight.ndim == 2:
        input_size = int(weight.shape[1])
        if input_size == source_channels:
            return weight @ transport.transpose(0, 1).to(weight)
        if input_size % source_channels == 0:
            spatial_size = input_size // source_channels
            reshaped = weight.reshape(weight.shape[0], source_channels, spatial_size)
            aligned = torch.einsum("ois,ri->ors", reshaped, transport.to(weight))
            return aligned.reshape_as(weight)
    if weight.ndim >= 3 and int(weight.shape[1]) == source_channels:
        return torch.einsum("oi...,ri->or...", weight, transport.to(weight))
    return weight


def align_state_dict_ot(
    model: nn.Module,
    reference_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    *,
    regularization: float = 0.05,
    iterations: int = 20,
) -> dict[str, torch.Tensor]:
    """Align hidden neurons in ``source_state`` to ``reference_state``.

    This is the layer-wise transport in Loci Eq. (3)-(4).  The final classifier
    rows are kept in their semantic class order; only its incoming features are
    transported.
    """
    aligned = _clone_tensor_state(source_state)
    alignable = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    if not alignable:
        return aligned
    last_module_name = alignable[-1][0]
    previous_transport: torch.Tensor | None = None

    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if previous_transport is None:
                continue
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = prefix + suffix
                value = aligned.get(key)
                if (
                    value is not None
                    and value.ndim == 1
                    and value.numel() == previous_transport.shape[1]
                ):
                    aligned[key] = previous_transport.to(value) @ value
            continue
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        weight_key = prefix + "weight"
        reference_weight = reference_state.get(weight_key)
        source_weight = aligned.get(weight_key)
        if reference_weight is None or source_weight is None:
            previous_transport = None
            continue
        if previous_transport is not None and getattr(module, "groups", 1) == 1:
            source_weight = _align_input_channels(source_weight, previous_transport)

        if module_name == last_module_name or reference_weight.shape[0] != source_weight.shape[0]:
            aligned[weight_key] = source_weight
            previous_transport = None
            continue

        reference_features = reference_weight.detach().reshape(reference_weight.shape[0], -1)
        source_features = source_weight.detach().reshape(source_weight.shape[0], -1)
        if reference_features.shape[1] != source_features.shape[1]:
            aligned[weight_key] = source_weight
            previous_transport = None
            continue
        transport = _sinkhorn_transport(
            reference_features,
            source_features,
            regularization=regularization,
            iterations=iterations,
        )
        aligned[weight_key] = (
            transport.to(source_weight) @ source_weight.reshape(source_weight.shape[0], -1)
        ).reshape_as(source_weight)
        bias_key = prefix + "bias"
        source_bias = aligned.get(bias_key)
        if source_bias is not None and source_bias.ndim == 1:
            aligned[bias_key] = transport.to(source_bias) @ source_bias
        previous_transport = transport.detach().cpu()
    return aligned


def fuse_states_ot(
    model: nn.Module,
    center_state: Mapping[str, torch.Tensor],
    candidate_states: Sequence[Mapping[str, torch.Tensor]],
    *,
    regularization: float = 0.05,
    iterations: int = 20,
) -> dict[str, torch.Tensor]:
    """Fuse a center with aligned neighbors using the weights in Loci Eq. (5)."""
    center = _clone_tensor_state(center_state)
    if not candidate_states:
        return center
    aligned_candidates = [
        align_state_dict_ot(
            model,
            center,
            candidate,
            regularization=regularization,
            iterations=iterations,
        )
        for candidate in candidate_states
    ]
    fused: dict[str, torch.Tensor] = {}
    for name, center_value in center.items():
        values = [candidate[name] for candidate in aligned_candidates if name in candidate]
        if not center_value.is_floating_point() or not values:
            fused[name] = center_value.clone()
            continue
        neighbor_mean = torch.stack(
            [value.to(dtype=center_value.dtype) for value in values], dim=0
        ).mean(dim=0)
        fused[name] = 0.5 * center_value + 0.5 * neighbor_mean
    return fused


@dataclass
class LociTaskKnowledge:
    knowledge_id: str
    client_id: str
    task_id: str
    state: StateDict
    mask_state: dict[str, torch.Tensor]
    num_samples: int = 0


@dataclass
class TaskMemoryPalace:
    """Server-side store for sparse past-task knowledge."""

    entries: dict[str, LociTaskKnowledge] = field(default_factory=dict)

    def add(self, knowledge: LociTaskKnowledge) -> None:
        self.entries[knowledge.knowledge_id] = knowledge

    def all(self) -> list[LociTaskKnowledge]:
        return list(self.entries.values())

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class LociClient:
    """Loci client with continual local learning and a communicable KD model."""

    client_id: str
    trainer: BaseTrainer
    kd_model: nn.Module
    task_loaders: Mapping[str, DataLoader]
    epochs: int = 1
    kd_epochs: int = 1
    kd_lr: float = 1e-3
    temperature: float = 2.0
    kd_alpha: float = 0.5
    integrator_weight: float = 1.0
    continual_method: str = "gem"
    ewc_lambda: float = 100.0
    fisher_batches: int = 0
    gem_memory_size: int = 256
    gem_memory_strength: float = 0.5
    gem_qp_eps: float = 1e-3
    knowledge_ratio: float = 0.05
    knowledge_finetune_epochs: int = 1
    task_classes: Mapping[str, Sequence[int]] | None = None
    optimizer_name: str = "sgd"
    lr: float | None = None
    weight_decay: float = 0.0
    current_task_id: str | None = None
    current_state: StateDict | None = None
    kd_state: StateDict | None = None
    fisher_state: dict[str, torch.Tensor] = field(default_factory=dict)
    anchor_state: dict[str, torch.Tensor] = field(default_factory=dict)
    episodic_memory: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=dict
    )
    memory_positions: dict[str, int] = field(default_factory=dict)
    consolidated_tasks: int = 0
    initial_kd_state: dict[str, torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < float(self.knowledge_ratio) <= 1.0):
            raise ValueError("knowledge_ratio must be in (0, 1].")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive.")
        if not (0.0 <= float(self.kd_alpha) <= 1.0):
            raise ValueError("kd_alpha must be in [0, 1].")
        self.continual_method = str(self.continual_method).lower()
        if self.continual_method not in {"gem", "ewc"}:
            raise ValueError("continual_method must be either 'gem' or 'ewc'.")
        self.optimizer_name = str(self.optimizer_name).lower()
        if self.optimizer_name not in {"sgd", "adam"}:
            raise ValueError("optimizer_name must be either 'sgd' or 'adam'.")
        if self.continual_method == "gem" and self.optimizer_name != "sgd":
            raise ValueError(
                "GEM requires SGD because adaptive optimizer preconditioning can "
                "invalidate the projected gradient constraints."
            )
        if self.continual_method == "gem" and float(self.weight_decay) != 0.0:
            raise ValueError(
                "GEM requires zero weight_decay because optimizer-side decay is "
                "applied after the gradient projection."
            )
        if int(self.gem_memory_size) <= 0:
            raise ValueError("gem_memory_size must be positive.")
        if float(self.gem_memory_strength) < 0.0:
            raise ValueError("gem_memory_strength must be non-negative.")
        if self.lr is None:
            self.lr = float(self.trainer.optimizer.param_groups[0].get("lr", 0.001))
        self.initial_kd_state = _clone_tensor_state(self.kd_model.state_dict())
        self.kd_state = _clone_tensor_state(self.initial_kd_state)
        self.current_state = _clone_tensor_state(self.trainer.model.state_dict())

    def _classes(self, task_id: str) -> Sequence[int] | None:
        return None if self.task_classes is None else self.task_classes.get(task_id)

    def _build_main_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                self.trainer.model.parameters(),
                lr=float(self.lr),
                # GEM constrains the actual update direction. Momentum would
                # add an unprojected historical velocity after that constraint.
                momentum=0.0 if self.continual_method == "gem" else 0.9,
                weight_decay=float(self.weight_decay),
            )
        return torch.optim.Adam(
            self.trainer.model.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

    def on_task_start(self, task_id: str) -> None:
        if self.current_task_id == task_id:
            return
        self.current_task_id = task_id
        self.episodic_memory.setdefault(task_id, [])
        self.memory_positions.setdefault(task_id, 0)
        self.kd_state = _clone_tensor_state(self.initial_kd_state)
        self.kd_model.load_state_dict(self.kd_state, strict=True)

    def _update_episodic_memory(
        self,
        task_id: str,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update GEM's fixed-size cyclic memory for the current task."""
        task_memory = self.episodic_memory.setdefault(task_id, [])
        position = self.memory_positions.setdefault(task_id, 0)
        for sample, target in zip(inputs.detach().cpu(), targets.detach().cpu()):
            item = (sample.clone(), target.clone())
            if len(task_memory) < int(self.gem_memory_size):
                task_memory.append(item)
            else:
                task_memory[position] = item
            position = (position + 1) % int(self.gem_memory_size)
        self.memory_positions[task_id] = position

    def _gem_memory_gradients(
        self,
        current_task_id: str,
        parameters: Sequence[nn.Parameter],
    ) -> list[torch.Tensor]:
        """Compute one task-aware constraint gradient per past task."""
        gradients: list[torch.Tensor] = []
        for task_id, samples in self.episodic_memory.items():
            if task_id == current_task_id or not samples:
                continue
            memory_inputs = torch.stack([sample for sample, _ in samples]).to(
                self.trainer.device
            )
            memory_targets = torch.stack([target for _, target in samples]).long().to(
                self.trainer.device
            )
            self.trainer.optimizer.zero_grad()
            logits, local_targets = _task_view(
                self.trainer.model(memory_inputs),
                memory_targets,
                self._classes(task_id),
            )
            assert local_targets is not None
            F.cross_entropy(logits, local_targets).backward()
            gradients.append(flatten_gradients(parameters).clone())
        return gradients

    def _ewc_penalty(self) -> torch.Tensor:
        parameters = dict(self.trainer.model.named_parameters())
        penalty = torch.zeros((), device=self.trainer.device)
        for name, fisher in self.fisher_state.items():
            if name not in parameters or name not in self.anchor_state:
                continue
            parameter = parameters[name]
            anchor = self.anchor_state[name].to(parameter)
            penalty = penalty + 0.5 * torch.sum(
                fisher.to(parameter) * (parameter - anchor).square()
            )
        return penalty

    def _evaluate_kd_state(
        self,
        state: Mapping[str, torch.Tensor],
        loader: DataLoader,
        task_id: str,
    ) -> float:
        self.kd_model.load_state_dict(state, strict=True)
        self.kd_model.to(self.trainer.device)
        self.kd_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                logits, local_targets = _task_view(
                    self.kd_model(inputs), targets, self._classes(task_id)
                )
                assert local_targets is not None
                correct += int((logits.argmax(dim=1) == local_targets).sum().item())
                total += int(targets.shape[0])
        return correct / max(1, total)

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
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
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
                logits, local_targets = _task_view(
                    self.trainer.model(inputs), targets, self._classes(task_id)
                )
                assert local_targets is not None
                with torch.no_grad():
                    teacher_logits, _ = _task_view(teacher(inputs), None, self._classes(task_id))
                classification_loss = F.cross_entropy(logits, local_targets)
                distillation_loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = (
                    classification_loss
                    + float(self.integrator_weight) * float(teacher_accuracy) * distillation_loss
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
                total_batches += 1
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_ce += float(classification_loss.detach().item()) * batch_size
                total_kd += float(distillation_loss.detach().item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == local_targets).sum().item())
        denominator = max(1, total_examples)
        return {
            "loss": total_loss / denominator,
            "accuracy": total_correct / denominator,
            "classification_loss": total_ce / denominator,
            "integrator_kd_loss": total_kd / denominator,
            "gem_constraint_violations": total_constraint_violations
            / max(1, total_batches),
            "gem_gradient_projections": total_gradient_projections
            / max(1, total_batches),
        }

    def _distill_kd_model(self, loader: DataLoader, task_id: str) -> float:
        self.kd_model.to(self.trainer.device)
        self.kd_model.train()
        self.trainer.model.to(self.trainer.device)
        self.trainer.model.eval()
        optimizer = torch.optim.Adam(self.kd_model.parameters(), lr=float(self.kd_lr))
        temperature = float(self.temperature)
        total_loss = 0.0
        total_examples = 0
        for _ in range(max(0, int(self.kd_epochs))):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                with torch.no_grad():
                    teacher_logits, _ = _task_view(
                        self.trainer.model(inputs), None, self._classes(task_id)
                    )
                student_logits, local_targets = _task_view(
                    self.kd_model(inputs), targets, self._classes(task_id)
                )
                assert local_targets is not None
                supervised_loss = F.cross_entropy(student_logits, local_targets)
                distillation_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = (1.0 - float(self.kd_alpha)) * supervised_loss + float(
                    self.kd_alpha
                ) * distillation_loss
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
        loader = self.task_loaders[task_id]
        base_state = _clone_tensor_state(self.kd_state or self.initial_kd_state)
        aggregated_state = _clone_tensor_state(global_state) if global_state else base_state
        base_accuracy = self._evaluate_kd_state(base_state, loader, task_id)
        aggregated_accuracy = self._evaluate_kd_state(aggregated_state, loader, task_id)
        use_aggregated = bool(global_state) and aggregated_accuracy >= base_accuracy
        teacher_state = aggregated_state if use_aggregated else base_state
        teacher_accuracy = aggregated_accuracy if use_aggregated else base_accuracy
        self.kd_model.load_state_dict(teacher_state, strict=True)

        metrics = self._train_main_model(loader, task_id, teacher_state, teacher_accuracy)
        metrics["kd_distill_loss"] = self._distill_kd_model(loader, task_id)
        metrics["kd_accuracy_before"] = float(base_accuracy)
        metrics["kd_accuracy_aggregated"] = float(aggregated_accuracy)
        metrics["used_aggregated_kd"] = float(use_aggregated)
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
            },
        )

    def _estimate_fisher(self, loader: DataLoader, task_id: str) -> dict[str, torch.Tensor]:
        self.trainer.model.to(self.trainer.device)
        self.trainer.model.eval()
        fisher = {
            name: torch.zeros_like(parameter, device=self.trainer.device)
            for name, parameter in self.trainer.model.named_parameters()
            if parameter.requires_grad
        }
        examples = 0
        for batch_index, (inputs, targets) in enumerate(loader):
            if int(self.fisher_batches) > 0 and batch_index >= int(self.fisher_batches):
                break
            inputs = move_to_device(inputs, self.trainer.device)
            targets = move_to_device(targets, self.trainer.device)
            logits, local_targets = _task_view(
                self.trainer.model(inputs), targets, self._classes(task_id)
            )
            assert local_targets is not None
            loss = F.cross_entropy(logits, local_targets)
            self.trainer.model.zero_grad()
            loss.backward()
            batch_size = int(targets.shape[0])
            examples += batch_size
            for name, parameter in self.trainer.model.named_parameters():
                if name in fisher and parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().square() * batch_size
        return {name: (value / max(1, examples)).cpu() for name, value in fisher.items()}

    def _pruning_masks(self, state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        parameter_names = [name for name, _ in self.kd_model.named_parameters()]
        eligible = [
            state[name].detach().abs().flatten()
            for name in parameter_names
            if name in state and state[name].is_floating_point()
        ]
        total = sum(value.numel() for value in eligible)
        keep = min(total, max(1, int(math.ceil(total * float(self.knowledge_ratio)))))
        masks = {
            name: torch.zeros_like(state[name], dtype=torch.bool)
            for name in parameter_names
            if name in state and state[name].is_floating_point()
        }
        if total == 0:
            return masks
        magnitudes = torch.cat(eligible)
        selected = torch.topk(magnitudes, k=keep, largest=True, sorted=False).indices
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
        dense_state = _clone_tensor_state(self.kd_state or self.kd_model.state_dict())
        masks = self._pruning_masks(dense_state)
        sparse_model = copy.deepcopy(self.kd_model).to(self.trainer.device)
        sparse_state = _clone_tensor_state(dense_state)
        for name, mask in masks.items():
            sparse_state[name] = sparse_state[name] * mask.to(sparse_state[name])
        sparse_model.load_state_dict(sparse_state, strict=True)
        optimizer = torch.optim.Adam(sparse_model.parameters(), lr=float(self.kd_lr))
        loader = self.task_loaders[task_id]
        named_parameters = dict(sparse_model.named_parameters())
        for _ in range(max(0, int(self.knowledge_finetune_epochs))):
            sparse_model.train()
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                logits, local_targets = _task_view(
                    sparse_model(inputs), targets, self._classes(task_id)
                )
                assert local_targets is not None
                loss = F.cross_entropy(logits, local_targets)
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
            mask_state={name: mask.detach().cpu().clone() for name, mask in masks.items()},
            num_samples=len(loader.dataset),
        )

    def on_task_end(self, task_id: str) -> LociTaskKnowledge:
        if self.continual_method == "ewc":
            loader = self.task_loaders[task_id]
            current_fisher = self._estimate_fisher(loader, task_id)
            if self.fisher_state:
                # Online EWC accumulates importance. Averaging it over tasks
                # makes the old-task constraint progressively weaker.
                self.fisher_state = {
                    name: self.fisher_state.get(name, torch.zeros_like(value)) + value
                    for name, value in current_fisher.items()
                }
            else:
                self.fisher_state = current_fisher
            self.anchor_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.trainer.model.named_parameters()
            }
        self.consolidated_tasks += 1
        return self.extract_task_knowledge(task_id)

    def build_eval_state(self, task_id: str) -> dict[str, torch.Tensor]:
        del task_id
        return _clone_tensor_state(self.current_state or self.trainer.model.state_dict())


@dataclass
class _Candidate:
    candidate_id: str
    client_id: str
    task_id: str
    state: dict[str, torch.Tensor]


@dataclass
class LociServer:
    """Personalized task-grained Loci server."""

    kd_model: nn.Module
    clients: Sequence[LociClient]
    public_loaders: Mapping[str, DataLoader]
    task_classes: Mapping[str, Sequence[int]] | None = None
    client_sample_ratio: float = 1.0
    similar_tasks: int = 4
    similarity: str = "activation"
    selector_candidates: int = 20
    selector_batches: int = 5
    temperature: float = 2.0
    ot_regularization: float = 0.05
    ot_iterations: int = 20
    memory_palace: TaskMemoryPalace = field(default_factory=TaskMemoryPalace)
    client_task_ids: dict[str, str] = field(default_factory=dict)
    client_kd_states: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    _activation_cache: dict[tuple[str, str], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not (0.0 < float(self.client_sample_ratio) <= 1.0):
            raise ValueError("client_sample_ratio must be in (0, 1].")
        if int(self.similar_tasks) < 0:
            raise ValueError("similar_tasks must be non-negative.")
        if self.similarity not in {"activation", "weight"}:
            raise ValueError("similarity must be 'activation' or 'weight'.")
        self.client_kd_states = {
            client.client_id: _clone_tensor_state(client.kd_state or client.initial_kd_state)
            for client in self.clients
        }

    def set_client_task_ids(self, client_task_ids: Mapping[str, str]) -> None:
        self.client_task_ids = {
            str(client_id): str(task_id) for client_id, task_id in client_task_ids.items()
        }

    def task_id_for_client(self, client_id: str, default_task_id: str) -> str:
        return self.client_task_ids.get(client_id, default_task_id)

    def on_client_tasks_start(self, tasks: Mapping[str, TaskDefinition]) -> None:
        clients = {client.client_id: client for client in self.clients}
        for client_id, task in tasks.items():
            clients[client_id].on_task_start(task.task_id)
            self.client_kd_states[client_id] = _clone_tensor_state(
                clients[client_id].initial_kd_state
            )

    def on_client_tasks_end(self, tasks: Mapping[str, TaskDefinition]) -> None:
        clients = {client.client_id: client for client in self.clients}
        for client_id, task in tasks.items():
            self.memory_palace.add(clients[client_id].on_task_end(task.task_id))

    def _activation_signature(
        self,
        candidate: _Candidate,
        task_id: str,
    ) -> torch.Tensor:
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
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_index, (inputs, _targets) in enumerate(loader):
                if int(self.selector_batches) > 0 and batch_index >= int(self.selector_batches):
                    break
                inputs = move_to_device(inputs, device)
                logits, _ = _task_view(model(inputs), None, class_ids)
                outputs.append(logits.detach().cpu())
        signature = torch.cat(outputs, dim=0) if outputs else torch.empty(0, 0)
        self._activation_cache[cache_key] = signature
        return signature

    def _distance(
        self,
        left: _Candidate,
        right: _Candidate,
    ) -> float:
        if self.similarity == "weight":
            return _weight_distance(left.state, right.state)
        left_logits = self._activation_signature(left, left.task_id)
        right_logits = self._activation_signature(right, left.task_id)
        if left_logits.numel() == 0 or right_logits.shape != left_logits.shape:
            return float("inf")
        temperature = float(self.temperature)
        loss = -torch.sum(
            F.log_softmax(left_logits / temperature, dim=1)
            * F.softmax(right_logits / temperature, dim=1),
            dim=1,
        ).mean()
        return float(loss.item())

    def _select_candidates(
        self,
        center: _Candidate,
        candidates: Sequence[_Candidate],
    ) -> tuple[list[_Candidate], list[float]]:
        if int(self.similar_tasks) == 0:
            return [], []
        eligible = [
            candidate for candidate in candidates if candidate.candidate_id != center.candidate_id
        ]
        if (
            self.similarity == "activation"
            and int(self.selector_candidates) > 0
            and len(eligible) > int(self.selector_candidates)
        ):
            eligible.sort(key=lambda candidate: _weight_distance(center.state, candidate.state))
            eligible = eligible[: int(self.selector_candidates)]
        scored = [(self._distance(center, candidate), candidate) for candidate in eligible]
        scored.sort(key=lambda item: item[0])
        selected = scored[: min(int(self.similar_tasks), len(scored))]
        return [candidate for _, candidate in selected], [distance for distance, _ in selected]

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        self._activation_cache.clear()
        clients = list(self.clients)
        if clients and float(self.client_sample_ratio) < 1.0:
            selected_count = max(1, int(len(clients) * float(self.client_sample_ratio)))
            clients = random.sample(clients, k=selected_count)
        results: list[TrainResult] = []
        centers: list[_Candidate] = []
        for client in clients:
            client_task_id = self.task_id_for_client(client.client_id, task_id)
            incoming_state = self.client_kd_states.get(client.client_id, {})
            result = client.fit(
                incoming_state,
                ClientContext(client.client_id, round_idx, client_task_id),
            )
            results.append(result)
            state = _clone_tensor_state(result.payload.get("kd_state", {}))
            centers.append(
                _Candidate(
                    candidate_id=f"round:{round_idx}:{client.client_id}",
                    client_id=client.client_id,
                    task_id=client_task_id,
                    state=state,
                )
            )

        historical = [
            _Candidate(
                candidate_id=f"memory:{knowledge.knowledge_id}",
                client_id=knowledge.client_id,
                task_id=knowledge.task_id,
                state=_clone_tensor_state(knowledge.state),
            )
            for knowledge in self.memory_palace.all()
        ]
        candidate_pool = centers + historical
        selected_metadata: dict[str, list[str]] = {}
        distances: list[float] = []
        for center in centers:
            selected, selected_distances = self._select_candidates(center, candidate_pool)
            fused_state = fuse_states_ot(
                self.kd_model,
                center.state,
                [candidate.state for candidate in selected],
                regularization=float(self.ot_regularization),
                iterations=int(self.ot_iterations),
            )
            self.client_kd_states[center.client_id] = fused_state
            selected_metadata[center.client_id] = [candidate.candidate_id for candidate in selected]
            distances.extend(selected_distances)

        total_samples = sum(max(0, int(result.num_samples)) for result in results)
        metric_names = {name for result in results for name in result.metrics}
        metrics: MetricDict = {}
        for name in metric_names:
            numerator = sum(
                float(result.metrics.get(name, 0.0)) * max(0, int(result.num_samples))
                for result in results
            )
            metrics[name] = numerator / max(1, total_samples)
        metrics["num_clients"] = float(len(results))
        metrics["total_samples"] = float(total_samples)
        metrics["memory_tasks"] = float(len(self.memory_palace))
        metrics["selected_distance"] = sum(distances) / max(1, len(distances))
        first_state = self.client_kd_states[centers[0].client_id] if centers else {}
        return AggregationResult(
            global_state=_clone_tensor_state(first_state),
            metrics=metrics,
            metadata={
                "aggregator": "loci_ot",
                "round_idx": round_idx,
                "task_id": task_id,
                "client_task_ids": dict(self.client_task_ids),
                "selected_knowledge": selected_metadata,
            },
        )

    def build_eval_state(
        self, task_id: str, client_id: str | None = None
    ) -> dict[str, torch.Tensor]:
        if client_id is not None:
            for client in self.clients:
                if client.client_id == client_id:
                    return client.build_eval_state(task_id)
        if not self.clients:
            return {}
        return self.clients[0].build_eval_state(task_id)


__all__ = [
    "LociClient",
    "LociServer",
    "LociTaskKnowledge",
    "TaskMemoryPalace",
    "align_state_dict_ot",
    "fuse_states_ot",
]
