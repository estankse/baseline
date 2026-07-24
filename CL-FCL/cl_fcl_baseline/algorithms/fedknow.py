from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader

from ..contracts import AggregationResult, ClientContext, MetricDict, StateDict, TaskDefinition, TrainResult
from ..trainers.trainer import BaseTrainer
from ..trainers.utils import detach_state_dict, move_to_device
from .fl import FedAvgAggregator


def _clone_tensor_state(state: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def _state_numel(state: Mapping[str, torch.Tensor]) -> int:
    return int(sum(value.numel() for value in state.values() if value.is_floating_point()))


def _state_nnz(state: Mapping[str, torch.Tensor]) -> int:
    return int(sum(torch.count_nonzero(value).item() for value in state.values() if value.is_floating_point()))


def _flatten_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    flat_parts: List[torch.Tensor] = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            flat_parts.append(torch.zeros_like(parameter, memory_format=torch.preserve_format).reshape(-1))
        else:
            flat_parts.append(gradient.detach().reshape(-1))
    if not flat_parts:
        return torch.empty(0)
    return torch.cat(flat_parts)


def _assign_flat_gradient(parameters: Sequence[torch.nn.Parameter], flat_gradient: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        length = parameter.numel()
        parameter.grad = flat_gradient[offset : offset + length].view_as(parameter).detach().clone()
        offset += length


def _clear_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        parameter.grad = None


@dataclass
class FedKNOWKnowledge:
    task_id: str
    state: StateDict
    mask_state: Dict[str, torch.Tensor]


@dataclass
class FedKNOWClient:
    """Client-side FedKNOW implementation.

    FedKNOW keeps compact per-task knowledge locally. During a new task it
    restores gradients from the most dissimilar signature tasks and projects
    the current gradient onto the acute-angle feasible region described in
    Eq. (3)-(5) of the paper.
    """

    client_id: str
    trainer: BaseTrainer
    task_loaders: Mapping[str, DataLoader]
    epochs: int = 1
    knowledge_ratio: float = 0.1
    signature_k: int = 10
    integrator_steps: int = 100
    knowledge_finetune_epochs: int = 1
    post_aggregation_epochs: int = 1
    distillation_warmup_epochs: int = 2
    restorer_loss: str = "soft"
    restorer_temperature: float = 2.0
    optimizer_name: str = "sgd"
    lr: float | None = None
    momentum: float = 0.9
    weight_decay: float = 0.0
    task_classes: Mapping[str, Sequence[int]] | None = None
    knowledge_base: Dict[str, FedKNOWKnowledge] = field(default_factory=dict)
    knowledge_model: torch.nn.Module = field(init=False, repr=False)
    current_state: StateDict | None = None
    personal_state: StateDict | None = None
    personal_task_id: str | None = None
    previous_task_state: StateDict | None = None
    previous_task_ids: List[str] = field(default_factory=list)
    pending_task_snapshot: bool = False

    def __post_init__(self) -> None:
        self.knowledge_model = copy.deepcopy(self.trainer.model)
        if self.lr is None:
            self.lr = float(self.trainer.optimizer.param_groups[0].get("lr", 0.01))
        if self.optimizer_name not in {"sgd", "adam"}:
            self.optimizer_name = self.trainer.optimizer.__class__.__name__.lower()
            if self.optimizer_name not in {"sgd", "adam"}:
                self.optimizer_name = "sgd"
        if not (0.0 < float(self.knowledge_ratio) <= 1.0):
            raise ValueError("knowledge_ratio must be in (0, 1].")
        if int(self.signature_k) < 0:
            raise ValueError("signature_k must be non-negative.")
        if self.restorer_loss not in {"hard", "soft"}:
            raise ValueError("restorer_loss must be 'hard' or 'soft'.")
        if float(self.restorer_temperature) <= 0.0:
            raise ValueError("restorer_temperature must be positive.")

    @property
    def parameter_names(self) -> List[str]:
        return [name for name, _ in self.trainer.model.named_parameters()]

    @property
    def classifier_parameter_names(self) -> set[str]:
        if not self.task_classes:
            return set()
        all_class_ids = [
            int(class_id)
            for class_ids in self.task_classes.values()
            for class_id in class_ids
        ]
        if not all_class_ids:
            return set()
        num_classes = max(all_class_ids) + 1
        exact_matches: set[str] = set()
        fallback_matches: set[str] = set()
        for module_name, module in self.trainer.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            prefix = f"{module_name}." if module_name else ""
            names = {f"{prefix}weight", f"{prefix}bias"}
            if int(module.out_features) == num_classes:
                exact_matches.update(names)
            elif int(module.out_features) > num_classes:
                fallback_matches.update(names)
        return exact_matches or fallback_matches

    @property
    def shared_parameter_names(self) -> set[str]:
        classifier_names = self.classifier_parameter_names
        return {
            name
            for name, _parameter in self.trainer.model.named_parameters()
            if name not in classifier_names
        }

    @property
    def shared_parameters(self) -> List[torch.nn.Parameter]:
        shared_names = self.shared_parameter_names
        return [
            parameter
            for name, parameter in self.trainer.model.named_parameters()
            if name in shared_names
        ]

    @property
    def local_parameters(self) -> List[torch.nn.Parameter]:
        shared_names = self.shared_parameter_names
        return [
            parameter
            for name, parameter in self.trainer.model.named_parameters()
            if name not in shared_names
        ]

    def _local_state_names(self) -> set[str]:
        # The reference Flower client communicates feature_net.state_dict(),
        # which includes BatchNorm buffers.  Only the classifier head remains
        # local; treating every buffer as personal silently changes FedKNOW
        # into a FedBN-like method.
        return set(self.classifier_parameter_names)

    def _merge_global_with_personal_state(
        self,
        global_state: StateDict,
        personal_state: StateDict | None = None,
    ) -> Dict[str, torch.Tensor]:
        state = _clone_tensor_state(global_state)
        source_state = personal_state or self.personal_state
        if not source_state:
            return state
        for name in self._local_state_names():
            value = source_state.get(name)
            if isinstance(value, torch.Tensor) and name in state:
                state[name] = value.detach().cpu().clone()
        return state

    def _occupied_masks(self) -> Dict[str, torch.Tensor]:
        occupied: Dict[str, torch.Tensor] = {}
        for knowledge in self.knowledge_base.values():
            for name, mask in knowledge.mask_state.items():
                mask_bool = mask.detach().cpu().bool()
                occupied[name] = mask_bool if name not in occupied else (occupied[name] | mask_bool)
        return occupied

    def on_task_start(self, task_id: str) -> None:
        # FedKNOW is a continual client-side method: the next task starts
        # from the client's latest local model, not from an empty task state.
        if self.personal_task_id is not None and self.personal_task_id != task_id:
            # The reference implementation freezes the dense main model at a
            # task boundary.  Its distillation gradient is the first GEM
            # constraint; the sparse PackNet teachers provide the remaining
            # task-specific constraints.
            self.previous_task_ids = list(self.knowledge_base.keys())
            self.previous_task_state = None
            self.pending_task_snapshot = bool(self.previous_task_ids)
        self.personal_task_id = task_id

    def _previous_model_gradient(
        self,
        inputs: torch.Tensor,
        parameters: Sequence[torch.nn.Parameter],
    ) -> torch.Tensor | None:
        if self.previous_task_state is None or not self.previous_task_ids:
            return None
        teacher_state = {
            name: value.detach().to(inputs.device)
            for name, value in self.previous_task_state.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.no_grad():
            teacher_logits = functional_call(self.trainer.model, teacher_state, (inputs,))
        student_logits = self.trainer.model(inputs)
        previous_classes = sorted({
            int(class_id)
            for previous_task_id in self.previous_task_ids
            for class_id in (self.task_classes or {}).get(previous_task_id, [])
        })
        if previous_classes:
            class_ids = torch.tensor(previous_classes, device=inputs.device, dtype=torch.long)
            teacher_logits = teacher_logits.index_select(1, class_ids)
            student_logits = student_logits.index_select(1, class_ids)
        temperature = float(self.restorer_temperature)
        teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
        loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            teacher_probs,
            reduction="batchmean",
        ) * (temperature * temperature)
        return self._gradient_for_loss(loss, parameters)

    def _build_optimizer(self, parameters: Sequence[torch.nn.Parameter] | None = None) -> torch.optim.Optimizer:
        params = list(parameters) if parameters is not None else list(self.trainer.model.parameters())
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=float(self.lr), weight_decay=float(self.weight_decay))
        return torch.optim.SGD(
            params,
            lr=float(self.lr),
            momentum=float(self.momentum),
            weight_decay=float(self.weight_decay),
        )

    def _forward_with_knowledge(
        self,
        knowledge: FedKNOWKnowledge,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # Rebuild the compact PackNet teacher for this task from cumulative,
        # non-overlapping allocations.
        functional_state = {
            name: value.detach().to(inputs.device)
            for name, value in self.trainer.model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
        for name in self.shared_parameter_names:
            if name in functional_state:
                functional_state[name] = torch.zeros_like(functional_state[name])
        for name, value in knowledge.state.items():
            if name in self.shared_parameter_names or name not in functional_state:
                continue
            functional_state[name] = value.detach().to(
                inputs.device,
                dtype=functional_state[name].dtype,
            )
        for previous_task_id, previous_knowledge in self.knowledge_base.items():
            for name, mask in previous_knowledge.mask_state.items():
                if name not in functional_state or name not in previous_knowledge.state:
                    continue
                mask_device = mask.detach().to(inputs.device, dtype=functional_state[name].dtype)
                retained = previous_knowledge.state[name].detach().to(
                    inputs.device, dtype=functional_state[name].dtype
                )
                functional_state[name] = (
                    functional_state[name] * (1.0 - mask_device) + retained * mask_device
                )
            if previous_task_id == knowledge.task_id:
                break
        return functional_call(self.trainer.model, functional_state, (inputs,))

    def _gradient_for_loss(
        self,
        loss: torch.Tensor,
        parameters: Sequence[torch.nn.Parameter],
        *,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        return _flatten_gradients(parameters, gradients)

    def _task_class_tensor(
        self,
        task_id: str,
        logits: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.task_classes:
            return None
        class_ids = self.task_classes.get(task_id)
        if not class_ids:
            return None
        ids = torch.tensor([int(class_id) for class_id in class_ids], device=logits.device, dtype=torch.long)
        if ids.numel() <= 0:
            return None
        if ids.numel() == logits.shape[1] and bool(torch.equal(ids, torch.arange(logits.shape[1], device=logits.device))):
            return None
        return ids

    def _task_loss_and_predictions(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        task_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_ids = self._task_class_tensor(task_id, logits)
        if class_ids is None:
            return F.cross_entropy(logits, targets), logits.argmax(dim=1)

        task_logits = logits.index_select(dim=1, index=class_ids)
        local_targets = torch.zeros_like(targets)
        for local_idx, class_id in enumerate(class_ids.tolist()):
            local_targets = torch.where(targets == int(class_id), torch.full_like(targets, local_idx), local_targets)
        predictions = class_ids[task_logits.argmax(dim=1)]
        return F.cross_entropy(task_logits, local_targets), predictions

    def _wasserstein_distance(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
        probs_a = F.softmax(logits_a.detach(), dim=1)
        probs_b = F.softmax(logits_b.detach(), dim=1)
        cdf_a = torch.cumsum(probs_a, dim=1)
        cdf_b = torch.cumsum(probs_b, dim=1)
        return float(torch.mean(torch.abs(cdf_a - cdf_b)).item())

    def _gradient_wasserstein_distance(self, gradient_a: torch.Tensor, gradient_b: torch.Tensor) -> float:
        if gradient_a.numel() == 0 or gradient_b.numel() == 0:
            return 0.0
        abs_a = torch.abs(gradient_a.detach().float()).flatten()
        abs_b = torch.abs(gradient_b.detach().float()).flatten()
        norm_a = torch.sum(abs_a)
        norm_b = torch.sum(abs_b)
        if float(norm_a.item()) <= 1e-12 or float(norm_b.item()) <= 1e-12:
            return 0.0
        probs_a = abs_a / norm_a
        probs_b = abs_b / norm_b
        return float(torch.mean(torch.abs(torch.cumsum(probs_a, dim=0) - torch.cumsum(probs_b, dim=0))).item())

    def _select_signature_gradients(
        self,
        task_id: str,
        inputs: torch.Tensor,
        current_gradient: torch.Tensor,
        parameters: Sequence[torch.nn.Parameter],
    ) -> tuple[List[torch.Tensor], List[float]]:
        previous_task_ids = [
            previous_task_id
            for previous_task_id in self.knowledge_base.keys()
            if previous_task_id != task_id
        ]
        if not previous_task_ids or int(self.signature_k) == 0:
            return [], []

        scored: List[tuple[float, torch.Tensor]] = []
        for previous_task_id in previous_task_ids:
            gradient = self._restored_gradient(self.knowledge_base[previous_task_id], inputs, parameters)
            scored.append((self._gradient_wasserstein_distance(current_gradient, gradient), gradient))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: min(int(self.signature_k), len(scored))]
        return [gradient for _, gradient in selected], [distance for distance, _ in selected]

    def _restored_gradient(
        self,
        knowledge: FedKNOWKnowledge,
        inputs: torch.Tensor,
        parameters: Sequence[torch.nn.Parameter],
    ) -> torch.Tensor:
        with torch.no_grad():
            teacher_logits = self._forward_with_knowledge(knowledge, inputs)
        return self._restored_gradient_from_teacher_logits(knowledge.task_id, inputs, teacher_logits, parameters)

    def _restored_gradient_from_teacher_logits(
        self,
        task_id: str,
        inputs: torch.Tensor,
        teacher_logits: torch.Tensor,
        parameters: Sequence[torch.nn.Parameter],
    ) -> torch.Tensor:
        student_logits = self.trainer.model(inputs)
        class_ids = self._task_class_tensor(task_id, student_logits)
        if class_ids is not None:
            student_task_logits = student_logits.index_select(dim=1, index=class_ids)
            teacher_task_logits = teacher_logits.index_select(dim=1, index=class_ids)
            if self.restorer_loss == "soft":
                temperature = float(self.restorer_temperature)
                teacher_probs = F.softmax(teacher_task_logits.detach() / temperature, dim=1)
                loss = F.kl_div(
                    F.log_softmax(student_task_logits / temperature, dim=1),
                    teacher_probs,
                    reduction="batchmean",
                ) * (temperature * temperature)
            else:
                pseudo_targets = teacher_task_logits.detach().argmax(dim=1)
                loss = F.cross_entropy(student_task_logits, pseudo_targets)
            return self._gradient_for_loss(loss, parameters)
        if self.restorer_loss == "soft":
            temperature = float(self.restorer_temperature)
            teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
            loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                teacher_probs,
                reduction="batchmean",
            ) * (temperature * temperature)
        else:
            pseudo_targets = teacher_logits.detach().argmax(dim=1)
            loss = F.cross_entropy(student_logits, pseudo_targets)
        return self._gradient_for_loss(loss, parameters)

    def _integrate_gradient(
        self,
        current_gradient: torch.Tensor,
        signature_gradients: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        valid_gradients = [
            gradient.detach().to(current_gradient.device, dtype=torch.float32)
            for gradient in signature_gradients
            if gradient.numel() == current_gradient.numel() and float(torch.norm(gradient.detach()).item()) > 1e-12
        ]
        if not valid_gradients or current_gradient.numel() == 0:
            return current_gradient, {"integrated_constraints": 0.0, "integrated_violations": 0.0}

        current = current_gradient.detach().to(dtype=torch.float32)
        gradients = torch.stack(valid_gradients, dim=0)
        dot_products = torch.mv(gradients, current)
        initial_violations = int(torch.sum(dot_products < 0.0).item())
        if initial_violations == 0:
            return current_gradient, {
                "integrated_constraints": float(len(valid_gradients)),
                "integrated_violations": 0.0,
                "projected_violations": 0.0,
                "projected_min_dot": float(torch.min(dot_products).item()),
            }

        gram = torch.mm(gradients, gradients.t())
        q = dot_products
        v = torch.zeros(gradients.shape[0], device=current.device, dtype=torch.float32)
        if gram.numel() == 1:
            lipschitz = float(torch.clamp(gram.reshape(()), min=1e-8).item())
        else:
            lipschitz = float(torch.clamp(torch.linalg.eigvalsh(gram).max(), min=1e-8).item())
        step_size = 1.0 / lipschitz
        for _ in range(max(1, int(self.integrator_steps))):
            dual_gradient = torch.mv(gram, v) + q
            v = torch.clamp(v - step_size * dual_gradient, min=0.0)

        integrated = current + torch.mv(gradients.t(), v)
        for _ in range(max(1, int(self.integrator_steps))):
            fixed = True
            for gradient in gradients:
                dot_value = torch.dot(gradient, integrated)
                if float(dot_value.item()) < 0.0:
                    integrated = integrated + (-dot_value / torch.clamp(torch.dot(gradient, gradient), min=1e-12)) * gradient
                    fixed = False
            if fixed:
                break
        final_dots = torch.mv(gradients, integrated)
        projected_violations = int(torch.sum(final_dots < -1e-6).item())
        return integrated.to(dtype=current_gradient.dtype), {
            "integrated_constraints": float(len(valid_gradients)),
            "integrated_violations": float(initial_violations),
            "projected_violations": float(projected_violations),
            "projected_min_dot": float(torch.min(final_dots).item()),
        }

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        task_id = context.task_id or next(iter(self.task_loaders))
        if task_id not in self.task_loaders:
            task_id = next(iter(self.task_loaders))

        initial_state = self._merge_global_with_personal_state(global_state)
        if self.pending_task_snapshot:
            # In the reference Flower client set_parameters(global) runs
            # before Appr.train() freezes model_old at a task boundary.
            self.previous_task_state = _clone_tensor_state(initial_state)
            self.pending_task_snapshot = False
        self.trainer.model.load_state_dict(_clone_tensor_state(initial_state), strict=True)
        self.trainer.model.to(self.trainer.device)
        self.trainer.model.train()
        self.trainer.optimizer = self._build_optimizer()
        loader = self.task_loaders[task_id]

        total_examples = 0
        total_loss = 0.0
        total_correct = 0
        total_constraints = 0.0
        total_violations = 0.0
        total_projected_violations = 0.0
        total_projected_min_dot = 0.0
        total_signature = 0.0

        shared_parameters = self.shared_parameters
        local_parameters = self.local_parameters
        for epoch_idx in range(int(self.epochs)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)

                is_distillation_warmup = (
                    self.previous_task_state is not None
                    and epoch_idx < int(self.distillation_warmup_epochs)
                )
                if is_distillation_warmup:
                    # Reference train_epoch_head(): first descend the dense
                    # previous-model distillation loss, then learn the current
                    # task normally.  This actively repairs old decision
                    # boundaries before the representation/GEM phase.
                    all_parameters = list(self.trainer.model.parameters())
                    memory_gradient = self._previous_model_gradient(inputs, all_parameters)
                    if memory_gradient is not None:
                        self.trainer.optimizer.zero_grad()
                        _clear_gradients(all_parameters)
                        _assign_flat_gradient(all_parameters, memory_gradient)
                        self.trainer.optimizer.step()

                logits = self.trainer.model(inputs)
                loss, predictions = self._task_loss_and_predictions(logits, targets, task_id)
                current_gradient = self._gradient_for_loss(
                    loss,
                    shared_parameters,
                    retain_graph=bool(local_parameters),
                )
                local_gradient = (
                    self._gradient_for_loss(loss, local_parameters)
                    if local_parameters
                    else torch.empty(0, device=current_gradient.device)
                )
                restored_gradients: List[torch.Tensor] = []
                if not is_distillation_warmup:
                    restored_gradients, _distances = self._select_signature_gradients(
                        task_id,
                        inputs,
                        current_gradient,
                        shared_parameters,
                    )
                    previous_gradient = self._previous_model_gradient(inputs, shared_parameters)
                    if previous_gradient is not None:
                        restored_gradients = [previous_gradient, *restored_gradients]
                integrated_gradient, integration_metrics = self._integrate_gradient(
                    current_gradient,
                    restored_gradients,
                )

                self.trainer.optimizer.zero_grad()
                _clear_gradients(list(self.trainer.model.parameters()))
                _assign_flat_gradient(shared_parameters, integrated_gradient)
                if local_parameters:
                    _assign_flat_gradient(local_parameters, local_gradient)
                self.trainer.optimizer.step()

                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int((predictions.detach() == targets).sum().item())
                total_constraints += integration_metrics["integrated_constraints"]
                total_violations += integration_metrics["integrated_violations"]
                total_projected_violations += integration_metrics.get("projected_violations", 0.0)
                total_projected_min_dot += integration_metrics.get("projected_min_dot", 0.0)
                total_signature += float(len(restored_gradients))

        if total_examples == 0:
            metrics: MetricDict = {"loss": 0.0, "accuracy": 0.0}
        else:
            num_batches = max(1.0, total_examples / max(1, int(getattr(loader, "batch_size", 1) or 1)))
            metrics = {
                "loss": total_loss / total_examples,
                "accuracy": total_correct / total_examples,
                "signature_tasks": total_signature / num_batches,
                "integrated_constraints": total_constraints / num_batches,
                "integrated_violations": total_violations / num_batches,
                "projected_violations": total_projected_violations / num_batches,
                "projected_min_dot": total_projected_min_dot / num_batches,
                "knowledge_base_size": float(len(self.knowledge_base)),
            }

        state = detach_state_dict(self.trainer.model.state_dict())
        self.current_state = state
        self.personal_state = state
        self.personal_task_id = task_id
        self._train_knowledge_model(task_id)
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics=metrics,
            payload={
                "model_state": state,
                "before_aggregation_state": state,
            },
        )

    def post_aggregation_finetune(
        self,
        task_id: str,
        before_state: StateDict,
        aggregated_state: StateDict,
    ) -> MetricDict:
        if int(self.post_aggregation_epochs) <= 0 or task_id not in self.task_loaders:
            synchronized_state = self._merge_global_with_personal_state(aggregated_state)
            self.personal_state = synchronized_state
            self.current_state = synchronized_state
            self.personal_task_id = task_id
            return {}

        loader = self.task_loaders[task_id]
        self.trainer.model.to(self.trainer.device)
        optimizer = self._build_optimizer()
        shared_parameters = self.shared_parameters
        local_parameters = self.local_parameters

        total_examples = 0
        total_loss = 0.0
        total_violations = 0.0
        total_projected_violations = 0.0
        current_after_state = self._merge_global_with_personal_state(
            aggregated_state,
            personal_state=before_state,
        )
        for _ in range(int(self.post_aggregation_epochs)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)

                self.trainer.model.load_state_dict(_clone_tensor_state(before_state), strict=True)
                self.trainer.model.train()
                before_logits = self.trainer.model(inputs)
                before_loss, _before_predictions = self._task_loss_and_predictions(before_logits, targets, task_id)
                before_gradient = self._gradient_for_loss(before_loss, shared_parameters)

                self.trainer.model.load_state_dict(current_after_state, strict=True)
                self.trainer.model.train()
                after_logits = self.trainer.model(inputs)
                after_loss, _after_predictions = self._task_loss_and_predictions(after_logits, targets, task_id)
                after_gradient = self._gradient_for_loss(
                    after_loss,
                    shared_parameters,
                    retain_graph=bool(local_parameters),
                )
                local_gradient = (
                    self._gradient_for_loss(after_loss, local_parameters)
                    if local_parameters
                    else torch.empty(0, device=after_gradient.device)
                )
                # Paper Section III-A/III-E defines the post-aggregation
                # integration using exactly two task-current gradients: g_b
                # before aggregation and g_a after aggregation.  Historical
                # PackNet constraints belong to local continual training;
                # adding them again here over-constrains this one-epoch
                # negative-transfer correction and was the main reason the
                # previous implementation degraded old-task accuracy.
                integrated_gradient, integration_metrics = self._integrate_gradient(
                    after_gradient,
                    [before_gradient],
                )

                optimizer.zero_grad()
                _clear_gradients(list(self.trainer.model.parameters()))
                _assign_flat_gradient(shared_parameters, integrated_gradient)
                if local_parameters:
                    _assign_flat_gradient(local_parameters, local_gradient)
                optimizer.step()
                current_after_state = detach_state_dict(self.trainer.model.state_dict())

                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(after_loss.detach().item()) * batch_size
                total_violations += integration_metrics["integrated_violations"]
                total_projected_violations += integration_metrics.get("projected_violations", 0.0)

        self.personal_state = current_after_state
        self.personal_task_id = task_id
        self.current_state = current_after_state
        if total_examples == 0:
            return {"post_loss": 0.0, "post_violations": 0.0}
        return {
            "post_loss": total_loss / total_examples,
            "post_violations": total_violations,
            "post_projected_violations": total_projected_violations,
        }

    def _train_knowledge_model(self, task_id: str) -> None:
        """Train the persistent local PackNet model in parallel with FedKNOW."""
        if task_id not in self.task_loaders:
            return
        model = self.knowledge_model.to(self.trainer.device)
        model.train()
        optimizer = self._build_optimizer(list(model.parameters()))
        occupied = self._occupied_masks()
        protected_reference = detach_state_dict(model.state_dict())
        for _ in range(int(self.epochs)):
            for inputs, targets in self.task_loaders[task_id]:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss, _predictions = self._task_loss_and_predictions(logits, targets, task_id)
                loss.backward()
                optimizer.step()
                # Old PackNet allocations are immutable.  Restoring after the
                # step also cancels optimizer momentum and weight decay.
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name not in occupied or name not in protected_reference:
                            continue
                        mask = occupied[name].to(parameter.device, dtype=torch.bool)
                        reference = protected_reference[name].to(parameter.device, dtype=parameter.dtype)
                        parameter.data[mask] = reference[mask]
                        state = optimizer.state.get(parameter, {})
                        for value in state.values():
                            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                                value[mask] = 0

    def _build_top_weight_masks(self, task_id: str, state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        shared_parameter_names = self.shared_parameter_names
        occupied_masks = self._occupied_masks()
        available_by_name: Dict[str, torch.Tensor] = {}
        available_values: List[torch.Tensor] = []
        fixed_bias_masks: Dict[str, torch.Tensor] = {}
        for name, value in state.items():
            if name not in shared_parameter_names or not value.is_floating_point():
                continue
            occupied = occupied_masks.get(name, torch.zeros_like(value, dtype=torch.bool)).to(value.device)
            available = ~occupied
            if "bias" in name.split("."):
                # Reference PackNet excludes every bias from magnitude
                # pruning, then fix_biases() freezes them after task 0.  A
                # full first-task mask represents that persistent allocation
                # in our task-knowledge format.
                fixed_bias_masks[name] = available.to(dtype=value.dtype).cpu()
                continue
            available_by_name[name] = available
            if bool(torch.any(available)):
                available_values.append(torch.abs(value.detach())[available].flatten())
        if not available_values:
            return fixed_bias_masks
        # The reference PackNet implementation computes one global quantile
        # over all still-free prunable weights, not a separate quota per layer.
        all_available = torch.cat(available_values)
        keep_count = max(1, int(math.ceil(float(self.knowledge_ratio) * all_available.numel())))
        threshold = torch.topk(
            all_available,
            k=min(keep_count, all_available.numel()),
            largest=True,
        ).values[-1]
        masks: Dict[str, torch.Tensor] = dict(fixed_bias_masks)
        for name, available in available_by_name.items():
            value = state[name]
            masks[name] = (
                available & (torch.abs(value.detach()) >= threshold)
            ).to(dtype=value.dtype).cpu()
        return masks

    def _finetune_knowledge_weights(
        self,
        task_id: str,
        masks: Mapping[str, torch.Tensor],
    ) -> None:
        if int(self.knowledge_finetune_epochs) <= 0 or task_id not in self.task_loaders or not masks:
            return
        cumulative_masks = self._occupied_masks()
        for name, mask in masks.items():
            current = mask.detach().cpu().bool()
            cumulative_masks[name] = (
                current if name not in cumulative_masks else (cumulative_masks[name] | current)
            )
        # PackNet prunes the unassigned weights before fine-tuning the compact
        # subnetwork.  Fine-tuning a dense model and only masking afterwards
        # trains a different function and makes the stored knowledge unusable.
        with torch.no_grad():
            for name, parameter in self.knowledge_model.named_parameters():
                if name in cumulative_masks:
                    parameter.mul_(cumulative_masks[name].to(parameter.device, dtype=parameter.dtype))
        compact_reference = detach_state_dict(self.knowledge_model.state_dict())
        optimizer = self._build_optimizer(list(self.knowledge_model.parameters()))
        self.knowledge_model.to(self.trainer.device)
        self.knowledge_model.train()
        for _ in range(int(self.knowledge_finetune_epochs)):
            for inputs, targets in self.task_loaders[task_id]:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                optimizer.zero_grad()
                logits = self.knowledge_model(inputs)
                loss, _predictions = self._task_loss_and_predictions(logits, targets, task_id)
                loss.backward()
                for name, parameter in self.knowledge_model.named_parameters():
                    if parameter.grad is not None and name in masks:
                        parameter.grad.mul_(masks[name].to(parameter.device))
                optimizer.step()
                with torch.no_grad():
                    for name, parameter in self.knowledge_model.named_parameters():
                        if name not in cumulative_masks:
                            continue
                        mask = masks.get(name, torch.zeros_like(parameter)).to(parameter.device)
                        reference = compact_reference[name].to(parameter.device)
                        parameter.data.mul_(mask).add_(reference * (1.0 - mask))

    def extract_task_knowledge(self, task_id: str) -> FedKNOWKnowledge | None:
        if task_id not in self.task_loaders:
            return None
        state = detach_state_dict(self.knowledge_model.state_dict())
        masks = self._build_top_weight_masks(task_id, state)
        self._finetune_knowledge_weights(task_id, masks)

        finetuned_state = detach_state_dict(self.knowledge_model.state_dict())
        sparse_state: Dict[str, torch.Tensor] = {}
        parameter_names = set(self.parameter_names)
        for name, value in finetuned_state.items():
            if name in masks:
                sparse_state[name] = value.detach().cpu() * masks[name].to(value.dtype)
            elif name in self.classifier_parameter_names:
                # PackNet masks the representation; the small task-aware head
                # is retained so its rows do not drift during later tasks.
                sparse_state[name] = value.detach().cpu().clone()
            elif name not in parameter_names and isinstance(value, torch.Tensor):
                sparse_state[name] = value.detach().cpu().clone()
        knowledge = FedKNOWKnowledge(
            task_id=task_id,
            state=sparse_state,
            mask_state={name: value.detach().cpu().clone() for name, value in masks.items()},
        )
        self.knowledge_base[task_id] = knowledge
        return knowledge

    def build_eval_state(self, global_state: StateDict, task_id: str) -> Dict[str, torch.Tensor]:
        # The reference client's evaluate() first loads the server's latest
        # feature_net parameters.  Evaluation therefore uses global shared
        # features plus the client's local multi-task classifier head.
        return self._merge_global_with_personal_state(
            global_state,
            personal_state=self.current_state or self.personal_state,
        )


@dataclass
class FedKNOWServer:
    model: torch.nn.Module
    clients: Sequence[FedKNOWClient]
    aggregator: FedAvgAggregator = field(default_factory=FedAvgAggregator)
    client_sample_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < float(self.client_sample_ratio) <= 1.0):
            raise ValueError("client_sample_ratio must be in (0, 1].")

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        return detach_state_dict(self.model.state_dict())

    def set_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        current_state = self.get_global_state()
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                current_state[name] = value.detach().cpu().clone()
        self.model.load_state_dict(current_state, strict=True)

    def _select_clients(self) -> List[FedKNOWClient]:
        clients = list(self.clients)
        if clients and self.client_sample_ratio < 1.0:
            num_selected = max(1, int(len(clients) * self.client_sample_ratio))
            clients = random.sample(clients, k=num_selected)
        return clients

    def on_task_start(self, task: TaskDefinition) -> None:
        for client in self.clients:
            client.on_task_start(task.task_id)

    def on_task_end(self, task: TaskDefinition) -> None:
        for client in self.clients:
            client.extract_task_knowledge(task.task_id)

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        global_state = self.get_global_state()
        selected_clients = self._select_clients()
        client_results: List[TrainResult] = []
        before_states: Dict[str, StateDict] = {}

        for client in selected_clients:
            context = ClientContext(client_id=client.client_id, round_idx=round_idx, task_id=task_id)
            result = client.fit(global_state, context)
            client_results.append(result)
            before_state = result.payload.get("before_aggregation_state", {})
            if isinstance(before_state, Mapping):
                before_states[client.client_id] = _clone_tensor_state(before_state)

        aggregation_result = self.aggregator.aggregate(client_results)
        if aggregation_result.global_state:
            self.set_global_state(aggregation_result.global_state)

        post_metric_sums: Dict[str, float] = {}
        post_metric_counts: Dict[str, float] = {}
        updated_global_state = self.get_global_state()
        for client in selected_clients:
            before_state = before_states.get(client.client_id)
            if before_state is None:
                continue
            post_metrics = client.post_aggregation_finetune(task_id, before_state, updated_global_state)
            for name, value in post_metrics.items():
                post_metric_sums[name] = post_metric_sums.get(name, 0.0) + float(value)
                post_metric_counts[name] = post_metric_counts.get(name, 0.0) + 1.0

        metrics = dict(aggregation_result.metrics)
        for name, value in post_metric_sums.items():
            if post_metric_counts.get(name, 0.0) > 0.0:
                metrics[f"client_{name}"] = value / post_metric_counts[name]

        metadata = dict(aggregation_result.metadata)
        metadata["aggregator"] = "fedknow"
        metadata["round_idx"] = round_idx
        metadata["task_id"] = task_id
        return AggregationResult(
            global_state=aggregation_result.global_state,
            metrics=metrics,
            metadata=metadata,
        )

    def build_eval_state(self, task_id: str, client_id: str | None = None) -> Dict[str, torch.Tensor]:
        global_state = self.get_global_state()
        if client_id is not None:
            for client in self.clients:
                if client.client_id == client_id:
                    return client.build_eval_state(global_state, task_id)
        for client in self.clients:
            if client.current_state is not None or client.personal_state is not None:
                return client.build_eval_state(global_state, task_id)
        return global_state
