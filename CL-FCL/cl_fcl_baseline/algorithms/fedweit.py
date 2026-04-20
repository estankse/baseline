from __future__ import annotations

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


def _clone_tensor_state(state: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[name] = value.detach().cpu().clone()
    return cloned


def _state_l2_norm(state: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for value in state.values():
        if value.is_floating_point():
            total += float(torch.sum(value.detach().float().pow(2)).item())
    return math.sqrt(total)


def _state_nnz(state: Mapping[str, torch.Tensor]) -> int:
    total = 0
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            total += int(torch.count_nonzero(value.detach()).item())
    return total


def _state_numel(state: Mapping[str, torch.Tensor]) -> int:
    total = 0
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            total += int(value.numel())
    return total


@dataclass
class FedWeITKnowledge:
    client_id: str
    task_id: str
    adaptive_state: StateDict


@dataclass
class FedWeITClient:
    """Client-side FedWeIT variables and Eq.2 optimization.

    The effective task model follows Eq.1 in the paper:
    theta_c^(t) = B_c * m_c^(t) + A_c^(t) + sum_i alpha_i A_i.
    """

    client_id: str
    trainer: BaseTrainer
    task_loaders: Mapping[str, DataLoader]
    epochs: int = 1
    lambda1: float = 1e-3
    lambda2: float = 1e-4
    lambda_mask: float = 0.0
    mask_init: float = -1.0
    optimizer_name: str = "sgd"
    lr: float | None = None
    momentum: float = 0.9
    weight_decay: float = 0.0
    mask_threshold: float = 0.5
    adaptive_threshold: float | None = None
    client_sparsity: float = 0.3
    base_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    mask_logits: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    adaptive_params: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    alpha_logits: Dict[str, torch.Tensor] = field(default_factory=dict)
    task_knowledge: Dict[str, List[FedWeITKnowledge]] = field(default_factory=dict)
    buffer_state: Dict[str, torch.Tensor] = field(default_factory=dict)
    task_buffer_states: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    retro_anchor_states: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = field(default_factory=dict)
    _seen_knowledge_tasks: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.lr is None:
            self.lr = float(self.trainer.optimizer.param_groups[0].get("lr", 0.01))
        if self.optimizer_name not in {"sgd", "adam"}:
            self.optimizer_name = self.trainer.optimizer.__class__.__name__.lower()
            if self.optimizer_name not in {"sgd", "adam"}:
                self.optimizer_name = "sgd"

    @property
    def parameter_names(self) -> List[str]:
        return [name for name, _ in self.trainer.model.named_parameters()]

    @property
    def buffer_names(self) -> List[str]:
        return [name for name, _ in self.trainer.model.named_buffers()]

    def _global_parameter_state(self, global_state: StateDict) -> Dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in global_state.items()
            if name in self.parameter_names and isinstance(value, torch.Tensor)
        }

    def _buffer_state(
        self,
        global_state: StateDict,
        device: torch.device | str,
        task_id: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        task_buffers = self.task_buffer_states.get(task_id, {}) if task_id is not None else {}
        buffers: Dict[str, torch.Tensor] = {}
        for name in self.buffer_names:
            value = task_buffers.get(name, self.buffer_state.get(name, global_state.get(name)))
            if isinstance(value, torch.Tensor):
                buffers[name] = value.detach().to(device)
        return buffers

    def _base_with_global_update(self, global_state: StateDict) -> Dict[str, torch.Tensor]:
        global_params = self._global_parameter_state(global_state)
        if not self.base_params:
            return _clone_tensor_state(global_params)

        updated = _clone_tensor_state(self.base_params)
        for name, global_value in global_params.items():
            if name not in updated:
                updated[name] = global_value.detach().cpu().clone()
                continue
            if global_value.is_floating_point():
                mask = global_value != 0
                updated[name][mask] = global_value[mask]
            else:
                updated[name] = global_value.detach().cpu().clone()
        return updated

    def _initialize_or_update_base(self, global_state: StateDict) -> None:
        # Algorithm 1 distributes theta_G every round; FedWeIT updates the
        # shareable B_c coordinates from non-zero global entries.
        self.base_params = self._base_with_global_update(global_state)

    def _ensure_task_state(self, task_id: str, knowledge: Sequence[FedWeITKnowledge]) -> None:
        if task_id not in self.mask_logits:
            if float(self.mask_init) < 0.0:
                self.mask_logits[task_id] = {
                    name: torch.empty_like(value).normal_(mean=0.0, std=0.01)
                    for name, value in self.base_params.items()
                    if value.is_floating_point()
                }
            else:
                init_logit = math.log(float(self.mask_init) / max(1e-6, 1.0 - float(self.mask_init)))
                self.mask_logits[task_id] = {
                    name: torch.full_like(value, init_logit)
                    for name, value in self.base_params.items()
                    if value.is_floating_point()
                }
        if task_id not in self.adaptive_params:
            self.adaptive_params[task_id] = {
                name: torch.zeros_like(value)
                for name, value in self.base_params.items()
                if value.is_floating_point()
            }
        if task_id not in self.alpha_logits or len(self.alpha_logits[task_id]) != len(knowledge):
            self.alpha_logits[task_id] = torch.zeros(len(knowledge), dtype=torch.float32)
        self._ensure_retro_anchors(task_id)

    def _ensure_retro_anchors(self, task_id: str) -> None:
        if task_id in self.retro_anchor_states:
            return
        anchors: Dict[str, Dict[str, torch.Tensor]] = {}
        for previous_task_id, previous_state in self.adaptive_params.items():
            if previous_task_id == task_id:
                continue
            masks = self.mask_logits.get(previous_task_id, {})
            task_anchor: Dict[str, torch.Tensor] = {}
            for name, previous_value in previous_state.items():
                if name not in self.base_params or name not in masks:
                    continue
                if not previous_value.is_floating_point() or not self.base_params[name].is_floating_point():
                    continue
                mask_value = torch.sigmoid(masks[name])
                task_anchor[name] = (
                    self.base_params[name] * mask_value + previous_value
                ).detach().cpu().clone()
            if task_anchor:
                anchors[previous_task_id] = task_anchor
        self.retro_anchor_states[task_id] = anchors

    def _prepare_knowledge(
        self,
        task_id: str,
        metadata: Mapping[str, object],
    ) -> List[FedWeITKnowledge]:
        incoming = metadata.get("knowledge_base")
        if incoming is not None and task_id not in self._seen_knowledge_tasks:
            if isinstance(incoming, list):
                self.task_knowledge[task_id] = [
                    entry
                    for entry in incoming
                    if isinstance(entry, FedWeITKnowledge) and entry.client_id != self.client_id
                ]
            else:
                self.task_knowledge[task_id] = []
            self._seen_knowledge_tasks.add(task_id)
        return self.task_knowledge.get(task_id, [])

    def _knowledge_to_device(
        self,
        knowledge: Sequence[FedWeITKnowledge],
        device: torch.device | str,
    ) -> List[FedWeITKnowledge]:
        device_knowledge: List[FedWeITKnowledge] = []
        for entry in knowledge:
            adaptive_state = {
                name: value.detach().to(device)
                for name, value in entry.adaptive_state.items()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            }
            device_knowledge.append(
                FedWeITKnowledge(
                    client_id=entry.client_id,
                    task_id=entry.task_id,
                    adaptive_state=adaptive_state,
                )
            )
        return device_knowledge

    def _task_tensor_params(
        self,
        task_id: str,
        knowledge: Sequence[FedWeITKnowledge],
        device: torch.device | str,
    ) -> tuple[
        Dict[str, torch.nn.Parameter],
        Dict[str, torch.nn.Parameter],
        Dict[str, torch.nn.Parameter],
        Dict[str, Dict[str, torch.nn.Parameter]],
        torch.nn.Parameter,
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        base = {
            name: torch.nn.Parameter(value.detach().to(device).clone())
            for name, value in self.base_params.items()
            if value.is_floating_point()
        }

        masks = {
            name: torch.nn.Parameter(value.detach().to(device).clone())
            for name, value in self.mask_logits[task_id].items()
        }
        adaptive = {
            name: torch.nn.Parameter(value.detach().to(device).clone())
            for name, value in self.adaptive_params[task_id].items()
        }
        previous_adaptive: Dict[str, Dict[str, torch.nn.Parameter]] = {}
        previous_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        previous_effective_anchor = {
            previous_task_id: {
                name: value.detach().to(device).clone()
                for name, value in previous_state.items()
                if value.is_floating_point()
            }
            for previous_task_id, previous_state in self.retro_anchor_states.get(task_id, {}).items()
        }
        for previous_task_id, previous_state in self.adaptive_params.items():
            if previous_task_id == task_id:
                continue
            previous_adaptive[previous_task_id] = {
                name: torch.nn.Parameter(value.detach().to(device).clone())
                for name, value in previous_state.items()
                if value.is_floating_point()
            }

            previous_masks[previous_task_id] = {
                name: torch.sigmoid(value.detach().to(device).clone())
                for name, value in self.mask_logits.get(previous_task_id, {}).items()
            }
        alpha = torch.nn.Parameter(self.alpha_logits[task_id].detach().to(device).clone())
        return (
            base,
            masks,
            adaptive,
            previous_adaptive,
            alpha,
            previous_masks,
            previous_effective_anchor,
        )

    def _knowledge_transfer_state(
        self,
        knowledge: Sequence[FedWeITKnowledge],
        alpha: torch.Tensor,
        device: torch.device | str,
    ) -> Dict[str, torch.Tensor]:
        if not knowledge:
            return {}
        weights = torch.softmax(alpha, dim=0)
        transfer: Dict[str, torch.Tensor] = {}
        for index, entry in enumerate(knowledge):
            for name, value in entry.adaptive_state.items():
                if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                    continue
                value_device = value.detach()
                if value_device.device != torch.device(device):
                    value_device = value_device.to(device)
                weighted = value_device * weights[index]
                if name not in transfer:
                    transfer[name] = weighted
                else:
                    transfer[name] = transfer[name] + weighted
        return transfer

    def _compose_parameters(
        self,
        base: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        adaptive: Mapping[str, torch.Tensor],
        transfer: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        composed: Dict[str, torch.Tensor] = {}
        for name, base_value in base.items():
            mask_value = torch.sigmoid(masks[name])
            composed[name] = base_value * mask_value + adaptive[name]
            if name in transfer:
                composed[name] = composed[name] + transfer[name].to(composed[name].dtype)
        return composed

    def _hard_mask_state(self, task_id: str) -> Dict[str, torch.Tensor]:
        sparsity = min(max(float(self.client_sparsity), 0.0), 1.0)
        hard_masks: Dict[str, torch.Tensor] = {}
        for name, value in self.mask_logits[task_id].items():
            flat = torch.abs(value.detach()).flatten()
            if flat.numel() == 0:
                hard_masks[name] = torch.zeros_like(value).detach().cpu().clone()
                continue
            if sparsity <= 0.0:
                hard_masks[name] = torch.ones_like(value).detach().cpu().clone()
                continue
            if sparsity >= 1.0:
                hard_masks[name] = torch.zeros_like(value).detach().cpu().clone()
                continue
            sorted_values = torch.sort(flat).values
            threshold_index = min(int(math.floor(flat.numel() * sparsity)), flat.numel() - 1)
            threshold = sorted_values[threshold_index]
            hard_masks[name] = (torch.abs(value) > threshold).to(dtype=value.dtype).detach().cpu().clone()
        return hard_masks

    def _sparsify_adaptive_state(self, adaptive_state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        threshold = float(self.lambda1 if self.adaptive_threshold is None else self.adaptive_threshold)
        sparse_state: Dict[str, torch.Tensor] = {}
        for name, value in adaptive_state.items():
            if value.is_floating_point():
                keep = torch.abs(value) > threshold
                sparse_state[name] = (value * keep.to(dtype=value.dtype)).detach().cpu().clone()
            else:
                sparse_state[name] = value.detach().cpu().clone()
        return sparse_state

    def _build_optimizer(self, params: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=float(self.lr), weight_decay=float(self.weight_decay))
        return torch.optim.SGD(
            params,
            lr=float(self.lr),
            momentum=float(self.momentum),
            weight_decay=float(self.weight_decay),
        )

    def _regularization_loss(
        self,
        base: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        adaptive: Mapping[str, torch.Tensor],
        previous_adaptive: Mapping[str, Mapping[str, torch.Tensor]],
        previous_masks: Mapping[str, Mapping[str, torch.Tensor]],
        previous_effective_anchor: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adaptive_sparse_terms: List[torch.Tensor] = []
        mask_sparse_terms: List[torch.Tensor] = []
        retro_terms: List[torch.Tensor] = []
        for name, mask_logit in masks.items():
            mask_sparse_terms.append(torch.sum(torch.abs(torch.sigmoid(mask_logit))))
            adaptive_sparse_terms.append(torch.sum(torch.abs(adaptive[name])))
        for previous_task_id, previous_state in previous_adaptive.items():
            for name, previous_value in previous_state.items():
                adaptive_sparse_terms.append(torch.sum(torch.abs(previous_value)))
                if name not in base:
                    continue
                mask_value = previous_masks.get(previous_task_id, {}).get(name)
                anchor_value = previous_effective_anchor.get(previous_task_id, {}).get(name)
                if mask_value is None or anchor_value is None:
                    continue
                restored = base[name] * mask_value + previous_value
                retro_terms.append(torch.sum(torch.square(restored - anchor_value)))
        device = next(iter(adaptive.values())).device
        adaptive_sparse_loss = (
            torch.stack(adaptive_sparse_terms).sum() if adaptive_sparse_terms else torch.tensor(0.0, device=device)
        )
        mask_sparse_loss = torch.stack(mask_sparse_terms).sum() if mask_sparse_terms else torch.tensor(0.0, device=device)
        retro_loss = torch.stack(retro_terms).sum() if retro_terms else torch.tensor(0.0, device=device)
        return adaptive_sparse_loss, mask_sparse_loss, retro_loss

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        task_id = context.task_id or next(iter(self.task_loaders))
        if task_id not in self.task_loaders:
            task_id = next(iter(self.task_loaders))

        device = self.trainer.device
        self.trainer.model.to(device)
        self._initialize_or_update_base(global_state)
        knowledge = self._prepare_knowledge(task_id, context.metadata)
        self._ensure_task_state(task_id, knowledge)
        device_knowledge = self._knowledge_to_device(knowledge, device)

        (
            base,
            masks,
            adaptive,
            previous_adaptive,
            alpha,
            previous_masks,
            previous_effective_anchor,
        ) = self._task_tensor_params(task_id, device_knowledge, device)
        trainable: List[torch.nn.Parameter] = [*base.values(), *masks.values(), *adaptive.values()]
        for previous_state in previous_adaptive.values():
            trainable.extend(previous_state.values())
        if len(device_knowledge) > 0:
            trainable.append(alpha)
        optimizer = self._build_optimizer(trainable)
        buffers = self._buffer_state(global_state, device, task_id=task_id)
        loader = self.task_loaders[task_id]

        total_examples = 0
        total_loss = 0.0
        total_ce = 0.0
        total_sparse = 0.0
        total_retro = 0.0
        total_correct = 0

        for _ in range(int(self.epochs)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, device)
                targets = move_to_device(targets, device)
                transfer = self._knowledge_transfer_state(device_knowledge, alpha, device)
                composed = self._compose_parameters(base, masks, adaptive, transfer)
                params_and_buffers = {**buffers, **composed}
                logits = functional_call(self.trainer.model, params_and_buffers, (inputs,))
                ce_loss = F.cross_entropy(logits, targets)
                adaptive_sparse_loss, mask_sparse_loss, retro_loss = self._regularization_loss(
                    base=base,
                    masks=masks,
                    adaptive=adaptive,
                    previous_adaptive=previous_adaptive,
                    previous_masks=previous_masks,
                    previous_effective_anchor=previous_effective_anchor,
                )
                sparse_loss = float(self.lambda1) * adaptive_sparse_loss + float(self.lambda_mask) * mask_sparse_loss
                loss = ce_loss + sparse_loss + float(self.lambda2) * retro_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_ce += float(ce_loss.detach().item()) * batch_size
                total_sparse += float(sparse_loss.detach().item()) * batch_size
                total_retro += float(retro_loss.detach().item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == targets).sum().item())

        self.base_params = {name: value.detach().cpu().clone() for name, value in base.items()}
        self.buffer_state = {name: value.detach().cpu().clone() for name, value in buffers.items()}
        self.task_buffer_states[task_id] = {
            name: value.detach().cpu().clone()
            for name, value in buffers.items()
        }
        self.mask_logits[task_id] = {name: value.detach().cpu().clone() for name, value in masks.items()}
        self.adaptive_params[task_id] = {name: value.detach().cpu().clone() for name, value in adaptive.items()}
        for previous_task_id, previous_state in previous_adaptive.items():
            self.adaptive_params[previous_task_id] = {
                name: value.detach().cpu().clone()
                for name, value in previous_state.items()
            }
        self.alpha_logits[task_id] = alpha.detach().cpu().clone()

        mask_state = {
            name: torch.sigmoid(value).detach().cpu().clone()
            for name, value in self.mask_logits[task_id].items()
        }
        hard_mask_state = self._hard_mask_state(task_id)
        shared_state = {
            name: self.base_params[name] * hard_mask_state[name]
            for name in self.base_params
            if name in hard_mask_state
        }
        adaptive_state = self._sparsify_adaptive_state(self.adaptive_params[task_id])
        alpha_state = torch.softmax(self.alpha_logits[task_id], dim=0).detach().cpu()
        shared_numel = _state_numel(shared_state)
        adaptive_numel = _state_numel(adaptive_state)
        shared_nnz = _state_nnz(shared_state)
        adaptive_nnz = _state_nnz(adaptive_state)

        if total_examples == 0:
            metrics: MetricDict = {
                "loss": 0.0,
                "ce_loss": 0.0,
                "sparse_loss": 0.0,
                "retro_loss": 0.0,
                "accuracy": 0.0,
                "kb_size": float(len(knowledge)),
            }
        else:
            metrics = {
                "loss": total_loss / total_examples,
                "ce_loss": total_ce / total_examples,
                "sparse_loss": total_sparse / total_examples,
                "retro_loss": total_retro / total_examples,
                "accuracy": total_correct / total_examples,
                "kb_size": float(len(knowledge)),
                "shared_norm": _state_l2_norm(shared_state),
                "adaptive_norm": _state_l2_norm(adaptive_state),
                "shared_nnz": float(shared_nnz),
                "adaptive_nnz": float(adaptive_nnz),
                "shared_density": float(shared_nnz / max(1, shared_numel)),
                "adaptive_density": float(adaptive_nnz / max(1, adaptive_numel)),
            }

        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics=metrics,
            payload={
                "base_state": detach_state_dict(self.base_params),
                "mask_state": mask_state,
                "hard_mask_state": hard_mask_state,
                "shared_state": detach_state_dict(shared_state),
                "buffer_state": detach_state_dict(self.buffer_state),
                "adaptive_state": adaptive_state,
                "alpha_state": alpha_state,
            },
        )

    def build_eval_state(self, global_state: StateDict, task_id: str) -> Dict[str, torch.Tensor]:
        if task_id not in self.mask_logits or task_id not in self.adaptive_params:
            return _clone_tensor_state(global_state)

        knowledge = self.task_knowledge.get(task_id, [])
        device = self.trainer.device
        alpha = self.alpha_logits.get(task_id, torch.zeros(len(knowledge))).to(device)
        device_knowledge = self._knowledge_to_device(knowledge, device)
        transfer = self._knowledge_transfer_state(device_knowledge, alpha, device)
        mask_state = {name: value.to(device) for name, value in self.mask_logits[task_id].items()}
        eval_base = self._base_with_global_update(global_state)
        base = {name: value.to(device) for name, value in eval_base.items() if name in mask_state}
        adaptive = {name: value.to(device) for name, value in self.adaptive_params[task_id].items()}
        composed = self._compose_parameters(base, mask_state, adaptive, transfer)
        state = _clone_tensor_state(global_state)
        eval_buffers = self.task_buffer_states.get(task_id, self.buffer_state)
        for name, value in eval_buffers.items():
            state[name] = value.detach().cpu().clone()
        for name, value in composed.items():
            state[name] = value.detach().cpu().clone()
        return state


@dataclass
class FedWeITAggregator:
    metric_prefix: str = "client_"

    def aggregate(self, client_results: List[TrainResult]) -> AggregationResult:
        if not client_results:
            return AggregationResult(global_state={}, metrics={"num_clients": 0.0, "total_samples": 0.0})

        shared_states: List[StateDict] = []
        hard_mask_states: List[StateDict] = []
        buffer_states: List[StateDict] = []
        metric_sums: Dict[str, float] = {}
        metric_weights: Dict[str, float] = {}
        for result in client_results:
            shared_state = result.payload.get("shared_state", {})
            hard_mask_state = result.payload.get("hard_mask_state", {})
            if isinstance(shared_state, Mapping) and shared_state:
                shared_states.append(shared_state)
                hard_mask_states.append(hard_mask_state if isinstance(hard_mask_state, Mapping) else {})
            buffer_state = result.payload.get("buffer_state", {})
            if isinstance(buffer_state, Mapping) and buffer_state:
                buffer_states.append(buffer_state)
            for metric_name, metric_value in result.metrics.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + float(metric_value)
                metric_weights[metric_name] = metric_weights.get(metric_name, 0.0) + 1.0

        if not shared_states:
            return AggregationResult(
                global_state={},
                metrics={"num_clients": float(len(client_results)), "total_samples": 0.0},
            )

        averaged_state: Dict[str, torch.Tensor] = {}
        for name in shared_states[0].keys():
            numerators = [state[name].detach().float() for state in shared_states if name in state]
            denominators = [
                mask_state[name].detach().float()
                for mask_state in hard_mask_states
                if isinstance(mask_state, Mapping) and name in mask_state
            ]
            if not numerators:
                continue
            numerator = torch.stack(numerators, dim=0).sum(dim=0)
            if len(denominators) == len(numerators):
                denominator = torch.stack(denominators, dim=0).sum(dim=0)
                averaged_state[name] = torch.where(
                    denominator > 0,
                    numerator / torch.clamp(denominator, min=1.0),
                    torch.zeros_like(numerator),
                )
            else:
                averaged_state[name] = numerator / float(len(numerators))
        if buffer_states:
            for name, first_value in buffer_states[0].items():
                if not isinstance(first_value, torch.Tensor):
                    continue
                values = [state[name] for state in buffer_states if name in state and isinstance(state[name], torch.Tensor)]
                if not values:
                    continue
                if first_value.is_floating_point():
                    averaged_state[name] = torch.stack([value.detach().float() for value in values], dim=0).mean(dim=0)
                else:
                    averaged_state[name] = values[0].detach().clone()

        averaged_metrics = {
            f"{self.metric_prefix}{metric_name}": metric_sums[metric_name] / metric_weights[metric_name]
            for metric_name in metric_sums
            if metric_weights.get(metric_name, 0.0) > 0.0
        }
        averaged_metrics["num_clients"] = float(len(shared_states))
        averaged_metrics["total_samples"] = float(sum(max(int(result.num_samples), 0) for result in client_results))

        return AggregationResult(
            global_state=averaged_state,
            metrics=averaged_metrics,
            metadata={"aggregator": "fedweit", "aggregation": "active_mask_mean_B_mask"},
        )


@dataclass
class FedWeITServer:
    model: torch.nn.Module
    clients: Sequence[FedWeITClient]
    aggregator: FedWeITAggregator = field(default_factory=FedWeITAggregator)
    client_sample_ratio: float = 1.0
    kb_sample_size: int = 0
    knowledge_base: List[FedWeITKnowledge] = field(default_factory=list)
    sampled_task_kb: Dict[str, List[FedWeITKnowledge]] = field(default_factory=dict)
    task_adaptive_buffer: Dict[str, Dict[str, FedWeITKnowledge]] = field(default_factory=dict)
    _clients_that_received_kb: set[tuple[str, str]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not (0.0 < float(self.client_sample_ratio) <= 1.0):
            raise ValueError("client_sample_ratio must be in (0, 1].")
        if int(self.kb_sample_size) < 0:
            raise ValueError("kb_sample_size must be non-negative.")

    def on_task_start(self, task: TaskDefinition) -> None:
        self.task_adaptive_buffer.setdefault(task.task_id, {})
        if not self.knowledge_base:
            self.sampled_task_kb[task.task_id] = []
        elif self.kb_sample_size <= 0 or self.kb_sample_size >= len(self.knowledge_base):
            self.sampled_task_kb[task.task_id] = list(self.knowledge_base)
        else:
            self.sampled_task_kb[task.task_id] = random.sample(
                self.knowledge_base,
                k=int(self.kb_sample_size),
            )

    def on_task_end(self, task: TaskDefinition) -> None:
        # Algorithm 1, line 11: kb <- kb union {A_j^(t)} for all clients.
        self.knowledge_base.extend(self.task_adaptive_buffer.get(task.task_id, {}).values())

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        return detach_state_dict(self.model.state_dict())

    def set_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        current_state = self.get_global_state()
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                current_state[name] = value.detach().cpu().clone()
        self.model.load_state_dict(current_state, strict=True)

    def _select_clients(self) -> List[FedWeITClient]:
        clients = list(self.clients)
        if clients and self.client_sample_ratio < 1.0:
            num_selected = max(1, int(len(clients) * self.client_sample_ratio))
            clients = random.sample(clients, k=num_selected)
        return clients

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        global_state = self.get_global_state()
        client_results: List[TrainResult] = []

        for client in self._select_clients():
            received_key = (client.client_id, task_id)
            metadata: dict[str, object] = {}
            if received_key not in self._clients_that_received_kb:
                metadata["knowledge_base"] = self.sampled_task_kb.get(task_id, [])
                self._clients_that_received_kb.add(received_key)
            context = ClientContext(
                client_id=client.client_id,
                round_idx=round_idx,
                task_id=task_id,
                metadata=metadata,
            )
            result = client.fit(global_state, context)
            client_results.append(result)

            adaptive_state = result.payload.get("adaptive_state", {})
            if isinstance(adaptive_state, Mapping) and adaptive_state:
                self.task_adaptive_buffer.setdefault(task_id, {})[client.client_id] = FedWeITKnowledge(
                    client_id=client.client_id,
                    task_id=task_id,
                    adaptive_state=detach_state_dict(dict(adaptive_state)),
                )

        aggregation_result = self.aggregator.aggregate(client_results)
        if aggregation_result.global_state:
            self.set_global_state(aggregation_result.global_state)

        metadata = dict(aggregation_result.metadata)
        metadata["round_idx"] = round_idx
        metadata["task_id"] = task_id
        metadata["knowledge_base_size"] = float(len(self.knowledge_base))
        metadata["sampled_kb_size"] = float(len(self.sampled_task_kb.get(task_id, [])))
        return AggregationResult(
            global_state=aggregation_result.global_state,
            metrics=dict(aggregation_result.metrics),
            metadata=metadata,
        )

    def build_eval_state(self, task_id: str, client_id: str | None = None) -> Dict[str, torch.Tensor]:
        if client_id is not None:
            for client in self.clients:
                if client.client_id == client_id:
                    return client.build_eval_state(self.get_global_state(), task_id)
        for client in self.clients:
            if task_id in client.mask_logits and task_id in client.adaptive_params:
                return client.build_eval_state(self.get_global_state(), task_id)
        return self.get_global_state()
