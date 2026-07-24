from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset

from ...contracts import AggregationResult, ClientContext, MetricDict, StateDict, TrainResult
from ...datasets.build import build_dataloader
from ...trainers.utils import detach_state_dict, move_to_device
from ..fedweit import (
    FedWeITClient,
    FedWeITKnowledge,
    FedWeITServer,
    _state_l2_norm,
    _state_nnz,
    _state_numel,
)


def _channel_tensor(
    value: float | Sequence[float] | torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=reference.device, dtype=reference.dtype)
    else:
        tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        return tensor
    shape = [1] * reference.ndim
    shape[1] = int(tensor.numel())
    return tensor.reshape(shape)


def _clamp_inputs(
    inputs: torch.Tensor,
    clip_min: float | Sequence[float] | torch.Tensor | None,
    clip_max: float | Sequence[float] | torch.Tensor | None,
) -> torch.Tensor:
    lower = _channel_tensor(clip_min, inputs)
    upper = _channel_tensor(clip_max, inputs)
    if lower is not None:
        inputs = torch.max(inputs, lower)
    if upper is not None:
        inputs = torch.min(inputs, upper)
    return inputs


def _cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    first_flat = first.detach().float().reshape(-1)
    second_flat = second.detach().float().reshape(-1)
    first_norm = torch.norm(first_flat)
    second_norm = torch.norm(second_flat)
    if float(first_norm.item()) == 0.0 or float(second_norm.item()) == 0.0:
        return 0.0
    return float(
        F.cosine_similarity(first_flat.unsqueeze(0), second_flat.unsqueeze(0), dim=1, eps=1e-8).item()
    )


class _FunctionalModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, params_and_buffers: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.model = model
        self.params_and_buffers = params_and_buffers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional_call(self.model, self.params_and_buffers, (inputs,))


@dataclass
class FedWeITOwnClient(FedWeITClient):
    """FedWeIT client with risk-aware UAP defense from the FedWeIT4_3 variant."""

    uap_epsilon: float | Sequence[float] | torch.Tensor = 10.0 / 255.0
    uap_lr: float = 0.05
    uap_gen_epochs: int = 2
    uap_data_ratio: float = 0.1
    uap_clip_min: float | Sequence[float] | torch.Tensor | None = None
    uap_clip_max: float | Sequence[float] | torch.Tensor | None = None
    adv_epochs: int = 4
    stage1_epochs: int = 2
    adv_mix_ratio: float = 0.1
    conf_threshold: float = 0.1
    defense_lr_scale: float = 0.5
    task_high_risk_indices: Dict[str, set[int]] = field(default_factory=dict)
    task_low_risk_indices: Dict[str, set[int]] = field(default_factory=dict)
    task_local_uaps: Dict[str, torch.Tensor] = field(default_factory=dict)
    best_uaps: Dict[str, torch.Tensor] = field(default_factory=dict)
    eval_base_override: Dict[str, torch.Tensor] = field(default_factory=dict)

    def _ordered_task_loader(self, task_id: str) -> DataLoader:
        base_loader = self.task_loaders[task_id]
        batch_size = int(base_loader.batch_size) if base_loader.batch_size is not None else max(1, len(base_loader.dataset))
        return build_dataloader(
            base_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    def _subset_loader(
        self,
        loader: DataLoader,
        task_id: str,
    ) -> DataLoader:
        data_ratio = min(max(float(self.uap_data_ratio), 0.0), 1.0)
        if data_ratio <= 0.0 or data_ratio >= 1.0 or len(loader.dataset) <= 1:
            return loader

        subset_len = max(1, int(math.ceil(len(loader.dataset) * data_ratio)))
        seed = sum(ord(ch) for ch in f"{self.client_id}:{task_id}")
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(loader.dataset), generator=generator)[:subset_len].tolist()
        subset = Subset(loader.dataset, indices)
        batch_size = int(loader.batch_size) if loader.batch_size is not None else max(1, len(subset))
        return build_dataloader(subset, batch_size=batch_size, shuffle=True)

    def _compose_current_task_tensors(
        self,
        task_id: str,
        global_state: StateDict,
        device: torch.device | str,
    ) -> tuple[
        Dict[str, torch.nn.Parameter],
        Dict[str, torch.nn.Parameter],
        Dict[str, torch.nn.Parameter],
        torch.nn.Parameter,
        Dict[str, torch.Tensor],
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
        alpha = torch.nn.Parameter(self.alpha_logits[task_id].detach().to(device).clone())
        buffers = self._buffer_state(global_state, device, task_id=task_id)
        return base, masks, adaptive, alpha, buffers

    def _build_local_optimizer(
        self,
        params: Sequence[torch.nn.Parameter],
        lr_scale: float = 1.0,
    ) -> torch.optim.Optimizer:
        lr = float(self.lr) * float(lr_scale)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=float(self.weight_decay))
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(self.momentum),
            weight_decay=float(self.weight_decay),
        )

    def _apply_uap(self, inputs: torch.Tensor, uap: torch.Tensor) -> torch.Tensor:
        perturbed = inputs + uap.to(device=inputs.device, dtype=inputs.dtype)
        return _clamp_inputs(perturbed, self.uap_clip_min, self.uap_clip_max)

    def _functional_params_and_buffers(
        self,
        base: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        adaptive: Mapping[str, torch.Tensor],
        alpha: torch.Tensor,
        knowledge: Sequence[FedWeITKnowledge],
        buffers: Mapping[str, torch.Tensor],
        device: torch.device | str,
        detach_tensors: bool = False,
    ) -> Dict[str, torch.Tensor]:
        transfer = self._knowledge_transfer_state(knowledge, alpha, device)
        composed = self._compose_parameters(base, masks, adaptive, transfer)
        params_and_buffers = {**buffers, **composed}
        if not detach_tensors:
            return params_and_buffers
        return {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name, value in params_and_buffers.items()
        }

    def _evaluate_risk_partition(
        self,
        task_id: str,
        base: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        adaptive: Mapping[str, torch.Tensor],
        alpha: torch.Tensor,
        knowledge: Sequence[FedWeITKnowledge],
        buffers: Mapping[str, torch.Tensor],
        device: torch.device | str,
    ) -> tuple[set[int], set[int]]:
        loader = self._ordered_task_loader(task_id)
        self.trainer.model.to(self.trainer.device)
        self.trainer.model.eval()
        params_and_buffers = self._functional_params_and_buffers(
            base=base,
            masks=masks,
            adaptive=adaptive,
            alpha=alpha,
            knowledge=knowledge,
            buffers=buffers,
            device=device,
            detach_tensors=True,
        )

        high_risk: set[int] = set()
        low_risk: set[int] = set()
        offset = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                logits = functional_call(self.trainer.model, params_and_buffers, (inputs,))
                confidences = torch.softmax(logits, dim=1)
                max_confidence, predictions = torch.max(confidences, dim=1)
                risk_mask = (max_confidence < float(self.conf_threshold)) | (predictions != targets)
                for sample_idx in range(int(targets.shape[0])):
                    absolute_idx = offset + sample_idx
                    if bool(risk_mask[sample_idx].item()):
                        high_risk.add(absolute_idx)
                    else:
                        low_risk.add(absolute_idx)
                offset += int(targets.shape[0])
        return high_risk, low_risk

    def _generate_local_uap(
        self,
        task_id: str,
        base: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        adaptive: Mapping[str, torch.Tensor],
        alpha: torch.Tensor,
        knowledge: Sequence[FedWeITKnowledge],
        buffers: Mapping[str, torch.Tensor],
        device: torch.device | str,
    ) -> torch.Tensor | None:
        loader = self._subset_loader(self._ordered_task_loader(task_id), task_id=task_id)
        if len(loader.dataset) == 0:
            return None

        self.trainer.model.to(self.trainer.device)
        self.trainer.model.eval()
        params_and_buffers = self._functional_params_and_buffers(
            base=base,
            masks=masks,
            adaptive=adaptive,
            alpha=alpha,
            knowledge=knowledge,
            buffers=buffers,
            device=device,
            detach_tensors=True,
        )

        delta: torch.nn.Parameter | None = None
        optimizer: torch.optim.Optimizer | None = None
        final_delta: torch.Tensor | None = None
        epsilon = None

        for _ in range(max(1, int(self.uap_gen_epochs))):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                if delta is None:
                    delta = torch.nn.Parameter(torch.zeros_like(inputs[:1]))
                    optimizer = torch.optim.Adam([delta], lr=float(self.uap_lr))
                    epsilon = _channel_tensor(self.uap_epsilon, inputs[:1])
                assert delta is not None
                assert optimizer is not None
                assert epsilon is not None
                perturbed = _clamp_inputs(inputs + delta, self.uap_clip_min, self.uap_clip_max)
                logits = functional_call(self.trainer.model, params_and_buffers, (perturbed,))
                loss = -F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
                delta.grad = grad.detach()
                optimizer.step()
                delta.data = torch.clamp(delta.data, min=-epsilon, max=epsilon)
                final_delta = delta.detach().clone()

        return None if final_delta is None else final_delta.detach().cpu().clone()

    def _refresh_best_uap(
        self,
        task_id: str,
        uap_scores: torch.Tensor,
        uap_counts: torch.Tensor,
        pool: Sequence[torch.Tensor],
    ) -> None:
        if not pool or int(uap_counts.numel()) == 0 or float(uap_counts.sum().item()) <= 0.0:
            return
        best_idx = int(torch.argmax(uap_scores / torch.clamp(uap_counts, min=1e-9)).item())
        self.best_uaps[task_id] = pool[best_idx].detach().cpu().clone()

    def _finalize_current_task_state(
        self,
        task_id: str,
        base: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        adaptive: Mapping[str, torch.Tensor],
        alpha: torch.Tensor,
        buffers: Mapping[str, torch.Tensor],
    ) -> tuple[StateDict, StateDict, StateDict, torch.Tensor, StateDict]:
        self.base_params = {name: value.detach().cpu().clone() for name, value in base.items()}
        self.buffer_state = {name: value.detach().cpu().clone() for name, value in buffers.items()}
        self.task_buffer_states[task_id] = {
            name: value.detach().cpu().clone()
            for name, value in buffers.items()
        }
        self.mask_logits[task_id] = {name: value.detach().cpu().clone() for name, value in masks.items()}
        self.adaptive_params[task_id] = {name: value.detach().cpu().clone() for name, value in adaptive.items()}
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
        return mask_state, hard_mask_state, adaptive_state, alpha_state, shared_state

    def build_eval_state(self, global_state: StateDict, task_id: str) -> Dict[str, torch.Tensor]:
        if task_id not in self.mask_logits or task_id not in self.adaptive_params:
            return super().build_eval_state(global_state, task_id)

        knowledge = self.task_knowledge.get(task_id, [])
        device = self.trainer.device
        alpha = self.alpha_logits.get(task_id, torch.zeros(len(knowledge))).to(device)
        device_knowledge = self._knowledge_to_device(knowledge, device)
        transfer = self._knowledge_transfer_state(device_knowledge, alpha, device)
        mask_state = {name: value.to(device) for name, value in self.mask_logits[task_id].items()}
        eval_base_state = self.eval_base_override if self.eval_base_override else self._base_with_global_update(global_state)
        base = {name: value.to(device) for name, value in eval_base_state.items() if name in mask_state}
        adaptive = {name: value.to(device) for name, value in self.adaptive_params[task_id].items()}
        composed = self._compose_parameters(base, mask_state, adaptive, transfer)
        state = {
            name: value.detach().cpu().clone()
            for name, value in self.trainer.model.state_dict().items()
        }
        eval_buffers = self.task_buffer_states.get(task_id, self.buffer_state)
        for name, value in eval_buffers.items():
            state[name] = value.detach().cpu().clone()
        for name, value in composed.items():
            state[name] = value.detach().cpu().clone()
        return state

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        task_id = context.task_id or next(iter(self.task_loaders))
        if task_id not in self.task_loaders:
            task_id = next(iter(self.task_loaders))

        device = self.trainer.device
        self.trainer.model.to(device)
        self.trainer.model.train()
        self.eval_base_override = {}
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

        high_risk_indices, low_risk_indices = self._evaluate_risk_partition(
            task_id=task_id,
            base=base,
            masks=masks,
            adaptive=adaptive,
            alpha=alpha,
            knowledge=device_knowledge,
            buffers=buffers,
            device=device,
        )
        self.task_high_risk_indices[task_id] = high_risk_indices
        self.task_low_risk_indices[task_id] = low_risk_indices
        local_uap = self._generate_local_uap(
            task_id=task_id,
            base=base,
            masks=masks,
            adaptive=adaptive,
            alpha=alpha,
            knowledge=device_knowledge,
            buffers=buffers,
            device=device,
        )
        if local_uap is not None:
            self.task_local_uaps[task_id] = local_uap.detach().cpu().clone()

        num_high_risk = float(len(high_risk_indices))
        num_low_risk = float(len(low_risk_indices))
        num_partitioned = max(1.0, num_high_risk + num_low_risk)
        uap_linf = 0.0
        uap_l2 = 0.0
        if local_uap is not None:
            uap_linf = float(local_uap.detach().abs().max().item())
            uap_l2 = float(local_uap.detach().float().norm().item())

        if total_examples == 0:
            metrics: MetricDict = {
                "loss": 0.0,
                "ce_loss": 0.0,
                "sparse_loss": 0.0,
                "retro_loss": 0.0,
                "accuracy": 0.0,
                "kb_size": float(len(knowledge)),
                "high_risk_ratio": 0.0,
                "num_high_risk": num_high_risk,
                "num_low_risk": num_low_risk,
                "uap_linf": uap_linf,
                "uap_l2": uap_l2,
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
                "high_risk_ratio": num_high_risk / num_partitioned,
                "num_high_risk": num_high_risk,
                "num_low_risk": num_low_risk,
                "uap_linf": uap_linf,
                "uap_l2": uap_l2,
            }

        payload: StateDict = {
            "base_state": detach_state_dict(self.base_params),
            "mask_state": mask_state,
            "hard_mask_state": hard_mask_state,
            "shared_state": detach_state_dict(shared_state),
            "buffer_state": detach_state_dict(self.buffer_state),
            "adaptive_state": adaptive_state,
            "alpha_state": alpha_state,
        }
        if local_uap is not None:
            payload["local_uap"] = local_uap.detach().cpu().clone()

        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics=metrics,
            payload=payload,
        )

    def run_defense(
        self,
        global_state: StateDict,
        task_id: str,
        uap_pool: Sequence[torch.Tensor],
    ) -> MetricDict:
        if task_id not in self.task_loaders:
            return {}

        initial_pool: List[torch.Tensor] = [
            uap.detach().to(self.trainer.device)
            for uap in uap_pool
            if isinstance(uap, torch.Tensor)
        ]
        for best_uap in self.best_uaps.values():
            if isinstance(best_uap, torch.Tensor):
                initial_pool.append(best_uap.detach().to(self.trainer.device))
        if not initial_pool:
            return {}

        device = self.trainer.device
        self.trainer.model.to(device)
        self.trainer.model.train()
        self._initialize_or_update_base(global_state)
        knowledge = self.task_knowledge.get(task_id, [])
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
        ) = self._task_tensor_params(
            task_id=task_id,
            knowledge=device_knowledge,
            device=device,
        )
        previous_adaptive = {
            previous_task_id: {
                name: value.detach()
                for name, value in previous_state.items()
            }
            for previous_task_id, previous_state in previous_adaptive.items()
        }
        buffers = self._buffer_state(global_state, device, task_id=task_id)
        trainable: List[torch.nn.Parameter] = [*base.values(), *masks.values(), *adaptive.values()]
        if len(device_knowledge) > 0:
            trainable.append(alpha)
        optimizer = self._build_local_optimizer(trainable, lr_scale=float(self.defense_lr_scale))
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        high_risk_indices = self.task_high_risk_indices.get(task_id, set())
        ordered_loader = self._ordered_task_loader(task_id)

        total_examples = 0
        total_loss = 0.0
        total_correct = 0
        total_adv_examples = 0

        stage1_epochs = min(max(int(self.stage1_epochs), 0), max(1, int(self.adv_epochs)))
        for epoch_idx in range(max(1, int(self.adv_epochs))):
            is_stage1 = epoch_idx < stage1_epochs
            current_pool = initial_pool
            if not is_stage1 and task_id in self.best_uaps:
                current_pool = [self.best_uaps[task_id].detach().to(device)]
            if not current_pool:
                current_pool = initial_pool
            if not current_pool:
                break

            uap_scores = torch.zeros(len(current_pool), device=device)
            uap_counts = torch.zeros(len(current_pool), device=device)
            sample_offset = 0

            for inputs, targets in ordered_loader:
                inputs = move_to_device(inputs, device)
                targets = move_to_device(targets, device)
                batch_size = int(targets.shape[0])
                batch_uap_indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
                mixed_inputs = inputs.detach().clone()

                for sample_idx in range(batch_size):
                    absolute_idx = sample_offset + sample_idx
                    is_high_risk = absolute_idx in high_risk_indices
                    if is_high_risk:
                        chosen_idx = int(torch.randint(0, len(current_pool), (1,), device=device).item())
                        mixed_inputs[sample_idx : sample_idx + 1] = self._apply_uap(
                            mixed_inputs[sample_idx : sample_idx + 1],
                            current_pool[chosen_idx],
                        )
                        batch_uap_indices[sample_idx] = chosen_idx
                    elif not is_stage1 and float(torch.rand(1, device=device).item()) < float(self.adv_mix_ratio):
                        chosen_idx = len(current_pool) - 1
                        mixed_inputs[sample_idx : sample_idx + 1] = self._apply_uap(
                            mixed_inputs[sample_idx : sample_idx + 1],
                            current_pool[chosen_idx],
                        )
                        batch_uap_indices[sample_idx] = chosen_idx

                transfer = self._knowledge_transfer_state(device_knowledge, alpha, device)
                composed = self._compose_parameters(base, masks, adaptive, transfer)
                params_and_buffers = {**buffers, **composed}
                logits = functional_call(self.trainer.model, params_and_buffers, (mixed_inputs,))
                full_losses = criterion(logits, targets)
                ce_loss = full_losses.mean()
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

                if epoch_idx in {max(0, stage1_epochs - 1), max(0, int(self.adv_epochs) - 1)}:
                    adv_mask = batch_uap_indices >= 0
                    if bool(adv_mask.any().item()):
                        active_indices = batch_uap_indices[adv_mask]
                        uap_scores.index_add_(0, active_indices, full_losses[adv_mask].detach())
                        uap_counts.index_add_(0, active_indices, torch.ones_like(full_losses[adv_mask]))

                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == targets).sum().item())
                total_adv_examples += int((batch_uap_indices >= 0).sum().item())
                sample_offset += batch_size

            if epoch_idx in {max(0, stage1_epochs - 1), max(0, int(self.adv_epochs) - 1)}:
                self._refresh_best_uap(task_id, uap_scores, uap_counts, current_pool)

        mask_state, hard_mask_state, adaptive_state, alpha_state, shared_state = self._finalize_current_task_state(
            task_id=task_id,
            base=base,
            masks=masks,
            adaptive=adaptive,
            alpha=alpha,
            buffers=buffers,
        )
        self.eval_base_override = {
            name: value.detach().cpu().clone()
            for name, value in self.base_params.items()
        }
        shared_numel = _state_numel(shared_state)
        adaptive_numel = _state_numel(adaptive_state)
        shared_nnz = _state_nnz(shared_state)
        adaptive_nnz = _state_nnz(adaptive_state)
        _ = mask_state, alpha_state

        num_high_risk = float(len(high_risk_indices))
        num_low_risk = float(len(self.task_low_risk_indices.get(task_id, set())))
        best_uap = self.best_uaps.get(task_id)
        best_uap_linf = float(best_uap.abs().max().item()) if isinstance(best_uap, torch.Tensor) and best_uap.numel() > 0 else 0.0

        if total_examples == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "adv_ratio": 0.0,
                "pool_size": float(len(initial_pool)),
                "num_high_risk": num_high_risk,
                "num_low_risk": num_low_risk,
                "best_uap_linf": best_uap_linf,
            }
        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
            "adv_ratio": float(total_adv_examples / max(1, total_examples)),
            "pool_size": float(len(initial_pool)),
            "num_high_risk": num_high_risk,
            "num_low_risk": num_low_risk,
            "best_uap_linf": best_uap_linf,
            "shared_norm": _state_l2_norm(shared_state),
            "adaptive_norm": _state_l2_norm(adaptive_state),
            "shared_nnz": float(shared_nnz),
            "adaptive_nnz": float(adaptive_nnz),
            "shared_density": float(shared_nnz / max(1, shared_numel)),
            "adaptive_density": float(adaptive_nnz / max(1, adaptive_numel)),
        }


@dataclass
class FedWeITOwnServer(FedWeITServer):
    uap_topk: int = 2
    client_latest_uaps: Dict[str, torch.Tensor] = field(default_factory=dict)

    def _update_uap_memory(
        self,
        selected_clients: Sequence[FedWeITOwnClient],
        client_results: Sequence[TrainResult],
    ) -> None:
        for result in client_results:
            local_uap = result.payload.get("local_uap")
            if isinstance(local_uap, torch.Tensor):
                self.client_latest_uaps[result.client_id] = local_uap.detach().cpu().clone()

    def _personalized_uap_pool(self, client_id: str) -> List[torch.Tensor]:
        if client_id not in self.client_latest_uaps:
            return []
        target_uap = self.client_latest_uaps[client_id].detach().cpu().clone()
        candidates: List[tuple[float, torch.Tensor]] = []
        for other_client_id, other_uap in self.client_latest_uaps.items():
            if other_client_id == client_id:
                continue
            candidates.append((_cosine_similarity(target_uap, other_uap), other_uap.detach().cpu().clone()))
        candidates.sort(key=lambda item: item[0], reverse=True)
        topk = max(0, int(self.uap_topk))
        pool = [target_uap]
        pool.extend(candidate.detach().cpu().clone() for _, candidate in candidates[:topk])
        return pool

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        global_state = self.get_global_state()
        client_results: List[TrainResult] = []
        selected_clients = [
            client
            for client in self._select_clients()
            if isinstance(client, FedWeITOwnClient)
        ]

        for client in selected_clients:
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
        self._update_uap_memory(selected_clients, client_results)

        defense_metric_sums: Dict[str, float] = {}
        defense_metric_counts: Dict[str, float] = {}
        defended_clients = 0
        updated_global_state = self.get_global_state()
        for client in selected_clients:
            defense_metrics = client.run_defense(
                global_state=updated_global_state,
                task_id=task_id,
                uap_pool=self._personalized_uap_pool(client.client_id),
            )
            if defense_metrics:
                defended_clients += 1
            for metric_name, metric_value in defense_metrics.items():
                defense_metric_sums[metric_name] = defense_metric_sums.get(metric_name, 0.0) + float(metric_value)
                defense_metric_counts[metric_name] = defense_metric_counts.get(metric_name, 0.0) + 1.0

        metrics = dict(aggregation_result.metrics)
        for metric_name, metric_sum in defense_metric_sums.items():
            count = defense_metric_counts.get(metric_name, 0.0)
            if count > 0.0:
                metrics[f"defense_{metric_name}"] = metric_sum / count
        metrics["num_defense_clients"] = float(defended_clients)
        metrics["num_uap_clients"] = float(len(self.client_latest_uaps))

        metadata = dict(aggregation_result.metadata)
        metadata["round_idx"] = round_idx
        metadata["task_id"] = task_id
        metadata["knowledge_base_size"] = float(len(self.knowledge_base))
        metadata["sampled_kb_size"] = float(len(self.sampled_task_kb.get(task_id, [])))
        metadata["uap_topk"] = float(self.uap_topk)
        return AggregationResult(
            global_state=aggregation_result.global_state,
            metrics=metrics,
            metadata=metadata,
        )
