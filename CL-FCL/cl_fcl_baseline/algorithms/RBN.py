from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.nn.modules.batchnorm import _BatchNorm

from ..contracts import AggregationResult, ClientContext, MetricDict, StateDict, TrainResult
from ..trainers.utils import detach_state_dict, move_to_device
from .PGD import PGDConfig, pgd_linf_attack
from .fedweit import (
    FedWeITAggregator,
    FedWeITClient,
    FedWeITServer,
    _clone_tensor_state,
    _state_l2_norm,
    _state_nnz,
    _state_numel,
)


class DualNormLayer(nn.Module):
    """FedRBN-style dual BatchNorm layer with shared affine parameters."""

    def __init__(
        self,
        num_features: int,
        *,
        bn_class: type[nn.Module],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        share_affine: bool = True,
    ) -> None:
        super().__init__()
        self.affine = bool(affine)
        self.share_affine = bool(share_affine)
        self.clean_bn = bn_class(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=self.affine and not self.share_affine,
            track_running_stats=track_running_stats,
        )
        self.noise_bn = bn_class(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=self.affine and not self.share_affine,
            track_running_stats=track_running_stats,
        )
        if self.affine and self.share_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.clean_input: bool | torch.Tensor = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(self.clean_input, torch.Tensor):
            clean_mask = torch.nonzero(self.clean_input, as_tuple=False).squeeze(1)
            noise_mask = torch.nonzero(~self.clean_input, as_tuple=False).squeeze(1)
            outputs = torch.zeros_like(inputs)
            if clean_mask.numel() > 0:
                outputs[clean_mask] = self.clean_bn(inputs.index_select(0, clean_mask))
            if noise_mask.numel() > 0:
                outputs[noise_mask] = self.noise_bn(inputs.index_select(0, noise_mask))
        elif bool(self.clean_input):
            outputs = self.clean_bn(inputs)
        else:
            outputs = self.noise_bn(inputs)
        if self.affine and self.share_affine and self.weight is not None and self.bias is not None:
            shape = [1] * outputs.ndim
            shape[1] = -1
            outputs = outputs * self.weight.view(*shape) + self.bias.view(*shape)
        return outputs


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    if not module_name:
        raise ValueError("Cannot replace the root module in-place.")
    parent = root
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _dual_from_batchnorm(module: _BatchNorm) -> DualNormLayer:
    dual_cls = nn.BatchNorm2d if isinstance(module, nn.BatchNorm2d) else nn.BatchNorm1d
    dual = DualNormLayer(
        int(module.num_features),
        bn_class=dual_cls,
        eps=float(module.eps),
        momentum=0.1 if module.momentum is None else float(module.momentum),
        affine=bool(module.affine),
        track_running_stats=bool(module.track_running_stats),
        share_affine=True,
    )
    with torch.no_grad():
        if module.affine and dual.weight is not None and dual.bias is not None:
            dual.weight.copy_(module.weight.detach())
            dual.bias.copy_(module.bias.detach())
        for bn in (dual.clean_bn, dual.noise_bn):
            if module.track_running_stats:
                bn.running_mean.copy_(module.running_mean.detach())
                bn.running_var.copy_(module.running_var.detach())
                if getattr(module, "num_batches_tracked", None) is not None:
                    bn.num_batches_tracked.copy_(module.num_batches_tracked.detach())
    return dual


def enable_dual_batch_norm(model: nn.Module) -> int:
    """Convert in-model BatchNorm layers into FedRBN dual BN layers."""

    replacements: List[tuple[str, DualNormLayer]] = []
    for module_name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            replacements.append((module_name, _dual_from_batchnorm(module)))
    for module_name, new_module in replacements:
        _replace_module(model, module_name, new_module)
    return len(replacements)


def set_dual_bn_mode(model: nn.Module, is_noised: bool | torch.Tensor) -> None:
    def _apply(module: nn.Module) -> None:
        if isinstance(module, DualNormLayer):
            module.clean_input = ~is_noised if isinstance(is_noised, torch.Tensor) else (not bool(is_noised))

    model.apply(_apply)


def set_dual_bn_noise_training(model: nn.Module, mode: bool) -> None:
    def _apply(module: nn.Module) -> None:
        if isinstance(module, DualNormLayer):
            module.noise_bn.train(mode)

    model.apply(_apply)


def _collect_dual_bn_state_names(model: nn.Module) -> tuple[set[str], set[str]]:
    parameter_names: set[str] = set()
    buffer_names: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, DualNormLayer):
            continue
        prefix = f"{module_name}." if module_name else ""
        if module.weight is not None:
            parameter_names.add(prefix + "weight")
        if module.bias is not None:
            parameter_names.add(prefix + "bias")
        for branch_name, branch in (("clean_bn", module.clean_bn), ("noise_bn", module.noise_bn)):
            branch_prefix = prefix + branch_name + "."
            for buffer_name, _ in branch.named_buffers(recurse=False):
                buffer_names.add(branch_prefix + buffer_name)
    return parameter_names, buffer_names


class _FunctionalModel(nn.Module):
    def __init__(self, model: nn.Module, params_and_buffers: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.model = model
        self.params_and_buffers = params_and_buffers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional_call(self.model, self.params_and_buffers, (inputs,))


def _client_local_bn_state(
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, value in params.items():
        state[name] = value.detach().cpu().clone()
    for name, value in buffers.items():
        state[name] = value.detach().cpu().clone()
    return state


@dataclass
class FedWeITRBNClient(FedWeITClient):
    """FedWeIT client with FedRBN's DBN, propagation, and PNC mechanisms."""

    pgd_config: PGDConfig = field(default_factory=PGDConfig)
    is_at_client: bool = True
    adv_lambda: float = 0.5
    pnc_coef: float = -1.0
    pnc_warmup: int = 10
    src_weight_mode: str = "cos"
    attack_noised_bn: bool = True
    local_bn_param_names: set[str] = field(init=False, default_factory=set)
    local_bn_buffer_names: set[str] = field(init=False, default_factory=set)
    task_local_bn_params: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    task_local_bn_buffers: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.local_bn_param_names, self.local_bn_buffer_names = _collect_dual_bn_state_names(self.trainer.model)

    @property
    def parameter_names(self) -> List[str]:
        return [
            name
            for name, _ in self.trainer.model.named_parameters()
            if name not in self.local_bn_param_names
        ]

    def _pnc_value(self, round_idx: int | None) -> float:
        if float(self.pnc_coef) < 0.0:
            return 0.0
        if round_idx is not None and int(round_idx) <= int(self.pnc_warmup):
            return 0.0
        return float(self.pnc_coef)

    def _shared_buffer_state(
        self,
        global_state: StateDict,
        device: torch.device | str,
        task_id: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        task_buffers = self.task_buffer_states.get(task_id, {}) if task_id is not None else {}
        buffers: Dict[str, torch.Tensor] = {}
        for name in self.buffer_names:
            if name in self.local_bn_buffer_names:
                continue
            value = task_buffers.get(name, self.buffer_state.get(name, global_state.get(name)))
            if isinstance(value, torch.Tensor):
                buffers[name] = value.detach().to(device)
        return buffers

    def _local_bn_state(
        self,
        global_state: StateDict,
        device: torch.device | str,
        task_id: str,
    ) -> tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.Tensor]]:
        task_params = self.task_local_bn_params.get(task_id, {})
        task_buffers = self.task_local_bn_buffers.get(task_id, {})
        model_state = self.trainer.model.state_dict()

        params: Dict[str, torch.nn.Parameter] = {}
        for name in sorted(self.local_bn_param_names):
            value = task_params.get(name, global_state.get(name, model_state.get(name)))
            if isinstance(value, torch.Tensor):
                params[name] = torch.nn.Parameter(value.detach().to(device).clone())

        buffers: Dict[str, torch.Tensor] = {}
        for name in sorted(self.local_bn_buffer_names):
            value = task_buffers.get(name, global_state.get(name, model_state.get(name)))
            if isinstance(value, torch.Tensor):
                buffers[name] = value.detach().to(device).clone()
        return params, buffers

    def prepare_local_bn(self, global_state: StateDict, task_id: str) -> None:
        device = self.trainer.device
        params, buffers = self._local_bn_state(global_state, device, task_id)
        self.task_local_bn_params[task_id] = {
            name: value.detach().cpu().clone()
            for name, value in params.items()
        }
        self.task_local_bn_buffers[task_id] = {
            name: value.detach().cpu().clone()
            for name, value in buffers.items()
        }

    def duplicate_clean_to_noise_bn(self, task_id: str) -> None:
        buffers = self.task_local_bn_buffers.get(task_id)
        if not buffers:
            return
        for name in list(buffers.keys()):
            if ".noise_bn." not in name:
                continue
            clean_name = name.replace(".noise_bn.", ".clean_bn.")
            if clean_name in buffers:
                buffers[name] = buffers[clean_name].detach().clone()

    def local_bn_state(self, task_id: str) -> Dict[str, torch.Tensor]:
        return _client_local_bn_state(
            self.task_local_bn_params.get(task_id, {}),
            self.task_local_bn_buffers.get(task_id, {}),
        )

    def propagate_noise_bn(
        self,
        task_id: str,
        source_states: Sequence[Mapping[str, torch.Tensor]],
        src_weight_mode: str = "cos",
    ) -> None:
        if task_id not in self.task_local_bn_buffers:
            return
        if not source_states:
            self.duplicate_clean_to_noise_bn(task_id)
            return

        dst_state = self.local_bn_state(task_id)
        candidates: dict[str, List[torch.Tensor]] = defaultdict(list)
        layer_scores: dict[str, List[torch.Tensor]] = defaultdict(list)
        found_noise_bn = False

        for src_state in source_states:
            for clean_key, dst_clean in dst_state.items():
                if ".clean_bn." not in clean_key:
                    continue
                if "running_mean" not in clean_key and "running_var" not in clean_key:
                    continue
                noise_key = clean_key.replace(".clean_bn.", ".noise_bn.")
                if clean_key not in src_state or noise_key not in src_state:
                    continue
                found_noise_bn = True
                candidates[noise_key].append(src_state[noise_key].detach().cpu().clone())

                if "running_mean" in clean_key:
                    mean_key = clean_key
                    var_key = clean_key.replace("running_mean", "running_var")
                else:
                    mean_key = clean_key.replace("running_var", "running_mean")
                    var_key = clean_key
                if mean_key not in src_state or var_key not in src_state or mean_key not in dst_state or var_key not in dst_state:
                    continue
                if src_weight_mode.lower() == "cos":
                    score = (
                        F.cosine_similarity(
                            src_state[mean_key].detach().float().reshape(-1),
                            dst_state[mean_key].detach().float().reshape(-1),
                            dim=0,
                            eps=1e-8,
                        )
                        + F.cosine_similarity(
                            src_state[var_key].detach().float().reshape(-1),
                            dst_state[var_key].detach().float().reshape(-1),
                            dim=0,
                            eps=1e-8,
                        )
                    ) * 50.0
                    layer_scores[noise_key].append(score)

        if not found_noise_bn:
            self.duplicate_clean_to_noise_bn(task_id)
            return

        if src_weight_mode.lower() == "eq":
            source_weights = torch.full(
                (len(source_states),),
                1.0 / max(1, len(source_states)),
                dtype=torch.float32,
            )
        elif src_weight_mode.lower() == "cos":
            per_layer: List[torch.Tensor] = []
            for noise_key in sorted(layer_scores.keys()):
                per_layer.append(torch.stack(layer_scores[noise_key], dim=0))
            stacked = torch.stack(per_layer, dim=1).mean(dim=1)
            source_weights = torch.softmax(stacked, dim=0)
        else:
            raise ValueError(f"Unsupported src_weight_mode: {src_weight_mode}")

        buffers = self.task_local_bn_buffers[task_id]
        for noise_key, values in candidates.items():
            if noise_key not in buffers or len(values) == 0:
                continue
            mixed = torch.zeros_like(buffers[noise_key])
            for value, weight in zip(values, source_weights):
                mixed = mixed + value.to(mixed.device, mixed.dtype) * float(weight.item())
            buffers[noise_key] = mixed.detach().cpu().clone()

    def _attack_inputs(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        params_and_buffers: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        attack_model = _FunctionalModel(self.trainer.model, params_and_buffers)
        was_training = self.trainer.model.training
        if self.attack_noised_bn:
            set_dual_bn_mode(self.trainer.model, True)
        self.trainer.model.eval()
        try:
            adversarial_inputs = pgd_linf_attack(
                attack_model,
                inputs,
                targets,
                self.pgd_config,
            )
        finally:
            if was_training:
                self.trainer.model.train()
        return adversarial_inputs

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        task_id = context.task_id or next(iter(self.task_loaders))
        if task_id not in self.task_loaders:
            task_id = next(iter(self.task_loaders))

        device = self.trainer.device
        self.trainer.model.to(device)
        self.trainer.model.train()
        self._initialize_or_update_base(global_state)
        knowledge = self._prepare_knowledge(task_id, context.metadata)
        self._ensure_task_state(task_id, knowledge)
        device_knowledge = self._knowledge_to_device(knowledge, device)
        self.prepare_local_bn(global_state, task_id)

        (
            base,
            masks,
            adaptive,
            previous_adaptive,
            alpha,
            previous_masks,
            previous_effective_anchor,
        ) = self._task_tensor_params(task_id, device_knowledge, device)
        local_bn_params, local_bn_buffers = self._local_bn_state(global_state, device, task_id)
        trainable: List[torch.nn.Parameter] = [
            *base.values(),
            *masks.values(),
            *adaptive.values(),
            *local_bn_params.values(),
        ]
        for previous_state in previous_adaptive.values():
            trainable.extend(previous_state.values())
        if len(device_knowledge) > 0:
            trainable.append(alpha)
        optimizer = self._build_optimizer(trainable)
        shared_buffers = self._shared_buffer_state(global_state, device, task_id=task_id)
        loader = self.task_loaders[task_id]
        pnc_value = self._pnc_value(context.round_idx)

        total_examples = 0
        total_loss = 0.0
        total_clean_ce = 0.0
        total_adv_ce = 0.0
        total_pnc_ce = 0.0
        total_sparse = 0.0
        total_retro = 0.0
        total_correct = 0

        for _ in range(int(self.epochs)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, device)
                targets = move_to_device(targets, device)
                transfer = self._knowledge_transfer_state(device_knowledge, alpha, device)
                composed = self._compose_parameters(base, masks, adaptive, transfer)
                params_and_buffers = {
                    **shared_buffers,
                    **local_bn_buffers,
                    **local_bn_params,
                    **composed,
                }

                set_dual_bn_mode(self.trainer.model, False)
                clean_logits = functional_call(self.trainer.model, params_and_buffers, (inputs,))
                clean_ce_loss = F.cross_entropy(clean_logits, targets)

                adv_ce_loss = torch.tensor(0.0, device=device)
                pnc_ce_loss = torch.tensor(0.0, device=device)
                if self.is_at_client:
                    adversarial_inputs = self._attack_inputs(inputs, targets, params_and_buffers)
                    set_dual_bn_mode(self.trainer.model, True)
                    adv_logits = functional_call(self.trainer.model, params_and_buffers, (adversarial_inputs,))
                    adv_ce_loss = F.cross_entropy(adv_logits, targets)
                    task_loss = (1.0 - float(self.adv_lambda)) * clean_ce_loss + float(self.adv_lambda) * adv_ce_loss
                    prediction_logits = adv_logits
                else:
                    if pnc_value > 0.0:
                        set_dual_bn_mode(self.trainer.model, True)
                        set_dual_bn_noise_training(self.trainer.model, False)
                        try:
                            noise_logits = functional_call(self.trainer.model, params_and_buffers, (inputs,))
                        finally:
                            set_dual_bn_noise_training(self.trainer.model, True)
                            set_dual_bn_mode(self.trainer.model, False)
                        pnc_ce_loss = F.cross_entropy(noise_logits, targets)
                        task_loss = (1.0 - pnc_value) * clean_ce_loss + pnc_value * pnc_ce_loss
                    else:
                        task_loss = clean_ce_loss
                    prediction_logits = clean_logits

                adaptive_sparse_loss, mask_sparse_loss, retro_loss = self._regularization_loss(
                    base=base,
                    masks=masks,
                    adaptive=adaptive,
                    previous_adaptive=previous_adaptive,
                    previous_masks=previous_masks,
                    previous_effective_anchor=previous_effective_anchor,
                )
                sparse_loss = float(self.lambda1) * adaptive_sparse_loss + float(self.lambda_mask) * mask_sparse_loss
                loss = task_loss + sparse_loss + float(self.lambda2) * retro_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_clean_ce += float(clean_ce_loss.detach().item()) * batch_size
                total_adv_ce += float(adv_ce_loss.detach().item()) * batch_size
                total_pnc_ce += float(pnc_ce_loss.detach().item()) * batch_size
                total_sparse += float(sparse_loss.detach().item()) * batch_size
                total_retro += float(retro_loss.detach().item()) * batch_size
                total_correct += int((prediction_logits.argmax(dim=1) == targets).sum().item())

        self.base_params = {name: value.detach().cpu().clone() for name, value in base.items()}
        self.buffer_state = {name: value.detach().cpu().clone() for name, value in shared_buffers.items()}
        self.task_buffer_states[task_id] = {
            name: value.detach().cpu().clone()
            for name, value in shared_buffers.items()
        }
        self.task_local_bn_params[task_id] = {
            name: value.detach().cpu().clone()
            for name, value in local_bn_params.items()
        }
        self.task_local_bn_buffers[task_id] = {
            name: value.detach().cpu().clone()
            for name, value in local_bn_buffers.items()
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
                "clean_ce_loss": 0.0,
                "adv_ce_loss": 0.0,
                "pnc_ce_loss": 0.0,
                "sparse_loss": 0.0,
                "retro_loss": 0.0,
                "accuracy": 0.0,
                "kb_size": float(len(knowledge)),
                "rbn_is_at_client": 1.0 if self.is_at_client else 0.0,
                "rbn_pnc_coef": float(pnc_value),
            }
        else:
            metrics = {
                "loss": total_loss / total_examples,
                "clean_ce_loss": total_clean_ce / total_examples,
                "adv_ce_loss": total_adv_ce / total_examples,
                "pnc_ce_loss": total_pnc_ce / total_examples,
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
                "rbn_is_at_client": 1.0 if self.is_at_client else 0.0,
                "rbn_pnc_coef": float(pnc_value),
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
        local_bn_params, local_bn_buffers = self._local_bn_state(global_state, device, task_id)
        composed = self._compose_parameters(base, mask_state, adaptive, transfer)
        state = _clone_tensor_state(global_state)
        eval_buffers = self.task_buffer_states.get(task_id, self.buffer_state)
        for name, value in eval_buffers.items():
            state[name] = value.detach().cpu().clone()
        for name, value in local_bn_params.items():
            state[name] = value.detach().cpu().clone()
        for name, value in local_bn_buffers.items():
            state[name] = value.detach().cpu().clone()
        for name, value in composed.items():
            state[name] = value.detach().cpu().clone()
        return state


@dataclass
class FedWeITRBNAggregator(FedWeITAggregator):
    def aggregate(self, client_results: List[TrainResult]) -> AggregationResult:
        result = super().aggregate(client_results)
        metadata = dict(result.metadata)
        metadata["aggregator"] = "fedweit_rbn"
        metadata["aggregation"] = "active_mask_mean_non_bn_local_dbn_private"
        return AggregationResult(
            global_state=result.global_state,
            metrics=dict(result.metrics),
            metadata=metadata,
        )


@dataclass
class FedWeITRBNServer(FedWeITServer):
    aggregator: FedWeITAggregator = field(default_factory=FedWeITRBNAggregator)
    src_weight_mode: str = "cos"
    propagate_before_training: bool = True

    def _propagate_task_bn(self, task_id: str) -> None:
        global_state = self.get_global_state()
        at_clients = [
            client
            for client in self.clients
            if isinstance(client, FedWeITRBNClient) and client.is_at_client
        ]
        source_states = [
            client.local_bn_state(task_id)
            for client in at_clients
            if task_id in client.task_local_bn_buffers
        ]
        for client in self.clients:
            if not isinstance(client, FedWeITRBNClient):
                continue
            client.prepare_local_bn(global_state, task_id)
            if client.is_at_client:
                continue
            client.propagate_noise_bn(task_id, source_states, src_weight_mode=self.src_weight_mode)

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        if self.propagate_before_training:
            self._propagate_task_bn(task_id)
        return super().run_round(round_idx, task_id)
