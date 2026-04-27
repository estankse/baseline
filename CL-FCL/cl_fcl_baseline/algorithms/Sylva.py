from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader

from ..contracts import AggregationResult, ClientContext, MetricDict, StateDict, TaskDefinition, TrainResult
from ..trainers.utils import detach_state_dict, move_to_device
from .PGD import PGDConfig, pgd_linf_attack
from .fedweit import FedWeITAggregator, FedWeITClient, FedWeITServer, _state_l2_norm, _state_nnz, _state_numel


class _FunctionalModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, params_and_buffers: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.model = model
        self.params_and_buffers = params_and_buffers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional_call(self.model, self.params_and_buffers, (inputs,))


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
        adversarial = torch.max(adversarial, clip_min)
    if clip_max is not None:
        adversarial = torch.min(adversarial, clip_max)
    return adversarial.detach()


def _trades_pgd_linf_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    clean_logits: torch.Tensor,
    config: PGDConfig,
) -> torch.Tensor:
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
    clean_probs = torch.softmax(clean_logits.detach(), dim=1)
    if config.random_start:
        random_delta = torch.empty_like(adversarial).uniform_(-1.0, 1.0) * epsilon
        adversarial = _project_linf(clean + random_delta, clean, epsilon, clip_min, clip_max)

    for _ in range(int(config.steps)):
        adversarial.requires_grad_(True)
        adversarial_logits = model(adversarial)
        loss = F.kl_div(F.log_softmax(adversarial_logits, dim=1), clean_probs, reduction="batchmean")
        grad = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + step_size * grad.sign()
        adversarial = _project_linf(adversarial, clean, epsilon, clip_min, clip_max)

    return adversarial.detach()


def _flatten_state(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    flattened: List[torch.Tensor] = []
    for name in sorted(state.keys()):
        value = state[name]
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            flattened.append(value.detach().float().reshape(-1))
    if not flattened:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(flattened, dim=0)


def _layer_group_name(parameter_name: str) -> str:
    if "." not in parameter_name:
        return parameter_name
    return parameter_name.rsplit(".", 1)[0]


@dataclass
class FedWeITSylvaClient(FedWeITClient):
    pgd_config: PGDConfig = field(default_factory=PGDConfig)
    class_balance_power: float = 0.6
    class_balance_smoothing: float = 1e-3
    dynamic_weight_rounds: int = 3
    clean_loss_weight: float = 0.8
    adv_loss_weight: float = 1.25
    kl_weight: float = 8.0
    global_reg: float = 1e-4
    phase2_epochs: int = 1
    phase2_topk_layers: int = 1
    phase2_tradeoff: float = 0.7
    phase2_lr_scale: float = 0.15
    phase2_max_batches: int = 1
    task_class_histograms: Dict[str, torch.Tensor] = field(default_factory=dict)
    phase2_selected_layers: Dict[str, List[str]] = field(default_factory=dict)

    def _infer_num_classes(self, loader: DataLoader, device: torch.device | str) -> int:
        try:
            sample_inputs, _ = next(iter(loader))
        except StopIteration:
            return 0
        sample_inputs = move_to_device(sample_inputs[:1], device)
        was_training = self.trainer.model.training
        self.trainer.model.eval()
        try:
            with torch.no_grad():
                logits = self.trainer.model(sample_inputs)
        finally:
            if was_training:
                self.trainer.model.train()
        return int(logits.shape[-1])

    def _task_class_counts(self, task_id: str, loader: DataLoader, device: torch.device | str) -> torch.Tensor:
        cached = self.task_class_histograms.get(task_id)
        if cached is not None:
            return cached.detach().to(device)

        num_classes = max(1, self._infer_num_classes(loader, device))
        counts = torch.full((num_classes,), float(self.class_balance_smoothing), dtype=torch.float32)
        dataset = loader.dataset
        for index in range(len(dataset)):
            _, target = dataset[index]
            target_value = int(target.item()) if isinstance(target, torch.Tensor) else int(target)
            if target_value >= counts.numel():
                padding = torch.full(
                    (target_value + 1 - counts.numel(),),
                    float(self.class_balance_smoothing),
                    dtype=torch.float32,
                )
                counts = torch.cat([counts, padding], dim=0)
            if target_value >= 0:
                counts[target_value] += 1.0

        self.task_class_histograms[task_id] = counts.detach().cpu().clone()
        return counts.to(device)

    def _task_class_weights(
        self,
        task_id: str,
        loader: DataLoader,
        device: torch.device | str,
        round_idx: int | None,
    ) -> torch.Tensor:
        counts = self._task_class_counts(task_id, loader, device)
        inverse_frequency = (counts.sum() / counts.clamp_min(float(self.class_balance_smoothing))).pow(
            float(self.class_balance_power)
        )
        inverse_frequency = inverse_frequency / inverse_frequency.mean().clamp_min(1e-12)
        if round_idx is None:
            blend = 1.0
        else:
            blend = min(1.0, float(round_idx + 1) / max(1, int(self.dynamic_weight_rounds)))
        return torch.ones_like(inverse_frequency).lerp(inverse_frequency, blend)

    def _global_alignment_loss(
        self,
        base: Mapping[str, torch.Tensor],
        global_anchor: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        losses = [
            torch.sum(torch.square(base[name] - global_anchor[name]))
            for name in base
            if name in global_anchor
        ]
        if losses:
            return torch.stack(losses).sum()
        device = next(iter(base.values())).device
        return torch.tensor(0.0, device=device)

    def _build_phase2_optimizer(self, params: Sequence[torch.nn.Parameter]) -> torch.optim.Optimizer:
        lr = float(self.lr) * max(float(self.phase2_lr_scale), 1e-6)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=float(self.weight_decay))
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(self.momentum),
            weight_decay=float(self.weight_decay),
        )

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
        global_anchor = {
            name: value.detach().to(device)
            for name, value in self._global_parameter_state(global_state).items()
            if name in base
        }
        loader = self.task_loaders[task_id]
        class_weights = self._task_class_weights(task_id, loader, device, context.round_idx)

        total_examples = 0
        total_loss = 0.0
        total_clean_ce = 0.0
        total_adv_ce = 0.0
        total_kl = 0.0
        total_alignment = 0.0
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

                clean_logits = functional_call(self.trainer.model, params_and_buffers, (inputs,))
                attack_model = _FunctionalModel(self.trainer.model, params_and_buffers)
                was_training = self.trainer.model.training
                self.trainer.model.eval()
                try:
                    adversarial_inputs = _trades_pgd_linf_attack(
                        attack_model,
                        inputs,
                        clean_logits,
                        self.pgd_config,
                    )
                finally:
                    if was_training:
                        self.trainer.model.train()
                adversarial_logits = functional_call(self.trainer.model, params_and_buffers, (adversarial_inputs,))

                clean_ce_loss = F.cross_entropy(clean_logits, targets, weight=class_weights)
                adv_ce_loss = F.cross_entropy(adversarial_logits, targets, weight=class_weights)
                robust_kl_loss = F.kl_div(
                    F.log_softmax(adversarial_logits, dim=1),
                    torch.softmax(clean_logits.detach(), dim=1),
                    reduction="batchmean",
                )
                alignment_loss = self._global_alignment_loss(base, global_anchor)
                adaptive_sparse_loss, mask_sparse_loss, retro_loss = self._regularization_loss(
                    base=base,
                    masks=masks,
                    adaptive=adaptive,
                    previous_adaptive=previous_adaptive,
                    previous_masks=previous_masks,
                    previous_effective_anchor=previous_effective_anchor,
                )
                sparse_loss = float(self.lambda1) * adaptive_sparse_loss + float(self.lambda_mask) * mask_sparse_loss
                primary_loss = (
                    float(self.clean_loss_weight) * clean_ce_loss
                    + float(self.adv_loss_weight) * adv_ce_loss
                )
                loss = (
                    primary_loss
                    + float(self.kl_weight) * robust_kl_loss
                    + float(self.global_reg) * alignment_loss
                    + sparse_loss
                    + float(self.lambda2) * retro_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_clean_ce += float(clean_ce_loss.detach().item()) * batch_size
                total_adv_ce += float(adv_ce_loss.detach().item()) * batch_size
                total_kl += float(robust_kl_loss.detach().item()) * batch_size
                total_alignment += float(alignment_loss.detach().item()) * batch_size
                total_sparse += float(sparse_loss.detach().item()) * batch_size
                total_retro += float(retro_loss.detach().item()) * batch_size
                total_correct += int((adversarial_logits.argmax(dim=1) == targets).sum().item())

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
        class_weight_min = float(class_weights.min().item()) if class_weights.numel() > 0 else 0.0
        class_weight_max = float(class_weights.max().item()) if class_weights.numel() > 0 else 0.0

        if total_examples == 0:
            metrics: MetricDict = {
                "loss": 0.0,
                "clean_ce_loss": 0.0,
                "adv_ce_loss": 0.0,
                "robust_kl_loss": 0.0,
                "alignment_loss": 0.0,
                "sparse_loss": 0.0,
                "retro_loss": 0.0,
                "accuracy": 0.0,
                "kb_size": float(len(knowledge)),
                "class_weight_min": class_weight_min,
                "class_weight_max": class_weight_max,
            }
        else:
            metrics = {
                "loss": total_loss / total_examples,
                "clean_ce_loss": total_clean_ce / total_examples,
                "adv_ce_loss": total_adv_ce / total_examples,
                "robust_kl_loss": total_kl / total_examples,
                "alignment_loss": total_alignment / total_examples,
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
                "class_weight_min": class_weight_min,
                "class_weight_max": class_weight_max,
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

    def refine_benign(self, global_state: StateDict, task_id: str) -> None:
        if int(self.phase2_epochs) <= 0 or int(self.phase2_topk_layers) <= 0:
            self.phase2_selected_layers[task_id] = []
            return
        if task_id not in self.task_loaders or task_id not in self.mask_logits or task_id not in self.adaptive_params:
            self.phase2_selected_layers[task_id] = []
            return

        device = self.trainer.device
        self.trainer.model.to(device)
        self.trainer.model.train()
        loader = self.task_loaders[task_id]
        class_weights = self._task_class_weights(task_id, loader, device, round_idx=None)
        knowledge = self.task_knowledge.get(task_id, [])
        device_knowledge = self._knowledge_to_device(knowledge, device)
        alpha_logits = self.alpha_logits.get(task_id, torch.zeros(len(device_knowledge), dtype=torch.float32))
        alpha = alpha_logits.detach().to(device)
        transfer = self._knowledge_transfer_state(device_knowledge, alpha, device)
        base_items = [
            (name, torch.nn.Parameter(value.detach().to(device).clone()))
            for name, value in self.base_params.items()
            if value.is_floating_point()
        ]
        adaptive_items = [
            (name, torch.nn.Parameter(value.detach().to(device).clone()))
            for name, value in self.adaptive_params[task_id].items()
        ]
        if not base_items and not adaptive_items:
            self.phase2_selected_layers[task_id] = []
            return

        base = dict(base_items)
        masks = {
            name: value.detach().to(device)
            for name, value in self.mask_logits[task_id].items()
        }
        adaptive = dict(adaptive_items)
        buffers = self._buffer_state(global_state, device, task_id=task_id)

        score_sums: Dict[str, float] = {}
        max_batches = None if int(self.phase2_max_batches) <= 0 else int(self.phase2_max_batches)
        ordered_names = [name for name, _ in base_items] + [name for name, _ in adaptive_items]
        ordered_params = [param for _, param in base_items] + [param for _, param in adaptive_items]

        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = move_to_device(inputs, device)
            targets = move_to_device(targets, device)
            composed = self._compose_parameters(base, masks, adaptive, transfer)
            params_and_buffers = {**buffers, **composed}
            clean_logits = functional_call(self.trainer.model, params_and_buffers, (inputs,))
            clean_loss = F.cross_entropy(clean_logits, targets, weight=class_weights)

            attack_model = _FunctionalModel(self.trainer.model, params_and_buffers)
            was_training = self.trainer.model.training
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
            adversarial_logits = functional_call(self.trainer.model, params_and_buffers, (adversarial_inputs,))
            adversarial_loss = F.cross_entropy(adversarial_logits, targets, weight=class_weights)

            clean_grads = torch.autograd.grad(clean_loss, ordered_params, retain_graph=True, allow_unused=True)
            adv_grads = torch.autograd.grad(adversarial_loss, ordered_params, allow_unused=True)
            for name, clean_grad, adv_grad in zip(ordered_names, clean_grads, adv_grads):
                group_name = _layer_group_name(name)
                clean_norm = 0.0 if clean_grad is None else float(clean_grad.detach().norm().item())
                adv_norm = 0.0 if adv_grad is None else float(adv_grad.detach().norm().item())
                score_sums[group_name] = score_sums.get(group_name, 0.0) + clean_norm - (
                    float(self.phase2_tradeoff) * adv_norm
                )

        if not score_sums:
            self.phase2_selected_layers[task_id] = []
            return

        selected_groups = [
            group_name
            for group_name, _ in sorted(score_sums.items(), key=lambda item: (-item[1], item[0]))[
                : max(1, int(self.phase2_topk_layers))
            ]
        ]
        selected = set(selected_groups)
        selected_params = [
            param
            for name, param in list(base.items()) + list(adaptive.items())
            if _layer_group_name(name) in selected
        ]
        if not selected_params:
            self.phase2_selected_layers[task_id] = []
            return

        optimizer = self._build_phase2_optimizer(selected_params)
        for _ in range(int(self.phase2_epochs)):
            for batch_idx, (inputs, targets) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                inputs = move_to_device(inputs, device)
                targets = move_to_device(targets, device)
                composed = self._compose_parameters(base, masks, adaptive, transfer)
                params_and_buffers = {**buffers, **composed}
                logits = functional_call(self.trainer.model, params_and_buffers, (inputs,))
                loss = F.cross_entropy(logits, targets, weight=class_weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.base_params.update({name: value.detach().cpu().clone() for name, value in base.items()})
        self.adaptive_params[task_id] = {
            name: value.detach().cpu().clone()
            for name, value in adaptive.items()
        }
        self.phase2_selected_layers[task_id] = selected_groups


@dataclass
class FedWeITSylvaAggregator(FedWeITAggregator):
    temperature: float = 0.7
    neighbors: int = 2

    def _client_weights(self, client_results: Sequence[TrainResult]) -> tuple[Dict[str, float], Dict[str, float]]:
        if not client_results:
            return {}, {}

        sample_counts = {
            result.client_id: float(max(int(result.num_samples), 0))
            for result in client_results
        }
        states = {
            result.client_id: result.payload.get("shared_state", {})
            for result in client_results
            if isinstance(result.payload.get("shared_state", {}), Mapping)
        }
        vectors = {
            client_id: _flatten_state(state)
            for client_id, state in states.items()
            if isinstance(state, Mapping)
        }
        if len(vectors) <= 1:
            weights = {
                result.client_id: sample_counts.get(result.client_id, 1.0) or 1.0
                for result in client_results
            }
            similarities = {client_id: 1.0 for client_id in weights}
            return weights, similarities

        client_ids = list(vectors.keys())
        pairwise: Dict[str, List[float]] = {client_id: [] for client_id in client_ids}
        all_distances: List[float] = []
        for idx, client_id in enumerate(client_ids):
            for other_id in client_ids[idx + 1 :]:
                distance = float(torch.norm(vectors[client_id] - vectors[other_id], p=2).item())
                pairwise[client_id].append(distance)
                pairwise[other_id].append(distance)
                if distance > 0.0:
                    all_distances.append(distance)
        scale = float(torch.tensor(all_distances, dtype=torch.float32).median().item()) if all_distances else 1.0
        scale = max(scale * max(float(self.temperature), 1e-6), 1e-12)

        similarities: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        for client_id in client_ids:
            distances = sorted(pairwise.get(client_id, []))
            if not distances:
                similarity = 1.0
            else:
                if int(self.neighbors) > 0:
                    distances = distances[: min(len(distances), int(self.neighbors))]
                similarity = float(
                    torch.exp(-torch.tensor(distances, dtype=torch.float32) / scale).mean().item()
                )
            similarities[client_id] = similarity
            weights[client_id] = max(sample_counts.get(client_id, 1.0), 1.0) * similarity
        return weights, similarities

    def aggregate(self, client_results: List[TrainResult]) -> AggregationResult:
        if not client_results:
            return AggregationResult(global_state={}, metrics={"num_clients": 0.0, "total_samples": 0.0})

        client_weights, similarities = self._client_weights(client_results)
        shared_states: List[tuple[StateDict, float]] = []
        hard_mask_states: List[tuple[StateDict, float]] = []
        buffer_states: List[tuple[StateDict, float]] = []
        metric_sums: Dict[str, float] = {}
        metric_weights: Dict[str, float] = {}
        for result in client_results:
            weight = float(client_weights.get(result.client_id, 1.0))
            shared_state = result.payload.get("shared_state", {})
            hard_mask_state = result.payload.get("hard_mask_state", {})
            if isinstance(shared_state, Mapping) and shared_state:
                shared_states.append((shared_state, weight))
                hard_mask_states.append((hard_mask_state if isinstance(hard_mask_state, Mapping) else {}, weight))
            buffer_state = result.payload.get("buffer_state", {})
            if isinstance(buffer_state, Mapping) and buffer_state:
                buffer_states.append((buffer_state, weight))
            for metric_name, metric_value in result.metrics.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + float(metric_value)
                metric_weights[metric_name] = metric_weights.get(metric_name, 0.0) + 1.0

        if not shared_states:
            return AggregationResult(
                global_state={},
                metrics={"num_clients": float(len(client_results)), "total_samples": 0.0},
            )

        averaged_state: Dict[str, torch.Tensor] = {}
        first_shared_state = shared_states[0][0]
        for name in first_shared_state.keys():
            weighted_values = [
                state[name].detach().float() * weight
                for state, weight in shared_states
                if name in state
            ]
            weighted_masks = [
                mask_state[name].detach().float() * weight
                for mask_state, weight in hard_mask_states
                if isinstance(mask_state, Mapping) and name in mask_state
            ]
            if not weighted_values:
                continue
            numerator = torch.stack(weighted_values, dim=0).sum(dim=0)
            if len(weighted_masks) == len(weighted_values):
                denominator = torch.stack(weighted_masks, dim=0).sum(dim=0)
                averaged_state[name] = torch.where(
                    denominator > 0,
                    numerator / torch.clamp(denominator, min=1e-12),
                    torch.zeros_like(numerator),
                )
            else:
                total_weight = sum(weight for state, weight in shared_states if name in state)
                averaged_state[name] = numerator / max(total_weight, 1e-12)

        if buffer_states:
            first_buffer_state = buffer_states[0][0]
            for name, first_value in first_buffer_state.items():
                if not isinstance(first_value, torch.Tensor):
                    continue
                values = [
                    (state[name], weight)
                    for state, weight in buffer_states
                    if name in state and isinstance(state[name], torch.Tensor)
                ]
                if not values:
                    continue
                if first_value.is_floating_point():
                    total_weight = sum(weight for _, weight in values)
                    averaged_state[name] = (
                        torch.stack([value.detach().float() * weight for value, weight in values], dim=0).sum(dim=0)
                        / max(total_weight, 1e-12)
                    )
                else:
                    averaged_state[name] = values[0][0].detach().clone()

        averaged_metrics = {
            f"{self.metric_prefix}{metric_name}": metric_sums[metric_name] / metric_weights[metric_name]
            for metric_name in metric_sums
            if metric_weights.get(metric_name, 0.0) > 0.0
        }
        averaged_metrics["num_clients"] = float(len(shared_states))
        averaged_metrics["total_samples"] = float(sum(max(int(result.num_samples), 0) for result in client_results))
        averaged_metrics["sylva_similarity_min"] = float(min(similarities.values())) if similarities else 0.0
        averaged_metrics["sylva_similarity_max"] = float(max(similarities.values())) if similarities else 0.0

        return AggregationResult(
            global_state=averaged_state,
            metrics=averaged_metrics,
            metadata={
                "aggregator": "fedweit_sylva",
                "aggregation": "similarity_weighted_active_mask_mean_B_mask",
                "sylva_client_weights": dict(client_weights),
                "sylva_client_similarities": dict(similarities),
                "sylva_temperature": float(self.temperature),
                "sylva_neighbors": float(max(int(self.neighbors), 0)),
            },
        )


@dataclass
class FedWeITSylvaServer(FedWeITServer):
    temperature: float = 0.7
    neighbors: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        self.aggregator = FedWeITSylvaAggregator(
            temperature=self.temperature,
            neighbors=self.neighbors,
        )

    def on_task_end(self, task: TaskDefinition) -> None:
        super().on_task_end(task)
        global_state = self.get_global_state()
        for client in self.clients:
            refine_benign = getattr(client, "refine_benign", None)
            if callable(refine_benign):
                refine_benign(global_state, task.task_id)
