from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader

from ..contracts import ClientContext, MetricDict, StateDict, TrainResult
from ..trainers.utils import detach_state_dict, move_to_device
from .PGD import PGDConfig
from .fedweit import FedWeITClient, FedWeITServer, _state_l2_norm, _state_nnz, _state_numel


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


def _expand_log_prior(log_prior: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return log_prior.detach().to(device=logits.device, dtype=logits.dtype).view(1, -1)


def _adjust_logits(logits: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
    return logits + _expand_log_prior(log_prior, logits)


def _calibrated_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(_adjust_logits(logits, log_prior), targets)


def _calibrated_kl_loss(
    clean_logits: torch.Tensor,
    adversarial_logits: torch.Tensor,
    log_prior: torch.Tensor,
) -> torch.Tensor:
    calibrated_clean = _adjust_logits(clean_logits, log_prior)
    calibrated_adv = _adjust_logits(adversarial_logits, log_prior)
    clean_probs = torch.softmax(calibrated_clean.detach(), dim=1)
    return -(clean_probs * F.log_softmax(calibrated_adv, dim=1)).sum(dim=1).mean()


def calibrated_pgd_linf_attack(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    log_prior: torch.Tensor,
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
    if config.random_start:
        random_delta = torch.empty_like(adversarial).uniform_(-1.0, 1.0) * epsilon
        adversarial = _project_linf(clean + random_delta, clean, epsilon, clip_min, clip_max)

    with torch.no_grad():
        clean_logits = model(clean)

    for _ in range(int(config.steps)):
        adversarial.requires_grad_(True)
        adversarial_logits = model(adversarial)
        loss = _calibrated_kl_loss(clean_logits, adversarial_logits, log_prior)
        grad = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + step_size * grad.sign()
        adversarial = _project_linf(adversarial, clean, epsilon, clip_min, clip_max)

    return adversarial.detach()


def evaluate_calfat_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    log_prior: torch.Tensor,
    device: str | torch.device = "cpu",
) -> MetricDict:
    model.to(device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    try:
        for inputs, targets in dataloader:
            inputs = move_to_device(inputs, device)
            targets = move_to_device(targets, device)
            with torch.no_grad():
                logits = model(inputs)
                adjusted_logits = _adjust_logits(logits, log_prior)
                loss = F.cross_entropy(adjusted_logits, targets)
                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int((adjusted_logits.argmax(dim=1) == targets).sum().item())
    finally:
        if was_training:
            model.train()

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "num_samples": 0.0}
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_samples": float(total_examples),
    }


def evaluate_calfat_pgd_robustness(
    model: torch.nn.Module,
    dataloader: DataLoader,
    log_prior: torch.Tensor,
    config: PGDConfig,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> MetricDict:
    model.to(device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    evaluated_batches = 0

    try:
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if max_batches is not None and int(max_batches) > 0 and batch_idx >= int(max_batches):
                break
            inputs = move_to_device(inputs, device)
            targets = move_to_device(targets, device)
            adversarial = calibrated_pgd_linf_attack(model, inputs, log_prior, config)
            with torch.no_grad():
                logits = model(adversarial)
                adjusted_logits = _adjust_logits(logits, log_prior)
                loss = F.cross_entropy(adjusted_logits, targets)
                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int((adjusted_logits.argmax(dim=1) == targets).sum().item())
            evaluated_batches += 1
    finally:
        if was_training:
            model.train()

    if total_examples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "num_batches": float(evaluated_batches),
            "num_samples": 0.0,
        }
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_batches": float(evaluated_batches),
        "num_samples": float(total_examples),
    }


@dataclass
class FedWeITCalFATClient(FedWeITClient):
    pgd_config: PGDConfig = field(default_factory=PGDConfig)
    prior_smoothing: float = 1e-6
    task_class_priors: Dict[str, torch.Tensor] = field(default_factory=dict)

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

    def _task_class_prior(self, task_id: str, loader: DataLoader, device: torch.device | str) -> torch.Tensor:
        cached = self.task_class_priors.get(task_id)
        if cached is not None:
            return cached.detach().to(device)

        num_classes = self._infer_num_classes(loader, device)
        prior = torch.full((max(1, num_classes),), float(self.prior_smoothing), dtype=torch.float32)
        dataset = loader.dataset
        total_samples = 0
        for index in range(len(dataset)):
            _, target = dataset[index]
            target_value = int(target.item()) if isinstance(target, torch.Tensor) else int(target)
            if 0 <= target_value < prior.numel():
                prior[target_value] += 1.0
            total_samples += 1

        if total_samples > 0:
            prior = (prior - float(self.prior_smoothing)) / float(total_samples) + float(self.prior_smoothing)
        self.task_class_priors[task_id] = prior.detach().cpu().clone()
        return prior.to(device)

    def class_log_prior(self, task_id: str, device: torch.device | str = "cpu") -> torch.Tensor:
        prior = self.task_class_priors.get(task_id)
        if prior is None:
            raise KeyError(f"Class prior for task {task_id!r} has not been initialized.")
        return torch.log(prior.detach().to(device))

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
        loader = self.task_loaders[task_id]
        class_prior = self._task_class_prior(task_id, loader, device)
        log_prior = torch.log(class_prior)

        total_examples = 0
        total_loss = 0.0
        total_cce = 0.0
        total_adv_ce = 0.0
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

                was_training = self.trainer.model.training
                self.trainer.model.eval()
                try:
                    adversarial_inputs = calibrated_pgd_linf_attack(
                        lambda batch: functional_call(self.trainer.model, params_and_buffers, (batch,)),
                        inputs,
                        log_prior,
                        self.pgd_config,
                    )
                finally:
                    if was_training:
                        self.trainer.model.train()

                logits = functional_call(self.trainer.model, params_and_buffers, (adversarial_inputs,))
                adjusted_logits = _adjust_logits(logits, log_prior)
                cce_loss = F.cross_entropy(adjusted_logits, targets)
                adv_ce_loss = F.cross_entropy(logits, targets)
                adaptive_sparse_loss, mask_sparse_loss, retro_loss = self._regularization_loss(
                    base=base,
                    masks=masks,
                    adaptive=adaptive,
                    previous_adaptive=previous_adaptive,
                    previous_masks=previous_masks,
                    previous_effective_anchor=previous_effective_anchor,
                )
                sparse_loss = float(self.lambda1) * adaptive_sparse_loss + float(self.lambda_mask) * mask_sparse_loss
                loss = cce_loss + sparse_loss + float(self.lambda2) * retro_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_cce += float(cce_loss.detach().item()) * batch_size
                total_adv_ce += float(adv_ce_loss.detach().item()) * batch_size
                total_sparse += float(sparse_loss.detach().item()) * batch_size
                total_retro += float(retro_loss.detach().item()) * batch_size
                total_correct += int((adjusted_logits.argmax(dim=1) == targets).sum().item())

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
                "cce_loss": 0.0,
                "adv_ce_loss": 0.0,
                "sparse_loss": 0.0,
                "retro_loss": 0.0,
                "accuracy": 0.0,
                "kb_size": float(len(knowledge)),
                "class_prior_min": float(class_prior.min().item()),
                "class_prior_max": float(class_prior.max().item()),
            }
        else:
            metrics = {
                "loss": total_loss / total_examples,
                "cce_loss": total_cce / total_examples,
                "adv_ce_loss": total_adv_ce / total_examples,
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
                "class_prior_min": float(class_prior.min().item()),
                "class_prior_max": float(class_prior.max().item()),
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


class FedWeITCalFATServer(FedWeITServer):
    pass
