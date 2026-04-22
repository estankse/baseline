from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..contracts import MetricDict
from ..trainers.utils import move_to_device


@dataclass(frozen=True)
class PGDConfig:
    epsilon: float | Sequence[float] | torch.Tensor = 8.0 / 255.0
    step_size: float | Sequence[float] | torch.Tensor = 2.0 / 255.0
    steps: int = 10
    random_start: bool = True
    clip_min: float | Sequence[float] | torch.Tensor | None = None
    clip_max: float | Sequence[float] | torch.Tensor | None = None


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
    if clip_min is not None or clip_max is not None:
        adversarial = torch.max(adversarial, clip_min) if clip_min is not None else adversarial
        adversarial = torch.min(adversarial, clip_max) if clip_max is not None else adversarial
    return adversarial.detach()


def pgd_linf_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: PGDConfig,
) -> torch.Tensor:
    """Generate untargeted L-infinity PGD adversarial examples."""
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

    for _ in range(int(config.steps)):
        adversarial.requires_grad_(True)
        logits = model(adversarial)
        loss = F.cross_entropy(logits, targets)
        grad = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        adversarial = adversarial.detach() + step_size * grad.sign()
        adversarial = _project_linf(adversarial, clean, epsilon, clip_min, clip_max)

    return adversarial.detach()


def evaluate_pgd_robustness(
    model: nn.Module,
    dataloader: DataLoader,
    config: PGDConfig,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> MetricDict:
    """Evaluate robust loss/accuracy on PGD adversarial examples."""
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
            adversarial = pgd_linf_attack(model, inputs, targets, config)
            with torch.no_grad():
                logits = model(adversarial)
                loss = F.cross_entropy(logits, targets)
                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == targets).sum().item())
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
