from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ...contracts import MetricDict, TaskDefinition
from ...trainers.utils import move_to_device
from .base import ContinualLearner


@dataclass
class EWCLearner(ContinualLearner):
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    The diagonal Fisher is estimated from *per-example* score gradients.  By
    default labels are sampled from the model predictive distribution, which
    is the Fisher estimator described in the paper.  ``empirical`` mode is
    provided for experiments that intentionally use ground-truth labels.
    """

    ewc_lambda: float = 400.0
    fisher_samples: int = 200
    fisher_mode: str = "sampled"
    parameter_means: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    fishers: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.ewc_lambda) < 0.0:
            raise ValueError("ewc_lambda must be non-negative.")
        if int(self.fisher_samples) <= 0:
            raise ValueError("fisher_samples must be positive.")
        if self.fisher_mode not in {"sampled", "empirical"}:
            raise ValueError("fisher_mode must be 'sampled' or 'empirical'.")

    def consolidation_loss(self) -> torch.Tensor:
        loss = torch.zeros((), device=self.device)
        named_parameters = dict(self.model.named_parameters())
        # Eq. (3): each previous posterior contributes its own quadratic
        # precision-weighted spring around that task's optimum.
        for task_id, fisher in self.fishers.items():
            means = self.parameter_means[task_id]
            for name, importance in fisher.items():
                parameter = named_parameters[name]
                mean = means[name].to(parameter.device, dtype=parameter.dtype)
                diagonal = importance.to(parameter.device, dtype=parameter.dtype)
                loss = loss + (diagonal * (parameter - mean).pow(2)).sum()
        return 0.5 * float(self.ewc_lambda) * loss

    def observe(self, inputs: torch.Tensor, targets: torch.Tensor, task_id: str) -> MetricDict:
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        task_loss = self.restricted_cross_entropy(logits, targets, task_id)
        ewc_loss = self.consolidation_loss()
        loss = task_loss + ewc_loss
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "task_loss": float(task_loss.detach().item()),
            "ewc_loss": float(ewc_loss.detach().item()),
        }

    def estimate_fisher(self, dataloader: DataLoader, task_id: str) -> Dict[str, torch.Tensor]:
        self.model.to(self.device)
        self.model.eval()
        named_parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        fisher = {
            name: torch.zeros_like(parameter, device=self.device)
            for name, parameter in named_parameters.items()
        }
        class_ids = self.classes_for_loss(task_id)
        class_index = torch.tensor(class_ids, device=self.device, dtype=torch.long)
        label_map = {class_id: offset for offset, class_id in enumerate(class_ids)}

        examples = 0
        for batch in dataloader:
            inputs = move_to_device(batch[0], self.device)
            targets = move_to_device(batch[1], self.device).long()
            for sample_idx in range(int(targets.shape[0])):
                if examples >= int(self.fisher_samples):
                    break
                self.model.zero_grad(set_to_none=True)
                logits = self.model(inputs[sample_idx : sample_idx + 1]).index_select(1, class_index)
                log_probabilities = torch.log_softmax(logits, dim=1)
                if self.fisher_mode == "sampled":
                    # F = E_{x,y~p_theta}[score(y) score(y)^T].
                    local_target = torch.multinomial(log_probabilities.exp().squeeze(0), 1)
                else:
                    target_value = int(targets[sample_idx].item())
                    if target_value not in label_map:
                        continue
                    local_target = torch.tensor([label_map[target_value]], device=self.device)
                score_loss = -log_probabilities[0, local_target[0]]
                gradients = torch.autograd.grad(
                    score_loss,
                    tuple(named_parameters.values()),
                    allow_unused=True,
                )
                for (name, parameter), gradient in zip(named_parameters.items(), gradients):
                    if gradient is not None:
                        fisher[name] += gradient.detach().pow(2)
                    else:
                        fisher[name] += torch.zeros_like(parameter)
                examples += 1
            if examples >= int(self.fisher_samples):
                break

        if examples == 0:
            raise ValueError(f"Cannot estimate Fisher information from empty task {task_id!r}.")
        return {
            name: (value / float(examples)).detach().cpu()
            for name, value in fisher.items()
        }

    def _on_task_end(self, task: TaskDefinition, train_loader: DataLoader) -> None:
        self.fishers[task.task_id] = self.estimate_fisher(train_loader, task.task_id)
        self.parameter_means[task.task_id] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
