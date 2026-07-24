from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...contracts import MetricDict, TaskDefinition
from .base import ContinualLearner, last_linear_module


@dataclass
class LwFLearner(ContinualLearner):
    """Learning without Forgetting (Li and Hoiem, 2016/2018).

    A pre-update network records responses on the new task images.  Each old
    task is distilled through its own softened softmax, matching the paper's
    multi-head loss instead of applying one softmax across unrelated tasks.
    """

    temperature: float = 2.0
    distillation_weight: float = 1.0
    warmup_epochs: int = 1
    teacher: torch.nn.Module | None = field(default=None, init=False, repr=False)
    old_task_ids: List[str] = field(default_factory=list, init=False)
    _warmup: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scenario != "task":
            raise ValueError(
                "The original multi-head LwF requires scenario='task'. "
                "Class-incremental evaluation requires the distinct LwF.MC variant."
            )
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive.")
        if float(self.distillation_weight) < 0.0:
            raise ValueError("distillation_weight must be non-negative.")
        if int(self.warmup_epochs) < 0:
            raise ValueError("warmup_epochs must be non-negative.")

    def _on_task_start(self, task: TaskDefinition) -> None:
        self.old_task_ids = [task_id for task_id in self.task_order if task_id != task.task_id]
        old_classes = {
            class_id
            for task_id in self.old_task_ids
            for class_id in self.task_classes[task_id]
        }
        overlap = old_classes.intersection(self.task_classes[task.task_id])
        if overlap:
            raise ValueError(f"LwF expects disjoint task heads; repeated classes: {sorted(overlap)}")
        if self.old_task_ids:
            self.teacher = copy.deepcopy(self.model).to(self.device)
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        else:
            self.teacher = None
        self._initialize_new_head(task)

    def _initialize_new_head(self, task: TaskDefinition) -> None:
        """Add the new task nodes using the paper's Xavier initialization."""
        classifier = last_linear_module(self.model)
        class_ids = self.task_classes[task.task_id]
        index = torch.tensor(class_ids, device=classifier.weight.device, dtype=torch.long)
        new_weight = torch.empty(
            (len(class_ids), int(classifier.in_features)),
            device=classifier.weight.device,
            dtype=classifier.weight.dtype,
        )
        torch.nn.init.xavier_uniform_(new_weight)
        with torch.no_grad():
            classifier.weight.index_copy_(0, index, new_weight)
            if classifier.bias is not None:
                classifier.bias.index_fill_(0, index, 0.0)

        # Pre-allocation means optimizers may already have momentum/state for
        # these rows.  Newly added output nodes must start with fresh state.
        for parameter in (classifier.weight, classifier.bias):
            if parameter is None:
                continue
            for value in self.optimizer.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                    value.index_fill_(0, index, 0.0)

    def _task_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        task_id: str,
    ) -> torch.Tensor:
        classes = self.task_classes[task_id]
        index = torch.tensor(classes, device=logits.device, dtype=torch.long)
        label_map = {class_id: offset for offset, class_id in enumerate(classes)}
        local_targets = torch.tensor(
            [label_map[int(value)] for value in targets.detach().cpu().tolist()],
            device=targets.device,
            dtype=torch.long,
        )
        return F.cross_entropy(logits.index_select(1, index), local_targets)

    def _distillation_loss(self, inputs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.teacher is None:
            return torch.zeros((), device=logits.device)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        temperature = float(self.temperature)
        loss = torch.zeros((), device=logits.device)
        # Equations (2)-(4): sum one modified cross-entropy per old task.
        # LwF does not multiply this loss by T^2.
        for old_task_id in self.old_task_ids:
            index = torch.tensor(
                self.task_classes[old_task_id], device=logits.device, dtype=torch.long
            )
            targets = F.softmax(teacher_logits.index_select(1, index) / temperature, dim=1)
            log_predictions = F.log_softmax(logits.index_select(1, index) / temperature, dim=1)
            loss = loss - (targets * log_predictions).sum(dim=1).mean()
        return loss

    def _warmup_step(self) -> tuple[torch.nn.Linear, torch.Tensor, torch.Tensor | None]:
        classifier = last_linear_module(self.model)
        current_classes = set(self.task_classes[self.current_task_id or ""])
        inactive = [
            class_id
            for class_id in range(int(classifier.out_features))
            if class_id not in current_classes
        ]
        inactive_index = torch.tensor(inactive, device=classifier.weight.device, dtype=torch.long)
        weight_reference = classifier.weight.detach().index_select(0, inactive_index).clone()
        bias_reference = None
        if classifier.bias is not None:
            bias_reference = classifier.bias.detach().index_select(0, inactive_index).clone()
        for parameter in self.model.parameters():
            if parameter is not classifier.weight and parameter is not classifier.bias:
                parameter.grad = None
        if classifier.weight.grad is not None and inactive:
            classifier.weight.grad.index_fill_(0, inactive_index, 0.0)
        if classifier.bias is not None and classifier.bias.grad is not None and inactive:
            classifier.bias.grad.index_fill_(0, inactive_index, 0.0)
        return classifier, weight_reference, bias_reference

    def observe(self, inputs: torch.Tensor, targets: torch.Tensor, task_id: str) -> MetricDict:
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        new_loss = self._task_cross_entropy(logits, targets, task_id)
        old_loss = (
            torch.zeros((), device=logits.device)
            if self._warmup
            else self._distillation_loss(inputs, logits)
        )
        loss = new_loss + float(self.distillation_weight) * old_loss
        loss.backward()

        warmup_state = self._warmup_step() if self._warmup else None
        self.optimizer.step()
        if warmup_state is not None:
            classifier, weight_reference, bias_reference = warmup_state
            current_classes = set(self.task_classes[task_id])
            inactive = [
                class_id
                for class_id in range(int(classifier.out_features))
                if class_id not in current_classes
            ]
            inactive_index = torch.tensor(inactive, device=classifier.weight.device, dtype=torch.long)
            with torch.no_grad():
                classifier.weight.index_copy_(0, inactive_index, weight_reference)
                if classifier.bias is not None and bias_reference is not None:
                    classifier.bias.index_copy_(0, inactive_index, bias_reference)
            # Momentum/Adam buffers for inactive rows must also stay frozen.
            for parameter in (classifier.weight, classifier.bias):
                if parameter is None:
                    continue
                for value in self.optimizer.state.get(parameter, {}).values():
                    if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                        value.index_fill_(0, inactive_index, 0.0)

        return {
            "loss": float(loss.detach().item()),
            "new_loss": float(new_loss.detach().item()),
            "distillation_loss": float(old_loss.detach().item()),
        }

    def train_task(
        self,
        task: TaskDefinition,
        dataloader: DataLoader,
        epochs: int,
        epoch_callback: Callable[[MetricDict], None] | None = None,
    ) -> List[MetricDict]:
        history: List[MetricDict] = []
        if self.old_task_ids:
            self._warmup = True
            for epoch in range(int(self.warmup_epochs)):
                metrics = self.train_epoch(dataloader, task.task_id)
                metrics.update({"epoch": float(epoch), "warmup": 1.0})
                history.append(metrics)
                if epoch_callback is not None:
                    epoch_callback(metrics)
        self._warmup = False
        for epoch in range(int(epochs)):
            metrics = self.train_epoch(dataloader, task.task_id)
            metrics.update({"epoch": float(epoch), "warmup": 0.0})
            history.append(metrics)
            if epoch_callback is not None:
                epoch_callback(metrics)
        return history
