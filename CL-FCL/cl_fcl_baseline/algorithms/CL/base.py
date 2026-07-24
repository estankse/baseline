from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ...contracts import MetricDict, TaskDefinition
from ...trainers.utils import move_to_device


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return parameters in the stable order used by all gradient methods."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def flatten_gradients(parameters: Sequence[nn.Parameter]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            parts.append(torch.zeros_like(parameter).reshape(-1))
        else:
            parts.append(parameter.grad.detach().reshape(-1))
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def assign_flat_gradient(parameters: Sequence[nn.Parameter], gradient: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        numel = parameter.numel()
        parameter.grad = gradient[offset : offset + numel].view_as(parameter).clone()
        offset += numel
    if offset != gradient.numel():
        raise ValueError("Flat gradient size does not match the model parameters.")


def last_linear_module(model: nn.Module) -> nn.Linear:
    linear_modules = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linear_modules:
        raise ValueError("The continual-learning model must contain a linear classifier.")
    return linear_modules[-1]


def extract_normalized_features(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Extract the input of the last linear layer and L2-normalize it.

    Existing repository models do not expose a common ``features`` method.  A
    short-lived pre-hook keeps iCaRL architecture agnostic without changing
    the forward contract used by the FL/FCL implementations.
    """
    classifier = last_linear_module(model)
    captured: List[torch.Tensor] = []

    def _capture(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured.append(args[0])

    handle = classifier.register_forward_pre_hook(_capture)
    try:
        model(inputs)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Unable to capture features before the classifier.")
    features = captured[-1]
    if features.ndim > 2:
        features = torch.flatten(features, start_dim=1)
    return F.normalize(features, p=2, dim=1)


@dataclass
class ContinualLearner:
    """Common single-model continual-learning interface.

    ``scenario`` controls evaluation and task loss masking:
    ``class`` predicts among all classes seen so far, ``task`` uses the task
    descriptor (as in GEM's incremental CIFAR experiment), and ``domain``
    keeps a shared output space (as in permuted MNIST).
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    device: str | torch.device = "cpu"
    scenario: str = "class"
    task_classes: Dict[str, List[int]] = field(default_factory=dict)
    task_order: List[str] = field(default_factory=list)
    seen_classes: List[int] = field(default_factory=list)
    current_task_id: str | None = None

    def __post_init__(self) -> None:
        if self.scenario not in {"class", "task", "domain"}:
            raise ValueError("scenario must be one of: class, task, domain.")
        self.device = torch.device(self.device)
        self.model.to(self.device)

    def begin_task(self, task: TaskDefinition) -> None:
        classes = [int(value) for value in task.metadata.get("classes", [])]
        if not classes:
            classes = list(range(int(task.num_classes)))
        self.current_task_id = task.task_id
        self.task_classes[task.task_id] = classes
        if task.task_id not in self.task_order:
            self.task_order.append(task.task_id)
        for class_id in classes:
            if class_id not in self.seen_classes:
                self.seen_classes.append(class_id)
        self._on_task_start(task)

    def _on_task_start(self, task: TaskDefinition) -> None:
        del task

    def end_task(self, task: TaskDefinition, train_loader: DataLoader) -> None:
        self._on_task_end(task, train_loader)

    def _on_task_end(self, task: TaskDefinition, train_loader: DataLoader) -> None:
        del task, train_loader

    def classes_for_loss(self, task_id: str) -> List[int]:
        if self.scenario == "task":
            return list(self.task_classes[task_id])
        if self.scenario == "class":
            return list(self.seen_classes)
        # Domain-incremental tasks share the complete output space.
        classifier = last_linear_module(self.model)
        return list(range(int(classifier.out_features)))

    def restricted_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        task_id: str,
    ) -> torch.Tensor:
        class_ids = self.classes_for_loss(task_id)
        if not class_ids:
            raise ValueError(f"Task {task_id!r} has no output classes.")
        index = torch.tensor(class_ids, device=logits.device, dtype=torch.long)
        selected_logits = logits.index_select(1, index)
        label_map = {class_id: offset for offset, class_id in enumerate(class_ids)}
        try:
            local_targets = torch.tensor(
                [label_map[int(value)] for value in targets.detach().cpu().tolist()],
                device=targets.device,
                dtype=torch.long,
            )
        except KeyError as exc:
            raise ValueError(
                f"Target class {exc.args[0]} is not active for task {task_id!r}."
            ) from exc
        return F.cross_entropy(selected_logits, local_targets)

    def observe(self, inputs: torch.Tensor, targets: torch.Tensor, task_id: str) -> MetricDict:
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = self.restricted_cross_entropy(logits, targets, task_id)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().item())}

    def train_epoch(self, dataloader: DataLoader, task_id: str) -> MetricDict:
        self.model.to(self.device)
        self.model.train()
        totals: Dict[str, float] = {}
        total_examples = 0
        for batch in dataloader:
            inputs = move_to_device(batch[0], self.device)
            targets = move_to_device(batch[1], self.device).long()
            metrics = self.observe(inputs, targets, task_id)
            batch_size = int(targets.shape[0])
            total_examples += batch_size
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        if total_examples == 0:
            return {"loss": 0.0}
        return {name: value / total_examples for name, value in totals.items()}

    def train_task(
        self,
        task: TaskDefinition,
        dataloader: DataLoader,
        epochs: int,
        epoch_callback: Callable[[MetricDict], None] | None = None,
    ) -> List[MetricDict]:
        history: List[MetricDict] = []
        for epoch in range(int(epochs)):
            metrics = self.train_epoch(dataloader, task.task_id)
            metrics["epoch"] = float(epoch)
            history.append(metrics)
            if epoch_callback is not None:
                epoch_callback(metrics)
        return history

    def predict(self, inputs: torch.Tensor, task_id: str | None = None) -> torch.Tensor:
        logits = self.model(inputs)
        if self.scenario == "task":
            if task_id is None:
                raise ValueError("task_id is required in the task-incremental scenario.")
            class_ids = self.task_classes[task_id]
        elif self.scenario == "class":
            class_ids = self.seen_classes
        else:
            return logits.argmax(dim=1)
        index = torch.tensor(class_ids, device=logits.device, dtype=torch.long)
        local_predictions = logits.index_select(1, index).argmax(dim=1)
        return index[local_predictions]

    def evaluate(self, dataloader: DataLoader, task_id: str | None = None) -> MetricDict:
        self.model.to(self.device)
        self.model.eval()
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = move_to_device(batch[0], self.device)
                targets = move_to_device(batch[1], self.device).long()
                predictions = self.predict(inputs, task_id=task_id)
                total_correct += int((predictions == targets).sum().item())
                total_examples += int(targets.shape[0])
        return {
            "accuracy": total_correct / max(1, total_examples),
            "num_samples": float(total_examples),
        }
