from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ...contracts import MetricDict, TaskDefinition
from .base import ContinualLearner, extract_normalized_features


@dataclass(frozen=True)
class ReplayableSample:
    """Raw exemplar image with separate train and evaluation transforms."""

    raw_sample: Any
    train_transform: Callable[[Any], torch.Tensor] | None = None
    eval_transform: Callable[[Any], torch.Tensor] | None = None

    def _copy_raw(self) -> Any:
        if isinstance(self.raw_sample, torch.Tensor):
            return self.raw_sample.clone()
        copier = getattr(self.raw_sample, "copy", None)
        return copier() if callable(copier) else copy.deepcopy(self.raw_sample)

    def transformed(self, *, train: bool, horizontal_flip: bool = False) -> torch.Tensor:
        transform = self.train_transform if train else self.eval_transform
        raw_sample = self._copy_raw()
        if horizontal_flip:
            if isinstance(raw_sample, torch.Tensor):
                raw_sample = torch.flip(raw_sample, dims=(-1,))
            else:
                try:
                    from torchvision.transforms.functional import hflip
                except ModuleNotFoundError as exc:  # pragma: no cover
                    raise ModuleNotFoundError(
                        "torchvision is required for iCaRL prototype flip augmentation."
                    ) from exc
                raw_sample = hflip(raw_sample)
        sample = transform(raw_sample) if transform is not None else raw_sample
        if not isinstance(sample, torch.Tensor):
            raise TypeError("iCaRL exemplar transforms must return tensors.")
        return sample


Exemplar = Tuple[Any, int]


def _stored_exemplar(dataset: Dataset, index: int) -> Exemplar:
    """Copy one raw item through common Subset/class-split wrappers."""
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent_index = int(getattr(dataset, "indices")[int(index)])
        sample, target = _stored_exemplar(getattr(dataset, "dataset"), parent_index)
        if bool(getattr(dataset, "remap_labels", False)):
            target = int(getattr(dataset, "label_map")[int(target)])
        return sample, int(target)

    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        raw_data = getattr(dataset, "data")[int(index)]
        try:
            from torchvision.transforms.functional import to_pil_image
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "torchvision is required to store raw image exemplars."
            ) from exc
        raw_image = to_pil_image(raw_data).copy()
        target = int(getattr(dataset, "targets")[int(index)])
        train_transform = getattr(dataset, "transform", None)
        eval_transform = getattr(dataset, "_cl_eval_transform", train_transform)
        return ReplayableSample(raw_image, train_transform, eval_transform), target

    sample, target = dataset[int(index)][:2]
    if not isinstance(sample, torch.Tensor):
        raise TypeError("iCaRL expects tensor samples or a raw image dataset.")
    return sample.detach().cpu().clone(), int(target)


class ExemplarDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, exemplars: Sequence[Exemplar]) -> None:
        self.exemplars = list(exemplars)

    def __len__(self) -> int:
        return len(self.exemplars)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample, target = self.exemplars[index]
        if isinstance(sample, ReplayableSample):
            sample = sample.transformed(train=True)
        return sample, torch.tensor(target, dtype=torch.long)


@dataclass
class ICaRLLearner(ContinualLearner):
    """iCaRL: Incremental Classifier and Representation Learning.

    Implements Algorithms 1-5 of Rebuffi et al. (CVPR 2017): binary
    classification/distillation targets, exemplar rehearsal, prioritized
    herding, fixed total memory, and nearest-mean-of-exemplars prediction.
    """

    memory_budget: int = 2000
    lr_decay: float = 5.0
    lr_milestones: tuple[float, float] = (0.7, 0.9)
    prototype_flip: bool = False
    exemplar_sets: Dict[int, List[Exemplar]] = field(default_factory=dict)
    teacher: torch.nn.Module | None = field(default=None, init=False, repr=False)
    old_classes: List[int] = field(default_factory=list, init=False)
    new_classes: List[int] = field(default_factory=list, init=False)
    _prototype_classes: List[int] = field(default_factory=list, init=False, repr=False)
    _prototypes: torch.Tensor | None = field(default=None, init=False, repr=False)
    _initial_lrs: List[float] = field(default_factory=list, init=False, repr=False)
    _exemplars_current: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scenario != "class":
            raise ValueError("iCaRL is defined for the class-incremental scenario.")
        if int(self.memory_budget) <= 0:
            raise ValueError("memory_budget must be positive.")
        if float(self.lr_decay) <= 1.0:
            raise ValueError("lr_decay must be greater than one.")
        if any(not (0.0 < float(value) < 1.0) for value in self.lr_milestones):
            raise ValueError("lr_milestones must be fractions in (0, 1).")
        self._initial_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

    def _on_task_start(self, task: TaskDefinition) -> None:
        self.new_classes = list(self.task_classes[task.task_id])
        if not self.new_classes:
            raise ValueError("iCaRL requires at least one new class per update.")
        self.old_classes = [
            class_id
            for previous_task_id in self.task_order
            if previous_task_id != task.task_id
            for class_id in self.task_classes[previous_task_id]
        ]
        overlap = set(self.new_classes).intersection(self.old_classes)
        if overlap:
            raise ValueError(f"iCaRL requires new classes at each update: {sorted(overlap)}")
        if self.old_classes:
            self.teacher = copy.deepcopy(self.model).to(self.device)
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        else:
            self.teacher = None
        self._prototypes = None
        self._prototype_classes = []
        self._exemplars_current = False

    def observe(self, inputs: torch.Tensor, targets: torch.Tensor, task_id: str) -> MetricDict:
        del task_id
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        binary_targets = torch.zeros_like(logits)

        if self.old_classes:
            if self.teacher is None:
                raise RuntimeError("iCaRL teacher is missing for old-class distillation.")
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
                old_index = torch.tensor(self.old_classes, device=logits.device, dtype=torch.long)
                binary_targets.index_copy_(
                    1,
                    old_index,
                    torch.sigmoid(teacher_logits.index_select(1, old_index)),
                )
        for class_id in self.new_classes:
            binary_targets[:, class_id] = (targets == int(class_id)).to(binary_targets.dtype)

        elementwise_loss = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none"
        )
        # The authors' CIFAR implementation uses a fixed 100-output sigmoid
        # head from the first task and averages BCE over every output.  Future
        # class targets therefore remain zero until their task arrives.  This
        # fixed scale is what makes the paper's unusually large SGD rate (2.0)
        # well-defined across all class increments.
        class_scale = 1.0 / float(logits.shape[1])
        per_class_loss = elementwise_loss.mean(dim=0)
        loss = per_class_loss.sum() * class_scale
        loss.backward()
        self.optimizer.step()
        old_loss = (
            per_class_loss.index_select(
                0, torch.tensor(self.old_classes, device=logits.device, dtype=torch.long)
            ).sum()
            * class_scale
            if self.old_classes
            else loss.new_zeros(())
        )
        classification_loss = loss - old_loss
        return {
            "loss": float(loss.detach().item()),
            "classification_loss": float(classification_loss.detach().item()),
            "distillation_loss": float(old_loss.detach().item()),
        }

    def _replay_loader(self, dataloader: DataLoader) -> DataLoader:
        old_exemplars = [
            exemplar
            for class_id in self.old_classes
            for exemplar in self.exemplar_sets.get(class_id, [])
        ]
        if not old_exemplars:
            return dataloader
        dataset = ConcatDataset([dataloader.dataset, ExemplarDataset(old_exemplars)])
        return DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    def train_task(
        self,
        task: TaskDefinition,
        dataloader: DataLoader,
        epochs: int,
        epoch_callback: Callable[[MetricDict], None] | None = None,
    ) -> List[MetricDict]:
        replay_loader = self._replay_loader(dataloader)
        history: List[MetricDict] = []
        milestones = [int(math.floor(float(value) * int(epochs))) for value in self.lr_milestones]
        for epoch in range(int(epochs)):
            decay_count = sum(epoch >= milestone for milestone in milestones)
            for group, initial_lr in zip(self.optimizer.param_groups, self._initial_lrs):
                group["lr"] = initial_lr / (float(self.lr_decay) ** decay_count)
            metrics = self.train_epoch(replay_loader, task.task_id)
            self._exemplars_current = False
            metrics.update({"epoch": float(epoch), "lr": float(self.optimizer.param_groups[0]["lr"])})
            history.append(metrics)
            if epoch_callback is not None:
                epoch_callback(metrics)
        return history

    def _class_samples(
        self,
        dataset: Dataset,
        class_id: int,
    ) -> List[Exemplar]:
        samples: List[Exemplar] = []
        for index in range(len(dataset)):
            sample, target = _stored_exemplar(dataset, index)
            target_value = int(target)
            if target_value == int(class_id):
                samples.append((sample, target_value))
        return samples

    def _features_for_samples(
        self,
        samples: Sequence[Exemplar],
        batch_size: int = 256,
        horizontal_flip: bool = False,
    ) -> torch.Tensor:
        features: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(samples), int(batch_size)):
                batch_samples: List[torch.Tensor] = []
                for sample, _target in samples[start : start + int(batch_size)]:
                    if isinstance(sample, ReplayableSample):
                        sample = sample.transformed(
                            train=False, horizontal_flip=horizontal_flip
                        )
                    elif horizontal_flip and isinstance(sample, torch.Tensor):
                        sample = torch.flip(sample, dims=(-1,))
                    if not isinstance(sample, torch.Tensor):
                        raise TypeError("iCaRL exemplar evaluation requires tensor samples.")
                    batch_samples.append(sample)
                inputs = torch.stack(batch_samples).to(self.device)
                features.append(extract_normalized_features(self.model, inputs).cpu())
        if not features:
            return torch.empty((0, 0))
        return torch.cat(features, dim=0)

    def construct_exemplar_set(
        self,
        samples: Sequence[Exemplar],
        count: int,
    ) -> List[Exemplar]:
        """Algorithm 4: prioritized herding toward the normalized class mean."""
        if count <= 0 or not samples:
            return []
        features = self._features_for_samples(samples)
        # Algorithm 4 averages the already normalized feature vectors.  It
        # does not normalize each candidate mean again during herding.
        class_mean = features.mean(dim=0)
        selected: List[int] = []
        selected_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        running_sum = torch.zeros_like(class_mean)
        for exemplar_number in range(1, min(int(count), len(samples)) + 1):
            candidate_means = (
                features + running_sum.unsqueeze(0)
            ) / float(exemplar_number)
            distances = torch.linalg.vector_norm(candidate_means - class_mean.unsqueeze(0), dim=1)
            distances[selected_mask] = float("inf")
            chosen = int(torch.argmin(distances).item())
            selected.append(chosen)
            selected_mask[chosen] = True
            running_sum += features[chosen]
        return [samples[index] for index in selected]

    def update_exemplar_sets(self, train_loader: DataLoader) -> None:
        """Refresh current-task herding sets for intermediate/final NME evaluation."""
        exemplars_per_class = int(self.memory_budget) // max(1, len(self.seen_classes))
        if exemplars_per_class < 1:
            raise ValueError(
                "iCaRL memory_budget must allow at least one exemplar per observed class."
            )
        # Algorithm 5 relies on the priority order produced by herding.
        for class_id in self.old_classes:
            self.exemplar_sets[class_id] = self.exemplar_sets.get(class_id, [])[:exemplars_per_class]
        for class_id in self.new_classes:
            class_samples = self._class_samples(train_loader.dataset, class_id)
            self.exemplar_sets[class_id] = self.construct_exemplar_set(
                class_samples, exemplars_per_class
            )
        self._prototypes = None
        self._prototype_classes = []
        self._exemplars_current = True

    def _on_task_end(self, task: TaskDefinition, train_loader: DataLoader) -> None:
        del task
        if not self._exemplars_current:
            self.update_exemplar_sets(train_loader)

    def _build_prototypes(self) -> None:
        classes: List[int] = []
        prototypes: List[torch.Tensor] = []
        for class_id in self.seen_classes:
            exemplars = self.exemplar_sets.get(class_id, [])
            if not exemplars:
                continue
            features = self._features_for_samples(exemplars)
            prototype_mean = features.mean(dim=0)
            if self.prototype_flip:
                flipped_features = self._features_for_samples(
                    exemplars, horizontal_flip=True
                )
                prototype_mean = (prototype_mean + flipped_features.mean(dim=0)) / 2.0
            prototype = F.normalize(
                prototype_mean.unsqueeze(0), p=2, dim=1
            ).squeeze(0)
            classes.append(class_id)
            prototypes.append(prototype)
        if not prototypes:
            raise RuntimeError("iCaRL cannot classify before exemplar sets are constructed.")
        self._prototype_classes = classes
        self._prototypes = torch.stack(prototypes).to(self.device)

    def predict(self, inputs: torch.Tensor, task_id: str | None = None) -> torch.Tensor:
        del task_id
        if self._prototypes is None:
            self._build_prototypes()
        features = extract_normalized_features(self.model, inputs)
        # With normalized features/prototypes, nearest Euclidean mean and
        # maximum dot product are equivalent (paper Eq. 2).
        prototype_indices = torch.matmul(features, self._prototypes.T).argmax(dim=1)
        class_ids = torch.tensor(
            self._prototype_classes, device=inputs.device, dtype=torch.long
        )
        return class_ids[prototype_indices]
