from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from ...contracts import MetricDict, TaskDefinition
from ...trainers.utils import move_to_device
from ..CL.base import extract_normalized_features
from ..CL.iCaRL import Exemplar, ExemplarDataset, _stored_exemplar
from .base import RobustReplayLearner, _local_targets, soft_cross_entropy


@dataclass
class TABALearner(RobustReplayLearner):
    """Task-Aware Boundary Augmentation (Bai et al., ICASSP 2023)."""

    mix_batch_size: int = 0
    mix_lambda_min: float = 0.45
    mix_lambda_max: float = 0.55
    _previous_old_boundary: List[Exemplar] = field(default_factory=list, init=False)
    _previous_new_boundary: List[Exemplar] = field(default_factory=list, init=False)
    _current_old_boundary: List[Exemplar] = field(default_factory=list, init=False)
    _current_new_boundary: List[Exemplar] = field(default_factory=list, init=False)
    _prototype_classes: List[int] = field(default_factory=list, init=False, repr=False)
    _prototypes: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.mix_batch_size) < 0:
            raise ValueError("mix_batch_size must be non-negative.")
        if not 0.0 <= float(self.mix_lambda_min) <= float(self.mix_lambda_max) <= 1.0:
            raise ValueError("TABA lambda bounds must satisfy 0 <= min <= max <= 1.")

    def _on_task_start(self, task: TaskDefinition) -> None:
        super()._on_task_start(task)
        # The feature extractor will change while learning this task. Keep
        # epoch-level diagnostics classifier-based until the new class means
        # are constructed from the final herding memory in _on_task_end.
        self._prototype_classes = []
        self._prototypes = None

    def _rcl_loss(
        self,
        adversarial: torch.Tensor,
        targets: torch.Tensor,
        *,
        soft_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = list(self.seen_classes)
        logits = self._select_logits(self.model, adversarial, active)
        positions = {class_id: offset for offset, class_id in enumerate(active)}
        if soft_targets is None:
            classification = F.cross_entropy(
                logits, _local_targets(targets, active)
            )
        else:
            classification = soft_cross_entropy(logits, soft_targets)

        # Paper Eq. (1) defines L_ce over the unified set C = C_o union C_n.
        # In particular, old replay labels remain hard classification targets;
        # restricting supervision to the new output block turns the mutually
        # exclusive classifier into independent Bernoulli tasks and makes the
        # first-stage classification gradient much too small.
        distillation = classification.new_zeros(())
        if self.old_classes:
            if self.teacher is None:
                raise RuntimeError("TABA teacher is missing after the first task.")
            old_active_positions = torch.tensor(
                [positions[class_id] for class_id in self.old_classes],
                device=logits.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                teacher_logits = self._select_logits(
                    self.teacher, adversarial, self.old_classes
                )
                teacher_probabilities = F.softmax(teacher_logits, dim=1)
            current_old_log_probabilities = F.log_softmax(
                logits, dim=1
            ).index_select(1, old_active_positions)
            # Eq. (2): cross-entropy from the old model's soft labels to the
            # corresponding probabilities of the unified current classifier.
            distillation = -(
                teacher_probabilities * current_old_log_probabilities
            ).sum(dim=1).mean()
        return classification + distillation, classification, distillation

    def _mixed_boundary_batch(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self._previous_old_boundary or not self._previous_new_boundary:
            return None
        count = int(count)
        old_items = random.choices(self._previous_old_boundary, k=count)
        new_items = random.choices(self._previous_new_boundary, k=count)
        old_transformed = [self._transform_exemplar(item, train=True) for item in old_items]
        new_transformed = [self._transform_exemplar(item, train=True) for item in new_items]
        old_inputs = torch.stack([sample for sample, _target in old_transformed]).to(self.device)
        new_inputs = torch.stack([sample for sample, _target in new_transformed]).to(self.device)
        old_labels = torch.tensor(
            [target for _sample, target in old_transformed],
            device=self.device,
            dtype=torch.long,
        )
        new_labels = torch.tensor(
            [target for _sample, target in new_transformed],
            device=self.device,
            dtype=torch.long,
        )
        lambdas = torch.empty(count, device=self.device).uniform_(
            float(self.mix_lambda_min), float(self.mix_lambda_max)
        )
        view_shape = (count,) + (1,) * (old_inputs.ndim - 1)
        mixed_inputs = lambdas.view(view_shape) * old_inputs + (
            1.0 - lambdas.view(view_shape)
        ) * new_inputs

        active = list(self.seen_classes)
        old_one_hot = F.one_hot(
            _local_targets(old_labels, active), num_classes=len(active)
        ).to(mixed_inputs.dtype)
        new_one_hot = F.one_hot(
            _local_targets(new_labels, active), num_classes=len(active)
        ).to(mixed_inputs.dtype)
        mixed_targets = lambdas.unsqueeze(1) * old_one_hot + (
            1.0 - lambdas.unsqueeze(1)
        ) * new_one_hot
        # Hard labels are used only for the common method signature; PGD and
        # the loss both consume the interpolated labels.
        return mixed_inputs, old_labels, mixed_targets

    def _replay_loader(self, dataloader: DataLoader) -> DataLoader:
        """Build the paper's stage dataset ``X_new union memory_old``.

        Algorithm 1 samples the original minibatch from the union once per
        epoch.  Appending a replay batch to every stream batch would instead
        repeat old exemplars and change both the old/new ratio and the
        original minibatch size used by Eq. (6).
        """

        memory = self._memory_items()
        if not memory:
            return dataloader
        return DataLoader(
            ConcatDataset([dataloader.dataset, ExemplarDataset(memory)]),
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            drop_last=False,
            pin_memory=dataloader.pin_memory,
        )

    def _boundary_predictions(
        self, adversarial: torch.Tensor, active: List[int]
    ) -> torch.Tensor:
        """Classify with the same evaluation-mode model used by PGD.

        Besides making Algorithm 1's boundary test consistent with the
        attack, this avoids updating BatchNorm running statistics in an
        unoptimised, no-gradient forward pass.
        """

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                local_predictions = self._select_logits(
                    self.model, adversarial, active
                ).argmax(dim=1)
        finally:
            self.model.train(was_training)
        return torch.tensor(active, device=self.device)[local_predictions]

    def observe(self, inputs: torch.Tensor, targets: torch.Tensor, task_id: str) -> MetricDict:
        del task_id
        # ``train_task`` already draws this batch from X_new union memory_old.
        # Task membership must be label-based because that union is shuffled.
        clean, labels = inputs, targets
        if self.old_classes:
            old_class_ids = torch.tensor(
                self.old_classes, device=labels.device, dtype=labels.dtype
            )
            old_mask = labels.unsqueeze(1).eq(old_class_ids.unsqueeze(0)).any(dim=1)
        else:
            old_mask = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.bool)

        active = list(self.seen_classes)
        adversarial = self.generate_adversarial(clean, labels, class_ids=active)
        predictions = self._boundary_predictions(adversarial, active)
        boundary_mask = predictions.ne(labels)
        for index in boundary_mask.nonzero(as_tuple=False).flatten().tolist():
            exemplar = (clean[index].detach().cpu().clone(), int(labels[index].item()))
            if bool(old_mask[index].item()):
                self._current_old_boundary.append(exemplar)
            else:
                self._current_new_boundary.append(exemplar)

        rcl_loss, classification_loss, distillation_loss = self._rcl_loss(
            adversarial, labels
        )
        taba_loss = rcl_loss.new_zeros(())
        mix_classification = rcl_loss.new_zeros(())
        mix_distillation = rcl_loss.new_zeros(())
        mix_count = int(self.mix_batch_size) or int(labels.shape[0])
        mixed = self._mixed_boundary_batch(mix_count)
        if mixed is not None:
            mixed_inputs, mixed_labels, mixed_targets = mixed
            mixed_adversarial = self.generate_adversarial(
                mixed_inputs,
                mixed_labels,
                class_ids=active,
                soft_targets=mixed_targets,
            )
            taba_loss, mix_classification, mix_distillation = self._rcl_loss(
                mixed_adversarial, mixed_labels, soft_targets=mixed_targets
            )

        loss = rcl_loss + taba_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "rcl_loss": float(rcl_loss.detach().item()),
            "taba_loss": float(taba_loss.detach().item()),
            "classification_loss": float(classification_loss.detach().item()),
            "distillation_loss": float(distillation_loss.detach().item()),
            "mix_classification_loss": float(mix_classification.detach().item()),
            "mix_distillation_loss": float(mix_distillation.detach().item()),
            "boundary_fraction": float(boundary_mask.float().mean().item()),
        }

    def train_epoch(self, dataloader: DataLoader, task_id: str) -> MetricDict:
        self.model.to(self.device).train()
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
        # Algorithm 1 initializes B_0 to the full current training set.
        self._previous_old_boundary = list(self._memory_items())
        self._previous_new_boundary = [
            _stored_exemplar(dataloader.dataset, index)
            for index in range(len(dataloader.dataset))
        ]
        replay_loader = self._replay_loader(dataloader)
        history: List[MetricDict] = []
        for epoch in range(int(epochs)):
            self._current_old_boundary = []
            self._current_new_boundary = []
            metrics = self.train_epoch(replay_loader, task.task_id)
            metrics.update(
                {
                    "epoch": float(epoch),
                    "old_boundary_size": float(len(self._current_old_boundary)),
                    "new_boundary_size": float(len(self._current_new_boundary)),
                }
            )
            history.append(metrics)
            self._previous_old_boundary = self._current_old_boundary
            self._previous_new_boundary = self._current_new_boundary
            if epoch_callback is not None:
                epoch_callback(metrics)
        return history

    def _build_prototypes(self) -> None:
        classes: List[int] = []
        prototypes: List[torch.Tensor] = []
        for class_id in self.seen_classes:
            exemplars = self.exemplar_sets.get(class_id, [])
            if not exemplars:
                continue
            features = self._features_for_samples(exemplars)
            prototype = F.normalize(
                features.mean(dim=0, keepdim=True), p=2, dim=1
            ).squeeze(0)
            classes.append(class_id)
            prototypes.append(prototype)
        self._prototype_classes = classes
        self._prototypes = (
            torch.stack(prototypes).to(self.device) if prototypes else None
        )

    def refresh_nme(self, train_loader: DataLoader) -> None:
        """Refresh herding exemplars and class means for evaluation."""

        self.update_herding_memory(train_loader)
        self._build_prototypes()

    def _on_task_end(self, task: TaskDefinition, train_loader: DataLoader) -> None:
        del task
        self.refresh_nme(train_loader)

    def predict(self, inputs: torch.Tensor, task_id: str | None = None) -> torch.Tensor:
        active = self.active_classes(task_id)
        prototype_positions = {
            class_id: offset for offset, class_id in enumerate(self._prototype_classes)
        }
        # Before task-end herding, prototypes do not yet contain the current
        # classes. Fall back only for intermediate epoch diagnostics.
        if self._prototypes is None or any(
            class_id not in prototype_positions for class_id in active
        ):
            return super().predict(inputs, task_id=task_id)
        positions = torch.tensor(
            [prototype_positions[class_id] for class_id in active],
            device=self.device,
            dtype=torch.long,
        )
        prototypes = self._prototypes.index_select(0, positions)
        features = extract_normalized_features(self.model, inputs)
        predictions = torch.matmul(features, prototypes.T).argmax(dim=1)
        return torch.tensor(active, device=self.device, dtype=torch.long)[predictions]


TABA = TABALearner

__all__ = ["TABA", "TABALearner"]
