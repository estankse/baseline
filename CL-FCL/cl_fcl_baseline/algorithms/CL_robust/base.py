from __future__ import annotations

import copy
from dataclasses import dataclass, field
import random
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ...contracts import MetricDict, TaskDefinition
from ...trainers.utils import move_to_device
from ..CL.base import ContinualLearner, extract_normalized_features
from ..CL.iCaRL import Exemplar, ReplayableSample, _stored_exemplar
from ..FL_robust.PGD import PGDConfig


def _channel_tensor(
    value: float | Sequence[float] | torch.Tensor | None,
    inputs: torch.Tensor,
) -> torch.Tensor | None:
    if value is None:
        return None
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
        adversarial = torch.maximum(adversarial, clip_min)
    if clip_max is not None:
        adversarial = torch.minimum(adversarial, clip_max)
    return adversarial.detach()


def _local_targets(targets: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    label_map = {int(class_id): offset for offset, class_id in enumerate(class_ids)}
    try:
        values = [label_map[int(value)] for value in targets.detach().cpu().tolist()]
    except KeyError as exc:
        raise ValueError(f"Target class {exc.args[0]} is not active in this attack.") from exc
    return torch.tensor(values, device=targets.device, dtype=torch.long)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("Soft targets must have the same shape as logits.")
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@dataclass
class RobustReplayLearner(ContinualLearner):
    """Shared adversarial replay machinery for robust continual learners.

    The robust methods in this package all use a fixed classifier, a frozen
    previous-task teacher when required, and a fixed-size replay memory.
    Attacks optimize the differentiable classifier logits; subclasses may
    override ``predict`` when their paper uses a different inference rule.
    """

    memory_budget: int = 2000
    replay_batch_size: int = 0
    attack_config: PGDConfig = field(default_factory=PGDConfig)
    eval_attack_config: PGDConfig | None = None
    exemplar_sets: Dict[int, List[Exemplar]] = field(default_factory=dict)
    teacher: nn.Module | None = field(default=None, init=False, repr=False)
    old_classes: List[int] = field(default_factory=list, init=False)
    new_classes: List[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scenario not in {"class", "task"}:
            raise ValueError("Robust class-incremental methods support class or task scenarios.")
        if int(self.memory_budget) < 0:
            raise ValueError("memory_budget must be non-negative.")
        if int(self.replay_batch_size) < 0:
            raise ValueError("replay_batch_size must be non-negative.")
        if float(torch.as_tensor(self.attack_config.epsilon).min().item()) < 0.0:
            raise ValueError("PGD epsilon must be non-negative.")
        if int(self.attack_config.steps) < 0:
            raise ValueError("PGD steps must be non-negative.")
        if self.eval_attack_config is None:
            self.eval_attack_config = self.attack_config

    def _on_task_start(self, task: TaskDefinition) -> None:
        self.new_classes = list(self.task_classes[task.task_id])
        self.old_classes = [
            class_id
            for previous_task_id in self.task_order
            if previous_task_id != task.task_id
            for class_id in self.task_classes[previous_task_id]
        ]
        overlap = set(self.old_classes).intersection(self.new_classes)
        if overlap:
            raise ValueError(
                "Robust class-incremental learning requires disjoint class tasks: "
                f"{sorted(overlap)}"
            )
        if self.old_classes:
            self.teacher = copy.deepcopy(self.model).to(self.device).eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)
        else:
            self.teacher = None

    def active_classes(self, task_id: str | None = None) -> List[int]:
        if self.scenario == "task" and task_id is not None:
            return list(self.task_classes[task_id])
        return list(self.seen_classes)

    @staticmethod
    def _select_logits(
        model: nn.Module,
        inputs: torch.Tensor,
        class_ids: Sequence[int],
    ) -> torch.Tensor:
        logits = model(inputs)
        index = torch.tensor(class_ids, device=logits.device, dtype=torch.long)
        return logits.index_select(1, index)

    def generate_adversarial(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        config: PGDConfig | None = None,
        model: nn.Module | None = None,
        class_ids: Sequence[int] | None = None,
        soft_targets: torch.Tensor | None = None,
        calibrate: Callable[[torch.Tensor], torch.Tensor] | None = None,
        return_difficulty: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Run untargeted L-inf PGD and optionally count successful steps.

        RAER's difficulty value is accumulated from the predictions already
        produced by PGD, so requesting it adds no backward passes.
        """

        attack = self.attack_config if config is None else config
        attacked_model = self.model if model is None else model
        selected_classes = list(self.seen_classes if class_ids is None else class_ids)
        if not selected_classes:
            raise ValueError("At least one active class is required for PGD.")
        local_targets = _local_targets(targets, selected_classes)
        if soft_targets is not None and soft_targets.shape[1] != len(selected_classes):
            raise ValueError("soft_targets must be expressed over class_ids.")

        clean = inputs.detach()
        epsilon = _channel_tensor(attack.epsilon, clean)
        step_size = _channel_tensor(attack.step_size, clean)
        clip_min = _channel_tensor(attack.clip_min, clean)
        clip_max = _channel_tensor(attack.clip_max, clean)
        if epsilon is None or step_size is None:
            raise ValueError("PGD epsilon and step_size must not be None.")

        difficulty = torch.zeros(clean.shape[0], device=clean.device, dtype=torch.long)
        adversarial = clean.clone()
        if attack.random_start:
            random_delta = torch.empty_like(adversarial).uniform_(-1.0, 1.0) * epsilon
            adversarial = _project_linf(
                clean + random_delta, clean, epsilon, clip_min, clip_max
            )

        was_training = attacked_model.training
        attacked_model.eval()
        try:
            for _ in range(int(attack.steps)):
                adversarial.requires_grad_(True)
                logits = self._select_logits(attacked_model, adversarial, selected_classes)
                if calibrate is not None:
                    logits = calibrate(logits)
                loss = (
                    F.cross_entropy(logits, local_targets)
                    if soft_targets is None
                    else soft_cross_entropy(logits, soft_targets)
                )
                gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
                adversarial = _project_linf(
                    adversarial.detach() + step_size * gradient.sign(),
                    clean,
                    epsilon,
                    clip_min,
                    clip_max,
                )
                if return_difficulty:
                    with torch.no_grad():
                        step_logits = self._select_logits(
                            attacked_model, adversarial, selected_classes
                        )
                        if calibrate is not None:
                            step_logits = calibrate(step_logits)
                        difficulty += step_logits.argmax(dim=1).ne(local_targets)
        finally:
            attacked_model.train(was_training)

        if return_difficulty:
            return adversarial.detach(), difficulty.detach()
        return adversarial.detach()

    def _memory_items(self) -> List[Exemplar]:
        return [
            exemplar
            for class_id in sorted(self.exemplar_sets)
            for exemplar in self.exemplar_sets[class_id]
        ]

    @staticmethod
    def _transform_exemplar(exemplar: Exemplar, *, train: bool) -> tuple[torch.Tensor, int]:
        sample, target = exemplar
        if isinstance(sample, ReplayableSample):
            sample = sample.transformed(train=train)
        if not isinstance(sample, torch.Tensor):
            raise TypeError("Replay exemplars must transform to tensors.")
        return sample, int(target)

    def sample_memory(
        self,
        count: int,
        *,
        exclude: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, List[int]] | None:
        memory = self._memory_items()
        excluded = set(int(index) for index in (exclude or []))
        available = [index for index in range(len(memory)) if index not in excluded]
        if not available or int(count) <= 0:
            return None
        count = min(int(count), len(available))
        chosen = random.sample(available, count)
        transformed = [self._transform_exemplar(memory[index], train=True) for index in chosen]
        inputs = torch.stack([sample for sample, _target in transformed]).to(self.device)
        targets = torch.tensor(
            [target for _sample, target in transformed], device=self.device, dtype=torch.long
        )
        return inputs, targets, chosen

    def replay_count(self, stream_batch_size: int) -> int:
        if int(self.replay_batch_size) > 0:
            return int(self.replay_batch_size)
        return int(stream_batch_size)

    def _class_samples(self, dataset: Dataset, class_id: int) -> List[Exemplar]:
        samples: List[Exemplar] = []
        for index in range(len(dataset)):
            exemplar = _stored_exemplar(dataset, index)
            if int(exemplar[1]) == int(class_id):
                samples.append(exemplar)
        return samples

    def _features_for_samples(
        self,
        samples: Sequence[Exemplar],
        batch_size: int = 256,
    ) -> torch.Tensor:
        features: List[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(samples), int(batch_size)):
                    batch = [
                        self._transform_exemplar(item, train=False)[0]
                        for item in samples[start : start + int(batch_size)]
                    ]
                    if batch:
                        features.append(
                            extract_normalized_features(
                                self.model, torch.stack(batch).to(self.device)
                            ).cpu()
                        )
        finally:
            self.model.train(was_training)
        return torch.cat(features, dim=0) if features else torch.empty((0, 0))

    def construct_exemplar_set(
        self,
        samples: Sequence[Exemplar],
        count: int,
    ) -> List[Exemplar]:
        """iCaRL herding (Algorithms 4-5), used by TABA/DAML/FLAIR."""

        if int(count) <= 0 or not samples:
            return []
        features = self._features_for_samples(samples)
        class_mean = features.mean(dim=0)
        running_sum = torch.zeros_like(class_mean)
        selected: List[int] = []
        selected_mask = torch.zeros(features.shape[0], dtype=torch.bool)
        for number in range(1, min(int(count), len(samples)) + 1):
            candidate_means = (features + running_sum.unsqueeze(0)) / float(number)
            distances = torch.linalg.vector_norm(
                candidate_means - class_mean.unsqueeze(0), dim=1
            )
            distances[selected_mask] = float("inf")
            chosen = int(distances.argmin().item())
            selected.append(chosen)
            selected_mask[chosen] = True
            running_sum += features[chosen]
        return [samples[index] for index in selected]

    def update_herding_memory(self, train_loader: DataLoader) -> None:
        if int(self.memory_budget) == 0:
            self.exemplar_sets.clear()
            return
        per_class = int(self.memory_budget) // max(1, len(self.seen_classes))
        if per_class <= 0:
            raise ValueError("memory_budget must hold at least one sample per seen class.")
        for class_id in self.old_classes:
            self.exemplar_sets[class_id] = self.exemplar_sets.get(class_id, [])[:per_class]
        for class_id in self.new_classes:
            self.exemplar_sets[class_id] = self.construct_exemplar_set(
                self._class_samples(train_loader.dataset, class_id), per_class
            )

    def _on_task_end(self, task: TaskDefinition, train_loader: DataLoader) -> None:
        del task
        self.update_herding_memory(train_loader)

    def evaluation_attack_calibrator(
        self, class_ids: Sequence[int]
    ) -> Callable[[torch.Tensor], torch.Tensor] | None:
        del class_ids
        return None

    def evaluate_robust(
        self,
        dataloader: DataLoader,
        task_id: str | None = None,
        max_batches: int | None = None,
    ) -> MetricDict:
        self.model.to(self.device)
        was_training = self.model.training
        self.model.eval()
        total_correct = 0
        total_examples = 0
        total_loss = 0.0
        class_ids = self.active_classes(task_id)
        try:
            for batch_index, batch in enumerate(dataloader):
                if (
                    max_batches is not None
                    and int(max_batches) > 0
                    and batch_index >= int(max_batches)
                ):
                    break
                inputs = move_to_device(batch[0], self.device)
                targets = move_to_device(batch[1], self.device).long()
                adversarial = self.generate_adversarial(
                    inputs,
                    targets,
                    config=self.eval_attack_config,
                    class_ids=class_ids,
                    calibrate=self.evaluation_attack_calibrator(class_ids),
                )
                with torch.no_grad():
                    logits = self._select_logits(self.model, adversarial, class_ids)
                    local_targets = _local_targets(targets, class_ids)
                    loss = F.cross_entropy(logits, local_targets)
                    predictions = self.predict(adversarial, task_id=task_id)
                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.item()) * batch_size
                total_correct += int(predictions.eq(targets).sum().item())
        finally:
            self.model.train(was_training)
        return {
            "robust_accuracy": total_correct / max(1, total_examples),
            "robust_loss": total_loss / max(1, total_examples),
            "num_samples": float(total_examples),
        }
