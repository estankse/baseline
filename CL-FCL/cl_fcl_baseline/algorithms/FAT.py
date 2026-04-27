from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping

import torch
import torch.nn.functional as F
from torch.func import functional_call

from ..contracts import ClientContext, MetricDict, StateDict, TrainResult
from ..trainers.utils import detach_state_dict, move_to_device
from .PGD import PGDConfig, pgd_linf_attack
from .fedweit import FedWeITClient, FedWeITServer, _state_l2_norm, _state_nnz, _state_numel


class _FunctionalModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, params_and_buffers: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.model = model
        self.params_and_buffers = params_and_buffers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional_call(self.model, self.params_and_buffers, (inputs,))


@dataclass
class FedWeITFATClient(FedWeITClient):
    """FedWeIT client with local Federated Adversarial Training.

    FAT keeps the federated protocol unchanged and changes each local minibatch
    update to use a mixture of clean and PGD adversarial examples.
    """

    pgd_config: PGDConfig = field(default_factory=PGDConfig)
    adversarial_ratio: float = 0.5
    warmup_rounds: int = 0
    warmup_adversarial_ratio: float = 0.1

    def _round_adversarial_ratio(self, round_idx: int | None) -> float:
        if int(self.warmup_rounds) > 0 and round_idx is not None and int(round_idx) < int(self.warmup_rounds):
            ratio = float(self.warmup_adversarial_ratio)
        else:
            ratio = float(self.adversarial_ratio)
        return min(max(ratio, 0.0), 1.0)

    def _adversarial_count(self, batch_size: int, ratio: float) -> int:
        if batch_size <= 0 or ratio <= 0.0:
            return 0
        return min(batch_size, max(1, int(round(batch_size * ratio))))

    def _mixed_fat_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        params_and_buffers: Mapping[str, torch.Tensor],
        adversarial_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(targets.shape[0])
        adv_count = self._adversarial_count(batch_size, adversarial_ratio)
        if adv_count <= 0:
            empty = torch.empty(0, dtype=torch.long, device=targets.device)
            return inputs, empty, torch.arange(batch_size, dtype=torch.long, device=targets.device)

        permutation = torch.randperm(batch_size, device=targets.device)
        adv_indices = permutation[:adv_count]
        clean_indices = permutation[adv_count:]

        attack_model = _FunctionalModel(self.trainer.model, params_and_buffers)
        was_training = self.trainer.model.training
        self.trainer.model.eval()
        try:
            adversarial_inputs = pgd_linf_attack(
                attack_model,
                inputs.index_select(0, adv_indices),
                targets.index_select(0, adv_indices),
                self.pgd_config,
            )
        finally:
            if was_training:
                self.trainer.model.train()

        mixed_inputs = inputs.detach().clone()
        mixed_inputs[adv_indices] = adversarial_inputs
        return mixed_inputs, adv_indices, clean_indices

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
        adversarial_ratio = self._round_adversarial_ratio(context.round_idx)

        total_examples = 0
        total_loss = 0.0
        total_ce = 0.0
        total_clean_ce = 0.0
        total_adv_ce = 0.0
        total_clean_examples = 0
        total_adv_examples = 0
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
                mixed_inputs, adv_indices, clean_indices = self._mixed_fat_batch(
                    inputs=inputs,
                    targets=targets,
                    params_and_buffers=params_and_buffers,
                    adversarial_ratio=adversarial_ratio,
                )

                logits = functional_call(self.trainer.model, params_and_buffers, (mixed_inputs,))
                ce_loss = F.cross_entropy(logits, targets)
                adaptive_sparse_loss, mask_sparse_loss, retro_loss = self._regularization_loss(
                    base=base,
                    masks=masks,
                    adaptive=adaptive,
                    previous_adaptive=previous_adaptive,
                    previous_masks=previous_masks,
                    previous_effective_anchor=previous_effective_anchor,
                )
                sparse_loss = float(self.lambda1) * adaptive_sparse_loss + float(self.lambda_mask) * mask_sparse_loss
                loss = ce_loss + sparse_loss + float(self.lambda2) * retro_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = int(targets.shape[0])
                adv_count = int(adv_indices.numel())
                clean_count = int(clean_indices.numel())
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_ce += float(ce_loss.detach().item()) * batch_size
                total_sparse += float(sparse_loss.detach().item()) * batch_size
                total_retro += float(retro_loss.detach().item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == targets).sum().item())
                if clean_count > 0:
                    clean_loss = F.cross_entropy(
                        logits.index_select(0, clean_indices),
                        targets.index_select(0, clean_indices),
                    )
                    total_clean_ce += float(clean_loss.detach().item()) * clean_count
                    total_clean_examples += clean_count
                if adv_count > 0:
                    adv_loss = F.cross_entropy(
                        logits.index_select(0, adv_indices),
                        targets.index_select(0, adv_indices),
                    )
                    total_adv_ce += float(adv_loss.detach().item()) * adv_count
                    total_adv_examples += adv_count

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
                "ce_loss": 0.0,
                "clean_ce_loss": 0.0,
                "adv_ce_loss": 0.0,
                "sparse_loss": 0.0,
                "retro_loss": 0.0,
                "accuracy": 0.0,
                "kb_size": float(len(knowledge)),
                "adversarial_ratio": float(adversarial_ratio),
                "num_adversarial_samples": 0.0,
                "num_clean_samples": 0.0,
            }
        else:
            metrics = {
                "loss": total_loss / total_examples,
                "ce_loss": total_ce / total_examples,
                "clean_ce_loss": total_clean_ce / max(1, total_clean_examples),
                "adv_ce_loss": total_adv_ce / max(1, total_adv_examples),
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
                "adversarial_ratio": float(adversarial_ratio),
                "num_adversarial_samples": float(total_adv_examples),
                "num_clean_samples": float(total_clean_examples),
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


class FedWeITFATServer(FedWeITServer):
    pass
