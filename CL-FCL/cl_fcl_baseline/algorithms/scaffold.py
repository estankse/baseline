
from dataclasses import dataclass, field
import random
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from ..contracts import AggregationResult, ClientContext, MetricDict, StateDict, TrainResult
from ..trainers.trainer import BaseTrainer, build_default_loss
from ..trainers.utils import detach_state_dict, move_to_device


def _clone_tensor_state(state: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[name] = value.detach().cpu().clone()
    return cloned


def _zeros_like(state: Mapping[str, object]) -> Dict[str, torch.Tensor]:
    zeros: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            zeros[name] = torch.zeros_like(value.detach().cpu())
    return zeros


def _to_device(state: Mapping[str, torch.Tensor], device: torch.device | str) -> Dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in state.items()}


@dataclass
class ScaffoldClient:
    client_id: str
    trainer: BaseTrainer
    train_loader: DataLoader
    epochs: int = 1
    local_lr: float = 0.01

    def fit(self, global_state: StateDict, context: ClientContext) -> TrainResult:
        self.trainer.model.load_state_dict(global_state, strict=True)
        self.trainer.model.to(self.trainer.device)
        self.trainer.model.train()

        control = context.metadata.get("control", {})
        client_control = context.metadata.get("client_control", {})
        if not isinstance(control, Mapping):
            control = {}
        if not isinstance(client_control, Mapping):
            client_control = {}

        control_device = _to_device(control, self.trainer.device) if control else {}
        client_control_device = _to_device(client_control, self.trainer.device) if client_control else {}

        loss_fn = self.trainer.loss_fn or build_default_loss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        step_count = 0

        for _ in range(int(self.epochs)):
            for inputs, targets in self.train_loader:
                inputs = move_to_device(inputs, self.trainer.device)
                targets = move_to_device(targets, self.trainer.device)
                self.trainer.optimizer.zero_grad()
                logits = self.trainer.model(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()

                with torch.no_grad():
                    for name, param in self.trainer.model.named_parameters():
                        if param.grad is None:
                            continue
                        if name in control_device and name in client_control_device:
                            # modify the gradient
                            param.grad.data.add_(control_device[name] - client_control_device[name])

                self.trainer.optimizer.step()

                batch_size = int(targets.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                predictions = logits.argmax(dim=1)
                total_correct += int((predictions == targets).sum().item())
                step_count += 1

        metrics: MetricDict
        if total_examples == 0:
            metrics = {"loss": 0.0, "accuracy": 0.0}
        else:
            metrics = {
                "loss": total_loss / total_examples,
                "accuracy": total_correct / total_examples,
            }

        global_state_cpu = _clone_tensor_state(global_state)
        local_state_cpu = detach_state_dict(self.trainer.model.state_dict())

        effective_lr = float(self.local_lr)
        if effective_lr <= 0:
            try:
                effective_lr = float(self.trainer.optimizer.param_groups[0].get("lr", 0.0))
            except Exception:
                effective_lr = 0.0

        if step_count > 0 and effective_lr > 0:
            scale = 1.0 / (float(step_count) * effective_lr)
        else:
            scale = 0.0

        control_cpu = _clone_tensor_state(control)
        client_control_cpu = _clone_tensor_state(client_control)

        new_client_control: Dict[str, torch.Tensor] = {}
        model_delta: Dict[str, torch.Tensor] = {}
        control_delta: Dict[str, torch.Tensor] = {}

        for name, global_tensor in global_state_cpu.items():
            local_tensor = local_state_cpu.get(name)
            if local_tensor is None:
                continue
            model_delta[name] = local_tensor - global_tensor

            ci_tensor = client_control_cpu.get(name)
            if ci_tensor is None:
                ci_tensor = torch.zeros_like(global_tensor)
            c_tensor = control_cpu.get(name)
            if c_tensor is None:
                c_tensor = torch.zeros_like(global_tensor)

            if scale > 0.0:
                ci_new = ci_tensor - c_tensor + (global_tensor - local_tensor) * scale
            else:
                ci_new = ci_tensor.clone()
            new_client_control[name] = ci_new
            control_delta[name] = ci_new - ci_tensor

        payload = {
            "model_delta": model_delta,
            "control_delta": control_delta,
            "control_state": new_client_control,
        }
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(self.train_loader.dataset),
            metrics=metrics,
            payload=payload,
        )


@dataclass
class ScaffoldServer:
    model: torch.nn.Module
    clients: Sequence[ScaffoldClient]
    client_sample_ratio: float = 1.0
    global_lr: float = 1.0
    global_control: Dict[str, torch.Tensor] = field(init=False)
    client_controls: Dict[str, Dict[str, torch.Tensor]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < float(self.client_sample_ratio) <= 1.0):
            raise ValueError("client_sample_ratio must be in (0, 1].")
        initial_state = _clone_tensor_state(self.model.state_dict())
        self.global_control = _zeros_like(initial_state)
        for client in self.clients:
            self.client_controls[client.client_id] = _zeros_like(initial_state)

    def get_global_state(self) -> StateDict:
        return detach_state_dict(self.model.state_dict())

    def set_global_state(self, state_dict: StateDict) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def _aggregate_metrics(self, results: Iterable[TrainResult]) -> MetricDict:
        metric_sums: Dict[str, float] = {}
        metric_weights: Dict[str, float] = {}
        total_samples = 0
        total_clients = 0
        for result in results:
            weight = max(int(result.num_samples), 0)
            total_samples += weight
            total_clients += 1
            for name, value in result.metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * float(weight)
                metric_weights[name] = metric_weights.get(name, 0.0) + float(weight)
        metrics: MetricDict = {}
        for name, total in metric_sums.items():
            if metric_weights.get(name, 0.0) > 0.0:
                metrics[f"client_{name}"] = total / metric_weights[name]
        metrics["num_clients"] = float(total_clients)
        metrics["total_samples"] = float(total_samples)
        return metrics

    def run_round(self, round_idx: int) -> AggregationResult:
        global_state = self.get_global_state()
        selected_clients = list(self.clients)
        if selected_clients and self.client_sample_ratio < 1.0:
            num_selected = max(1, int(len(selected_clients) * self.client_sample_ratio))
            selected_clients = random.sample(selected_clients, k=num_selected)

        client_results: List[TrainResult] = []
        for client in selected_clients:
            client_control = self.client_controls.get(client.client_id)
            if client_control is None:
                client_control = _zeros_like(global_state)
                self.client_controls[client.client_id] = client_control
            context = ClientContext(
                client_id=client.client_id,
                round_idx=round_idx,
                metadata={
                    "control": self.global_control,
                    "client_control": client_control,
                },
            )
            result = client.fit(global_state, context)
            client_results.append(result)
            new_control = result.payload.get("control_state")
            if isinstance(new_control, Mapping):
                self.client_controls[client.client_id] = _clone_tensor_state(new_control)

        if not client_results:
            return AggregationResult(
                global_state={},
                metrics={"num_clients": 0.0, "total_samples": 0.0},
                metadata={"aggregator": "scaffold"},
            )

        delta_x_acc: Dict[str, torch.Tensor] = {}
        delta_c_acc: Dict[str, torch.Tensor] = {}
        num_clients = len(client_results)

        for result in client_results:
            model_delta = result.payload.get("model_delta", {})
            control_delta = result.payload.get("control_delta", {})
            if isinstance(model_delta, Mapping):
                for name, tensor in model_delta.items():
                    if isinstance(tensor, torch.Tensor):
                        delta_x_acc[name] = delta_x_acc.get(name, torch.zeros_like(tensor)) + tensor
            if isinstance(control_delta, Mapping):
                for name, tensor in control_delta.items():
                    if isinstance(tensor, torch.Tensor):
                        delta_c_acc[name] = delta_c_acc.get(name, torch.zeros_like(tensor)) + tensor

        if num_clients > 0:
            for name in delta_x_acc:
                delta_x_acc[name] = delta_x_acc[name] / float(num_clients)
            for name in delta_c_acc:
                delta_c_acc[name] = delta_c_acc[name] / float(num_clients)

        new_state: Dict[str, torch.Tensor] = {}
        for name, tensor in global_state.items():
            delta = delta_x_acc.get(name)
            if isinstance(delta, torch.Tensor):
                new_state[name] = tensor + float(self.global_lr) * delta
            else:
                new_state[name] = tensor

        total_clients = len(self.clients)
        if total_clients > 0 and num_clients > 0:
            scale = float(num_clients) / float(total_clients)
        else:
            scale = 0.0
        for name, tensor in self.global_control.items():
            delta = delta_c_acc.get(name)
            if isinstance(delta, torch.Tensor):
                self.global_control[name] = tensor + scale * delta

        self.set_global_state(new_state)
        metrics = self._aggregate_metrics(client_results)
        return AggregationResult(
            global_state=detach_state_dict(self.model.state_dict()),
            metrics=metrics,
            metadata={"aggregator": "scaffold"},
        )
