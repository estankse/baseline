from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ...contracts import AggregationResult, ClientContext, TrainResult
from ...models.fcl_models import (
    BottleneckAdapter,
    PromptPool,
    PromptedVisionTransformer,
    SparseAdapterGate,
)
from ...models.clip_vlm import CLIPVisionLanguageModel
from ...trainers.utils import move_to_device
from ._common import PartialStateServer, clone_state, mean_metrics, task_loader, weighted_average_state


def routing_consistency_loss(
    image_routes: torch.Tensor,
    text_routes: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    image_routes = F.normalize(image_routes, dim=1)
    text_routes = F.normalize(text_routes, dim=1)
    similarity = image_routes @ text_routes.t() / float(temperature)
    targets = torch.arange(similarity.shape[0], device=similarity.device)
    return 0.5 * (
        F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.t(), targets)
    )


def expert_stability_loss(
    current_routes: torch.Tensor,
    historical_routes: torch.Tensor | None,
) -> torch.Tensor:
    if historical_routes is None:
        return current_routes.new_zeros(())
    historical = historical_routes.to(current_routes)
    if historical.ndim == 1:
        historical = historical.unsqueeze(0).expand_as(current_routes)
    return F.kl_div(
        current_routes.clamp_min(1e-8).log(),
        historical.clamp_min(1e-8),
        reduction="batchmean",
    )


class _WeightedAdapter(nn.Module):
    def __init__(
        self,
        shared: BottleneckAdapter,
        local: Sequence[BottleneckAdapter],
        gates: torch.Tensor,
    ) -> None:
        super().__init__()
        self.shared = shared
        self.local = nn.ModuleList(local)
        self.gates = gates

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.shared(inputs)
        local_outputs = torch.stack([adapter(inputs) for adapter in self.local], dim=1)
        shape = [self.gates.shape[0], self.gates.shape[1]] + [1] * (inputs.ndim - 1)
        mixed = (local_outputs * self.gates.view(*shape)).sum(dim=1)
        return 0.5 * (output + mixed)


@dataclass
class FedDuetClient:
    client_id: str
    backbone: PromptedVisionTransformer
    shared_adapters: nn.ModuleList
    local_adapters: nn.ModuleList
    router: nn.Module
    local_prompt: nn.Parameter
    task_loaders: Mapping[str, DataLoader]
    device: str | torch.device = "cpu"
    epochs: int = 5
    lr: float = 3e-5
    weight_decay: float = 0.01
    phase_switch_round: int = 5
    shared_logit_weight: float = 0.5
    moe_weight: float = 1.0
    cross_modal_weight: float = 1.0
    stability_weight: float = 1.0
    top_k_adapters: int = 1
    historical_routes: torch.Tensor | None = None
    clip_model: CLIPVisionLanguageModel | None = None
    text_router: nn.Module | None = None
    fusion_gating: nn.MultiheadAttention | None = None
    dp_clip: float = 0.0
    dp_noise_multiplier: float = 0.0

    def _router_at(self, layer: int, text: bool = False) -> SparseAdapterGate:
        router = self.text_router if text else self.router
        if router is None:
            raise RuntimeError("The requested Fed-Duet router is unavailable.")
        selected = router[layer] if isinstance(router, nn.ModuleList) else router
        if not isinstance(selected, SparseAdapterGate):
            raise TypeError("Fed-Duet routers must be SparseAdapterGate modules.")
        return selected

    def summarize(
        self,
        task_id: str,
        max_batches: int = 1,
        apply_dp: bool = True,
    ) -> torch.Tensor:
        loader = task_loader(self.task_loaders, task_id)
        summaries = []
        model: nn.Module = self.clip_model if self.clip_model is not None else self.backbone
        model.to(self.device).eval()
        with torch.no_grad():
            for index, (inputs, _) in enumerate(loader):
                inputs = move_to_device(inputs, self.device)
                if self.clip_model is not None:
                    features = self.clip_model.encode_image(inputs)
                else:
                    features = self.backbone.encode(inputs)
                    assert isinstance(features, torch.Tensor)
                summaries.append(features.mean(0))
                if max_batches > 0 and index + 1 >= max_batches:
                    break
        summary = torch.stack(summaries).mean(0)
        if apply_dp and self.dp_clip > 0:
            norm = summary.norm().clamp_min(1e-12)
            summary = summary * min(1.0, float(self.dp_clip) / float(norm))
            if self.dp_noise_multiplier > 0:
                summary = summary + torch.randn_like(summary) * (
                    float(self.dp_clip) * float(self.dp_noise_multiplier)
                )
        return summary.cpu()

    def _semantic_logits(
        self,
        inputs: torch.Tensor,
        shared_prompts: torch.Tensor,
    ) -> torch.Tensor:
        batch = inputs.shape[0]
        local = self.local_prompt.unsqueeze(0).expand(batch, -1, -1)
        local_features = self.backbone.encode(inputs, input_prompts=local)
        assert isinstance(local_features, torch.Tensor)
        local_logits = self.backbone.classifier(local_features)
        shared_logits = []
        for prompt in shared_prompts:
            expanded = prompt.unsqueeze(0).expand(batch, -1, -1)
            features = self.backbone.encode(inputs, input_prompts=expanded)
            assert isinstance(features, torch.Tensor)
            shared_logits.append(self.backbone.classifier(features))
        shared = torch.stack(shared_logits).mean(0) if shared_logits else local_logits
        weight = float(self.shared_logit_weight)
        return weight * local_logits + (1.0 - weight) * shared

    def _semantic_clip_logits(
        self,
        image_features: torch.Tensor,
        shared_prompts: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse local/shared text experts with image-conditioned attention."""
        assert self.clip_model is not None
        local_text = self.clip_model.encode_text(context=self.local_prompt)
        scale = self.clip_model.logit_scale.exp().clamp(max=100.0)
        local_logits = scale * image_features @ local_text.t()
        if shared_prompts.numel() == 0:
            return local_logits
        shared_text = torch.stack(
            [self.clip_model.encode_text(context=prompt) for prompt in shared_prompts],
            dim=1,
        )
        experts = torch.cat((local_text.unsqueeze(1), shared_text), dim=1)
        batch, classes = image_features.shape[0], local_text.shape[0]
        if self.fusion_gating is None:
            fused = experts.mean(1).unsqueeze(0).expand(batch, -1, -1)
        else:
            query = image_features[:, None, None, :].expand(-1, classes, -1, -1)
            query = query.reshape(batch * classes, 1, image_features.shape[-1])
            keys = experts.unsqueeze(0).expand(batch, -1, -1, -1)
            keys = keys.reshape(batch * classes, experts.shape[1], experts.shape[2])
            fused = self.fusion_gating(query, keys, keys, need_weights=False)[0]
            fused = F.normalize(fused.reshape(batch, classes, -1), dim=-1)
        shared_logits = scale * torch.einsum("bd,bcd->bc", image_features, fused)
        weight = float(self.shared_logit_weight)
        return weight * local_logits + (1.0 - weight) * shared_logits

    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
        shared_prompts: torch.Tensor,
    ) -> torch.Tensor:
        inputs = inputs.to(self.device)
        self.backbone.to(self.device).eval()
        if self.clip_model is not None:
            self.clip_model.to(self.device).eval()
        self.shared_adapters.to(self.device).eval()
        self.local_adapters.to(self.device).eval()
        self.router.to(self.device).eval()
        if self.text_router is not None:
            self.text_router.to(self.device).eval()
        if self.fusion_gating is not None:
            self.fusion_gating.to(self.device).eval()
        shared_prompts = shared_prompts.to(self.device)
        base_features = self.backbone.encode(inputs)
        assert isinstance(base_features, torch.Tensor)
        base_output = self.backbone.encode(inputs, return_intermediates=True)
        assert isinstance(base_output, tuple)
        _base_final, intermediate_features = base_output
        gates_per_layer = [
            self._router_at(layer)(
                intermediate_features[layer], top_k=self.top_k_adapters
            )["gates"]
            for layer in range(self.backbone.depth)
        ]
        per_layer = [
            _WeightedAdapter(
                self.shared_adapters[layer],
                list(self.local_adapters[layer]),
                gates_per_layer[layer],
            )
            for layer in range(self.backbone.depth)
        ]
        if self.clip_model is not None:
            image_features = self.clip_model.encode_image(inputs, adapters=per_layer)
            return self._semantic_clip_logits(image_features, shared_prompts)
        features = self.backbone.encode(inputs, adapters=per_layer)
        assert isinstance(features, torch.Tensor)
        parametric_logits = self.backbone.classifier(features)
        semantic_logits = self._semantic_logits(inputs, shared_prompts)
        return 0.5 * (parametric_logits + semantic_logits)

    def fit(
        self,
        shared_adapter_state: Mapping[str, object],
        head_state: Mapping[str, object],
        shared_prompts: torch.Tensor,
        prompt_indices: torch.Tensor,
        context: ClientContext,
    ) -> TrainResult:
        self.shared_adapters.load_state_dict(clone_state(shared_adapter_state), strict=True)
        if self.clip_model is None:
            self.backbone.classifier.load_state_dict(clone_state(head_state), strict=True)
        loader = task_loader(self.task_loaders, context.task_id)
        modules: list[nn.Module] = [
            self.backbone,
            self.shared_adapters,
            self.local_adapters,
            self.router,
        ]
        if self.clip_model is not None:
            modules.append(self.clip_model)
        if self.text_router is not None:
            modules.append(self.text_router)
        if self.fusion_gating is not None:
            modules.append(self.fusion_gating)
        for module in modules:
            module.to(self.device)
        self.local_prompt.data = self.local_prompt.data.to(self.device)
        shared_prompts = shared_prompts.to(self.device)
        parametric_phase = context.round_idx < int(self.phase_switch_round)
        self.local_prompt.requires_grad = False
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
        if parametric_phase:
            for parameter in self.shared_adapters.parameters():
                parameter.requires_grad = True
            for parameter in self.local_adapters.parameters():
                parameter.requires_grad = True
            for parameter in self.router.parameters():
                parameter.requires_grad = True
            if self.text_router is not None:
                for parameter in self.text_router.parameters():
                    parameter.requires_grad = True
        else:
            self.local_prompt.requires_grad = True
            if self.fusion_gating is not None:
                for parameter in self.fusion_gating.parameters():
                    parameter.requires_grad = True
            if self.clip_model is None:
                for parameter in self.backbone.classifier.parameters():
                    parameter.requires_grad = True
        trainable = [
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if self.local_prompt.requires_grad:
            trainable.append(self.local_prompt)
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        route_average = None
        for _ in range(max(int(self.epochs), 1)):
            for inputs, targets in loader:
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                with torch.no_grad():
                    base_output = self.backbone.encode(inputs, return_intermediates=True)
                assert isinstance(base_output, tuple)
                base_features, layer_features = base_output
                routing = [
                    self._router_at(layer)(
                        layer_features[layer], top_k=self.top_k_adapters
                    )
                    for layer in range(self.backbone.depth)
                ]
                per_layer = [
                    _WeightedAdapter(
                        self.shared_adapters[layer],
                        list(self.local_adapters[layer]),
                        routing[layer]["gates"],
                    )
                    for layer in range(self.backbone.depth)
                ]
                if self.clip_model is not None:
                    image_features = self.clip_model.encode_image(inputs, adapters=per_layer)
                    logits = self._semantic_clip_logits(image_features, shared_prompts)
                else:
                    parametric_features = self.backbone.encode(inputs, adapters=per_layer)
                    assert isinstance(parametric_features, torch.Tensor)
                    parametric_logits = self.backbone.classifier(parametric_features)
                    semantic_logits = self._semantic_logits(inputs, shared_prompts)
                    logits = 0.5 * (parametric_logits + semantic_logits)
                loss = F.cross_entropy(logits, targets)
                moe_loss = torch.stack(
                    [route["load_balance_loss"] for route in routing]
                ).mean()
                loss = loss + float(self.moe_weight) * moe_loss
                if self.clip_model is not None and self.text_router is not None:
                    class_text = self.clip_model.encode_text(context=self.local_prompt)
                    paired_text = class_text.index_select(0, targets)
                    text_routes = [
                        F.softmax(self._router_at(layer, text=True).network(paired_text), dim=1)
                        for layer in range(self.backbone.depth)
                    ]
                else:
                    text_feature = self.local_prompt.mean(dim=0).unsqueeze(0).expand_as(base_features)
                    text_routes = [
                        F.softmax(self._router_at(layer).network(text_feature), dim=1)
                        for layer in range(self.backbone.depth)
                    ]
                image_routes = [F.softmax(route["logits"], dim=1) for route in routing]
                cross_modal = torch.stack(
                    [routing_consistency_loss(image, text) for image, text in zip(image_routes, text_routes)]
                ).mean()
                stability_terms = []
                for layer, image in enumerate(image_routes):
                    history = None if self.historical_routes is None else self.historical_routes[layer]
                    stability_terms.append(expert_stability_loss(image, history))
                stability = torch.stack(stability_terms).mean()
                loss = loss + float(self.cross_modal_weight) * cross_modal
                loss = loss + float(self.stability_weight) * stability
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                batch_routes = torch.stack(
                    [route.detach().mean(0) for route in image_routes]
                )
                route_average = batch_routes if route_average is None else route_average + batch_routes
                count = int(targets.shape[0])
                total_examples += count
                total_loss += float(loss.detach()) * count
                total_correct += int((logits.argmax(1) == targets).sum())
        if route_average is not None:
            route_average = route_average / max(int(self.epochs) * len(loader), 1)
            if self.historical_routes is None:
                self.historical_routes = route_average.cpu()
            else:
                self.historical_routes = (
                    0.9 * self.historical_routes + 0.1 * route_average.cpu()
                )
        return TrainResult(
            client_id=self.client_id,
            num_samples=len(loader.dataset),
            metrics={
                "loss": total_loss / max(1, total_examples),
                "accuracy": total_correct / max(1, total_examples),
                "parametric_phase": float(parametric_phase),
            },
            payload={
                "shared_adapter_state": clone_state(self.shared_adapters.state_dict()),
                "head_state": (
                    {} if self.clip_model is not None
                    else clone_state(self.backbone.classifier.state_dict())
                ),
                "local_prompt": self.local_prompt.detach().cpu().clone(),
                "prompt_indices": prompt_indices.detach().cpu().clone(),
                "feature_summary": self.summarize(context.task_id or next(iter(self.task_loaders))),
            },
        )


@dataclass
class FedDuetServer(PartialStateServer):
    prompt_repository: PromptPool = field(default_factory=lambda: None)  # type: ignore[arg-type]
    dispatch_gate: nn.Module = field(default_factory=nn.Identity)
    shared_adapters: nn.ModuleList = field(default_factory=nn.ModuleList)
    head: nn.Module = field(default_factory=nn.Identity)
    device: str | torch.device = "cpu"
    experts_per_client: int = 4
    gate_lr: float = 1e-3

    def _dispatch(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.dispatch_gate.to(self.device)
        self.prompt_repository.to(self.device)
        logits = self.dispatch_gate(feature.to(self.device).unsqueeze(0)).squeeze(0)
        count = min(max(int(self.experts_per_client), 1), self.prompt_repository.pool_size)
        indices = logits.topk(count).indices
        return self.prompt_repository.prompts[indices].detach(), indices

    def _train_dispatch_gate(self, results: Sequence[TrainResult]) -> None:
        if not results:
            return
        self.dispatch_gate.to(self.device).train()
        optimizer = torch.optim.Adam(self.dispatch_gate.parameters(), lr=self.gate_lr)
        features = torch.stack([result.payload["feature_summary"] for result in results]).to(
            self.device
        )
        targets = torch.zeros(
            len(results), self.prompt_repository.pool_size, device=self.device
        )
        losses = torch.tensor(
            [float(result.metrics.get("loss", 0.0)) for result in results],
            device=self.device,
        )
        for row, result in enumerate(results):
            targets[row, result.payload["prompt_indices"].long()] = 1.0
        weights = 1.0 / (losses + 1e-6)
        weights = weights / weights.mean().clamp_min(1e-12)
        logits = self.dispatch_gate(features)
        row_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(1)
        loss = (weights * row_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @torch.no_grad()
    def _update_repository(self, results: Sequence[TrainResult]) -> None:
        contributions: dict[int, list[tuple[torch.Tensor, float]]] = {}
        for result in results:
            prompt = result.payload["local_prompt"]
            score = 1.0 / (float(result.metrics.get("loss", 0.0)) + 1e-6)
            for index in result.payload["prompt_indices"].tolist():
                contributions.setdefault(int(index), []).append((prompt, score))
        for index, prompts in contributions.items():
            denominator = sum(score for _, score in prompts)
            reference = self.prompt_repository.prompts[index]
            self.prompt_repository.prompts[index].copy_(
                sum(prompt.to(reference) * score for prompt, score in prompts)
                / denominator
            )

    def run_round(self, round_idx: int, task_id: str) -> AggregationResult:
        results: list[TrainResult] = []
        for client in self.selected_clients(round_idx):
            assert isinstance(client, FedDuetClient)
            context = self.client_context(client, round_idx, task_id)
            summary = client.summarize(context.task_id or task_id)
            prompts, indices = self._dispatch(summary)
            results.append(
                client.fit(
                    self.shared_adapters.state_dict(),
                    self.head.state_dict(),
                    prompts,
                    indices,
                    context,
                )
            )
        if not results:
            return self.empty_result(round_idx, task_id)
        parametric_phase = any(
            bool(result.metrics.get("parametric_phase", 0.0)) for result in results
        )
        clip_mode = any(not result.payload.get("head_state") for result in results)
        if parametric_phase:
            shared = weighted_average_state(
                [(result.payload["shared_adapter_state"], result.num_samples) for result in results]
            )
            self.shared_adapters.load_state_dict(shared, strict=True)
        if not clip_mode:
            head = weighted_average_state(
                [(result.payload["head_state"], result.num_samples) for result in results]
            )
            self.head.load_state_dict(head, strict=True)
        self._update_repository(results)
        self._train_dispatch_gate(results)
        state = {
            f"shared_adapters.{name}": value
            for name, value in clone_state(self.shared_adapters.state_dict()).items()
        }
        state.update(
            {f"repository.{name}": value for name, value in clone_state(self.prompt_repository.state_dict()).items()}
        )
        state.update({f"gate.{name}": value for name, value in clone_state(self.dispatch_gate.state_dict()).items()})
        if not clip_mode:
            state.update({f"head.{name}": value for name, value in clone_state(self.head.state_dict()).items()})
        return AggregationResult(
            global_state=state,
            metrics=mean_metrics(results),
            metadata={"round_idx": round_idx, "task_id": task_id, "algorithm": "fedduet", "clip_vlm": clip_mode, "parametric_phase": parametric_phase},
        )
