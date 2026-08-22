from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def l2_normalize(inputs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(inputs, p=2, dim=dim, eps=1e-12)


class PromptPool(nn.Module):
    """Key/value prompt pool shared by FedMGP, Powder and Fed-Duet."""

    def __init__(
        self,
        pool_size: int,
        prompt_length: int,
        embed_dim: int,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        if pool_size <= 0 or prompt_length <= 0:
            raise ValueError("pool_size and prompt_length must be positive.")
        if not 0 < top_k <= pool_size:
            raise ValueError("top_k must be in [1, pool_size].")
        self.pool_size = int(pool_size)
        self.prompt_length = int(prompt_length)
        self.embed_dim = int(embed_dim)
        self.top_k = int(top_k)
        self.prompts = nn.Parameter(torch.empty(pool_size, prompt_length, embed_dim))
        self.keys = nn.Parameter(torch.empty(pool_size, embed_dim))
        self.attention = nn.Parameter(torch.ones(pool_size, embed_dim))
        nn.init.uniform_(self.prompts, -1.0, 1.0)
        nn.init.uniform_(self.keys, -1.0, 1.0)

    def similarity(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2 or query.shape[-1] != self.embed_dim:
            raise ValueError("query must have shape [batch, embed_dim].")
        attended_query = query.unsqueeze(1) * self.attention.unsqueeze(0)
        return F.cosine_similarity(
            attended_query,
            self.keys.unsqueeze(0),
            dim=-1,
            eps=1e-12,
        )

    def forward(
        self,
        query: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        similarity = self.similarity(query)
        if indices is None:
            indices = similarity.topk(self.top_k, dim=1).indices
        if indices.ndim == 1:
            indices = indices.unsqueeze(1)
        selected = self.prompts[indices]
        prompts = selected.flatten(1, 2)
        selected_keys = l2_normalize(self.keys[indices])
        normalized_query = l2_normalize(query).unsqueeze(1)
        pull_similarity = (selected_keys * normalized_query).sum(dim=-1).mean()
        return {
            "prompts": prompts,
            "indices": indices,
            "similarity": similarity,
            "pull_similarity": pull_similarity,
        }


class PrefixPromptPool(nn.Module):
    """Class-wise key/value prefixes injected into transformer attention."""

    def __init__(
        self,
        num_layers: int,
        pool_size: int,
        prompt_length: int,
        embed_dim: int,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.pool_size = int(pool_size)
        self.prompt_length = int(prompt_length)
        self.embed_dim = int(embed_dim)
        self.top_k = int(top_k)
        self.key_prompts = nn.Parameter(
            torch.empty(num_layers, pool_size, prompt_length, embed_dim)
        )
        self.value_prompts = nn.Parameter(
            torch.empty(num_layers, pool_size, prompt_length, embed_dim)
        )
        self.keys = nn.Parameter(torch.empty(pool_size, embed_dim))
        nn.init.uniform_(self.key_prompts, -1.0, 1.0)
        nn.init.uniform_(self.value_prompts, -1.0, 1.0)
        nn.init.uniform_(self.keys, -1.0, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> dict[str, object]:
        similarity = l2_normalize(query) @ l2_normalize(self.keys).t()
        if indices is None:
            indices = similarity.topk(self.top_k, dim=1).indices
        if indices.ndim == 1:
            indices = indices.unsqueeze(1)
        prefixes: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.num_layers):
            keys = self.key_prompts[layer, indices].flatten(1, 2)
            values = self.value_prompts[layer, indices].flatten(1, 2)
            prefixes.append((keys, values))
        selected_keys = l2_normalize(self.keys[indices])
        pull_similarity = (selected_keys * l2_normalize(query).unsqueeze(1)).sum(-1).mean()
        return {
            "prefixes": prefixes,
            "indices": indices,
            "similarity": similarity,
            "pull_similarity": pull_similarity,
        }


class BottleneckAdapter(nn.Module):
    """Parallel residual adapter used by MultiFCL, MoAFCL and Fed-Duet."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(embed_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(scale)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.up(self.dropout(F.relu(self.down(inputs))))
        return inputs + self.scale * output


class SparseAdapterGate(nn.Module):
    """Top-k gate with the load-balancing term used by MoAFCL."""

    def __init__(self, embed_dim: int, num_adapters: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.num_adapters = int(num_adapters)
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_adapters),
        )

    def forward(
        self,
        features: torch.Tensor,
        top_k: int = 1,
        temperature: float = 5.0,
    ) -> dict[str, torch.Tensor]:
        logits = self.network(features)
        top_k = min(max(int(top_k), 1), self.num_adapters)
        values, indices = logits.topk(top_k, dim=1)
        selected = F.softmax(values / float(temperature), dim=1)
        gates = torch.zeros_like(logits).scatter(1, indices, selected)
        importance = gates.sum(dim=0)
        load = (gates > 0).sum(dim=0).to(gates.dtype)

        def _cv_squared(vector: torch.Tensor) -> torch.Tensor:
            if vector.numel() <= 1:
                return vector.new_zeros(())
            return vector.var(unbiased=False) / vector.mean().square().clamp_min(1e-10)

        return {
            "gates": gates,
            "logits": logits,
            "indices": indices,
            "load_balance_loss": _cv_squared(importance) + _cv_squared(load),
        }


class _PromptAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        if activation == "gelu":
            activation_layer: nn.Module = nn.GELU()
        elif activation == "quick_gelu":
            activation_layer = QuickGELU()
        else:
            raise ValueError(f"Unsupported transformer activation: {activation}")
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        prefix: tuple[torch.Tensor, torch.Tensor] | None = None,
        adapter: nn.Module | None = None,
    ) -> torch.Tensor:
        query = self.norm1(inputs)
        if prefix is None:
            keys = values = query
        else:
            prefix_keys, prefix_values = prefix
            keys = torch.cat((prefix_keys, query), dim=1)
            values = torch.cat((prefix_values, query), dim=1)
        attended, _ = self.attention(query, keys, values, need_weights=False)
        output = inputs + attended
        output = output + self.mlp(self.norm2(output))
        if adapter is not None:
            output = adapter(output)
        return output


class QuickGELU(nn.Module):
    """Activation used by the original OpenAI CLIP checkpoints."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * torch.sigmoid(1.702 * inputs)


@dataclass(frozen=True)
class PromptViTSpec:
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3


class PromptedVisionTransformer(nn.Module):
    """Small ViT exposing the prompt, adapter and multi-scale hooks used by FCL methods."""

    def __init__(
        self,
        input_shape: Sequence[int],
        num_classes: int,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        backbone_source: str = "vit",
    ) -> None:
        super().__init__()
        source = str(backbone_source).lower()
        if source not in {"vit", "clip"}:
            raise ValueError("backbone_source must be either 'vit' or 'clip'.")
        channels, height, width = (int(value) for value in input_shape)
        if height % patch_size or width % patch_size:
            raise ValueError("patch_size must divide both image dimensions.")
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.backbone_source = source
        self.patch_embed = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (height // patch_size) * (width // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.input_norm: nn.Module
        self.input_norm = nn.LayerNorm(embed_dim) if source == "clip" else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                _PromptAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    activation="quick_gelu" if source == "clip" else "gelu",
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, int(num_classes))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def load_pretrained_checkpoint(self, checkpoint: str | Path) -> dict[str, object]:
        """Load a native state dict, official Google ViT NPZ, or OpenAI CLIP checkpoint.

        The classifier is intentionally kept task-specific.  Unlike a permissive
        ``strict=False`` load, this method verifies that every backbone tensor was
        populated so an incompatible checkpoint cannot silently leave a frozen,
        randomly initialized backbone behind.
        """
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ViT backbone checkpoint does not exist: {path}")

        if path.suffix.lower() == ".npz":
            if self.backbone_source != "vit":
                raise ValueError(
                    "Google ViT .npz checkpoints require --backbone-source vit."
                )
            state = self._google_vit_npz_state(path)
            checkpoint_format = "google_vit_npz"
        else:
            raw_state = self._load_torch_checkpoint(path)
            state = self._unwrap_pytorch_state_dict(raw_state)
            is_clip = any(key.startswith("visual.") for key in state)
            if is_clip:
                if self.backbone_source != "clip":
                    raise ValueError(
                        "This is a CLIP checkpoint; construct the backbone with "
                        "--backbone-source clip."
                    )
                state = self._openai_clip_visual_state(state)
                checkpoint_format = "openai_clip"
            else:
                checkpoint_format = "pytorch"

        target = self.state_dict()
        backbone_keys = {key for key in target if not key.startswith("classifier.")}
        matching = {
            key: value
            for key, value in state.items()
            if key in backbone_keys and tuple(value.shape) == tuple(target[key].shape)
        }
        missing = sorted(backbone_keys - matching.keys())
        if missing:
            preview = ", ".join(missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise ValueError(
                f"Checkpoint {path} is incompatible with this PromptedVisionTransformer: "
                f"loaded {len(matching)}/{len(backbone_keys)} backbone tensors; "
                f"missing or shape-mismatched: {preview}{suffix}"
            )

        self.load_state_dict(matching, strict=False)
        return {
            "path": str(path.resolve()),
            "format": checkpoint_format,
            "loaded_backbone_tensors": len(matching),
            "total_backbone_tensors": len(backbone_keys),
            "classifier_loaded": False,
        }

    @staticmethod
    def _load_torch_checkpoint(path: Path) -> object:
        """Load both regular state dictionaries and OpenAI's TorchScript ``.pt`` files."""
        try:
            return torch.jit.load(str(path), map_location="cpu").state_dict()
        except RuntimeError:
            return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def _unwrap_pytorch_state_dict(raw_state: object) -> dict[str, torch.Tensor]:
        if not isinstance(raw_state, Mapping):
            raise ValueError("PyTorch checkpoint must contain a state-dict mapping.")
        for wrapper in ("state_dict", "model", "model_state_dict"):
            candidate = raw_state.get(wrapper)
            if isinstance(candidate, Mapping):
                raw_state = candidate
                break

        state: dict[str, torch.Tensor] = {}
        for raw_key, value in raw_state.items():
            if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
                continue
            key = raw_key
            for prefix in ("module.", "model.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            state[key] = value
        return state

    def _openai_clip_visual_state(
        self,
        clip_state: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert the visual tower of an original OpenAI CLIP state dict.

        CLIP's final 768-to-512 projection is intentionally omitted: prompt-based
        classifiers and adapters in this repository operate on the visual width.
        The text tower is likewise not part of these supervised classifiers.
        """
        required = {
            "visual.conv1.weight",
            "visual.class_embedding",
            "visual.positional_embedding",
            "visual.ln_pre.weight",
            "visual.ln_pre.bias",
            "visual.ln_post.weight",
            "visual.ln_post.bias",
        }
        absent = sorted(required - set(clip_state))
        if absent:
            raise ValueError(
                "CLIP checkpoint has no compatible ViT visual tower; missing keys: "
                + ", ".join(absent)
            )

        patch_weight = clip_state["visual.conv1.weight"]
        if patch_weight.shape[1] == 3 and self.patch_embed.in_channels == 1:
            patch_weight = patch_weight.mean(dim=1, keepdim=True)
        state: dict[str, torch.Tensor] = {
            "patch_embed.weight": patch_weight,
            # OpenAI CLIP's patch convolution has no bias, while this model keeps
            # one for compatibility with Google ViT checkpoints.
            "patch_embed.bias": torch.zeros_like(self.patch_embed.bias),
            "cls_token": clip_state["visual.class_embedding"].reshape(1, 1, -1),
            "position_embedding": self._resize_position_embedding(
                clip_state["visual.positional_embedding"].unsqueeze(0),
                self.position_embedding.shape[1],
            ),
            "input_norm.weight": clip_state["visual.ln_pre.weight"],
            "input_norm.bias": clip_state["visual.ln_pre.bias"],
            "norm.weight": clip_state["visual.ln_post.weight"],
            "norm.bias": clip_state["visual.ln_post.bias"],
        }
        for index in range(self.depth):
            source = f"visual.transformer.resblocks.{index}"
            destination = f"blocks.{index}"
            mapping = {
                f"{source}.ln_1.weight": f"{destination}.norm1.weight",
                f"{source}.ln_1.bias": f"{destination}.norm1.bias",
                f"{source}.attn.in_proj_weight": f"{destination}.attention.in_proj_weight",
                f"{source}.attn.in_proj_bias": f"{destination}.attention.in_proj_bias",
                f"{source}.attn.out_proj.weight": f"{destination}.attention.out_proj.weight",
                f"{source}.attn.out_proj.bias": f"{destination}.attention.out_proj.bias",
                f"{source}.ln_2.weight": f"{destination}.norm2.weight",
                f"{source}.ln_2.bias": f"{destination}.norm2.bias",
                f"{source}.mlp.c_fc.weight": f"{destination}.mlp.0.weight",
                f"{source}.mlp.c_fc.bias": f"{destination}.mlp.0.bias",
                f"{source}.mlp.c_proj.weight": f"{destination}.mlp.3.weight",
                f"{source}.mlp.c_proj.bias": f"{destination}.mlp.3.bias",
            }
            missing_block = sorted(key for key in mapping if key not in clip_state)
            if missing_block:
                raise ValueError(
                    f"CLIP checkpoint has no compatible visual block {index}; "
                    f"missing key: {missing_block[0]}"
                )
            state.update(
                {
                    destination_key: clip_state[source_key]
                    for source_key, destination_key in mapping.items()
                }
            )
        return state

    def _google_vit_npz_state(self, path: Path) -> dict[str, torch.Tensor]:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - numpy is a project dependency
            raise RuntimeError("Loading a Google ViT .npz requires numpy.") from exc

        def tensor(array: object) -> torch.Tensor:
            return torch.from_numpy(np.asarray(array).copy())

        state: dict[str, torch.Tensor] = {}
        with np.load(path, allow_pickle=False) as weights:
            required = {
                "embedding/kernel",
                "embedding/bias",
                "cls",
                "Transformer/posembed_input/pos_embedding",
                "Transformer/encoder_norm/scale",
                "Transformer/encoder_norm/bias",
            }
            absent = sorted(required - set(weights.files))
            if absent:
                raise ValueError(
                    f"{path} is not an official Google ViT checkpoint; "
                    f"missing keys: {', '.join(absent)}"
                )

            patch_weight = tensor(weights["embedding/kernel"]).permute(3, 2, 0, 1)
            target_channels = self.patch_embed.in_channels
            if patch_weight.shape[1] == 3 and target_channels == 1:
                patch_weight = patch_weight.mean(dim=1, keepdim=True)
            state["patch_embed.weight"] = patch_weight
            state["patch_embed.bias"] = tensor(weights["embedding/bias"])
            state["cls_token"] = tensor(weights["cls"])

            position = tensor(weights["Transformer/posembed_input/pos_embedding"])
            state["position_embedding"] = self._resize_position_embedding(
                position,
                self.position_embedding.shape[1],
            )

            for index in range(self.depth):
                source = f"Transformer/encoderblock_{index}"
                destination = f"blocks.{index}"
                block_keys = {
                    f"{source}/LayerNorm_0/scale",
                    f"{source}/LayerNorm_0/bias",
                    f"{source}/LayerNorm_2/scale",
                    f"{source}/LayerNorm_2/bias",
                    f"{source}/MlpBlock_3/Dense_0/kernel",
                    f"{source}/MlpBlock_3/Dense_0/bias",
                    f"{source}/MlpBlock_3/Dense_1/kernel",
                    f"{source}/MlpBlock_3/Dense_1/bias",
                }
                for part in ("query", "key", "value", "out"):
                    block_keys.add(
                        f"{source}/MultiHeadDotProductAttention_1/{part}/kernel"
                    )
                    block_keys.add(
                        f"{source}/MultiHeadDotProductAttention_1/{part}/bias"
                    )
                absent = sorted(block_keys - set(weights.files))
                if absent:
                    raise ValueError(
                        f"Google ViT checkpoint has no compatible encoder block {index}; "
                        f"missing key: {absent[0]}"
                    )

                state[f"{destination}.norm1.weight"] = tensor(
                    weights[f"{source}/LayerNorm_0/scale"]
                )
                state[f"{destination}.norm1.bias"] = tensor(
                    weights[f"{source}/LayerNorm_0/bias"]
                )
                state[f"{destination}.norm2.weight"] = tensor(
                    weights[f"{source}/LayerNorm_2/scale"]
                )
                state[f"{destination}.norm2.bias"] = tensor(
                    weights[f"{source}/LayerNorm_2/bias"]
                )

                attention = f"{source}/MultiHeadDotProductAttention_1"
                qkv_weights = []
                qkv_biases = []
                for part in ("query", "key", "value"):
                    kernel = tensor(weights[f"{attention}/{part}/kernel"])
                    qkv_weights.append(kernel.reshape(kernel.shape[0], -1).t())
                    qkv_biases.append(tensor(weights[f"{attention}/{part}/bias"]).reshape(-1))
                state[f"{destination}.attention.in_proj_weight"] = torch.cat(
                    qkv_weights, dim=0
                )
                state[f"{destination}.attention.in_proj_bias"] = torch.cat(
                    qkv_biases, dim=0
                )
                out_kernel = tensor(weights[f"{attention}/out/kernel"])
                state[f"{destination}.attention.out_proj.weight"] = out_kernel.reshape(
                    -1, out_kernel.shape[-1]
                ).t()
                state[f"{destination}.attention.out_proj.bias"] = tensor(
                    weights[f"{attention}/out/bias"]
                )

                state[f"{destination}.mlp.0.weight"] = tensor(
                    weights[f"{source}/MlpBlock_3/Dense_0/kernel"]
                ).t()
                state[f"{destination}.mlp.0.bias"] = tensor(
                    weights[f"{source}/MlpBlock_3/Dense_0/bias"]
                )
                state[f"{destination}.mlp.3.weight"] = tensor(
                    weights[f"{source}/MlpBlock_3/Dense_1/kernel"]
                ).t()
                state[f"{destination}.mlp.3.bias"] = tensor(
                    weights[f"{source}/MlpBlock_3/Dense_1/bias"]
                )

            state["norm.weight"] = tensor(weights["Transformer/encoder_norm/scale"])
            state["norm.bias"] = tensor(weights["Transformer/encoder_norm/bias"])
        return state

    @staticmethod
    def _resize_position_embedding(
        position: torch.Tensor,
        target_tokens: int,
    ) -> torch.Tensor:
        if position.shape[1] == target_tokens:
            return position
        source_grid = int(math.sqrt(position.shape[1] - 1))
        target_grid = int(math.sqrt(target_tokens - 1))
        if source_grid * source_grid != position.shape[1] - 1:
            raise ValueError("Source ViT position embedding does not use a square patch grid.")
        if target_grid * target_grid != target_tokens - 1:
            raise ValueError("Target ViT position embedding does not use a square patch grid.")
        cls_position = position[:, :1]
        patch_position = position[:, 1:].reshape(
            1, source_grid, source_grid, position.shape[-1]
        )
        position_dtype = patch_position.dtype
        patch_position = patch_position.permute(0, 3, 1, 2).float()
        patch_position = F.interpolate(
            patch_position,
            size=(target_grid, target_grid),
            mode="bicubic",
            align_corners=False,
        )
        patch_position = patch_position.to(position_dtype).permute(0, 2, 3, 1).reshape(
            1, target_tokens - 1, position.shape[-1]
        )
        return torch.cat((cls_position, patch_position), dim=1)

    def freeze_backbone(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((cls_token, patches), dim=1) + self.position_embedding
        return self.input_norm(tokens)

    def extract_layer_features(
        self,
        inputs: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Return the sequence-mean feature at an intermediate visual block.

        MoAFCL attaches its feature-aware prompt adapter to an intermediate
        CLIP visual block and averages that block's token/channel sequence.
        Exposing this explicitly avoids hooks and retains autograd when needed.
        """
        layer = int(layer)
        if not 0 <= layer < self.depth:
            raise ValueError(f"layer must be in [0, {self.depth - 1}], got {layer}.")
        tokens = self.embed(inputs)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index == layer:
                return tokens.mean(dim=1)
        raise RuntimeError("unreachable")

    def encode(
        self,
        inputs: torch.Tensor,
        input_prompts: torch.Tensor | None = None,
        layer_prompts: Mapping[int, torch.Tensor] | None = None,
        prefixes: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        adapters: Sequence[nn.Module | None] | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        tokens = self.embed(inputs)
        if input_prompts is not None:
            tokens = torch.cat((tokens[:, :1], input_prompts, tokens[:, 1:]), dim=1)
        intermediates: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            prefix = prefixes[index] if prefixes is not None and index < len(prefixes) else None
            adapter = adapters[index] if adapters is not None and index < len(adapters) else None
            prompt = None if layer_prompts is None else layer_prompts.get(index)
            if prompt is not None:
                if prompt.ndim != 3 or prompt.shape[0] != inputs.shape[0]:
                    raise ValueError(
                        "Each layer prompt must have shape [batch, length, embed_dim]."
                    )
                prompt_length = int(prompt.shape[1])
                prompted_tokens = torch.cat((tokens[:, :1], prompt, tokens[:, 1:]), dim=1)
                prompted_tokens = block(prompted_tokens, prefix=prefix, adapter=adapter)
                # Prompt-tuning inserts the prompt into Q/K/V for this block only.
                # Removing its output keeps token positions stable for the next layer.
                tokens = torch.cat(
                    (prompted_tokens[:, :1], prompted_tokens[:, 1 + prompt_length :]),
                    dim=1,
                )
            else:
                tokens = block(tokens, prefix=prefix, adapter=adapter)
            intermediates.append(self.norm(tokens[:, 0]))
        features = self.norm(tokens[:, 0])
        if return_intermediates:
            return features, intermediates
        return features

    def forward(
        self,
        inputs: torch.Tensor,
        input_prompts: torch.Tensor | None = None,
        layer_prompts: Mapping[int, torch.Tensor] | None = None,
        prefixes: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        adapters: Sequence[nn.Module | None] | None = None,
    ) -> torch.Tensor:
        features = self.encode(
            inputs,
            input_prompts=input_prompts,
            layer_prompts=layer_prompts,
            prefixes=prefixes,
            adapters=adapters,
        )
        assert isinstance(features, torch.Tensor)
        return self.classifier(features)


def clone_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def load_partial_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    module.load_state_dict(dict(state), strict=False)
