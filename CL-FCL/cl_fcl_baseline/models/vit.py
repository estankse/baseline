from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class ViTVariantSpec:
    embed_dim: int
    depth: int
    num_heads: int


VIT_VARIANTS = {
    "ViTTiny": ViTVariantSpec(embed_dim=192, depth=12, num_heads=3),
    "ViTSmall": ViTVariantSpec(embed_dim=384, depth=12, num_heads=6),
    "ViTBase": ViTVariantSpec(embed_dim=768, depth=12, num_heads=12),
    "ViTBasePatch16": ViTVariantSpec(embed_dim=768, depth=12, num_heads=12),
}


class _PatchEmbedding(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_size: int,
        image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        height, width = image_size
        if patch_size <= 0:
            raise ValueError("patch_size must be positive for ViT backbones.")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"ViT patch_size={patch_size} must divide the input spatial size {(height, width)}."
            )
        self.patch_size = int(patch_size)
        self.grid_size = (height // self.patch_size, width // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            int(input_channels),
            int(embed_dim),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(inputs)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens


class _TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = max(int(embed_dim * mlp_ratio), embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        tokens = self.norm1(inputs)
        tokens, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = residual + self.dropout1(tokens)

        residual = tokens
        tokens = self.norm2(tokens)
        tokens = residual + self.mlp(tokens)
        return tokens


class _ViTFeatures(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_channels, height, width = (int(dim) for dim in input_shape)
        self.patch_embed = _PatchEmbedding(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size=(height, width),
        )
        num_tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(inputs)
        cls_token = self.cls_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = tokens + self.position_embedding
        tokens = self.dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int] = (3, 32, 32),
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.features = _ViTFeatures(
            input_shape=input_shape,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.classifier = nn.Linear(embed_dim, int(num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


class ViTTiny(VisionTransformer):
    def __init__(
        self,
        input_shape: Sequence[int] = (3, 32, 32),
        num_classes: int = 10,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        spec = VIT_VARIANTS["ViTTiny"]
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            embed_dim=spec.embed_dim,
            depth=spec.depth,
            num_heads=spec.num_heads,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )


class ViTSmall(VisionTransformer):
    def __init__(
        self,
        input_shape: Sequence[int] = (3, 32, 32),
        num_classes: int = 10,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        spec = VIT_VARIANTS["ViTSmall"]
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            embed_dim=spec.embed_dim,
            depth=spec.depth,
            num_heads=spec.num_heads,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )


class ViTBase(VisionTransformer):
    def __init__(
        self,
        input_shape: Sequence[int] = (3, 32, 32),
        num_classes: int = 10,
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        spec = VIT_VARIANTS["ViTBase"]
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            embed_dim=spec.embed_dim,
            depth=spec.depth,
            num_heads=spec.num_heads,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )


class ViTBasePatch16(ViTBase):
    """ViT-B/16 architecture used by the prompt/CLIP-based FCL papers."""

    def __init__(
        self,
        input_shape: Sequence[int] = (3, 224, 224),
        num_classes: int = 100,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        del patch_size
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            patch_size=16,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
