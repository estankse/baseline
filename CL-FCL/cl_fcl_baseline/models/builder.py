from __future__ import annotations

import argparse
from typing import Sequence

from torch import nn

from .model import ResNet18, ResNet20, ResNet32, VGG11
from .simple_model import MLPClassifier, SimpleCNN
from .vit import ViTBase, ViTSmall, ViTTiny


DEFAULT_VIT_PATCH_SIZE = 4
DEFAULT_VIT_DROPOUT = 0.0
DEFAULT_VIT_ATTENTION_DROPOUT = 0.0
DEFAULT_VIT_MLP_RATIO = 4.0

MODEL_NAMES = (
    "mlp",
    "simplecnn",
    "VGG11",
    "ResNet18",
    "ResNet20",
    "ResNet32",
    "ViTTiny",
    "ViTSmall",
    "ViTBase",
)


def build_model(
    model_name: str,
    input_shape: Sequence[int],
    num_classes: int,
    hidden_dim: int = 200,
    vit_patch_size: int = DEFAULT_VIT_PATCH_SIZE,
    vit_dropout: float = DEFAULT_VIT_DROPOUT,
    vit_attention_dropout: float = DEFAULT_VIT_ATTENTION_DROPOUT,
    vit_mlp_ratio: float = DEFAULT_VIT_MLP_RATIO,
) -> nn.Module:
    normalized_input_shape = tuple(int(dim) for dim in input_shape)
    input_channels = int(normalized_input_shape[0])
    if model_name == "mlp":
        return MLPClassifier(
            input_shape=normalized_input_shape,
            hidden_dim=int(hidden_dim),
            num_classes=int(num_classes),
        )
    if model_name == "simplecnn":
        return SimpleCNN(
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
        )
    if model_name == "VGG11":
        return VGG11(input_channels=input_channels, num_classes=int(num_classes))
    if model_name == "ResNet18":
        return ResNet18(input_channels=input_channels, num_classes=int(num_classes))
    if model_name == "ResNet20":
        return ResNet20(input_channels=input_channels, num_classes=int(num_classes))
    if model_name == "ResNet32":
        return ResNet32(input_channels=input_channels, num_classes=int(num_classes))
    if model_name == "ViTTiny":
        return ViTTiny(
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
            patch_size=int(vit_patch_size),
            mlp_ratio=float(vit_mlp_ratio),
            dropout=float(vit_dropout),
            attention_dropout=float(vit_attention_dropout),
        )
    if model_name == "ViTSmall":
        return ViTSmall(
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
            patch_size=int(vit_patch_size),
            mlp_ratio=float(vit_mlp_ratio),
            dropout=float(vit_dropout),
            attention_dropout=float(vit_attention_dropout),
        )
    if model_name == "ViTBase":
        return ViTBase(
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
            patch_size=int(vit_patch_size),
            mlp_ratio=float(vit_mlp_ratio),
            dropout=float(vit_dropout),
            attention_dropout=float(vit_attention_dropout),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_model_from_args(
    args: argparse.Namespace,
    input_shape: Sequence[int],
    num_classes: int,
) -> nn.Module:
    return build_model(
        model_name=str(args.model),
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_dim=int(getattr(args, "hidden_dim", 200)),
        vit_patch_size=int(getattr(args, "vit_patch_size", DEFAULT_VIT_PATCH_SIZE)),
        vit_dropout=float(getattr(args, "vit_dropout", DEFAULT_VIT_DROPOUT)),
        vit_attention_dropout=float(
            getattr(args, "vit_attention_dropout", DEFAULT_VIT_ATTENTION_DROPOUT)
        ),
        vit_mlp_ratio=float(getattr(args, "vit_mlp_ratio", DEFAULT_VIT_MLP_RATIO)),
    )

