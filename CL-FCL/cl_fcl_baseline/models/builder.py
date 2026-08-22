from __future__ import annotations

import argparse
from typing import Sequence

from torch import nn

from .loci_models import SixCNN, build_extended_image_model
from .model import ResNet18, ResNet20, ResNet32, VGG11
from .simple_model import MLPClassifier, SimpleCNN
from .vit import ViTBase, ViTBasePatch16, ViTSmall, ViTTiny


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
    "ViTBasePatch16",
    "SixCNN",
    "MobileNetV2",
    "DenseNet121",
    "WideResNet50",
    "LociResNet18",
    "SixLayerViT",
    "TinyPiT",
)

_MODEL_ALIASES = {
    "mlp": "mlp",
    "simplecnn": "simplecnn",
    "vgg11": "VGG11",
    "resnet": "ResNet18",
    "resnet18": "ResNet18",
    "resnet20": "ResNet20",
    "resnet32": "ResNet32",
    "vittiny": "ViTTiny",
    "vitsmall": "ViTSmall",
    "vitbase": "ViTBase",
    "vitbasepatch16": "ViTBasePatch16",
    "vitb16": "ViTBasePatch16",
    "vitbasepatch16224": "ViTBasePatch16",
    "vitbasepatch16224in21k": "ViTBasePatch16",
    "sixcnn": "SixCNN",
    "6layercnn": "SixCNN",
    "6layerscnn": "SixCNN",
    "mobilenet": "MobileNetV2",
    "mobilenetv2": "MobileNetV2",
    "mobinet": "MobileNetV2",
    "densenet": "DenseNet121",
    "densenet121": "DenseNet121",
    "wideresnet": "WideResNet50",
    "wideresnet50": "WideResNet50",
    "wideresnet502": "WideResNet50",
    "lociresnet18": "LociResNet18",
    "vit": "SixLayerViT",
    "pinyvit": "SixLayerViT",
    "sixlayervit": "SixLayerViT",
    "locisixlayervit": "SixLayerViT",
    "pit": "TinyPiT",
    "tinypit": "TinyPiT",
    "reptailtinypit": "TinyPiT",
}


def normalize_model_name(model_name: str) -> str:
    key = "".join(character for character in str(model_name).lower() if character.isalnum())
    if key not in _MODEL_ALIASES:
        raise ValueError(f"Unsupported model: {model_name}")
    return _MODEL_ALIASES[key]


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
    key = normalize_model_name(model_name).lower()
    if key == "mlp":
        return MLPClassifier(
            input_shape=normalized_input_shape,
            hidden_dim=int(hidden_dim),
            num_classes=int(num_classes),
        )
    if key == "simplecnn":
        return SimpleCNN(
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
        )
    if key == "vgg11":
        return VGG11(input_channels=input_channels, num_classes=int(num_classes))
    if key in {"resnet", "resnet18"}:
        return ResNet18(input_channels=input_channels, num_classes=int(num_classes))
    if key == "resnet20":
        return ResNet20(input_channels=input_channels, num_classes=int(num_classes))
    if key == "resnet32":
        return ResNet32(input_channels=input_channels, num_classes=int(num_classes))
    if key == "vittiny":
        return ViTTiny(
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
            patch_size=int(vit_patch_size),
            mlp_ratio=float(vit_mlp_ratio),
            dropout=float(vit_dropout),
            attention_dropout=float(vit_attention_dropout),
        )
    if key == "vitsmall":
        return ViTSmall(
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
            patch_size=int(vit_patch_size),
            mlp_ratio=float(vit_mlp_ratio),
            dropout=float(vit_dropout),
            attention_dropout=float(vit_attention_dropout),
        )
    if key == "vitbase":
        return ViTBase(
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
            patch_size=int(vit_patch_size),
            mlp_ratio=float(vit_mlp_ratio),
            dropout=float(vit_dropout),
            attention_dropout=float(vit_attention_dropout),
        )
    if key == "vitbasepatch16":
        return ViTBasePatch16(
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
            mlp_ratio=float(vit_mlp_ratio),
            dropout=float(vit_dropout),
            attention_dropout=float(vit_attention_dropout),
        )
    if key == "sixcnn":
        return SixCNN(input_shape=normalized_input_shape, num_classes=int(num_classes))
    if key in {
        "mobilenetv2",
        "densenet121",
        "wideresnet50",
        "tinypit",
        "lociresnet18",
        "sixlayervit",
    }:
        return build_extended_image_model(
            model_name=key,
            input_shape=normalized_input_shape,
            num_classes=int(num_classes),
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
