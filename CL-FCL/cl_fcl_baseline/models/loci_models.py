from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .vit import VisionTransformer


class SixCNN(nn.Module):
    """The compact six-convolution network used as Loci's CV KD model."""

    def __init__(self, input_shape: Sequence[int], num_classes: int = 100) -> None:
        super().__init__()
        channels, height, width = (int(value) for value in input_shape)
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Dropout(0.25),
        )
        with torch.no_grad():
            output = self.features(torch.zeros(1, channels, height, width))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output.numel(), 1024, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(1024, int(num_classes), bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fc(self.features(inputs)))


class _ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class _InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expansion: int) -> None:
        super().__init__()
        hidden_channels = int(round(in_channels * expansion))
        self.use_residual = stride == 1 and in_channels == out_channels
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.append(_ConvBNReLU(in_channels, hidden_channels, kernel_size=1))
        layers.extend(
            [
                _ConvBNReLU(
                    hidden_channels,
                    hidden_channels,
                    stride=stride,
                    groups=hidden_channels,
                ),
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.block(inputs)
        return inputs + output if self.use_residual else output


class LociMobileNetV2(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        settings = (
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        )
        current_channels = 32
        layers: list[nn.Module] = [_ConvBNReLU(input_channels, current_channels, stride=2)]
        for expansion, output_channels, repeats, first_stride in settings:
            for index in range(repeats):
                stride = first_stride if index == 0 else 1
                layers.append(
                    _InvertedResidual(current_channels, output_channels, stride, expansion)
                )
                current_channels = output_channels
        layers.append(_ConvBNReLU(current_channels, 1280, kernel_size=1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, int(num_classes)))
        _initialize(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.features(inputs)
        output = nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
        return self.classifier(output)


class _DenseLayer(nn.Module):
    def __init__(self, input_features: int, growth_rate: int, bn_size: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(input_features)
        self.conv1 = nn.Conv2d(input_features, bn_size * growth_rate, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, previous_features: Sequence[torch.Tensor]) -> torch.Tensor:
        inputs = torch.cat(list(previous_features), dim=1)
        output = self.conv1(nn.functional.relu(self.norm1(inputs), inplace=True))
        return self.conv2(nn.functional.relu(self.norm2(output), inplace=True))


class _DenseBlock(nn.Module):
    def __init__(self, layers: int, input_features: int, growth_rate: int, bn_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DenseLayer(input_features + index * growth_rate, growth_rate, bn_size)
                for index in range(layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = [inputs]
        for layer in self.layers:
            features.append(layer(features))
        return torch.cat(features, dim=1)


class _DenseTransition(nn.Sequential):
    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__(
            nn.BatchNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_features, output_features, 1, bias=False),
            nn.AvgPool2d(2, stride=2),
        )


class LociDenseNet121(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        growth_rate = 32
        features = 64
        modules: list[nn.Module] = [
            nn.Conv2d(input_channels, features, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        ]
        for block_index, layer_count in enumerate((6, 12, 24, 16)):
            modules.append(_DenseBlock(layer_count, features, growth_rate, bn_size=4))
            features += layer_count * growth_rate
            if block_index != 3:
                modules.append(_DenseTransition(features, features // 2))
                features //= 2
        modules.append(nn.BatchNorm2d(features))
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(features, int(num_classes))
        _initialize(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = nn.functional.relu(self.features(inputs), inplace=True)
        output = nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
        return self.classifier(output)


class _WideBottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        width = channels * 2
        self.conv1 = nn.Conv2d(input_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or input_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * self.expansion),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs if self.downsample is None else self.downsample(inputs)
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        return self.relu(output + identity)


class LociWideResNet50(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.current_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        self.classifier = nn.Linear(512 * _WideBottleneck.expansion, int(num_classes))
        _initialize(self)

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [_WideBottleneck(self.current_channels, channels, stride=stride)]
        self.current_channels = channels * _WideBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_WideBottleneck(self.current_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(inputs)))))
        output = nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
        return self.classifier(output)


class _LociBasicBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs if self.downsample is None else self.downsample(inputs)
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + identity)


class LociResNet18(nn.Module):
    """ResNet-18 with the ImageNet stem copied by the Loci reference code."""

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.classifier = nn.Linear(512, int(num_classes))
        _initialize(self)

    @staticmethod
    def _make_layer(
        input_channels: int,
        output_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [_LociBasicBlock(input_channels, output_channels, stride=stride)]
        layers.extend(_LociBasicBlock(output_channels, output_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(inputs)))))
        output = nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
        return self.classifier(output)


class LociSixLayerViT(VisionTransformer):
    """The 192-wide, six-block, twelve-head ViT from Loci's factory.py."""

    def __init__(self, input_shape: Sequence[int], num_classes: int) -> None:
        super().__init__(
            input_shape=input_shape,
            num_classes=int(num_classes),
            embed_dim=192,
            depth=6,
            num_heads=12,
            patch_size=4,
        )


class _PiTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.norm1(inputs)
        attended, _weights = self.attention(tokens, tokens, tokens, need_weights=False)
        inputs = inputs + attended
        return inputs + self.mlp(self.norm2(inputs))


class _PiTPooling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.spatial_pool = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=input_dim,
        )
        self.cls_pool = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, int, int]:
        cls_token, spatial_tokens = tokens[:, :1], tokens[:, 1:]
        spatial = spatial_tokens.transpose(1, 2).reshape(
            tokens.shape[0], spatial_tokens.shape[2], height, width
        )
        spatial = self.spatial_pool(spatial)
        next_height, next_width = int(spatial.shape[2]), int(spatial.shape[3])
        spatial_tokens = spatial.flatten(2).transpose(1, 2)
        return (
            torch.cat((self.cls_pool(cls_token), spatial_tokens), dim=1),
            next_height,
            next_width,
        )


class LociTinyPiT(nn.Module):
    """Dependency-free PiT-Tiny variant used by the original Loci CV suite."""

    def __init__(self, input_shape: Sequence[int], num_classes: int) -> None:
        super().__init__()
        input_channels, height, width = (int(value) for value in input_shape)
        if height < 16 or width < 16:
            raise ValueError("TinyPiT requires image height and width of at least 16 pixels.")
        self.patch_embed = nn.Conv2d(
            input_channels,
            64,
            kernel_size=16,
            stride=8,
        )
        patch_height = (height - 16) // 8 + 1
        patch_width = (width - 16) // 8 + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_height * patch_width + 1, 64))
        self.stage1 = nn.ModuleList([_PiTBlock(64, 2) for _ in range(2)])
        self.pool1 = _PiTPooling(64, 128)
        self.stage2 = nn.ModuleList([_PiTBlock(128, 4) for _ in range(6)])
        self.pool2 = _PiTPooling(128, 256)
        self.stage3 = nn.ModuleList([_PiTBlock(256, 8) for _ in range(4)])
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, int(num_classes))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        _initialize(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial = self.patch_embed(inputs)
        height, width = int(spatial.shape[2]), int(spatial.shape[3])
        tokens = spatial.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1) + self.position_embedding
        for block in self.stage1:
            tokens = block(tokens)
        tokens, height, width = self.pool1(tokens, height, width)
        for block in self.stage2:
            tokens = block(tokens)
        tokens, height, width = self.pool2(tokens, height, width)
        for block in self.stage3:
            tokens = block(tokens)
        return self.classifier(self.norm(tokens[:, 0]))


def _initialize(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def build_extended_image_model(
    model_name: str,
    input_shape: Sequence[int],
    num_classes: int,
) -> nn.Module:
    """Build the additional image backbones introduced with Loci."""
    key = model_name.lower()
    input_channels = int(input_shape[0])
    if key in {"mobilenet", "mobilenetv2", "mobinet"}:
        return LociMobileNetV2(input_channels, int(num_classes))
    if key in {"densenet", "densenet121"}:
        return LociDenseNet121(input_channels, int(num_classes))
    if key in {"wideresnet", "wideresnet50", "wide_resnet50_2"}:
        return LociWideResNet50(input_channels, int(num_classes))
    if key == "lociresnet18":
        return LociResNet18(input_channels, int(num_classes))
    if key in {"vit", "pinyvit", "sixlayervit", "locisixlayervit"}:
        return LociSixLayerViT(input_shape, int(num_classes))
    if key in {"pit", "tinypit", "reptailtinypit"}:
        return LociTinyPiT(input_shape, int(num_classes))
    raise ValueError(f"Unsupported image model: {model_name}")


build_torchvision_loci_model = build_extended_image_model


__all__ = [
    "SixCNN",
    "LociMobileNetV2",
    "LociDenseNet121",
    "LociWideResNet50",
    "LociResNet18",
    "LociSixLayerViT",
    "LociTinyPiT",
    "build_extended_image_model",
    "build_torchvision_loci_model",
]
