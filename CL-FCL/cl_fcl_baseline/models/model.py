from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn


class VGG11(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        features = self.avgpool(features)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


@dataclass
class _ResNetConfig:
    depth: int
    num_classes: int
    input_channels: int


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(inputs)
        out = out + identity
        out = self.relu(out)
        return out


class _ResNetCIFAR(nn.Module):
    def __init__(self, config: _ResNetConfig) -> None:
        super().__init__()
        if (config.depth - 2) % 6 != 0:
            raise ValueError("ResNet depth should be 6n+2 for CIFAR.")
        num_blocks = (config.depth - 2) // 6

        self.in_channels = 16
        self.conv1 = _conv3x3(config.input_channels, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, num_blocks, stride=1)
        self.layer2 = self._make_layer(32, num_blocks, stride=2)
        self.layer3 = self._make_layer(64, num_blocks, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, config.num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(_BasicBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class ResNet20(_ResNetCIFAR):
    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__(_ResNetConfig(depth=20, num_classes=num_classes, input_channels=input_channels))


class ResNet32(_ResNetCIFAR):
    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__(_ResNetConfig(depth=32, num_classes=num_classes, input_channels=input_channels))

class ResNet44(_ResNetCIFAR):
    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__(_ResNetConfig(depth=44, num_classes=num_classes, input_channels=input_channels))


class _ResNet18(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = _conv3x3(input_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(_BasicBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class ResNet18(_ResNet18):
    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__(input_channels=input_channels, num_classes=num_classes)
