from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int] = (1, 28, 28),
        hidden_dim: int = 200,
        num_classes: int = 10,
        input_dim: Optional[int] = None,
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if input_dim is None:
            computed_dim = 1
            for dim in input_shape:
                computed_dim *= int(dim)
            input_dim = computed_dim

        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_layers = [int(hidden_dim)]
        else:
            hidden_layers = [int(dim) for dim in hidden_dims]

        layers = [nn.Flatten(), nn.Linear(int(input_dim), hidden_layers[0]), nn.ReLU()]
        for prev_dim, next_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], int(num_classes)))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        # features = torch.flatten(features, start_dim=1)
        return self.classifier(features)
