from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset


def dataset_info(name: str) -> Tuple[Tuple[int, int, int], int]:
    key = name.lower()
    if key == "mnist":
        return (1, 28, 28), 10
    if key == "cifar10":
        return (3, 32, 32), 10
    raise ValueError(f"Unknown dataset: {name}")


def build_torchvision_dataset(
    name: str,
    train: bool,
    data_dir: str | Path,
    num_samples: int | None = None,
    seed: int = 0,
) -> Dataset:
    key = name.lower()
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for MNIST/CIFAR datasets. Install torchvision to use this option."
        ) from exc

    if key == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST official mean/std.
        ])
    elif key == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    else:
        raise ValueError(f"Unsupported torchvision dataset: {name}")

    root = Path(data_dir)
    if key == "mnist":
        dataset: Dataset = datasets.MNIST(root=str(root), train=train, download=True, transform=transform)
    elif key == "cifar10":
        dataset = datasets.CIFAR10(root=str(root), train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported torchvision dataset: {name}")

    if num_samples is None or num_samples <= 0 or num_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[: int(num_samples)].tolist()
    return Subset(dataset, indices)
