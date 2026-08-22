from __future__ import annotations

from pathlib import Path
from bisect import bisect_right

import torch
from torch.utils.data import Dataset, Subset

from .cifar_datasets import build_cifar_pickle_dataset
from .loci_datasets import IMAGE_DATASET_ALIASES, build_image_dataset, image_dataset_info


DATASET_NAMES = (
    "mnist",
    "fashionmnist",
    "svhn",
    "cifar10",
    "cifar100",
    "five_datasets",
    "miniimagenet",
    "tinyimagenet",
    "fc100",
    "core50",
    "imagenet",
    "imagenet100",
    "imagenetr",
    "cub200",
    "domainnet",
    "domainnetsub",
    "officehome",
    "adaptiope",
    "pacs",
    "flowers102",
    "oxfordpets",
    "food101",
    "caltech101",
    "dtd",
    "notmnist",
)


_DATASET_ALIASES = {
    "fashion-mnist": "fashionmnist",
    "fashion_mnist": "fashionmnist",
    "5datasets": "five_datasets",
    "five-datasets": "five_datasets",
    "fivedatasets": "five_datasets",
}


def normalize_dataset_name(name: str) -> str:
    key = str(name).lower()
    key = _DATASET_ALIASES.get(key, key)
    if key in DATASET_NAMES:
        return key
    if key in IMAGE_DATASET_ALIASES:
        return IMAGE_DATASET_ALIASES[key]
    raise ValueError(f"Unknown dataset: {name}")


def dataset_info(name: str) -> tuple[tuple[int, int, int], int]:
    key = normalize_dataset_name(name)
    if key == "mnist":
        return (1, 28, 28), 10
    if key == "fashionmnist":
        return (1, 28, 28), 10
    if key == "svhn":
        return (3, 32, 32), 10
    if key == "cifar10":
        return (3, 32, 32), 10
    if key == "cifar100":
        return (3, 32, 32), 100
    if key == "five_datasets":
        return (3, 224, 224), 50
    if key in IMAGE_DATASET_ALIASES:
        return image_dataset_info(key)
    raise ValueError(f"Unknown dataset: {name}")


def _targets(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        parent = _targets(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    for attribute in ("targets", "labels"):
        if hasattr(dataset, attribute):
            values = getattr(dataset, attribute)
            if isinstance(values, torch.Tensor):
                values = values.tolist()
            return [int(value) for value in values]
    raise ValueError(f"Dataset {type(dataset).__name__} does not expose targets or labels.")


class _FiveDatasetsBenchmark(Dataset):
    """Five-Datasets as five disjoint ten-class tasks (labels 0..49)."""

    def __init__(self, datasets: list[Dataset], image_size: int) -> None:
        self.datasets = list(datasets)
        self.image_size = int(image_size)
        self.cumulative_sizes: list[int] = []
        self.targets: list[int] = []
        total = 0
        for task_index, dataset in enumerate(self.datasets):
            total += len(dataset)
            self.cumulative_sizes.append(total)
            self.targets.extend(target + 10 * task_index for target in _targets(dataset))

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, index: int):
        dataset_index = bisect_right(self.cumulative_sizes, int(index))
        start = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        sample, target = self.datasets[dataset_index][int(index) - start]
        if not isinstance(sample, torch.Tensor):
            raise TypeError("Five-Datasets components must return tensor images.")
        if sample.shape[0] == 1:
            sample = sample.repeat(3, 1, 1)
        if tuple(sample.shape[-2:]) != (self.image_size, self.image_size):
            sample = torch.nn.functional.interpolate(
                sample.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        return sample, torch.tensor(int(target) + 10 * dataset_index, dtype=torch.long)


def build_torchvision_dataset(
    name: str,
    train: bool,
    data_dir: str | Path,
    num_samples: int | None = None,
    seed: int = 0,
    image_size: int = 32,
    download: bool = True,
    normalization: str = "dataset",
) -> Dataset:
    key = normalize_dataset_name(name)
    if key in IMAGE_DATASET_ALIASES:
        return build_image_dataset(
            name=key,
            train=train,
            data_dir=data_dir,
            num_samples=num_samples,
            seed=seed,
            image_size=image_size,
            normalization=normalization,
        )
    if key in {"cifar10", "cifar100"}:
        return build_cifar_pickle_dataset(
            name=key,
            train=train,
            data_dir=data_dir,
            num_samples=num_samples,
            seed=seed,
            download=download,
            image_size=image_size,
            normalization=normalization,
        )
    try:
        from torchvision import datasets, transforms
    except (ImportError, RuntimeError) as exc:
        raise ModuleNotFoundError(
            "torchvision is required for MNIST/FashionMNIST/SVHN. Install torchvision to use this option."
        ) from exc

    if normalization == "clip" and key in {"mnist", "fashionmnist", "five_datasets"}:
        raise ValueError(
            "CLIP normalization requires RGB inputs and is not supported for "
            f"{key}; use --backbone-source vit for this dataset."
        )
    if normalization not in {"dataset", "clip"}:
        raise ValueError("normalization must be either 'dataset' or 'clip'.")
    if key in {"mnist", "fashionmnist"}:
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST official mean/std.
            ]
        )
        transform = eval_transform
    else:
        mean = (
            (0.48145466, 0.4578275, 0.40821073)
            if normalization == "clip"
            else (0.4377, 0.4438, 0.4728)
        )
        std = (
            (0.26862954, 0.26130258, 0.27577711)
            if normalization == "clip"
            else (0.1980, 0.2010, 0.1970)
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size))),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform = eval_transform

    root = Path(data_dir)
    if key == "mnist":
        dataset: Dataset = datasets.MNIST(
            root=str(root), train=train, download=bool(download), transform=transform
        )
    elif key == "fashionmnist":
        dataset = datasets.FashionMNIST(
            root=str(root), train=train, download=bool(download), transform=transform
        )
    elif key == "svhn":
        dataset = datasets.SVHN(
            root=str(root),
            split="train" if train else "test",
            download=bool(download),
            transform=transform,
        )
    elif key == "five_datasets":
        components = [
            build_torchvision_dataset(
                component,
                train=train,
                data_dir=root,
                num_samples=None,
                seed=seed,
                image_size=image_size,
                download=download,
                normalization=normalization,
            )
            for component in ("mnist", "fashionmnist", "notmnist", "cifar10", "svhn")
        ]
        dataset = _FiveDatasetsBenchmark(components, image_size=image_size)
    else:
        raise ValueError(f"Unsupported torchvision dataset: {name}")

    # iCaRL stores raw exemplar images. Replay uses the training transform,
    # while herding/NME use this deterministic view of the same image.
    setattr(dataset, "_cl_eval_transform", eval_transform)

    if num_samples is None or num_samples <= 0 or num_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[: int(num_samples)].tolist()
    return Subset(dataset, indices)
