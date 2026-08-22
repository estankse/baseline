from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


_CIFAR_ARCHIVES = {
    "cifar10": (
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "cifar-10-python.tar.gz",
        "cifar-10-batches-py",
    ),
    "cifar100": (
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "cifar-100-python.tar.gz",
        "cifar-100-python",
    ),
}


class _CifarTransform:
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        train: bool,
        image_size: int = 32,
    ) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.train = bool(train)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")

    def __call__(self, image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
            if array.shape[0] == 3:
                array = array.transpose(1, 2, 0)
        else:
            array = np.asarray(image)
        array = np.asarray(array, dtype=np.uint8)
        if self.train:
            array = np.pad(array, ((4, 4), (4, 4), (0, 0)), mode="reflect")
            top = int(torch.randint(9, (1,)).item())
            left = int(torch.randint(9, (1,)).item())
            array = array[top : top + 32, left : left + 32]
            if bool(torch.rand(()) < 0.5):
                array = np.flip(array, axis=1)
        if self.image_size != 32:
            array = np.asarray(
                Image.fromarray(np.asarray(array, dtype=np.uint8)).resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.BICUBIC,
                )
            )
        writable = np.array(array, dtype=np.uint8, order="C", copy=True)
        tensor = torch.from_numpy(writable).permute(2, 0, 1).float()
        tensor = tensor / 255.0
        return (tensor - self.mean) / self.std


class CifarPickleDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: Path, name: str, train: bool, transform) -> None:
        self.root = Path(root)
        self.name = str(name).lower()
        self.train = bool(train)
        self.transform = transform
        filenames = self._filenames()
        data_parts: list[np.ndarray] = []
        targets: list[int] = []
        for filename in filenames:
            with (self.root / filename).open("rb") as handle:
                batch = pickle.load(handle, encoding="latin1")
            data_parts.append(np.asarray(batch["data"], dtype=np.uint8))
            label_key = "fine_labels" if self.name == "cifar100" else "labels"
            targets.extend(int(value) for value in batch[label_key])
        flat_data = np.concatenate(data_parts, axis=0)
        self.data = flat_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = targets

    def _filenames(self) -> list[str]:
        if self.name == "cifar100":
            return ["train" if self.train else "test"]
        if self.name == "cifar10":
            if self.train:
                return [f"data_batch_{index}" for index in range(1, 6)]
            return ["test_batch"]
        raise ValueError(f"Unsupported CIFAR dataset: {self.name}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.fromarray(self.data[int(index)])
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.targets[int(index)], dtype=torch.long)


def _find_extracted_root(data_dir: Path, directory_name: str) -> Path | None:
    candidates = (
        data_dir / directory_name,
        data_dir / "cifar10" / directory_name,
        data_dir / "cifar100" / directory_name,
        data_dir / "CIFAR10" / directory_name,
        data_dir / "CIFAR100" / directory_name,
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination != member_path and destination not in member_path.parents:
                raise ValueError(f"Unsafe path in CIFAR archive: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in CIFAR archive: {member.name}")
        archive.extractall(destination)


def _download_cifar(data_dir: Path, name: str) -> Path:
    url, archive_name, directory_name = _CIFAR_ARCHIVES[name]
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / archive_name
    urllib.request.urlretrieve(url, archive_path)
    try:
        _safe_extract(archive_path, data_dir)
    finally:
        archive_path.unlink(missing_ok=True)
    extracted_root = data_dir / directory_name
    if not extracted_root.is_dir():
        raise RuntimeError(f"CIFAR archive did not create the expected directory: {extracted_root}")
    return extracted_root


def build_cifar_pickle_dataset(
    name: str,
    train: bool,
    data_dir: str | Path,
    num_samples: int | None = None,
    seed: int = 0,
    download: bool = True,
    image_size: int = 32,
    normalization: str = "dataset",
) -> Dataset:
    key = str(name).lower()
    if key not in _CIFAR_ARCHIVES:
        raise ValueError(f"Unsupported CIFAR dataset: {name}")
    _url, _archive_name, directory_name = _CIFAR_ARCHIVES[key]
    root = _find_extracted_root(Path(data_dir).expanduser(), directory_name)
    if root is None:
        if not download:
            raise FileNotFoundError(
                f"Could not find {directory_name} below {data_dir}; enable --download or extract it there."
            )
        root = _download_cifar(Path(data_dir).expanduser(), key)

    if normalization == "clip":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    elif normalization == "dataset" and key == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif normalization == "dataset":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError("normalization must be either 'dataset' or 'clip'.")
    eval_transform = _CifarTransform(mean, std, train=False, image_size=image_size)
    dataset: Dataset = CifarPickleDataset(
        root=root,
        name=key,
        train=bool(train),
        transform=(
            _CifarTransform(mean, std, train=True, image_size=image_size)
            if train
            else eval_transform
        ),
    )
    setattr(dataset, "_cl_eval_transform", eval_transform)
    if num_samples is None or int(num_samples) <= 0 or int(num_samples) >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[: int(num_samples)].tolist()
    return Subset(dataset, indices)


__all__ = ["CifarPickleDataset", "build_cifar_pickle_dataset"]
