from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}

IMAGE_DATASET_ALIASES: Mapping[str, str] = {
    "miniimagenet": "miniimagenet",
    "mini-imagenet": "miniimagenet",
    "miniimagenet100": "miniimagenet",
    "tinyimagenet": "tinyimagenet",
    "tiny-imagenet": "tinyimagenet",
    "tiny-imagenet-200": "tinyimagenet",
    "fc100": "fc100",
    "core50": "core50",
    "core-50": "core50",
    "imagenet": "imagenet",
    "imagenet1k": "imagenet",
    "imagenet100": "imagenet100",
    "imagenet-100": "imagenet100",
    "imagenetr": "imagenetr",
    "imagenet-r": "imagenetr",
    "cub200": "cub200",
    "cub-200": "cub200",
    "cub_200_2011": "cub200",
    "domainnet": "domainnet",
    "domain-net": "domainnet",
    "domainnetsub": "domainnetsub",
    "domainnet-sub": "domainnetsub",
    "officehome": "officehome",
    "office-home": "officehome",
    "adaptiope": "adaptiope",
    "pacs": "pacs",
    "flowers102": "flowers102",
    "flowers-102": "flowers102",
    "oxfordpets": "oxfordpets",
    "oxford-pets": "oxfordpets",
    "oxford-iiit-pet": "oxfordpets",
    "food101": "food101",
    "food-101": "food101",
    "caltech101": "caltech101",
    "caltech-101": "caltech101",
    "dtd": "dtd",
    "notmnist": "notmnist",
    "not-mnist": "notmnist",
}

IMAGE_DATASET_INFO: Mapping[str, tuple[tuple[int, int, int], int]] = {
    "miniimagenet": ((3, 32, 32), 100),
    "tinyimagenet": ((3, 32, 32), 200),
    "fc100": ((3, 32, 32), 100),
    # The reference code treats 50 objects across 11 sessions as 550 labels.
    "core50": ((3, 32, 32), 550),
    "imagenet": ((3, 32, 32), 1000),
    "imagenet100": ((3, 224, 224), 100),
    "imagenetr": ((3, 224, 224), 200),
    "cub200": ((3, 224, 224), 200),
    "domainnet": ((3, 224, 224), 345),
    "domainnetsub": ((3, 224, 224), 100),
    "officehome": ((3, 224, 224), 65),
    "adaptiope": ((3, 224, 224), 123),
    "pacs": ((3, 224, 224), 7),
    "flowers102": ((3, 224, 224), 102),
    "oxfordpets": ((3, 224, 224), 37),
    "food101": ((3, 224, 224), 101),
    "caltech101": ((3, 224, 224), 101),
    "dtd": ((3, 224, 224), 47),
    "notmnist": ((3, 224, 224), 10),
}

_ROOT_NAMES: Mapping[str, Sequence[str]] = {
    "miniimagenet": ("mini-imagenet", "miniimagenet", "MiniImageNet"),
    "tinyimagenet": ("tiny-imagenet-200", "tinyimagenet", "TinyImageNet"),
    "fc100": ("FC100", "fc100"),
    "core50": ("core50", "CORE50"),
    "imagenet": ("imagenet", "ImageNet", "ILSVRC2012"),
    "imagenet100": ("imagenet100", "imagenet-100", "ImageNet100"),
    "imagenetr": ("imagenet-r", "imagenetr", "ImageNet-R"),
    "cub200": ("cub200", "CUB_200_2011", "CUB200"),
    "domainnet": ("domainnet", "domain_net", "DomainNet"),
    "domainnetsub": ("domainnetsub", "domainnet", "DomainNet"),
    "officehome": ("officehome", "OfficeHome", "Office-Home"),
    "adaptiope": ("adaptiope", "Adaptiope"),
    "pacs": ("pacs", "PACS"),
    "flowers102": ("flowers-102", "flowers102", "Flowers102"),
    "oxfordpets": ("oxford-iiit-pet", "oxfordpets", "OxfordPets"),
    "food101": ("food-101", "food101", "Food101"),
    "caltech101": ("caltech-101", "caltech101", "Caltech101"),
    "dtd": ("dtd", "DTD"),
    "notmnist": ("notMNIST", "notmnist", "NotMNIST"),
}


def normalize_image_dataset_name(name: str) -> str:
    key = str(name).lower()
    if key not in IMAGE_DATASET_ALIASES:
        raise ValueError(f"Unsupported image dataset: {name}")
    return IMAGE_DATASET_ALIASES[key]


def image_dataset_info(name: str) -> tuple[tuple[int, int, int], int]:
    return IMAGE_DATASET_INFO[normalize_image_dataset_name(name)]


class ImagePathDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: Iterable[tuple[str | Path, int]],
        transform,
    ) -> None:
        self.samples = [(str(path), int(target)) for path, target in samples]
        self.targets = [target for _path, target in self.samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[int(index)]
        with Image.open(path) as image:
            sample = image.convert("RGB")
            if self.transform is not None:
                sample = self.transform(sample)
        return sample, torch.tensor(target, dtype=torch.long)


class _ImageTransform:
    def __init__(
        self,
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        train: bool,
        random_resized_crop: bool,
    ) -> None:
        self.image_size = int(image_size)
        self.resize_size = int(round(self.image_size * 1.125))
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.train = bool(train)
        self.random_resized_crop = bool(random_resized_crop)

    def _training_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        area = float(width * height)
        for _ in range(10):
            target_area = area * float(torch.empty(()).uniform_(0.08, 1.0).item())
            aspect_ratio = math.exp(
                float(torch.empty(()).uniform_(math.log(3.0 / 4.0), math.log(4.0 / 3.0)).item())
            )
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = int(torch.randint(width - crop_width + 1, (1,)).item())
                top = int(torch.randint(height - crop_height + 1, (1,)).item())
                return image.crop((left, top, left + crop_width, top + crop_height)).resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
        source_ratio = width / height
        if source_ratio < 3.0 / 4.0:
            crop_width, crop_height = width, int(round(width / (3.0 / 4.0)))
        elif source_ratio > 4.0 / 3.0:
            crop_width, crop_height = int(round(height * (4.0 / 3.0))), height
        else:
            crop_width, crop_height = width, height
        left = max(0, (width - crop_width) // 2)
        top = max(0, (height - crop_height) // 2)
        return image.crop((left, top, left + crop_width, top + crop_height)).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.train and self.random_resized_crop:
            image = self._training_crop(image)
        elif self.random_resized_crop:
            image = image.resize((self.resize_size, self.resize_size), Image.Resampling.BILINEAR)
            maximum_offset = max(0, self.resize_size - self.image_size)
            left = maximum_offset // 2
            top = maximum_offset // 2
            image = image.crop((left, top, left + self.image_size, top + self.image_size))
        else:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if self.train and self.random_resized_crop and bool(torch.rand(()) < 0.5):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


def _transforms(
    dataset_name: str,
    train: bool,
    image_size: int,
    normalization: str = "dataset",
):
    if normalization == "clip":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    elif normalization == "dataset" and dataset_name in {"fc100", "core50"}:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif normalization == "dataset":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError("normalization must be either 'dataset' or 'clip'.")
    return _ImageTransform(
        image_size=int(image_size),
        mean=mean,
        std=std,
        train=bool(train),
        random_resized_crop=dataset_name not in {"fc100", "core50"},
    )


def _looks_like_root(root: Path, name: str) -> bool:
    markers = {
        "miniimagenet": ("images", "new_train.csv", "train"),
        "tinyimagenet": ("wnids.txt", "train.csv", "train"),
        "fc100": ("train_csv", "train.csv", "train"),
        "core50": ("task_label", "core50", "train.csv"),
        "imagenet": ("train", "val"),
        "imagenet100": ("train", "val"),
        "imagenetr": ("README.txt", "train", "images"),
        "cub200": ("images.txt", "CUB_200_2011", "images"),
        "domainnet": ("clipart", "real", "splits"),
        "domainnetsub": ("clipart", "real", "splits"),
        "officehome": ("Art", "Clipart", "Product", "Real World"),
        "adaptiope": ("synthetic", "real_life", "product_images"),
        "pacs": ("art_painting", "cartoon", "photo", "sketch"),
        "flowers102": ("jpg", "train", "images"),
        "oxfordpets": ("images", "annotations", "train"),
        "food101": ("images", "meta", "train"),
        "caltech101": ("101_ObjectCategories", "train", "images"),
        "dtd": ("images", "labels", "train"),
        "notmnist": ("train", "test", "A"),
    }
    return any((root / marker).exists() for marker in markers[name])


def _resolve_root(data_dir: str | Path, name: str) -> Path:
    base = Path(data_dir).expanduser()
    candidates = [base, *(base / child for child in _ROOT_NAMES[name])]
    for candidate in candidates:
        if candidate.exists() and _looks_like_root(candidate, name):
            return candidate
    # Some datasets (notably ImageNet-R) are distributed directly as class
    # folders without a manifest.  Prefer named children before treating
    # --data-dir itself as that dataset root.
    for candidate in [*candidates[1:], candidates[0]]:
        if candidate.exists() and any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in candidate.rglob("*")
        ):
            return candidate
    expected = ", ".join(str(base / child) for child in _ROOT_NAMES[name])
    raise FileNotFoundError(
        f"Could not find {name} under {base}. Expected a dataset root at one of: {expected}."
    )


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _folder_samples(
    split_directory: Path,
    class_to_idx: Mapping[str, int] | None = None,
) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    classes = sorted(path.name for path in split_directory.iterdir() if path.is_dir())
    mapping = dict(class_to_idx or {name: index for index, name in enumerate(classes)})
    samples: list[tuple[Path, int]] = []
    for class_name, class_index in mapping.items():
        class_directory = split_directory / class_name
        if class_directory.is_dir():
            samples.extend((path, int(class_index)) for path in _image_files(class_directory))
    return samples, mapping


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_value(row: Mapping[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"CSV row is missing one of the required columns: {', '.join(names)}")


def _label_mapping(root: Path, csv_paths: Sequence[Path]) -> dict[str, int]:
    json_path = root / "classes_name.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            raw_mapping = json.load(handle)
        mapping: dict[str, int] = {}
        for key, value in raw_mapping.items():
            mapped = value[0] if isinstance(value, list) else value
            mapping[str(key)] = int(mapped)
        return mapping
    labels = {
        _row_value(row, ("label", "target", "class", "wnid"))
        for path in csv_paths
        if path.exists()
        for row in _read_csv(path)
    }
    numeric = True
    for label in labels:
        try:
            int(label)
        except ValueError:
            numeric = False
            break
    if numeric:
        return {label: int(label) for label in labels}
    return {label: index for index, label in enumerate(sorted(labels))}


def _csv_samples(
    root: Path,
    csv_path: Path,
    image_bases: Sequence[Path],
    mapping: Mapping[str, int],
) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for row in _read_csv(csv_path):
        relative = Path(_row_value(row, ("filename", "dir", "image", "path")))
        label = _row_value(row, ("label", "target", "class", "wnid"))
        path_candidates = (
            [relative] if relative.is_absolute() else [base / relative for base in image_bases]
        )
        image_path = next((path for path in path_candidates if path.exists()), path_candidates[0])
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in {csv_path} does not exist: {image_path}")
        samples.append((image_path, int(mapping[label])))
    return samples


def _miniimagenet_samples(root: Path, train: bool) -> list[tuple[Path, int]]:
    csv_paths = [root / "new_train.csv", root / "new_val.csv", root / "new_test.csv"]
    csv_path = (
        csv_paths[0]
        if train
        else next((path for path in csv_paths[1:] if path.exists()), csv_paths[1])
    )
    if csv_path.exists():
        mapping = _label_mapping(root, csv_paths)
        return _csv_samples(root, csv_path, (root / "images", root), mapping)
    split = root / ("train" if train else "val")
    if not split.exists() and not train:
        split = root / "test"
    samples, _mapping = _folder_samples(split)
    return samples


def _tinyimagenet_samples(root: Path, train: bool) -> list[tuple[Path, int]]:
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    csv_path = train_csv if train else test_csv
    if csv_path.exists():
        mapping = _label_mapping(root, (train_csv, test_csv))
        bases = (root / ("train" if train else "val"), root)
        return _csv_samples(root, csv_path, bases, mapping)

    train_directory = root / "train"
    class_names = sorted(path.name for path in train_directory.iterdir() if path.is_dir())
    mapping = {name: index for index, name in enumerate(class_names)}
    if train:
        samples, _mapping = _folder_samples(train_directory, mapping)
        return samples
    annotations = root / "val" / "val_annotations.txt"
    images = root / "val" / "images"
    if annotations.exists():
        samples = []
        with annotations.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip().split("\t")
                if len(fields) >= 2 and fields[1] in mapping:
                    samples.append((images / fields[0], mapping[fields[1]]))
        return samples
    samples, _mapping = _folder_samples(root / "val", mapping)
    return samples


def _fc100_samples(root: Path, train: bool) -> list[tuple[Path, int]]:
    train_csv = root / "train_csv"
    test_csv = root / "test_csv"
    if not train_csv.exists():
        train_csv = root / "train.csv"
    if not test_csv.exists():
        test_csv = root / "test.csv"
    csv_path = train_csv if train else test_csv
    if csv_path.exists():
        mapping = _label_mapping(root, (train_csv, test_csv))
        return _csv_samples(root, csv_path, (root / "train", root), mapping)
    split = root / ("train" if train else "test")
    samples, _mapping = _folder_samples(split)
    return samples


def _core50_samples(root: Path, train: bool) -> list[tuple[Path, int]]:
    nested_root = root / "core50" if (root / "core50").is_dir() else root
    task_root = nested_root / "task_label"
    csv_name = "train.csv" if train else "test.csv"
    csv_paths = sorted(task_root.glob(f"*/{csv_name}"))
    if not csv_paths and (root / csv_name).exists():
        csv_paths = [root / csv_name]
    if csv_paths:
        mapping = _label_mapping(root, csv_paths)
        samples: list[tuple[Path, int]] = []
        bases = (root, nested_root, nested_root / "core50_128x128")
        for csv_path in csv_paths:
            samples.extend(_csv_samples(root, csv_path, bases, mapping))
        return samples
    split = nested_root / ("train" if train else "test")
    samples, _mapping = _folder_samples(split)
    return samples


def _imagenet_samples(root: Path, train: bool) -> list[tuple[Path, int]]:
    train_directory = root / "train"
    class_names = sorted(path.name for path in train_directory.iterdir() if path.is_dir())
    mapping = {name: index for index, name in enumerate(class_names)}
    split = train_directory if train else root / "val"
    samples, _mapping = _folder_samples(split, mapping)
    return samples


def _listed_split_samples(root: Path, train: bool) -> list[tuple[Path, int]]:
    split_name = "train" if train else "test"
    candidates = sorted(root.glob(f"*_{split_name}.txt"))
    candidates.extend(sorted((root / "splits").glob(f"*_{split_name}.txt")))
    records: list[tuple[Path, str | int]] = []
    for list_path in candidates:
        with list_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.strip().rsplit(maxsplit=1)
                if not fields:
                    continue
                relative = Path(fields[0])
                label: str | int = fields[1] if len(fields) == 2 else relative.parent.name
                try:
                    label = int(label)
                except ValueError:
                    label = str(label)
                image_path = relative if relative.is_absolute() else root / relative
                if image_path.exists():
                    records.append((image_path, label))
    if not records:
        return []
    string_labels = sorted({str(label) for _path, label in records})
    mapping = {label: index for index, label in enumerate(string_labels)}
    return [(path, mapping[str(label)]) for path, label in records]


def _cub200_samples(root: Path, train: bool) -> list[tuple[Path, int]]:
    metadata_root = root / "CUB_200_2011" if (root / "CUB_200_2011").is_dir() else root
    images_txt = metadata_root / "images.txt"
    split_txt = metadata_root / "train_test_split.txt"
    if not images_txt.exists() or not split_txt.exists():
        return _generic_folder_samples(root, train)
    split = {}
    with split_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            image_id, is_train = line.strip().split(maxsplit=1)
            split[image_id] = bool(int(is_train))
    class_names: set[str] = set()
    rows: list[tuple[str, Path]] = []
    with images_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            image_id, relative = line.strip().split(maxsplit=1)
            relative_path = Path(relative)
            class_names.add(relative_path.parent.name)
            if split.get(image_id, False) == bool(train):
                rows.append((relative_path.parent.name, metadata_root / "images" / relative_path))
    mapping = {name: index for index, name in enumerate(sorted(class_names))}
    return [(path, mapping[class_name]) for class_name, path in rows]


def _metadata_split_samples(root: Path, name: str, train: bool) -> list[tuple[Path, int]]:
    if name == "food101":
        metadata = root / "meta" / ("train.txt" if train else "test.txt")
        if metadata.exists():
            rows = [line.strip() for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
            classes = sorted({Path(row).parent.as_posix() for row in rows})
            mapping = {label: index for index, label in enumerate(classes)}
            return [
                (root / "images" / f"{row}.jpg", mapping[Path(row).parent.as_posix()])
                for row in rows
            ]
    if name == "oxfordpets":
        metadata = root / "annotations" / ("trainval.txt" if train else "test.txt")
        if metadata.exists():
            records = []
            with metadata.open("r", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.split()
                    if len(fields) >= 2:
                        records.append((fields[0], int(fields[1]) - 1))
            return [(root / "images" / f"{stem}.jpg", label) for stem, label in records]
    if name == "dtd":
        label_dir = root / "labels"
        files = [label_dir / "train1.txt"] if train else [label_dir / "val1.txt", label_dir / "test1.txt"]
        rows = [
            line.strip()
            for path in files
            if path.exists()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows:
            classes = sorted({Path(row).parent.name for row in rows})
            mapping = {label: index for index, label in enumerate(classes)}
            return [(root / "images" / row, mapping[Path(row).parent.name]) for row in rows]
    return []


def _generic_folder_samples(root: Path, train: bool) -> list[tuple[Path, int]]:
    split_names = ("train",) if train else ("test", "val", "validation")
    for split_name in split_names:
        split = root / split_name
        if split.is_dir():
            samples, _mapping = _folder_samples(split)
            if samples:
                return samples
    image_root = root / "101_ObjectCategories" if (root / "101_ObjectCategories").is_dir() else root
    all_paths = _image_files(image_root)
    by_class: dict[str, list[Path]] = {}
    for path in all_paths:
        by_class.setdefault(path.parent.name, []).append(path)
    mapping = {name: index for index, name in enumerate(sorted(by_class))}
    samples: list[tuple[Path, int]] = []
    for class_name, paths in sorted(by_class.items()):
        paths = sorted(paths)
        if len(paths) <= 1:
            chosen = paths
        else:
            boundary = min(len(paths) - 1, max(1, int(round(len(paths) * 0.8))))
            chosen = paths[:boundary] if train else paths[boundary:]
        samples.extend((path, mapping[class_name]) for path in chosen)
    return samples


def _paper_dataset_samples(root: Path, name: str, train: bool) -> list[tuple[Path, int]]:
    if name == "cub200":
        return _cub200_samples(root, train)
    listed = _listed_split_samples(root, train)
    if listed:
        return listed
    metadata = _metadata_split_samples(root, name, train)
    if metadata:
        return metadata
    return _generic_folder_samples(root, train)


def build_image_dataset(
    name: str,
    train: bool,
    data_dir: str | Path,
    num_samples: int | None = None,
    seed: int = 0,
    image_size: int = 32,
    normalization: str = "dataset",
) -> Dataset:
    key = normalize_image_dataset_name(name)
    root = _resolve_root(data_dir, key)
    builders = {
        "miniimagenet": _miniimagenet_samples,
        "tinyimagenet": _tinyimagenet_samples,
        "fc100": _fc100_samples,
        "core50": _core50_samples,
        "imagenet": _imagenet_samples,
        "imagenet100": _imagenet_samples,
        "imagenetr": lambda root, train: _paper_dataset_samples(root, "imagenetr", train),
        "cub200": lambda root, train: _paper_dataset_samples(root, "cub200", train),
        "domainnet": lambda root, train: _paper_dataset_samples(root, "domainnet", train),
        "domainnetsub": lambda root, train: _paper_dataset_samples(root, "domainnetsub", train),
        "officehome": lambda root, train: _paper_dataset_samples(root, "officehome", train),
        "adaptiope": lambda root, train: _paper_dataset_samples(root, "adaptiope", train),
        "pacs": lambda root, train: _paper_dataset_samples(root, "pacs", train),
        "flowers102": lambda root, train: _paper_dataset_samples(root, "flowers102", train),
        "oxfordpets": lambda root, train: _paper_dataset_samples(root, "oxfordpets", train),
        "food101": lambda root, train: _paper_dataset_samples(root, "food101", train),
        "caltech101": lambda root, train: _paper_dataset_samples(root, "caltech101", train),
        "dtd": lambda root, train: _paper_dataset_samples(root, "dtd", train),
        "notmnist": lambda root, train: _paper_dataset_samples(root, "notmnist", train),
    }
    samples = builders[key](root, bool(train))
    if not samples:
        raise ValueError(f"No {'training' if train else 'test'} images found in {root}.")
    dataset: Dataset = ImagePathDataset(
        samples=samples,
        transform=_transforms(
            key,
            train=bool(train),
            image_size=int(image_size),
            normalization=normalization,
        ),
    )
    if num_samples is None or int(num_samples) <= 0 or int(num_samples) >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[: int(num_samples)].tolist()
    return Subset(dataset, indices)


LOCI_DATASET_ALIASES = IMAGE_DATASET_ALIASES
LOCI_DATASET_INFO = IMAGE_DATASET_INFO
normalize_loci_dataset_name = normalize_image_dataset_name
loci_dataset_info = image_dataset_info
build_loci_image_dataset = build_image_dataset


__all__ = [
    "ImagePathDataset",
    "IMAGE_DATASET_ALIASES",
    "IMAGE_DATASET_INFO",
    "build_image_dataset",
    "image_dataset_info",
    "normalize_image_dataset_name",
    "LOCI_DATASET_INFO",
    "build_loci_image_dataset",
    "loci_dataset_info",
    "normalize_loci_dataset_name",
]
