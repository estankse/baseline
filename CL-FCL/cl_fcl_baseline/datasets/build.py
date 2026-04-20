from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np


class RandomClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        num_samples: int = 256,
        input_shape: Sequence[int] = (1, 28, 28),
        num_classes: int = 10,
        seed: int = 0,
    ) -> None:
        self.num_samples = int(num_samples)
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.num_classes = int(num_classes)
        generator = torch.Generator().manual_seed(int(seed))
        self.features = torch.randn((self.num_samples, *self.input_shape), generator=generator)
        self.targets = torch.randint(0, self.num_classes, (self.num_samples,), generator=generator)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class ClassIncrementalSubset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        class_ids: Sequence[int],
        remap_labels: bool = True,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.class_ids = [int(class_id) for class_id in class_ids]
        self.remap_labels = bool(remap_labels)
        self.label_map = {class_id: idx for idx, class_id in enumerate(self.class_ids)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample, target = self.dataset[self.indices[index]]
        target_value = int(target)
        if self.remap_labels:
            target_value = self.label_map[target_value]
        return sample, torch.tensor(target_value, dtype=torch.long)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    if batch_size is None or int(batch_size) <= 0:
        batch_size = max(1, len(dataset))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def partition_dataset_iid(dataset: Dataset, num_clients: int, seed: int = 0) -> List[Subset]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(int(seed))).tolist()
    partitions: List[Subset] = []
    chunk_size = len(indices) // num_clients
    remainder = len(indices) % num_clients
    start = 0
    for client_idx in range(num_clients):
        extra = 1 if client_idx < remainder else 0
        end = start + chunk_size + extra
        partitions.append(Subset(dataset, indices[start:end]))
        start = end
    return partitions


def _extract_labels(dataset: Dataset) -> List[int]:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, torch.Tensor):
            return [int(value) for value in targets.tolist()]
        return [int(value) for value in list(targets)]
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if isinstance(labels, torch.Tensor):
            return [int(value) for value in labels.tolist()]
        return [int(value) for value in list(labels)]
    raise ValueError("Dataset does not expose labels via `targets` or `labels`.")


def _subset_labels(dataset: Dataset, indices: Iterable[int]) -> List[int]:
    base = dataset
    if isinstance(dataset, Subset):
        base = dataset.dataset
    labels = _extract_labels(base)
    return [labels[int(idx)] for idx in indices]


def build_class_incremental_tasks(
    dataset: Dataset,
    classes_per_task: int,
    num_tasks: int | None = None,
    class_order: Sequence[int] | None = None,
    seed: int = 0,
    shuffle_classes: bool = False,
    remap_labels: bool = True,
) -> List[ClassIncrementalSubset]:
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive.")

    labels = _subset_labels(dataset, range(len(dataset)))
    unique_classes = sorted(set(labels))
    if class_order is None:
        ordered_classes = unique_classes[:]
        if shuffle_classes:
            generator = torch.Generator().manual_seed(int(seed))
            permutation = torch.randperm(len(ordered_classes), generator=generator).tolist()
            ordered_classes = [ordered_classes[idx] for idx in permutation]
    else:
        ordered_classes = [int(class_id) for class_id in class_order]

    if num_tasks is None or int(num_tasks) <= 0:
        num_tasks = len(ordered_classes) // int(classes_per_task)

    required_classes = int(num_tasks) * int(classes_per_task)
    if required_classes > len(ordered_classes):
        raise ValueError("Not enough classes to build the requested number of tasks.")

    ordered_classes = ordered_classes[:required_classes]
    label_to_indices: dict[int, list[int]] = {}
    for sample_idx, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(sample_idx)

    task_datasets: List[ClassIncrementalSubset] = []
    for task_idx in range(int(num_tasks)):
        task_classes = ordered_classes[
            task_idx * int(classes_per_task): (task_idx + 1) * int(classes_per_task)
        ]
        task_indices: List[int] = []
        for class_id in task_classes:
            task_indices.extend(label_to_indices.get(int(class_id), []))
        task_datasets.append(
            ClassIncrementalSubset(
                dataset=dataset,
                indices=task_indices,
                class_ids=task_classes,
                remap_labels=remap_labels,
            )
        )
    return task_datasets


def partition_dataset_noniid(
    dataset: Dataset,
    num_clients: int,
    num_shards: int = 2,
    seed: int = 0,
) -> List[Subset]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")

    total_indices = list(range(len(dataset)))
    labels = _subset_labels(dataset, total_indices)
    sorted_pairs = sorted(zip(total_indices, labels), key=lambda pair: pair[1])
    sorted_indices = [idx for idx, _ in sorted_pairs]

    shards = num_clients * num_shards
    shard_size = len(sorted_indices) // shards
    if shard_size == 0:
        raise ValueError("Dataset is too small for the requested number of shards.")

    trimmed = sorted_indices[: shard_size * shards]
    shard_indices = [
        trimmed[i * shard_size : (i + 1) * shard_size]
        for i in range(shards)
    ]

    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(shards, generator=generator).tolist()

    client_partitions: List[Subset] = []
    for client_idx in range(num_clients):
        assigned: List[int] = []
        for j in range(num_shards):
            shard_id = permutation[client_idx * num_shards + j]
            assigned.extend(shard_indices[shard_id])
        client_partitions.append(Subset(dataset, assigned))
    return client_partitions


def partition_dataset_dirichlet(
        dataset: Dataset,
        num_clients: int,
        beta: float = 0.1,
        num_classes: int = 10,
        seed: int = 0
) -> List[Subset]:
    np.random.seed(seed)

    # 提取所有数据的标签并记录原始索引
    # 假设 dataset 可以通过索引访问，且返回 (data, label)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # 记录每个客户端分到的数据索引
    client_indices = {i: [] for i in range(num_clients)}

    # 对每一个类别，利用狄利克雷分布计算将其分配给各个客户端的比例
    for k in range(num_classes):
        # 找出属于类别 k 的所有数据的索引
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)  # 打乱该类别的索引

        # 生成基于 beta 的狄利克雷分布比例 (长度为 num_clients)
        # proportion 的和为 1，例如 [0.05, 0.9, ..., 0.01]
        proportions = np.random.dirichlet(np.repeat(beta, num_clients))


        proportions = np.array(
            [p * (len(idx_j) < len(labels) / num_clients) for p, idx_j in zip(proportions, client_indices.values())])
        proportions = proportions / proportions.sum()


        split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        # 将该类别的数据按切分点分给各个客户端
        split_idx = np.split(idx_k, split_points)
        for client_id, indices in enumerate(split_idx):
            client_indices[client_id].extend(indices.tolist())

    # 将索引转化为 Subset 列表
    client_partitions = [Subset(dataset, indices) for indices in client_indices.values()]
    return client_partitions
