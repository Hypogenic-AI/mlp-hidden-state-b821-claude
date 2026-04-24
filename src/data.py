"""Data loading for CIFAR-10 and MNIST.

Preloads the entire dataset into GPU tensors to eliminate CPU-side data
loading (transforms, PIL decoding, shuffling) which was a major
bottleneck. Provides a fixed 5000-sample probe subset drawn from the
test split for all representation-similarity analyses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torchvision import datasets, transforms


DATA_ROOT = Path("datasets")


def _cifar10_stats() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def _mnist_stats() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (0.1307,), (0.3081,)


def _tensorize(dset) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the entire dataset into two big tensors (no workers)."""
    import numpy as np

    # torchvision datasets provide .data and .targets. For MNIST these are
    # torch.Tensor; for CIFAR they're numpy arrays / Python lists.
    data = dset.data
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if hasattr(dset, "targets"):
        targets = dset.targets
    else:
        targets = dset.labels
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy().astype(np.int64)
    else:
        targets = np.asarray(targets, dtype=np.int64)
    if data.ndim == 3:  # MNIST, (N, 28, 28)
        data = data[:, None, :, :]  # add channel
    elif data.ndim == 4:  # CIFAR, (N, 32, 32, 3)
        data = data.transpose(0, 3, 1, 2)
    t = torch.from_numpy(data).float() / 255.0
    y = torch.from_numpy(targets)
    return t, y


def _normalize(x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    m = torch.tensor(mean).view(1, -1, 1, 1)
    s = torch.tensor(std).view(1, -1, 1, 1)
    return (x - m) / s


@dataclass
class TensorDataset:
    x: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        return self.x.shape[0]


@dataclass
class GPUSplit:
    """A dataset already loaded on GPU. Provides (x, y) batches.

    The shuffle / order is controlled per-epoch via `iter_batches`.
    """

    x: torch.Tensor  # (N, C, H, W), on device
    y: torch.Tensor  # (N,), on device
    batch_size: int
    shuffle: bool

    def __len__(self) -> int:
        return (self.x.shape[0] + self.batch_size - 1) // self.batch_size

    def iter_batches(self, generator: torch.Generator | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        n = self.x.shape[0]
        if self.shuffle:
            idx = torch.randperm(n, generator=generator, device=self.x.device)
        else:
            idx = torch.arange(n, device=self.x.device)
        for i in range(0, n, self.batch_size):
            sel = idx[i : i + self.batch_size]
            yield self.x[sel], self.y[sel]

    # Keep the DataLoader-style iteration contract
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return self.iter_batches()


def load_dataset_to_gpu(
    name: str, device: torch.device, batch_size: int = 512
) -> tuple[GPUSplit, GPUSplit, GPUSplit, int, int, torch.Tensor]:
    """Return (train_split, val_split, probe_split, input_dim, num_classes, train_labels_cpu).

    The `train_labels_cpu` is a CPU copy of the original (non-shuffled)
    labels — used to apply label-shuffling reproducibly.
    """
    if name == "cifar10":
        mean, std = _cifar10_stats()
        train = datasets.CIFAR10(root=str(DATA_ROOT / "cifar10"), train=True, download=False)
        test = datasets.CIFAR10(root=str(DATA_ROOT / "cifar10"), train=False, download=False)
        input_dim = 3 * 32 * 32
        num_classes = 10
    elif name == "mnist":
        mean, std = _mnist_stats()
        train = datasets.MNIST(root=str(DATA_ROOT / "mnist"), train=True, download=False)
        test = datasets.MNIST(root=str(DATA_ROOT / "mnist"), train=False, download=False)
        input_dim = 28 * 28
        num_classes = 10
    else:
        raise ValueError(name)

    xtr, ytr = _tensorize(train)
    xte, yte = _tensorize(test)
    xtr = _normalize(xtr, mean, std).to(device, non_blocking=True)
    xte_all = _normalize(xte, mean, std).to(device, non_blocking=True)
    ytr_gpu = ytr.to(device)
    yte_gpu = yte.to(device)

    probe_x = xte_all[:5000]
    probe_y = yte_gpu[:5000]
    val_x = xte_all[5000:]
    val_y = yte_gpu[5000:]

    train_split = GPUSplit(x=xtr, y=ytr_gpu, batch_size=batch_size, shuffle=True)
    val_split = GPUSplit(x=val_x, y=val_y, batch_size=1024, shuffle=False)
    probe_split = GPUSplit(x=probe_x, y=probe_y, batch_size=1024, shuffle=False)
    return train_split, val_split, probe_split, input_dim, num_classes, ytr
