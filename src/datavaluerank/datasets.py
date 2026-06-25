from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from torch.utils.data import Dataset


def _load_or_fetch_fashion_mnist_cache(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    cache_file = root / "fashion_mnist.npz"
    if cache_file.exists():
        cached = np.load(cache_file)
        return cached["images"], cached["labels"]

    root.mkdir(parents=True, exist_ok=True)
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    images = ds.data.astype(np.float32).reshape(-1, 28, 28) / 255.0
    labels = ds.target.astype(np.int64)
    np.savez_compressed(cache_file, images=images, labels=labels)
    return images, labels


def load_fashion_mnist(root: str | Path) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    root = Path(root)
    images, labels = _load_or_fetch_fashion_mnist_cache(root)
    train_x, test_x = images[:60000], images[60000:]
    train_y, test_y = labels[:60000], labels[60000:]
    return (train_x, train_y), (test_x, test_y)


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = torch.from_numpy(self.images[index]).unsqueeze(0).float()
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label
