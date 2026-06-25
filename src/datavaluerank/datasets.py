from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

FASHION_MNIST_BASE_URL = "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com"


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    urllib.request.urlretrieve(url, dst)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        _, num, rows, cols = struct.unpack(">IIII", fh.read(16))
        data = np.frombuffer(fh.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        _, num = struct.unpack(">II", fh.read(8))
        data = np.frombuffer(fh.read(), dtype=np.uint8)
    return data.reshape(num)


def load_fashion_mnist(root: str | Path) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    root = Path(root)
    raw_dir = root / "fashion_mnist"
    files = {
        "train_images": raw_dir / "train-images-idx3-ubyte.gz",
        "train_labels": raw_dir / "train-labels-idx1-ubyte.gz",
        "test_images": raw_dir / "t10k-images-idx3-ubyte.gz",
        "test_labels": raw_dir / "t10k-labels-idx1-ubyte.gz",
    }
    urls = {
        key: f"{FASHION_MNIST_BASE_URL}/{path.name}"
        for key, path in files.items()
    }

    for key, path in files.items():
        _download(urls[key], path)

    train_x = _read_idx_images(files["train_images"]).astype(np.float32) / 255.0
    train_y = _read_idx_labels(files["train_labels"]).astype(np.int64)
    test_x = _read_idx_images(files["test_images"]).astype(np.float32) / 255.0
    test_y = _read_idx_labels(files["test_labels"]).astype(np.int64)
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

