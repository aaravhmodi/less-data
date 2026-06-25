from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from .datasets import ArrayDataset
from .models import FashionCNN


@dataclass
class TrainResult:
    model: FashionCNN
    history: List[Dict[str, float]]
    train_seconds: float


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def train_model(
    train_dataset,
    val_dataset,
    *,
    device: torch.device,
    seed: int,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> TrainResult:
    torch.manual_seed(seed)
    model = FashionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, batch_size=batch_size, shuffle=False)

    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        val_metrics = evaluate_model(model, val_loader, device=device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": running_loss / max(1, seen),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
    train_seconds = time.perf_counter() - start
    return TrainResult(model=model, history=history, train_seconds=train_seconds)


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, *, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_pred = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


@torch.no_grad()
def predict_with_embedding(model, dataset, *, device: torch.device, batch_size: int = 128):
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs = []
    all_embeddings = []
    all_labels = []
    for x, y in loader:
        x = x.to(device)
        logits, embedding = model(x, return_embedding=True)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_embeddings.append(embedding.cpu())
        all_labels.append(y)
    return (
        torch.cat(all_probs, dim=0).numpy(),
        torch.cat(all_embeddings, dim=0).numpy(),
        torch.cat(all_labels, dim=0).numpy(),
    )

