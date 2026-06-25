from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Subset

from .datasets import ArrayDataset, load_fashion_mnist
from .scoring import ExampleSignals, ScoreWeights, keep_top_k
from .training import evaluate_model, predict_with_embedding, train_model, make_loader
from .utils import ensure_dir, set_seed


def _class_rarity(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels)
    inv = np.zeros_like(counts, dtype=np.float32)
    inv[counts > 0] = 1.0 / counts[counts > 0]
    rarity = inv[labels]
    return rarity / rarity.max() if rarity.max() > 0 else rarity


def _embedding_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) <= 1:
        return np.ones(len(embeddings), dtype=np.float32)
    k = min(6, len(embeddings))
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    uniq = distances[:, 1:].mean(axis=1)
    max_val = float(uniq.max()) if float(uniq.max()) > 0 else 1.0
    return uniq / max_val


def _signals_from_predictions(
    probs: np.ndarray,
    labels: np.ndarray,
    signal_noise: np.ndarray | None = None,
) -> List[ExampleSignals]:
    confidences = probs.max(axis=1)
    true_probs = probs[np.arange(len(labels)), labels]
    losses = -np.log(np.clip(true_probs, 1e-8, 1.0))
    predicted_labels = probs.argmax(axis=1)
    disagreement = (predicted_labels != labels).astype(np.float32)
    label_noise = 1.0 - true_probs
    if signal_noise is not None:
        label_noise = 0.5 * label_noise + 0.5 * signal_noise
    return [
        ExampleSignals(
            training_loss=float(losses[i]),
            prediction_confidence=float(confidences[i]),
            embedding_uniqueness=float(0.0),
            class_rarity=float(0.0),
            model_disagreement=float(disagreement[i]),
            label_noise_probability=float(label_noise[i]),
        )
        for i in range(len(labels))
    ]


def _assemble_signals(
    avg_probs: np.ndarray,
    avg_embeddings: np.ndarray,
    disagreement: np.ndarray,
    labels: np.ndarray,
) -> List[ExampleSignals]:
    class_rarity = _class_rarity(labels)
    uniqueness = _embedding_uniqueness(avg_embeddings)
    base = _signals_from_predictions(avg_probs, labels, signal_noise=disagreement)
    enriched = []
    for i, signal in enumerate(base):
        enriched.append(
            ExampleSignals(
                training_loss=signal.training_loss,
                prediction_confidence=signal.prediction_confidence,
                embedding_uniqueness=float(uniqueness[i]),
                class_rarity=float(class_rarity[i]),
                model_disagreement=float(signal.model_disagreement),
                label_noise_probability=signal.label_noise_probability,
            )
        )
    return enriched


def run_fashion_mnist_experiment(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    seeds: Sequence[int] = (0, 1, 2),
    keep_fractions: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
    teacher_folds: int = 3,
    epochs: int = 6,
    batch_size: int = 128,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(output_dir)
    (train_x, train_y), (test_x, test_y) = load_fashion_mnist(data_dir)
    full_train = ArrayDataset(train_x, train_y)
    test_dataset = ArrayDataset(test_x, test_y)
    train_indices = np.arange(len(train_y))
    base_train_idx, val_idx = train_test_split(
        train_indices, test_size=0.1, random_state=0, stratify=train_y
    )
    base_train_set = Subset(full_train, base_train_idx.tolist())
    val_set = Subset(full_train, val_idx.tolist())

    records: List[Dict[str, float]] = []

    for seed in seeds:
        set_seed(seed)

        full_result = train_model(
            base_train_set,
            val_set,
            device=device,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
        )
        test_metrics = evaluate_model(full_result.model, make_loader(test_dataset, batch_size=batch_size, shuffle=False), device=device)
        records.append(
            {
                "dataset": "Fashion-MNIST",
                "method": "full_data",
                "seed": seed,
                "keep_fraction": 1.0,
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "training_time_seconds": full_result.train_seconds,
                "examples_removed": 0,
                "notes": "baseline",
            }
        )

        skf = StratifiedKFold(n_splits=teacher_folds, shuffle=True, random_state=seed)
        from sklearn.model_selection import RepeatedStratifiedKFold

        rskf = RepeatedStratifiedKFold(
            n_splits=teacher_folds,
            n_repeats=2,
            random_state=seed,
        )
        base_labels = train_y[base_train_idx]
        num_examples = len(base_train_idx)
        num_classes = int(np.max(base_labels)) + 1
        prob_sum = np.zeros((num_examples, num_classes), dtype=np.float64)
        prob_sq_sum = np.zeros((num_examples, num_classes), dtype=np.float64)
        embedding_sum = None
        counts = np.zeros(num_examples, dtype=np.int32)

        for fold, (fold_train_idx, fold_holdout_idx) in enumerate(rskf.split(train_x[base_train_idx], base_labels)):
            fold_train_global = base_train_idx[fold_train_idx]
            fold_holdout_local = fold_holdout_idx
            fold_train_set = Subset(full_train, fold_train_global.tolist())
            fold_val_set = Subset(full_train, base_train_idx[fold_holdout_local].tolist())
            teacher = train_model(
                fold_train_set,
                fold_val_set,
                device=device,
                seed=seed + fold + 100,
                epochs=max(2, epochs // 2),
                batch_size=batch_size,
            ).model
            probs, embeddings, labels = predict_with_embedding(
                teacher,
                Subset(full_train, base_train_idx[fold_holdout_local].tolist()),
                device=device,
                batch_size=batch_size,
            )
            if embedding_sum is None:
                embedding_sum = np.zeros((num_examples, embeddings.shape[1]), dtype=np.float64)
            prob_sum[fold_holdout_local] += probs
            prob_sq_sum[fold_holdout_local] += probs ** 2
            embedding_sum[fold_holdout_local] += embeddings
            counts[fold_holdout_local] += 1

        if np.any(counts == 0):
            raise RuntimeError("Cross-validation did not score every training example.")

        avg_probs = prob_sum / counts[:, None]
        avg_embeddings = embedding_sum / counts[:, None]
        disagreement = (prob_sq_sum / counts[:, None] - avg_probs ** 2).mean(axis=1)

        signals = _assemble_signals(avg_probs, avg_embeddings, disagreement, base_labels)
        for keep_fraction in keep_fractions:
            selected_local = keep_top_k(signals, keep_fraction=keep_fraction)
            selected_indices = base_train_idx[np.array(selected_local)]
            selected_set = Subset(full_train, selected_indices.tolist())
            subset_result = train_model(
                selected_set,
                val_set,
                device=device,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
            )
            subset_metrics = evaluate_model(
                subset_result.model,
                make_loader(test_dataset, batch_size=batch_size, shuffle=False),
                device=device,
            )
            records.append(
                {
                    "dataset": "Fashion-MNIST",
                    "method": "DataValueRank",
                    "seed": seed,
                    "keep_fraction": float(keep_fraction),
                    "accuracy": subset_metrics["accuracy"],
                    "macro_f1": subset_metrics["macro_f1"],
                    "training_time_seconds": subset_result.train_seconds,
                    "examples_removed": int(len(base_train_idx) - len(selected_indices)),
                    "notes": "cross_validated_scoring",
                }
            )

    results = pd.DataFrame.from_records(records)
    results.to_csv(Path(output_dir) / "fashion_mnist_results.csv", index=False)
    return results
