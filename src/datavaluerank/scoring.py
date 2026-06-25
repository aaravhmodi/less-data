from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ExampleSignals:
    """Signals used to rank the value of a training example.

    All values are expected to be precomputed for a single example.
    Higher `datavalue_rank` output means the example is more valuable to keep.
    """

    training_loss: float
    prediction_confidence: float
    embedding_uniqueness: float
    class_rarity: float
    model_disagreement: float
    label_noise_probability: float


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for the ranking formula.

    Positive weights increase the score. Negative weights reduce it.
    """

    training_loss: float = 0.20
    prediction_confidence: float = 0.15
    embedding_uniqueness: float = 0.25
    class_rarity: float = 0.15
    model_disagreement: float = 0.15
    label_noise_probability: float = -0.30


def min_max_normalize(values: Sequence[float]) -> List[float]:
    """Normalize values to the [0, 1] range.

    If all values are the same, return zeros to avoid division by zero.
    """

    if not values:
        return []

    low = min(values)
    high = max(values)
    if high == low:
        return [0.0 for _ in values]

    scale = high - low
    return [(value - low) / scale for value in values]


def rank_examples(
    examples: Sequence[ExampleSignals],
    weights: ScoreWeights | None = None,
) -> List[float]:
    """Return a rank score for each example.

    The scoring rule is intentionally simple and interpretable:

    - informative examples are rewarded
    - likely noisy examples are penalized
    """

    if weights is None:
        weights = ScoreWeights()

    losses = min_max_normalize([ex.training_loss for ex in examples])
    confidences = min_max_normalize([ex.prediction_confidence for ex in examples])
    uniqueness = min_max_normalize([ex.embedding_uniqueness for ex in examples])
    rarity = min_max_normalize([ex.class_rarity for ex in examples])
    disagreement = min_max_normalize([ex.model_disagreement for ex in examples])
    noise = min_max_normalize([ex.label_noise_probability for ex in examples])

    scores: List[float] = []
    for i, _ in enumerate(examples):
        score = (
            weights.training_loss * losses[i]
            + weights.prediction_confidence * confidences[i]
            + weights.embedding_uniqueness * uniqueness[i]
            + weights.class_rarity * rarity[i]
            + weights.model_disagreement * disagreement[i]
            + weights.label_noise_probability * noise[i]
        )
        scores.append(score)

    return scores


def keep_top_k(
    examples: Sequence[ExampleSignals],
    keep_fraction: float,
    weights: ScoreWeights | None = None,
) -> List[int]:
    """Return the indices of the highest-value examples to keep."""

    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in the interval (0, 1].")

    scores = rank_examples(examples, weights=weights)
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    keep_count = max(1, round(len(ranked_indices) * keep_fraction))
    return ranked_indices[:keep_count]
