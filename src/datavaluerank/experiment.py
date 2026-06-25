from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .scoring import ExampleSignals, ScoreWeights, keep_top_k


@dataclass
class ExperimentConfig:
    dataset_name: str
    model_name: str
    keep_fractions: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2)
    seeds: Sequence[int] = (0, 1, 2)
    weights: ScoreWeights = field(default_factory=ScoreWeights)


@dataclass
class ExperimentResult:
    keep_fraction: float
    seed: int
    metrics: Dict[str, float]


def select_subset(examples: Sequence[ExampleSignals], keep_fraction: float) -> List[int]:
    return keep_top_k(examples, keep_fraction=keep_fraction)


def build_retention_schedule() -> List[float]:
    return [1.0, 0.8, 0.6, 0.4, 0.2]
