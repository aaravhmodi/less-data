from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class ResultSchema:
    """Expected result columns for paper-ready tables and plots."""

    columns: Sequence[str] = (
        "dataset",
        "method",
        "seed",
        "keep_fraction",
        "accuracy",
        "macro_f1",
        "training_time_seconds",
        "parameter_count",
        "notes",
    )


def validate_results_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = set(ResultSchema().columns)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected result columns: {sorted(missing)}")
    return df
