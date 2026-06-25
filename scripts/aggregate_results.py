from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_numeric(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=1)
    count = series.count()
    stderr = std / (count ** 0.5) if count and pd.notna(std) else float("nan")
    ci95 = 1.96 * stderr if pd.notna(stderr) else float("nan")
    return pd.Series(
        {
            "mean": mean,
            "std": std,
            "ci95_low": mean - ci95 if pd.notna(ci95) else float("nan"),
            "ci95_high": mean + ci95 if pd.notna(ci95) else float("nan"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-seed results into summary tables.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="CSV files to merge")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs]
    df = pd.concat(frames, ignore_index=True)

    group_cols = [col for col in ["dataset", "method", "keep_fraction"] if col in df.columns]
    if not group_cols:
        raise ValueError("Expected at least one grouping column among dataset, method, keep_fraction.")

    metric_columns = [col for col in ["accuracy", "macro_f1", "training_time_seconds", "memory_mb"] if col in df.columns]
    if not metric_columns:
        raise ValueError("No supported metric columns found in input data.")

    grouped = df.groupby(group_cols, dropna=False)
    records = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        for metric in metric_columns:
            stats = summarize_numeric(group[metric])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci95_low"] = stats["ci95_low"]
            row[f"{metric}_ci95_high"] = stats["ci95_high"]
        records.append(row)

    summary = pd.DataFrame.from_records(records)

    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
