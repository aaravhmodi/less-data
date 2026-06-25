from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    group_cols = [col for col in ["dataset", "method", "keep_fraction"] if col in df.columns]
    if not group_cols:
        raise ValueError("Expected dataset, method, or keep_fraction columns.")
    summary = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std"})
    )
    summary[f"{value_col}_stderr"] = summary[f"{value_col}_std"] / summary["count"].pow(0.5)
    summary[f"{value_col}_ci95"] = 1.96 * summary[f"{value_col}_stderr"]
    return summary


def save_lineplot(df: pd.DataFrame, output: Path, title: str, value_col: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 6), dpi=300)
    for method, group in df.groupby("method"):
        group = group.sort_values("keep_fraction")
        plt.plot(
            group["keep_fraction"] * 100,
            group[value_col],
            marker="o",
            linewidth=2,
            label=method,
        )
        if f"{value_col}_ci95" in group.columns:
            lower = group[value_col] - group[f"{value_col}_ci95"]
            upper = group[value_col] + group[f"{value_col}_ci95"]
            plt.fill_between(
                group["keep_fraction"] * 100,
                lower,
                upper,
                alpha=0.12,
            )
    plt.title(title)
    plt.xlabel("Data kept (%)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, format="jpg", dpi=300)
    plt.close()


def save_barplot(df: pd.DataFrame, output: Path, title: str, value_col: str, label_col: str) -> None:
    plt.figure(figsize=(12, 6), dpi=300)
    ordered = df.sort_values(value_col, ascending=False)
    plt.bar(ordered[label_col], ordered[value_col], color="#2a6f97")
    plt.title(title)
    plt.xlabel(label_col.replace("_", " ").title())
    plt.ylabel(value_col.replace("_", " ").title())
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, format="jpg", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from results CSV files.")
    parser.add_argument("--results", type=Path, required=True, help="CSV file with experiment results")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JPEG figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results)

    required_columns = {"method", "keep_fraction"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "accuracy" in df.columns:
        accuracy_df = _aggregate(df, "accuracy")
        save_lineplot(
            accuracy_df,
            args.output_dir / "accuracy_vs_retention.jpg",
            "Accuracy vs Retention",
            "accuracy_mean",
            "Accuracy",
        )

    if "training_time_seconds" in df.columns:
        time_df = _aggregate(df, "training_time_seconds")
        save_lineplot(
            time_df,
            args.output_dir / "time_vs_retention.jpg",
            "Training Time vs Retention",
            "training_time_seconds_mean",
            "Training Time (Seconds)",
        )

    if {"class_name", "class_accuracy"}.issubset(df.columns):
        save_barplot(
            df[["class_name", "class_accuracy"]].dropna().drop_duplicates(),
            args.output_dir / "class_performance.jpg",
            "Per-Class Performance",
            "class_accuracy",
            "class_name",
        )


if __name__ == "__main__":
    main()
