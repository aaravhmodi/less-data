from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_lineplot(df: pd.DataFrame, output: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 6), dpi=300)
    for method, group in df.groupby("method"):
        group = group.sort_values("keep_fraction")
        plt.plot(
            group["keep_fraction"] * 100,
            group[ylabel],
            marker="o",
            linewidth=2,
            label=method,
        )
    plt.title(title)
    plt.xlabel("Data kept (%)")
    plt.ylabel(ylabel.replace("_", " ").title())
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
        save_lineplot(
            df.rename(columns={"accuracy": "accuracy"}),
            args.output_dir / "accuracy_vs_retention.jpg",
            "Accuracy vs Retention",
            "accuracy",
        )

    if "training_time_seconds" in df.columns:
        time_df = df.copy()
        time_df["training_time_seconds"] = time_df["training_time_seconds"]
        save_lineplot(
            time_df.rename(columns={"training_time_seconds": "training_time_seconds"}),
            args.output_dir / "time_vs_retention.jpg",
            "Training Time vs Retention",
            "training_time_seconds",
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
