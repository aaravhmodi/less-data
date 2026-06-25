from __future__ import annotations

import argparse
from pathlib import Path

from datavaluerank.benchmark import run_fashion_mnist_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fashion-MNIST DataValueRank benchmark.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-folds", type=int, default=3)
    args = parser.parse_args()

    run_fashion_mnist_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        teacher_folds=args.teacher_folds,
    )


if __name__ == "__main__":
    main()
