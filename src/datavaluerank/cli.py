from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_fashion_mnist_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="DataValueRank research benchmark runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    fm = sub.add_parser("fashion-mnist", help="Run the Fashion-MNIST benchmark")
    fm.add_argument("--data-dir", type=Path, required=True)
    fm.add_argument("--output-dir", type=Path, required=True)
    fm.add_argument("--epochs", type=int, default=6)
    fm.add_argument("--batch-size", type=int, default=128)
    fm.add_argument("--teacher-folds", type=int, default=3)

    args = parser.parse_args()

    if args.command == "fashion-mnist":
        run_fashion_mnist_experiment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            teacher_folds=args.teacher_folds,
        )


if __name__ == "__main__":
    main()
