# DataValueRank

Can we train better models by throwing away bad data?

This repository is a research scaffold for the paper idea:

**DataValueRank: Can We Train Better Models by Throwing Away Bad Data?**

The core hypothesis is simple:

- many datasets contain duplicates, noisy labels, low-information samples, and redundant easy examples
- if we can score each example by usefulness, we can keep only the highest-value subset
- a smaller, cleaner training set may match full-data accuracy while reducing training time

## What This Project Contains

- a research-grade experiment plan
- a ranking model for scoring individual training examples
- a baseline protocol for comparison
- a manuscript outline with figure plan
- figure generation scripts that export JPEGs
- a place to plug in datasets, models, and training loops

## Research Question

Can we keep only the most useful training examples and still preserve accuracy?

## Method Summary

For each example, DataValueRank combines:

- training loss
- prediction confidence
- embedding uniqueness
- class rarity
- model disagreement
- label noise probability

The examples with the highest final score are kept for training.

Important: the raw signals should be computed in a leakage-safe way. For example, use
cross-validated predictions, a held-out teacher model, or a frozen reference encoder
instead of the final model trained on the same example.

## Baselines

- train on full dataset
- random subset
- remove duplicates
- remove high-loss examples
- remove low-loss examples

## Target Datasets

- CIFAR-10
- Fashion-MNIST
- AG News
- IMDb sentiment
- tabular fraud/churn datasets

## Metrics

- accuracy vs percentage of data kept
- training time reduction
- robustness
- class-level performance
- examples removed

## What You Need To Do

1. Choose the model family for each dataset.
2. Download or prepare the datasets.
3. Run the full-data baseline first.
4. Compute the per-example scoring signals in a leakage-safe way.
5. Rank examples and keep the top percentage.
6. Retrain on the selected subset.
7. Compare against the baselines.
8. Save the results in a table and generate JPEG plots.
9. Repeat across seeds and datasets.
10. Report confidence intervals, not just single-run numbers.

## Recommended Initial Experiments

Start small and reproducible:

1. Fashion-MNIST with a simple CNN
2. IMDb with a lightweight text classifier
3. One tabular dataset with XGBoost or logistic regression

Then scale to the rest of the benchmark suite.

## Repository Layout

- `docs/` project notes and experiment protocol
- `src/datavaluerank/` scoring and experiment scaffold

## Status

This is a scaffold, not a finished paper. The code structure and documentation are in place, but you still need to connect real dataset loaders, model training, and result collection.
