# Benchmark Protocol

## Benchmark Goal

Make DataValueRank comparable to standard pruning baselines on multiple datasets under a consistent protocol.

## Common Settings

For each dataset:

- use the same train/validation/test split across methods
- keep the model architecture fixed within a dataset
- use the same optimizer, batch size, and number of epochs
- tune pruning weights only on the validation split

## Must-Run Baselines

- full data
- random subset
- duplicate removal
- high-loss removal
- low-loss removal
- DataValueRank

## Retention Sweep

Evaluate at:

- 20 percent
- 40 percent
- 60 percent
- 80 percent
- 100 percent

## Reporting Checklist

- report accuracy and macro F1
- report training time
- report per-class performance
- report at least 3 seeds
- report confidence intervals where possible

## Figure Output

All main plots should be exported as JPEG.

## Result Interpretation

The benchmark is successful if the method can consistently reduce training data while preserving performance and not disproportionately harming minority classes.
