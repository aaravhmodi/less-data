# Research Design

## Positioning

DataValueRank is a data pruning and example valuation study, not just a heuristic subset selection demo.

The paper should answer three questions:

1. Can example scoring match full-data accuracy with fewer samples?
2. Which signal combinations matter most?
3. When does pruning fail?

## Core Hypothesis

Training examples are not equally valuable.

If we can approximate the marginal utility of each example, then training on a smaller high-value subset should preserve or improve accuracy while reducing compute.

## Why This Is Not a Lightweight Heuristic Paper

The method should be treated as a research pipeline with:

- controlled baselines
- multiple datasets
- multiple seeds
- ablation studies
- confidence intervals
- class-level analysis
- robustness checks
- error analysis on removed examples

## Measurement Principles

### Avoid Leakage

Do not compute all scores from the same model that is directly optimized on the target training example.

Prefer one of:

- K-fold out-of-fold predictions
- a frozen teacher model
- a separate scoring model
- cross-validated embeddings

### Compare Like With Like

Use the same architecture family and training budget when comparing pruning methods.

### Report Uncertainty

For each main number, report:

- mean over seeds
- standard deviation
- 95 percent confidence interval if possible

### Report Cost

Always report the compute tradeoff:

- number of examples kept
- wall-clock training time
- percent time saved
- percent memory saved if relevant

## Theoretical Framing

The score should reflect several distinct notions of value:

- informativeness: difficult but learnable examples
- coverage: examples that represent under-covered regions
- rarity: minority class protection
- disagreement: informative borderline cases
- cleanliness: lower estimated label noise

## Important Ablations

At minimum, test:

- loss only
- confidence only
- uniqueness only
- rarity only
- disagreement only
- noise only
- full DataValueRank

Then test the interactions:

- loss + noise
- uniqueness + rarity
- disagreement + confidence
- all signals without noise penalty

## Expected Failure Cases

- minority classes lose performance when rarity is underweighted
- noisy or mislabeled hard examples dominate if loss is overweighted
- confidence can collapse on overconfident wrong predictions
- uniqueness can over-favor outliers

These are not edge cases. They are central to the paper and should be analyzed.
