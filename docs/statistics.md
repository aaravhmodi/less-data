# Statistical Reporting

## Why This Matters

If the method wins by 0.2 points on a single seed, that is not enough evidence.

The paper should report variance and uncertainty.

## Minimum Reporting Standard

For each dataset and method:

- mean accuracy across seeds
- standard deviation across seeds
- mean training time across seeds
- standard deviation across seeds
- best and worst run if relevant

## Recommended Significance Tests

Use paired comparisons when the same seeds or splits are shared:

- paired t-test
- Wilcoxon signed-rank test
- bootstrap confidence interval

## Recommended Effect Size Reporting

Report one or more of:

- absolute accuracy difference
- relative training time reduction
- Cohen's d for repeated measurements

## Suggested Table Format

| dataset | method | keep % | accuracy mean | accuracy std | time mean | time std |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | DataValueRank | 60 | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

## Interpretation Rule

Do not claim a method is better unless the improvement is stable across seeds and does not come at an unacceptable cost to minority classes or robustness.
