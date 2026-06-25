# Figure Plan

This paper should include figures that are directly tied to the claims.

## Required Figures

### 1. Accuracy vs Retention

Line chart.

X-axis:

- percentage of data kept

Y-axis:

- test accuracy or macro F1

Series:

- full data
- random subset
- duplicate removal
- high-loss removal
- low-loss removal
- DataValueRank

### 2. Training Time vs Retention

Line chart.

Show wall-clock training time reduction as data is pruned.

### 3. Per-Class Performance

Grouped bar chart or heatmap.

Show whether pruning hurts minority or difficult classes.

### 4. Ablation Study

Bar chart.

Show performance for each signal removed from the scoring function.

### 5. Score Distribution

Histogram or density plot.

Compare score distributions of kept vs removed examples.

### 6. Confusion Matrix Delta

Heatmap.

Compare full-data vs pruned-data confusion matrices.

## Output Format

Export all figures as JPEG files for easy insertion into a paper draft.

Suggested folder:

- `figures/accuracy_vs_retention.jpg`
- `figures/time_vs_retention.jpg`
- `figures/class_performance.jpg`
- `figures/ablation.jpg`
- `figures/score_distribution.jpg`

## Figure Quality Requirements

- 300 DPI if possible
- readable axis labels
- consistent color palette
- legend outside the plotting area when crowded
- no default notebook styling

## What To Avoid

- one-off screenshots
- unlabeled plots
- mixed scales without explanation
- hiding the baseline
