# What You Need To Do

This is the shortest practical checklist for turning this idea into a paper.

## Phase 1: Setup

1. Pick the first dataset.
2. Pick one model architecture.
3. Decide where the data files will live.
4. Install the Python dependencies.

## Phase 2: Baseline

1. Train on the full dataset.
2. Save validation and test metrics.
3. Save the training time.
4. Repeat with at least 3 seeds.

## Phase 3: Scoring

1. Compute per-example training loss.
2. Compute prediction confidence.
3. Compute embedding uniqueness.
4. Compute class rarity.
5. Compute model disagreement.
6. Estimate label noise probability.

## Phase 4: Pruning

1. Normalize the scores.
2. Rank all examples.
3. Keep the top 20, 40, 60, 80, and 100 percent.
4. Retrain each subset.

## Phase 5: Comparison

1. Run the random-subset baseline.
2. Run duplicate removal.
3. Run high-loss and low-loss filters.
4. Compare against DataValueRank.

## Phase 6: Analysis

1. Make accuracy-vs-retention plots.
2. Make training-time-vs-retention plots.
3. Inspect which examples were removed.
4. Check class-level performance.

## Phase 7: Writeup

1. Describe the scoring rule clearly.
2. Explain the datasets and models.
3. Report all baselines.
4. Include ablations.
5. State failure cases honestly.

## Important Practical Note

Do not start with all datasets at once. Validate the pipeline on one dataset first, otherwise debugging becomes much harder.
