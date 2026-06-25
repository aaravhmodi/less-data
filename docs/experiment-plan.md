# Experiment Plan

## Objective

Test whether DataValueRank can reduce training data without materially reducing model performance.

## Main Claim To Validate

Training on the highest-value 60 percent of examples can match full-dataset accuracy while reducing training time.

That claim should only be treated as valid if it holds across multiple datasets, model families, and random seeds.

## Experimental Design

### 1. Baseline Training

Train each model on the full dataset.

Record:

- validation accuracy
- test accuracy
- training time
- class-wise metrics
- calibration or robustness metrics if available

### 2. Random Subset Baseline

Train on random subsets of:

- 20 percent
- 40 percent
- 60 percent
- 80 percent

This establishes the cost of simply using less data.

### 3. Heuristic Filtering Baselines

Compare against:

- duplicate removal
- high-loss removal
- low-loss removal

These are important because they isolate whether the value comes from the ranking method or from one of the signals alone.

### 4. DataValueRank

For every training example, compute:

- training loss
- prediction confidence
- embedding uniqueness
- class rarity
- model disagreement
- label noise probability

Then compute a single rank score and keep the top-scoring subset.

## Suggested Scoring Interpretation

- higher confidence often means the model already understands the example
- higher loss may mean the example is difficult or mislabeled
- higher embedding uniqueness means the sample covers a distinct region of input space
- higher class rarity protects minority classes
- higher disagreement can indicate informative edge cases
- higher label noise probability should penalize an example

The key design choice is balancing informativeness against noise.

## Recommended Initial Weighting

Use a simple weighted sum first:

- `loss`: 0.20
- `confidence`: 0.15
- `embedding_uniqueness`: 0.25
- `class_rarity`: 0.15
- `model_disagreement`: 0.15
- `label_noise_probability`: -0.30

Treat this as a starting point, not a final answer.

## Ranking Rule

1. Compute each signal per example.
2. Normalize the signals across the training set.
3. Combine them into a value score.
4. Sort examples from highest value to lowest value.
5. Keep the top `k` percent.

## Retention Sweep

For each dataset, run the following retention rates:

- 20 percent
- 40 percent
- 60 percent
- 80 percent
- 100 percent

This gives a full tradeoff curve.

## Evaluation Table

For each dataset and retention rate, report:

- accuracy
- F1 if relevant
- training time
- percentage of data kept
- percentage of time saved
- per-class accuracy
- robustness score

## Recommended Robustness Checks

- run 3 to 5 seeds
- test on an out-of-distribution split if available
- test class imbalance sensitivity
- inspect examples removed from minority classes

## Failure Modes To Watch

- the method removes hard-but-important examples
- the method overfits to easy samples
- label-noise estimation is poor
- rarity helps minority classes but hurts calibration
- ranking signals correlate too strongly and collapse into one heuristic

## Success Criteria

The method is interesting if it can show one or more of the following:

- same accuracy with less data
- same accuracy with lower training time
- better robustness at fixed training budget
- better class balance than random pruning

## Paper Structure

1. Introduction
2. Related Work
3. DataValueRank Method
4. Experimental Setup
5. Results
6. Ablations
7. Discussion
8. Limitations

## Practical Execution Order

1. Implement the scoring code.
2. Wire in one dataset and one model.
3. Verify the full-data baseline.
4. Verify one 60 percent retention run.
5. Expand to all retention rates.
6. Add the remaining datasets.
7. Add baselines and ablations.
8. Write the paper once the tables are stable.
