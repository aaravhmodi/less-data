# Paper Outline

## Title

DataValueRank: Can We Train Better Models by Throwing Away Bad Data?

## Abstract

State the problem, method, datasets, and headline result.

## 1. Introduction

- why redundant and noisy training data matter
- why data selection is harder than random pruning
- the contribution of DataValueRank

## 2. Related Work

- dataset pruning
- sample selection
- curriculum learning
- active learning
- noisy label detection
- example influence and importance scoring

## 3. Method

- signal definitions
- leakage-safe scoring pipeline
- score aggregation
- retention policy
- training protocol

## 4. Experimental Setup

- datasets
- models
- preprocessing
- evaluation metrics
- compute budget
- seeds and splits

## 5. Results

- accuracy vs retention
- training time reductions
- robustness
- class-level analysis

## 6. Ablations

- signal-by-signal analysis
- weighting sensitivity
- cross-dataset consistency

## 7. Error Analysis

- examples removed incorrectly
- minority class failures
- noise vs hardness confusion

## 8. Limitations

- dependence on a scoring model
- potential leakage
- dataset-specific tuning

## 9. Conclusion

- what was learned
- when the method is worth using

## Figure Checklist

- accuracy retention curve
- training time curve
- per-class performance chart
- ablation chart
- score distribution plot
