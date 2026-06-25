# Data and Result Schema

## Example-Level Input Schema

Each training example should have the following metadata available before pruning:

- `example_id`
- `dataset`
- `label`
- `features` or raw input reference
- `training_loss`
- `prediction_confidence`
- `embedding_uniqueness`
- `class_rarity`
- `model_disagreement`
- `label_noise_probability`

## Result-Level Output Schema

Each experiment row should contain:

- `dataset`
- `method`
- `seed`
- `keep_fraction`
- `accuracy`
- `macro_f1`
- `training_time_seconds`
- `parameter_count`
- `notes`

## Optional Analysis Columns

Add these if available:

- `precision`
- `recall`
- `auroc`
- `balanced_accuracy`
- `memory_mb`
- `examples_removed`
- `class_name`
- `class_accuracy`

## Why This Schema Helps

It makes it easier to:

- aggregate results
- generate plots
- compare methods across datasets
- write the paper tables without manual cleanup
