# Evaluation Scripts

### Features

- **Evaluate a specific model checkpoint**: Evaluate a single trained model
- **Evaluate all epochs**: Evaluate all checkpoints in a directory
- **Flexible dataset selection**: Choose which splits to evaluate (train, valid, test, or all)
- **Configuration-based**: Uses YAML config files for easy parameter management
- **Clean argument parsing**: Uses argparse for better CLI experience

### Usage

#### Evaluate a Specific Model Checkpoint

```bash
python scripts/evaluation/evaluate.py \
    --config path/to/config.yaml \
    --model_path path/to/checkpoint-1000 \
    --output_dir results/eval_checkpoint_1000 \
    --eval_set test
```

**Arguments:**
- `--config`: Path to YAML configuration file (required)
- `--model_path`: Path to specific model checkpoint to evaluate (required for specific mode)
- `--output_dir`: Output directory for evaluation results (required when using --model_path)
- `--eval_set`: Which dataset split(s) to evaluate on: `train`, `valid`, `test`, or `all` (default: `all`)

#### Evaluate All Epochs/Checkpoints

```bash
python scripts/evaluation/evaluate.py \
    --config path/to/config.yaml \
    --models_dir path/to/models_directory \
    --eval_set all
```

**Arguments:**
- `--config`: Path to YAML configuration file (required)
- `--models_dir`: Directory containing multiple model checkpoints (required for all-epochs mode)
- `--eval_set`: Which dataset split(s) to evaluate on: `valid`, `test`, or `all` (default: `all`)

### Configuration File

The YAML configuration file should contain:

```yaml
dataset_dir: /path/to/dataset
prompt_template_func_path: module.path.to.prompt_function
template_name: clinic_instruction  # optional, default: clinic_instruction
dataset_type: parquet  # optional, default: parquet
report_pred_threshold: 0.5  # optional, default: 0.5
model_prefix: checkpoint-  # optional, default: checkpoint-
```

### Examples

**Example 1: Evaluate a specific checkpoint on test set**
```bash
python scripts/evaluation/evaluate.py \
    --config configurations/config_model.yaml \
    --model_path outputs/training_run/checkpoint-5000 \
    --output_dir results/checkpoint_5000_eval \
    --eval_set test
```

**Example 2: Evaluate all checkpoints on validation set**
```bash
python scripts/evaluation/evaluate.py \
    --config configurations/config_model.yaml \
    --models_dir outputs/training_run \
    --eval_set valid
```

**Example 3: Evaluate a checkpoint on all available splits**
```bash
python scripts/evaluation/evaluate.py \
    --config configurations/config_model.yaml \
    --model_path outputs/training_run/checkpoint-final \
    --output_dir results/final_eval \
    --eval_set all
```
