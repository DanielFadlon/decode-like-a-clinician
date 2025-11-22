# Evaluation Scripts

## Evaluate - `evaluate.py`

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


## Zero Shot Evaluation - `zero_shot_evaluate.py`


Main zero-shot evaluation script that:
- Loads a pre-trained language model
- Creates clinical prompts using domain-specific templates
- Extracts next-token probabilities for classification tokens (0/1 or False/True)
- Calculates evaluation metrics (accuracy, precision, recall, F1, AUC-ROC)
- Identifies and logs invalid responses

### Usage
_________
### Using Configuration File (Recommended)

```bash
python scripts/evaluation/zero_shot_evaluate.py --config configurations/zero_shot_config.yaml
```

### Using Command Line Arguments

```bash
python scripts/evaluation/zero_shot_evaluate.py \
    --model_path path/to/model \
    --data_path data/test.parquet \
    --output_dir results/zero-shot/ \
    --data_name mimic \
    --true_token_id 29896 \
    --false_token_id 29900 \
    --trust_remote_code
```

### Configuration File Format
-------
Create a YAML configuration file with the following fields:

```yaml
# Model configuration
model_path: "path/to/model"
trust_remote_code: true
torch_dtype: float16

# Data configuration
data_path: "data/test.parquet"
data_name: mimic  # Options: mimic, tasmc, shebamc

# Token IDs for binary classification
true_token_id: 29896   # Token ID for positive class
false_token_id: 29900  # Token ID for negative class

# Output configuration
output_dir: "results/zero-shot/"
```

## Determining Token IDs

To find the correct token IDs for your model:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/model", trust_remote_code=True)

# For models that predict '1' and '0'
true_id = tokenizer.encode('1', add_special_tokens=False)[0]
false_id = tokenizer.encode('0', add_special_tokens=False)[0]

print(f"True Token ID: {true_id}")
print(f"False Token ID: {false_id}")
```
