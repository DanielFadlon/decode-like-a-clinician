# Baseline Models

This directory contains implementations of baseline machine learning models for clinical prediction tasks. These models serve as comparison baselines for more advanced methods.

## Available Models

### 1. XGBoost Model (`xgb/`)

A gradient boosting classifier using the XGBoost library.

**Features:**
- Hyperparameter configuration via dictionary
- Automatic train/validation/test split handling
- AUC evaluation metrics
- Model persistence (save/load)

**Usage:**
```python
from src.baselines import XGBoostModel

# Initialize model
model = XGBoostModel(data=df, hyperparams=config)

# Train
model.train()

# Evaluate
results = model.evaluate(output_dir_path='results/')

# Save model
model.save_model('models/')
```

### 2. LSTM-RNN Model (`lstm_rnn/`)

A sequence-based LSTM model for time-series classification.

**Features:**
- Automatic sequence creation from time-series data
- Stacked LSTM layers with batch normalization
- Early stopping and learning rate scheduling
- Training curve visualization
- StandardScaler normalization

**Usage:**
```python
from src.baselines import LSTMRNNModel

# Initialize model
model = LSTMRNNModel(df=data, hyperparams=config)

# Train
model.train()

# Evaluate
results = model.evaluate(output_dir_path='results/')

# Save model
model.save_model('models/')
```

### 3. Logistic Regression Model (`logistic_regression/`)

A logistic regression baseline with grid search hyperparameter tuning.

**Features:**
- Grid search with cross-validation
- StandardScaler normalization
- Predefined train/validation splits
- AUC evaluation

**Usage (as class):**
```python
from src.baselines import LogisticRegressionModel

# Initialize model
model = LogisticRegressionModel(data=df, normalize=True)

# Train
model.train()

# Evaluate
results = model.evaluate(output_dir_path='results/')

# Save model
model.save_model('models/')
```

**Usage (as script):**
```bash
python scripts/baselines/run_lor.py \
    --data_path data/train_data.parquet \
    --output_dir results/logistic_regression/ \
    --normalize
```

## Common Interface

All baseline models follow a common pattern:

1. **Initialization**: Pass data and hyperparameters
2. **Training**: Call `train()` method
3. **Evaluation**: Call `evaluate()` method with optional output directory
4. **Persistence**: Use `save_model()` and `load_model()` methods

## Data Format Requirements

All models expect data in the following format:

- **DataFrame columns**: 
  - `set_type`: One of 'train', 'valid', or 'test'
  - `label`: Binary target variable (0 or 1)
  - Additional columns: Feature variables

## Hyperparameter Configuration

### XGBoost
```python
hyperparams = {
    'learning': {
        'model_params': {
            'xgboost': {
                'train_params': {
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc'
                },
                'fit_params': {
                    'verbose': True
                }
            }
        }
    }
}
```

### LSTM-RNN
```python
hyperparams = {
    'history_volume': 10,  # Sequence length
    'layers': [64, 32, 16, 1],  # LSTM units + output
    'learning_rate': 0.001,
    'lambda': 0.01,  # L2 regularization
    'dropout': 0.2,
    'recurrent_dropout': 0.2,
    'activation': 'tanh',
    'recurrent_activation': 'sigmoid',
    'batch_size': 512,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'lr_patience': 2
}
```

### Logistic Regression
```python
hyperparams = {
    'param_grid': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000, 5000]
    }
}
```

## Output Structure

All models generate consistent output when evaluated:

```python
{
    'results_type': 'auc',
    'train': 0.85,  # Train AUC
    'valid': 0.82,  # Validation AUC
    'test': 0.80    # Test AUC
}
```

When `output_dir_path` is provided, models save:
- `auc_scores.txt` or `results.txt`: Text file with AUC scores
- `{split}_predictions.csv`: Predictions for validation and test sets
- `training_curves.png` (LSTM only): Training and validation curves

## Directory Structure

```
src/baselines/
├── __init__.py                     # Package initialization with exports
├── README.md                       # This file
├── baseline_ml_model.py            # Abstract base class
├── xgb/                            # XGBoost implementation
│   ├── __init__.py
│   └── xgb.py
├── lstm_rnn/                       # LSTM-RNN implementation
│   ├── __init__.py
│   └── lstm_rnn.py
└── logistic_regression/            # Logistic Regression implementation
    ├── __init__.py
    └── logistic_regression.py

scripts/baselines/
├── __init__.py
└── run_lor.py                      # Logistic regression training script
```

## Contributing

When adding new baseline models:

1. Follow the interface defined in `BaselineMLModel` (optional but recommended)
2. Include comprehensive docstrings with Google-style format
3. Add type hints for all public methods and function parameters
4. Implement `train()`, `evaluate()`, `save_model()`, and `load_model()` methods
5. Ensure consistent return format for `evaluate()` method
6. Update this README with usage examples
7. Add appropriate `__init__.py` exports
8. Write clear, professional code suitable for pull requests

## Code Quality Standards

All code in this directory follows these standards:

- **Type Hints**: All function signatures include type hints
- **Docstrings**: Google-style docstrings for all classes and methods
- **Error Handling**: Proper validation and informative error messages
- **Logging**: Clear print statements for training progress
- **Naming**: Descriptive variable and function names (PEP 8 compliant)
- **Organization**: Clean file structure with logical grouping

## Dependencies

- **XGBoost**: `xgboost`, `scikit-learn`, `pandas`, `numpy`, `joblib`
- **LSTM-RNN**: `tensorflow`, `keras`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Logistic Regression**: `scikit-learn`, `pandas`, `numpy`, `joblib`

## Examples

### Complete Training Pipeline

```python
import pandas as pd
from src.baselines import XGBoostModel, LSTMRNNModel, LogisticRegressionModel

# Load data
df = pd.read_parquet('data/train_data.parquet')
df = df.fillna(-1)  # Handle missing values

# XGBoost
xgb_model = XGBoostModel(data=df, hyperparams=xgb_config)
xgb_model.train()
xgb_results = xgb_model.evaluate(output_dir_path='results/xgb/')
xgb_model.save_model('models/xgb/')

# LSTM
lstm_model = LSTMRNNModel(df=df, hyperparams=lstm_config)
lstm_model.train()
lstm_results = lstm_model.evaluate(output_dir_path='results/lstm/')
lstm_model.save_model('models/lstm/')

# Logistic Regression
lr_model = LogisticRegressionModel(data=df, normalize=True)
lr_model.train()
lr_results = lr_model.evaluate(output_dir_path='results/lr/')
lr_model.save_model('models/lr/')

# Compare results
print("XGBoost Test AUC:", xgb_results['test'])
print("LSTM Test AUC:", lstm_results['test'])
print("Logistic Regression Test AUC:", lr_results['test'])
```

