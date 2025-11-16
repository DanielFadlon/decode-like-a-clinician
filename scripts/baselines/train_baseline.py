"""Unified training script for all baseline models.

This script trains any baseline model (XGBoost, LSTM-RNN, or Logistic Regression)
with a unified interface. Models are selected via the --model argument.

Usage:
    python train_baseline.py --model xgb --data_path data.parquet --output_dir results/xgb/
    python train_baseline.py --model lstm --data_path data.parquet --output_dir results/lstm/
    python train_baseline.py --model logistic_regression --data_path data.parquet --output_dir results/lr/
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional

import pandas as pd
import yaml

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.baselines import XGBoostModel, LSTMRNNModel, LogisticRegressionModel


AVAILABLE_MODELS = {
    'xgb': 'XGBoost',
    'lstm': 'LSTM-RNN',
    'logistic_regression': 'Logistic Regression',
    'lr': 'Logistic Regression'  # Alias
}


def load_config(config_path: Optional[str], model_name: str) -> Optional[Dict[str, Any]]:
    """Load hyperparameter configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        model_name: Name of the model

    Returns:
        Dictionary of hyperparameters or None
    """
    if config_path and os.path.exists(config_path):
        print(f"   Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"   Using default configuration for {AVAILABLE_MODELS[model_name]}")
    return None


def main(
    model_name: str,
    data_path: str,
    output_dir: str,
    hyperparams_config_path: Optional[str] = None,
    fill_value: float = -1,
    normalize: bool = True
) -> None:
    """Main function to run baseline model training.

    Args:
        model_name: Name of the model to train
        data_path: Path to input parquet file with data
        output_dir: Directory to save model and results
        config_path: Optional path to YAML config file with hyperparameters
        fill_value: Value to use for filling missing data
        normalize: Whether to normalize features (for logistic regression)
    """
    # Validate model name
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(set(AVAILABLE_MODELS.keys()))}"
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Print header
    print("=" * 70)
    print(f"{AVAILABLE_MODELS[model_name].upper()} BASELINE MODEL TRAINING")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"   Data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")

    # Validate required columns
    required_cols = ['set_type', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check data splits
    split_counts = df['set_type'].value_counts().to_dict()
    print(f"\n📊 Data splits:")
    for split_name in ['train', 'valid', 'test']:
        count = split_counts.get(split_name, 0)
        print(f"   {split_name.capitalize()}: {count:,} samples")

    # Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n⚠️  Filling {missing_count:,} missing values with {fill_value}")
        df = df.fillna(fill_value)

    # Load hyperparameters
    print(f"\n⚙️  Loading hyperparameters...")
    hyperparams = load_config(hyperparams_config_path, model_name)

    # Train model based on type
    if model_name == 'xgb':
        model = XGBoostModel(data=df, hyperparams=hyperparams)
    elif model_name == 'lstm':
        model = model = LSTMRNNModel(df=df, hyperparams=hyperparams)
    elif model_name in ['logistic_regression', 'lr']:
        resmodelults = LogisticRegressionModel(data=df, hyperparams=hyperparams, normalize=normalize)

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    model.train()

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    results = model.evaluate(output_dir_path=output_dir)

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    model.save_model(output_dir)


    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED")
    print("=" * 70)
    print(f"\n📊 Results Summary:")
    print(f"   Model:          {AVAILABLE_MODELS[model_name]}")
    print(f"   Train AUC:      {results['train']:.4f}")
    print(f"   Validation AUC: {results['valid']:.4f}")
    print(f"   Test AUC:       {results['test']:.4f}")
    print(f"\n💾 Output saved to: {output_dir}")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baseline machine learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Train XGBoost with default hyperparameters
    python train_baseline.py --model xgb --data_path data/train_data.parquet --output_dir results/xgb/

    # Train XGBoost with custom config
    python train_baseline.py --model xgb --data_path data/train_data.parquet --output_dir results/xgb/ --config_path configurations/config_xgb.yaml

    # Train LSTM
    python train_baseline.py --model lstm --data_path data/train_data.parquet --output_dir results/lstm/ --config_path configurations/config_lstm.yaml

    # Train Logistic Regression
    python train_baseline.py --model lr --data_path data/train_data.parquet --output_dir results/lr/ --normalize

    # Train all models (bash loop)
    for model in xgb lstm lr; do
        python train_baseline.py --model $model --data_path data/train_data.parquet --output_dir results/$model/
    done

    Available models:
    xgb, logistic_regression (or lr), lstm
            """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['xgb', 'lstm', 'logistic_regression', 'lr'],
        help="Model type to train: xgb, lstm, logistic_regression (or lr)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input parquet file containing train/valid/test data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trained model and results"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to YAML configuration file with hyperparameters (optional)"
    )
    parser.add_argument(
        "--fill_value",
        type=float,
        default=-1,
        help="Value to use for filling missing data (default: -1)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Whether to normalize features (for logistic regression, default: True)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    if args.config_path and not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    main(
        args.model,
        args.data_path,
        args.output_dir,
        args.config_path,
        args.fill_value,
        args.normalize
    )

