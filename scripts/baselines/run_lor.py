"""Logistic Regression baseline model training script.

This script trains a logistic regression model with hyperparameter tuning using
grid search with cross-validation. The model is trained on provided data with
train/validation/test splits and evaluated using AUC score.

Usage:
    python run_logistic_regression.py --data_path path/to/data.parquet --output_dir path/to/results
"""

import argparse
import os
import sys

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.baselines.logistic_regression import LogisticRegressionModel


def main(data_path: str, output_dir: str, normalize: bool = True) -> None:
    """Main function to run logistic regression pipeline.

    Args:
        data_path: Path to input parquet file with data
        output_dir: Directory to save model and results
        normalize: Whether to normalize features
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    # Initialize model
    model = LogisticRegressionModel(data=df, normalize=normalize)

    # Train model
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    model.train()

    # Evaluate model
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    results = model.evaluate(output_dir_path=output_dir)

    # Save model
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    model.save_model(output_dir)

    print("\n" + "="*50)
    print("COMPLETED")
    print("="*50)
    print(f"Results: {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression baseline model with hyperparameter tuning"
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
        "--normalize",
        action="store_true",
        default=True,
        help="Whether to normalize features using StandardScaler (default: True)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    main(args.data_path, args.output_dir, args.normalize)
