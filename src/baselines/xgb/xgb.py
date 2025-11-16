"""XGBoost baseline model implementation."""

import os
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


class XGBoostModel:
    """XGBoost classifier for baseline comparison.

    This model provides a gradient boosting baseline using XGBoost library.
    It supports hyperparameter configuration via dictionary and handles
    train/validation/test splits automatically.

    Attributes:
        data: DataFrame with 'set_type' and 'label' columns
        hyperparams: Dictionary containing XGBoost hyperparameters
        model: Trained XGBClassifier instance
    """

    def __init__(self, data: pd.DataFrame, hyperparams: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model.

        Args:
            data: DataFrame with 'set_type' and 'label' columns
            hyperparams: Dictionary containing hyperparameters for XGBoost.
                        Expected structure:
                        {
                            'learning': {
                                'model_params': {
                                    'xgboost': {
                                        'train_params': {...},
                                        'fit_params': {...}
                                    }
                                }
                            }
                        }
        """
        self.data = data
        self.hyperparams = hyperparams or {}
        self.model: Optional[XGBClassifier] = None

    @staticmethod
    def _get_split_data(df: pd.DataFrame, set_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and labels for a specific data split.

        Args:
            df: Input dataframe with 'set_type', 'label' columns
            set_type: One of 'train', 'valid', or 'test'

        Returns:
            Tuple of (features_df, labels_series)
        """
        split_df = df[df['set_type'] == set_type]
        X = split_df.drop(columns=['label', 'set_type'])
        y = split_df['label']
        return X, y

    def train(self) -> None:
        """Train XGBoost model on training data.

        Extracts train_params and fit_params from hyperparams and trains the model.
        """
        X_train, y_train = self._get_split_data(self.data, set_type='train')

        # Extract hyperparameters
        train_params = self.hyperparams.get('learning', {}).get('model_params', {}).get('xgboost', {}).get('train_params', {})
        fit_params = self.hyperparams.get('learning', {}).get('model_params', {}).get('xgboost', {}).get('fit_params', {})

        # Initialize and train model
        self.model = XGBClassifier(**train_params)
        self.model.fit(X_train, y_train.astype(int), **fit_params)

        print(f"XGBoost model trained with {X_train.shape[0]} samples and {X_train.shape[1]} features")

    def evaluate(self, output_dir_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate model on all data splits.

        Args:
            output_dir_path: Optional directory to save evaluation results and predictions

        Returns:
            Dictionary with AUC scores for train, validation, and test sets

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation. Call train() first.")

        # Evaluate on all splits
        results = {}
        for split_name in ['train', 'valid', 'test']:
            X, y = self._get_split_data(self.data, set_type=split_name)
            y_proba = self.model.predict_proba(X)[:, 1]
            auc_score = roc_auc_score(y, y_proba)
            results[split_name] = {
                'auc': float(auc_score),
                'predictions': y_proba
            }
            print(f"{split_name.capitalize()} AUC: {auc_score:.4f}")

        # Save results if output directory provided
        if output_dir_path is not None:
            os.makedirs(output_dir_path, exist_ok=True)

            # Save predictions for validation and test sets
            for split_name in ['valid', 'test']:
                split_df = self.data[self.data['set_type'] == split_name].copy()
                split_df['predicted_probability'] = results[split_name]['predictions']
                output_path = os.path.join(output_dir_path, f'{split_name}_predictions.csv')
                split_df.to_csv(output_path, index=False)

            # Save AUC scores
            results_path = os.path.join(output_dir_path, 'auc_scores.txt')
            with open(results_path, 'w') as f:
                for split_name in ['train', 'valid', 'test']:
                    f.write(f"{split_name.capitalize()} AUC: {results[split_name]['auc']:.4f}\n")

        # Return clean results without predictions array
        return {
            'results_type': 'auc',
            'train': results['train']['auc'],
            'valid': results['valid']['auc'],
            'test': results['test']['auc']
        }

    def save_model(self, output_dir_path: str) -> None:
        """Save trained model to disk.

        Args:
            output_dir_path: Directory where model will be saved

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        os.makedirs(output_dir_path, exist_ok=True)
        model_path = os.path.join(output_dir_path, 'xgboost_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk.

        Args:
            model_path: Path to the saved model file

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
