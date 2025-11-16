"""Logistic Regression baseline model implementation."""

import os
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:
    """Logistic Regression classifier with hyperparameter tuning.

    This model provides a linear baseline using scikit-learn's LogisticRegression
    with automatic grid search for hyperparameter optimization. It handles
    train/validation/test splits and optional feature normalization.

    Attributes:
        data: DataFrame with 'set_type' and 'label' columns
        hyperparams: Dictionary containing model hyperparameters
        model: Trained LogisticRegression instance
        scaler: StandardScaler for feature normalization (if normalize=True)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        hyperparams: Optional[Dict[str, Any]] = None,
        normalize: bool = True
    ):
        """Initialize Logistic Regression model.

        Args:
            data: DataFrame with 'set_type' and 'label' columns
            hyperparams: Dictionary containing hyperparameters for grid search.
                        If None, uses default parameter grid.
            normalize: Whether to normalize features using StandardScaler
        """
        self.data = data
        self.hyperparams = hyperparams or self._default_hyperparams()
        self.normalize = normalize
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.best_params_: Optional[Dict[str, Any]] = None

    @staticmethod
    def _default_hyperparams() -> Dict[str, Any]:
        """Return default hyperparameter grid for grid search."""
        return {
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [1000, 2000, 5000]
            }
        }

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
        """Train Logistic Regression model with grid search hyperparameter tuning.

        Uses combined train+validation data with predefined splits for grid search.
        Trains final model on best hyperparameters found during search.
        """
        # Extract data splits
        X_train, y_train = self._get_split_data(self.data, set_type='train')
        X_valid, y_valid = self._get_split_data(self.data, set_type='valid')

        print(f"Training with {X_train.shape[0]} train samples and {X_valid.shape[0]} validation samples")
        print(f"Number of features: {X_train.shape[1]}")

        # Combine train and validation for grid search
        X_combined = pd.concat([X_train, X_valid])
        y_combined = pd.concat([y_train, y_valid])

        # Normalize features if requested
        if self.normalize:
            print("Normalizing features using StandardScaler...")
            self.scaler = StandardScaler()
            X_combined = self.scaler.fit_transform(X_combined)

        # Create predefined split for grid search
        # -1 indicates training set, 0 indicates validation set
        test_fold = [-1] * len(X_train) + [0] * len(X_valid)
        predefined_split = PredefinedSplit(test_fold)

        # Get parameter grid
        param_grid = self.hyperparams.get('param_grid', self._default_hyperparams()['param_grid'])

        print("\nPerforming grid search with cross-validation...")
        print(f"Hyperparameter grid: {param_grid}")

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000),
            param_grid=param_grid,
            scoring='roc_auc',
            cv=predefined_split,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_combined, y_combined)

        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        print(f"\nBest Hyperparameters: {self.best_params_}")
        print(f"Best Validation AUC: {grid_search.best_score_:.4f}")

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

            # Normalize if scaler was fitted during training
            if self.scaler is not None:
                X = self.scaler.transform(X)

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

            # Save AUC scores and best hyperparameters
            results_path = os.path.join(output_dir_path, 'results.txt')
            with open(results_path, 'w') as f:
                for split_name in ['train', 'valid', 'test']:
                    f.write(f"{split_name.capitalize()} AUC: {results[split_name]['auc']:.4f}\n")
                f.write("\nBest Hyperparameters:\n")
                if self.best_params_:
                    for param_name, param_value in self.best_params_.items():
                        f.write(f"  {param_name}: {param_value}\n")

        # Return clean results without predictions array
        return {
            'results_type': 'auc',
            'train': results['train']['auc'],
            'valid': results['valid']['auc'],
            'test': results['test']['auc']
        }

    def save_model(self, output_dir_path: str) -> None:
        """Save trained model and scaler to disk.

        Args:
            output_dir_path: Directory where model will be saved

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        os.makedirs(output_dir_path, exist_ok=True)

        # Save model
        model_path = os.path.join(output_dir_path, 'logistic_regression_model.joblib')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        # Save scaler if it exists
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir_path, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path: str, scaler_path: Optional[str] = None) -> None:
        """Load a trained model from disk.

        Args:
            model_path: Path to the saved model file
            scaler_path: Optional path to the saved scaler file

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

        # Load scaler if path provided
        if scaler_path is not None:
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")

