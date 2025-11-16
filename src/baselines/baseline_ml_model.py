"""Base class for baseline machine learning models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class BaselineMLModel(ABC):
    """Abstract base class for all baseline ML models.

    This class defines a common interface that all baseline models should implement,
    ensuring consistency across different model types (XGBoost, LSTM, Logistic Regression, etc.).

    Attributes:
        data: The input dataframe containing features and labels
        hyperparams: Dictionary of hyperparameters for model training
        model: The trained model instance
    """

    def __init__(self, data: pd.DataFrame, hyperparams: Optional[Dict[str, Any]] = None):
        """Initialize the baseline model.

        Args:
            data: Input dataframe with 'set_type' and 'label' columns
            hyperparams: Dictionary of hyperparameters (optional)
        """
        self.data = data
        self.hyperparams = hyperparams or {}
        self.model = None

    @abstractmethod
    def train(self) -> None:
        """Train the model on the training data.

        This method should:
        - Extract training data from self.data
        - Initialize and train self.model
        - Store any training artifacts needed for evaluation
        """
        pass

    @abstractmethod
    def evaluate(self, output_dir_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the model on train, validation, and test sets.

        Args:
            output_dir_path: Optional path to save evaluation results

        Returns:
            Dictionary containing evaluation metrics for each dataset split
        """
        pass

    @abstractmethod
    def save_model(self, output_dir_path: str) -> None:
        """Save the trained model to disk.

        Args:
            output_dir_path: Directory path where model should be saved
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk.

        Args:
            model_path: Path to the saved model file
        """
        pass
