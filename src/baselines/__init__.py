"""Baseline models for clinical prediction tasks.

This package contains implementations of various baseline machine learning models
used for comparison with more advanced methods. All models follow a common interface
defined in the BaselineMLModel abstract class.

Available models:
- XGBoostModel: Gradient boosting baseline
- LSTMRNNModel: LSTM-based sequence model
- LogisticRegressionModel: Linear baseline with hyperparameter tuning
"""

from src.baselines.baseline_ml_model import BaselineMLModel
from src.baselines.xgb import XGBoostModel
from src.baselines.lstm_rnn import LSTMRNNModel
from src.baselines.logistic_regression import LogisticRegressionModel

__all__ = [
    'BaselineMLModel',
    'XGBoostModel',
    'LSTMRNNModel',
    'LogisticRegressionModel'
]

