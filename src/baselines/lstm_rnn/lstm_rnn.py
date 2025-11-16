"""LSTM-RNN baseline model implementation."""

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import LSTM, BatchNormalization, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2


class LSTMRNNModel:
    """LSTM-RNN model for sequence-based classification.

    This model creates sequences from time-series data and uses stacked LSTM layers
    for classification. It automatically handles data preprocessing, normalization,
    sequence creation, and model training.

    Attributes:
        df: Input dataframe with 'set_type' and 'label' columns
        hyperparams: Dictionary containing model hyperparameters
        X_train, y_train: Training sequences and labels
        X_valid, y_valid: Validation sequences and labels
        X_test, y_test: Test sequences and labels
        model: Compiled Keras Sequential model
        history: Training history from model.fit()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        hyperparams: Optional[Dict[str, Any]] = None,
        split_data_fn: Optional[callable] = None
    ):
        """Initialize LSTM-RNN model.

        Args:
            df: Input dataframe with 'set_type' and 'label' columns
            hyperparams: Dictionary containing model hyperparameters
            split_data_fn: Optional function to split data into train/valid/test.
                          If None, expects data to already be split with 'set_type' column.

        Raises:
            ValueError: If hyperparams is None
        """
        if hyperparams is None:
            raise ValueError("hyperparams dictionary is required")

        self.df = df
        self.hyperparams = hyperparams
        self.split_data_fn = split_data_fn

        print("Model Hyperparameters:")
        print(self.hyperparams)
        print()

        # Initialize data containers
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_valid: Optional[np.ndarray] = None
        self.y_valid: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Preprocess data and build model
        self._preprocess_data()
        self.model = self._build_model()
        self.history = None


    def _normalize(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize features using StandardScaler for each data split.

        Returns:
            Tuple of (train_df, valid_df, test_df) with normalized features
        """
        # Fill missing values
        self.df = self.df.fillna(0)

        def normalize_split(df: pd.DataFrame) -> pd.DataFrame:
            """Normalize a single data split."""
            features = df.drop(columns=['set_type', 'label'])
            scaler = StandardScaler()
            scaled_features = pd.DataFrame(
                scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
            # Add back metadata columns
            scaled_features['set_type'] = df['set_type'].values
            scaled_features['label'] = df['label'].values
            return scaled_features

        # Split data based on set_type column or use provided function
        if self.split_data_fn is not None:
            train_df, valid_df, test_df = self.split_data_fn(self.df)
        else:
            train_df = self.df[self.df['set_type'] == 'train']
            valid_df = self.df[self.df['set_type'] == 'valid']
            test_df = self.df[self.df['set_type'] == 'test']

        # Normalize each split independently
        train_normalized = normalize_split(train_df)
        valid_normalized = normalize_split(valid_df)
        test_normalized = normalize_split(test_df)

        return train_normalized, valid_normalized, test_normalized

    @staticmethod
    def _create_sequences(
        df: pd.DataFrame,
        sequence_length: int,
        split_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from time-series data.

        Args:
            df: DataFrame with normalized features and labels
            sequence_length: Number of time steps in each sequence
            split_name: Name of the split ('train', 'valid', or 'test')

        Returns:
            Tuple of (sequences, labels) as numpy arrays

        Raises:
            ValueError: If sequence_length is larger than available data
        """
        num_rows = df.shape[0]
        labels = df['label']
        features = df.drop(columns=['set_type', 'label'])

        if num_rows < sequence_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) is larger than "
                f"available data ({num_rows}) for {split_name} split"
            )

        X, y = [], []
        for i in range(num_rows - sequence_length + 1):
            # Extract sequence of features
            X.append(features.iloc[i:i + sequence_length].values)
            # Label is from the last time step in the sequence
            y.append(labels.iloc[i + sequence_length - 1])

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        print(f'{split_name.capitalize()} data dimensions: X={X.shape}, y={y.shape}')
        return X, y


    def _preprocess_data(self) -> None:
        """Preprocess data: normalize and create sequences."""
        sequence_length = self.hyperparams['history_volume']

        # Normalize data
        train_df, valid_df, test_df = self._normalize()

        # Create sequences for each split
        self.X_train, self.y_train = self._create_sequences(
            train_df, sequence_length=sequence_length, split_name='train'
        )
        self.X_valid, self.y_valid = self._create_sequences(
            valid_df, sequence_length=sequence_length, split_name='valid'
        )
        self.X_test, self.y_test = self._create_sequences(
            test_df, sequence_length=sequence_length, split_name='test'
        )


    def _build_model(self) -> Sequential:
        """Build and compile the LSTM model architecture.

        Returns:
            Compiled Keras Sequential model
        """
        # Extract hyperparameters
        layers = self.hyperparams.get('layers', [16, 16, 16, 1])
        learning_rate = float(self.hyperparams.get('learning_rate', 1e-3))
        l2_lambda = float(self.hyperparams.get('lambda', 3e-2))
        dropout = self.hyperparams.get('dropout', 0.2)
        recurrent_dropout = self.hyperparams.get('recurrent_dropout', 0.2)

        num_features = self.X_train.shape[2]
        sequence_length = self.hyperparams['history_volume']
        activation = self.hyperparams.get('activation', 'tanh')
        recurrent_activation = self.hyperparams.get('recurrent_activation', 'sigmoid')

        print("Model Architecture:")
        print(f"  Layers: {layers}")
        print(f"  Sequence length: {sequence_length}, Features: {num_features}")
        print(f"  Learning rate: {learning_rate}, L2 lambda: {l2_lambda}")
        print(f"  Dropout: {dropout}, Recurrent dropout: {recurrent_dropout}")
        print()

        # Build model
        model = Sequential()

        # Add LSTM layers (all but the last element in layers list)
        for i, units in enumerate(layers[:-1]):
            is_last_lstm = (i == len(layers) - 2)
            return_sequences = not is_last_lstm

            if i == 0:
                # First LSTM layer with input shape
                model.add(LSTM(
                    units=units,
                    input_shape=(sequence_length, num_features),
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    kernel_regularizer=l2(l2_lambda),
                    recurrent_regularizer=l2(l2_lambda),
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=return_sequences
                ))
            else:
                # Subsequent LSTM layers
                model.add(LSTM(
                    units=units,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    kernel_regularizer=l2(l2_lambda),
                    recurrent_regularizer=l2(l2_lambda),
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=return_sequences
                ))

            model.add(BatchNormalization())

        # Output layer
        model.add(Dense(units=layers[-1], activation='sigmoid'))

        # Compile model
        model.compile(
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="auc")],
            optimizer=Adam(learning_rate=learning_rate)
        )

        print(model.summary())
        return model



    def train(self) -> None:
        """Train the LSTM model with early stopping and learning rate reduction.

        Training uses validation set for monitoring and applies callbacks for
        early stopping and learning rate reduction on plateau.
        """
        # Extract training hyperparameters
        batch_size = self.hyperparams.get('batch_size', 512)
        num_epochs = self.hyperparams.get('num_epochs', 50)
        early_stopping_patience = self.hyperparams.get('early_stopping_patience', 10)
        lr_patience = self.hyperparams.get('lr_patience', 2)

        print(f"Training Examples: {self.X_train.shape[0]}")
        print(f"Validation Examples: {self.X_valid.shape[0]}")
        print(f"Test Examples: {self.X_test.shape[0]}")
        print("\nTraining Hyperparameters:")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print(f"  Learning rate patience: {lr_patience}")
        print()

        # Setup callbacks
        lr_decay = ReduceLROnPlateau(
            monitor='val_auc',
            patience=lr_patience,
            verbose=1,
            factor=0.5,
            min_lr=1e-6,
            mode='max'
        )

        early_stop = EarlyStopping(
            monitor='val_auc',
            patience=early_stopping_patience,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )

        # Train model
        start_time = time.time()
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(self.X_valid, self.y_valid),
            shuffle=True,
            verbose=1,
            callbacks=[lr_decay, early_stop]
        )

        elapsed_time = time.time() - start_time
        print(f'\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')


    def evaluate(self, output_dir_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate model on all data splits and optionally save results.

        Args:
            output_dir_path: Optional directory to save evaluation results and plots

        Returns:
            Dictionary with AUC scores for train, validation, and test sets

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.history is None:
            raise ValueError("Model must be trained before evaluation. Call train() first.")

        # Generate predictions
        y_train_pred = self.model.predict(self.X_train, verbose=0).flatten()
        y_valid_pred = self.model.predict(self.X_valid, verbose=0).flatten()
        y_test_pred = self.model.predict(self.X_test, verbose=0).flatten()

        # Calculate AUC scores
        train_auc = roc_auc_score(self.y_train, y_train_pred)
        valid_auc = roc_auc_score(self.y_valid, y_valid_pred)
        test_auc = roc_auc_score(self.y_test, y_test_pred)

        print("\nEvaluation Results:")
        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Validation AUC: {valid_auc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")

        # Save results if output directory provided
        if output_dir_path:
            os.makedirs(output_dir_path, exist_ok=True)

            # Save AUC scores
            results_path = os.path.join(output_dir_path, 'auc_scores.txt')
            with open(results_path, 'w') as f:
                f.write(f'Train AUC: {train_auc:.4f}\n')
                f.write(f'Validation AUC: {valid_auc:.4f}\n')
                f.write(f'Test AUC: {test_auc:.4f}\n')

            # Plot training curves
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            # Loss curves
            axes[0].plot(self.history.history['loss'], label='Training Loss', color='blue')
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss', color='red')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # AUC curves
            axes[1].plot(self.history.history['auc'], label='Training AUC', color='blue')
            axes[1].plot(self.history.history['val_auc'], label='Validation AUC', color='red')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('AUC')
            axes[1].set_title('Training and Validation AUC')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(output_dir_path, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\nResults saved to {output_dir_path}")

        return {
            "results_type": "auc",
            "train": float(train_auc),
            "valid": float(valid_auc),
            "test": float(test_auc)
        }

    def save_model(self, output_dir_path: str) -> None:
        """Save trained model to disk.

        Args:
            output_dir_path: Directory where model will be saved

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("No model to save. Build the model first.")

        os.makedirs(output_dir_path, exist_ok=True)
        model_path = os.path.join(output_dir_path, 'lstm_model.h5')
        self.model.save(model_path)
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

        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
