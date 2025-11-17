"""
Base Verbalizer

Provides base functionality for converting structured data into natural language.
"""

from typing import Optional, Dict
import pandas as pd
import json


class BaseVerbalizer:
    """Base class for all verbalizers."""

    def __init__(
        self,
        table_data: Optional[pd.DataFrame] = None,
        feature_names_config: Optional[Dict[str, str]] = None,
        feature_names_config_path: Optional[str] = None
    ):
        """
        Initialize the base verbalizer.

        Args:
            table_data: DataFrame containing the data to verbalize
            feature_names_config: Dictionary mapping feature keys to human-readable names
            feature_names_config_path: Path to JSON file containing feature names mapping
        """
        self.data = table_data

        # Load feature names from config
        if feature_names_config is not None:
            self.feature_names = feature_names_config
        elif feature_names_config_path is not None:
            self.feature_names = self._load_feature_names(feature_names_config_path)
        else:
            self.feature_names = {}

    @staticmethod
    def _load_feature_names(config_path: str) -> Dict[str, str]:
        """Load feature names from a JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_feature_name(self, feature_key: str) -> str:
        """
        Get the human-readable name for a feature.

        Args:
            feature_key: The feature key from the data

        Returns:
            Human-readable feature name, or the key itself if no mapping exists
        """
        return self.feature_names.get(feature_key, feature_key)

    def set_table_data(self, table_data: pd.DataFrame) -> None:
        """Set or update the table data."""
        self.data = table_data

    def get_text(self, row: pd.Series) -> str:
        """
        Convert a single row of data into text.

        This method should be implemented by subclasses.

        Args:
            row: A single row from the DataFrame

        Returns:
            Natural language description of the row
        """
        raise NotImplementedError("Subclasses must implement get_text()")

    def get_textual_data(self) -> pd.DataFrame:
        """
        Convert all rows in the table to textual descriptions.

        Returns:
            DataFrame with columns: ['text', 'set_type', 'label']
        """
        if self.data is None:
            raise ValueError("Table data must be set before calling get_textual_data()")

        textual_data = self.data.copy()
        textual_data['text'] = self.data.apply(lambda row: self.get_text(row), axis=1)

        # Return only relevant columns
        output_columns = ['text', 'set_type', 'label']
        return textual_data[output_columns].copy()

