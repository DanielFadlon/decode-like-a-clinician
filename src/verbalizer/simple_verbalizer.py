"""
Simple Verbalizer

Converts tabular data into technical "feature: value" format with optional forward filling.
"""

from typing import Optional, Dict, Callable
import pandas as pd

from .base_verbalizer import BaseVerbalizer
from .formatters import TechnicalFormatter


class SimpleVerbalizer(BaseVerbalizer):
    """
    Simple verbalizer that converts each row into a comma-separated list of feature-value pairs.

    Example output: "Age: 65, Blood Pressure: 120/80, Heart Rate: 75."
    """

    def __init__(
        self,
        table_data: Optional[pd.DataFrame] = None,
        feature_names_config: Optional[Dict[str, str]] = None,
        feature_names_config_path: Optional[str] = None,
        apply_forward_fill: bool = False,
        skip_null_values: bool = False,
        include_time: bool = True,
        value_formatter: Optional[Callable] = None
    ):
        """
        Initialize the simple verbalizer.

        Args:
            table_data: DataFrame containing the data to verbalize
            feature_names_config: Dictionary mapping feature keys to human-readable names
            feature_names_config_path: Path to JSON file containing feature names mapping
            apply_forward_fill: If True, apply forward fill to the data before verbalizing
            skip_null_values: If True, skip features with null values in the output
            include_time: If True, include time-related features in the output
            value_formatter: Optional function to format values before converting to text
        """
        super().__init__(table_data, feature_names_config, feature_names_config_path)

        self.apply_forward_fill = apply_forward_fill

        # Create technical formatter
        self.formatter = TechnicalFormatter(
            feature_names=self.feature_names,
            skip_null_values=skip_null_values,
            include_time=include_time,
            value_formatter=value_formatter
        )

        # Apply forward fill if requested
        if self.apply_forward_fill and self.data is not None:
            self._apply_forward_fill()

    def _apply_forward_fill(self) -> None:
        """Apply forward fill to the data, preserving set_type and label columns."""
        if self.data is None:
            return

        # Identify columns to exclude from forward fill
        exclude_columns = ['set_type', 'label']
        columns_to_fill = [col for col in self.data.columns if col not in exclude_columns]

        # Apply forward fill
        self.data[columns_to_fill] = self.data[columns_to_fill].ffill()

    def get_text(self, row: pd.Series) -> str:
        """
        Convert a single row into simple text format.

        Args:
            row: A single row from the DataFrame

        Returns:
            Text in format "feature1: value1, feature2: value2, ..."
        """
        return self.formatter.format_row(row)

