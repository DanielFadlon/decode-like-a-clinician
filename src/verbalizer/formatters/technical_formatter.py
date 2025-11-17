"""
Technical Formatter

Formats features as precise "Feature: value" pairs in technical/structured format.
"""

from typing import Dict, Optional, Callable
import pandas as pd

from .base_formatter import BaseFormatter


class TechnicalFormatter(BaseFormatter):
    """
    Technical formatter that creates precise "Feature: value" style text.

    This is the technical/structured format suitable for data tables,
    logs, and precise reporting.

    Example: "Age: 65, Blood Pressure: 120/80, Heart Rate: 75."
    """

    def __init__(
        self,
        feature_names: Optional[Dict[str, str]] = None,
        skip_null_values: bool = False,
        include_time: bool = True,
        value_formatter: Optional[Callable] = None
    ):
        """
        Initialize the technical formatter.

        Args:
            feature_names: Dictionary mapping feature keys to human-readable names
            skip_null_values: If True, skip features with null values
            include_time: If True, include TimeFromHospFeat in output
            value_formatter: Optional function to format values
        """
        super().__init__(feature_names, skip_null_values)
        self.include_time = include_time
        self.value_formatter = value_formatter

    def _format_value(self, value) -> str:
        """Format a single value."""
        if self.value_formatter is not None:
            return str(self.value_formatter(value))
        return str(value)

    def format_row(self, row: pd.Series) -> str:
        """
        Format a row as technical feature-value pairs.

        Args:
            row: A single row from the DataFrame

        Returns:
            Text in format "feature1: value1, feature2: value2, ..."
        """
        text_parts = []

        for feature_name, value in row.items():
            # Skip time column if not including time
            if feature_name == "TimeFromHospFeat" and not self.include_time:
                continue

            # Check if we should skip this feature
            if self.should_skip_feature(feature_name, value):
                continue

            # Format the feature-value pair
            readable_name = self.get_feature_name(feature_name)
            formatted_value = self._format_value(value)
            text_parts.append(f"{readable_name}: {formatted_value}")

        # Join all parts
        if text_parts:
            return ", ".join(text_parts) + "."
        else:
            return "No data available."

