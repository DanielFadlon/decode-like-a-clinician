"""
Base Formatter

Abstract base class for all formatters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd


class BaseFormatter(ABC):
    """Abstract base class for value formatters."""

    def __init__(
        self,
        feature_names: Optional[Dict[str, str]] = None,
        skip_null_values: bool = False
    ):
        """
        Initialize the formatter.

        Args:
            feature_names: Dictionary mapping feature keys to human-readable names
            skip_null_values: If True, skip features with null values
        """
        self.feature_names = feature_names or {}
        self.skip_null_values = skip_null_values

    def get_feature_name(self, feature_key: str) -> str:
        """Get human-readable name for a feature."""
        return self.feature_names.get(feature_key, feature_key)

    @abstractmethod
    def format_row(self, row: pd.Series) -> str:
        """
        Format a single row into text.

        Args:
            row: A single row from the DataFrame

        Returns:
            Formatted text representation
        """
        pass

    def should_skip_feature(self, feature_name: str, value) -> bool:
        """
        Determine if a feature should be skipped.

        Args:
            feature_name: Name of the feature
            value: Value of the feature

        Returns:
            True if the feature should be skipped
        """
        # Always skip metadata columns
        if feature_name in ['set_type', 'label']:
            return True

        # Skip null values if configured
        if self.skip_null_values and pd.isna(value):
            return True

        return False

