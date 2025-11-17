"""
Indicator Narrative Formatter

Creates narrative-style temporal descriptions for indicator histories.
"""

from typing import Dict, Optional, List
import pandas as pd

from .base_formatter import BaseFormatter
from ..time_encoding import extract_time_from_history_data


class IndicatorNarrativeFormatter(BaseFormatter):
    """
    Formatter that creates temporal narrative descriptions for indicator histories.

    Instead of listing values like "120 (t=0), 125 (t=6), 130 (t=12)",
    creates narratives like: "was measured 3 times from t=0 to t=12,
    with values 120, 125, and 130 (final: 130 at t=12)."

    This formatter is specifically designed for history-based data where
    each feature contains a list of {time, value} dictionaries.
    """

    def __init__(
        self,
        feature_names: Optional[Dict[str, str]] = None,
        num_values_to_show: int = 5,
        static_features: Optional[List[str]] = None
    ):
        """
        Initialize the indicator narrative formatter.

        Args:
            feature_names: Dictionary mapping feature keys to human-readable names
            num_values_to_show: Maximum number of values to include in narrative
            static_features: List of features to show without narrative (e.g., ['Age', 'CumulativeCost'])
        """
        super().__init__(feature_names, skip_null_values=False)
        self.num_values_to_show = num_values_to_show
        self.static_features = static_features or ['Age', 'CumulativeCost', 'TimeFromHospFeat']

    def _get_time_text(self, time_hours: float) -> str:
        """Format time for display."""
        return f"t={int(time_hours)}"

    def _create_feature_narrative(
        self,
        feature_history: List[dict],
        current_time: float
    ) -> str:
        """
        Create a narrative description for a feature's history.

        Args:
            feature_history: List of {time, value} dictionaries
            current_time: Current time in hours

        Returns:
            Narrative text describing the feature's temporal pattern
        """
        if not feature_history:
            return "None."

        num_changes = len(feature_history)
        num_to_show = min(num_changes, self.num_values_to_show)

        # Skip initial None values if that's all we have
        if feature_history[0].get("value") is None and num_changes == num_to_show:
            num_to_show = max(0, num_to_show - 1)

        if num_to_show == 0:
            return "None."

        # Get time information
        last_value_time = self._get_time_text(feature_history[-1].get("time", 0))
        first_value_time = self._get_time_text(feature_history[-num_to_show].get("time", 0))
        current_time_text = self._get_time_text(current_time)
        start_time = "t=0"

        # Single measurement
        if num_to_show == 1:
            value = feature_history[-1].get("value")
            return (
                f"was measured once from {start_time} to {current_time_text}, "
                f"with value {value} (final: {value} at {last_value_time})."
            )

        # Multiple measurements
        narrative_parts = []
        narrative_parts.append(
            f"was measured {num_to_show} times from {first_value_time} to {current_time_text}, "
            f"with values "
        )

        # Build values list
        value_parts = []
        for i in range(num_to_show, 0, -1):
            value_info = feature_history[-i]
            value = value_info.get("value")

            if i == 1:  # Last value
                last_value = feature_history[-1].get("value")
                last_time = self._get_time_text(feature_history[-1].get("time", 0))
                value_parts.append(f"and {value} (final: {last_value} at {last_time})")
            else:
                value_parts.append(str(value))

        narrative_parts.append(", ".join(value_parts[:-1]))
        if len(value_parts) > 1:
            narrative_parts.append(" " + value_parts[-1])
        narrative_parts.append(".")

        return "".join(narrative_parts)

    def format_row(self, row: pd.Series) -> str:
        """
        Format a row with history data into narrative descriptions.

        Args:
            row: A single row where each feature contains a list of {time, value} dicts

        Returns:
            Narrative text with temporal descriptions for each indicator
        """
        text_lines = []

        # Get current time from history data
        current_time = extract_time_from_history_data(row)

        # Process each feature
        for feature_name, feature_data in row.items():
            # Skip metadata
            if self.should_skip_feature(feature_name, feature_data):
                continue

            # Skip if not a list (not history format)
            if not isinstance(feature_data, list):
                continue

            readable_name = self.get_feature_name(feature_name)

            # Handle static features (show only latest value)
            if feature_name in self.static_features:
                if feature_name == 'TimeFromHospFeat':
                    continue
                latest_value = feature_data[-1].get('value') if len(feature_data) > 0 else None
                text_lines.append(f"{readable_name}: {latest_value}.")
            else:
                # Create narrative for temporal features
                narrative = self._create_feature_narrative(feature_data, current_time)
                text_lines.append(f"{readable_name} {narrative}")

        return "\n".join(text_lines) if text_lines else "No data available."

