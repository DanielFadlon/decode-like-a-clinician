"""
Indicator Series Formatter

Formats indicator histories as simple value series lists.
"""

from typing import Dict, Optional, List
import pandas as pd

from .base_formatter import BaseFormatter
from ..time_encoding import (
    TimeApproach,
    get_relative_position_encoded_text,
    get_date_by_time_from_hosp,
    get_day_and_hour_text,
    extract_time_from_history_data,
    INTERVAL_SIZE
)


class IndicatorSeriesFormatter(BaseFormatter):
    """
    Formatter that displays indicator histories as simple series lists.

    Example output: "HeartRate: 72 (t=0), 75 (t=6), 78 (t=12), 82 (t=18)."

    This is the default format for PerIndicatorVerbalizer.
    """

    def __init__(
        self,
        feature_names: Optional[Dict[str, str]] = None,
        num_values_to_show: int = 5,
        time_approach: TimeApproach = TimeApproach.ABSOLUTE_POSITION,
        num_report_hours: Optional[int] = None,
        static_features: Optional[List[str]] = None
    ):
        """
        Initialize the indicator series formatter.

        Args:
            feature_names: Dictionary mapping feature keys to human-readable names
            num_values_to_show: Maximum number of values to show per indicator
            time_approach: How to encode time (absolute, relative, etc.)
            num_report_hours: If set, only show changes within this window
            static_features: List of features to show without history
        """
        super().__init__(feature_names, skip_null_values=False)
        self.num_values_to_show = num_values_to_show
        self.time_approach = time_approach
        self.num_report_hours = num_report_hours
        self.static_features = static_features or ['Age', 'CumulativeCost', 'TimeFromHospFeat']

    def _get_time_text_for_value(
        self,
        value_time_in_hours: float,
        current_time_in_hours: float
    ) -> str:
        """
        Get the time encoding text for a specific value.

        Args:
            value_time_in_hours: Time when the value was recorded
            current_time_in_hours: Current time from hospitalization

        Returns:
            Time encoding text, e.g., " (t=3)" or " (3 hours ago)"
        """
        if self.time_approach == TimeApproach.ABSOLUTE_POSITION:
            return f" (t={int(value_time_in_hours)})"

        elif self.time_approach == TimeApproach.RELATIVE_POSITION:
            relative_text = get_relative_position_encoded_text(
                current_time_in_hours,
                value_time_in_hours
            )
            return f" ({relative_text})"

        elif self.time_approach == TimeApproach.ABSOLUTE_POSITION_INTERVALS:
            intervals = int(value_time_in_hours // INTERVAL_SIZE)
            return f" (t={intervals})"

        elif self.time_approach == TimeApproach.RELATIVE_POSITION_INTERVALS:
            relative_text = get_relative_position_encoded_text(
                current_time_in_hours,
                value_time_in_hours,
                should_count_intervals=True
            )
            return f" ({relative_text})"

        elif self.time_approach == TimeApproach.REAL_DATE:
            date = get_date_by_time_from_hosp(value_time_in_hours)
            return f" ({date})"

        elif self.time_approach == TimeApproach.REAL_DAY_AND_HOUR:
            day_hour = get_day_and_hour_text(value_time_in_hours)
            return f" ({day_hour})"

        elif self.time_approach == TimeApproach.NONE:
            return ""

        else:
            raise ValueError(f"Unknown time approach: {self.time_approach}")

    def format_row(self, row: pd.Series) -> str:
        """
        Format a row with history data into series lists.

        Args:
            row: A single row where each feature contains a list of {time, value} dicts

        Returns:
            Text with series lists for each indicator
        """
        text_lines = []

        # Get current time from history data
        current_time = extract_time_from_history_data(row)

        # Handle static features first (Age, CumulativeCost)
        for feature_name in ['Age', 'CumulativeCost']:
            if feature_name in row:
                feature_data = row[feature_name]
                if isinstance(feature_data, list) and len(feature_data) > 0:
                    value = feature_data[-1].get('value')
                    if value is not None:
                        if feature_name == 'Age':
                            value = round(float(value), 3)
                        readable_name = self.get_feature_name(feature_name)
                        text_lines.append(f"{readable_name}: {value}.")

        # Process temporal features
        skip_features = {'set_type', 'label', 'TimeFromHospFeat', 'Age', 'CumulativeCost'}

        for feature_name, feature_data in row.items():
            if feature_name in skip_features:
                continue

            # Skip if not a list (not history format)
            if not isinstance(feature_data, list):
                continue

            feature_history = feature_data
            readable_name = self.get_feature_name(feature_name)

            # Build the feature series
            value_parts = []

            # Get the last N changes
            num_values = min(self.num_values_to_show, len(feature_history))
            for i in range(num_values, 0, -1):
                value_info = feature_history[-i]

                # Validate
                if not isinstance(value_info, dict) or "time" not in value_info:
                    continue

                value_time = value_info.get("time", -1)
                value = value_info.get("value")

                # Skip if outside report window
                if self.num_report_hours is not None:
                    if current_time - value_time > self.num_report_hours:
                        continue

                # Format the value with time encoding
                if value is not None:
                    time_text = self._get_time_text_for_value(value_time, current_time)
                    value_parts.append(f"{value}{time_text}")
                else:
                    value_parts.append("None")

            # Add the feature to the text if it has values
            if value_parts:
                series_text = ", ".join(value_parts)
                text_lines.append(f"{readable_name}: {series_text}.")

        return "\n".join(text_lines) if text_lines else "No data available."

