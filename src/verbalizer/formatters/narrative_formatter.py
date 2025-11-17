"""
Narrative Formatter

Formats features using rule-based narrative templates for more natural language.
"""

from typing import Dict, Optional
import pandas as pd
import json

from .base_formatter import BaseFormatter


class NarrativeFormatter(BaseFormatter):
    """
    Rule-based narrative formatter that generates natural language descriptions.

    Uses a rules configuration to generate contextual narratives like:
    - "Heart rate is elevated at 120 bpm, indicating tachycardia."
    - "Blood pressure is normal at 120/80."

    Rules Config Format:
    {
        "HeartRate": {
            "type": "numeric",
            "mid_min_th": 60,
            "mid_max_th": 100,
            "low_text": "Heart rate is low at <VALUE> bpm.",
            "mid_text": "Heart rate is normal at <VALUE> bpm.",
            "high_text": "Heart rate is elevated at <VALUE> bpm."
        },
        "Anticoagulant": {
            "type": "Anticoagulant",
            "high_text": "Anticoagulant medication was administered.",
            "low_text": "No anticoagulant medication was given."
        }
    }
    """

    def __init__(
        self,
        rules_config: Dict[str, dict],
        feature_names: Optional[Dict[str, str]] = None,
        skip_null_values: bool = False,
        include_time: bool = True,
        rules_config_path: Optional[str] = None
    ):
        """
        Initialize the narrative formatter.

        Args:
            rules_config: Dictionary containing formatting rules for features
            feature_names: Dictionary mapping feature keys to human-readable names
            skip_null_values: If True, skip features with null values
            include_time: If True, include TimeFromHospFeat in output
            rules_config_path: Path to JSON file containing rules (alternative to rules_config)
        """
        super().__init__(feature_names, skip_null_values)

        # Load rules from file if path provided
        if rules_config_path is not None:
            with open(rules_config_path, 'r') as f:
                self.rules = json.load(f)
        else:
            self.rules = rules_config or {}

        self.include_time = include_time

    def _apply_rule(self, feature_name: str, value) -> Optional[str]:
        """
        Apply rule-based formatting to a feature value.

        Args:
            feature_name: Name of the feature
            value: Value of the feature

        Returns:
            Formatted narrative text, or None if no rule applies
        """
        # Get rules for this feature
        feature_rules = self.rules.get(feature_name)
        if feature_rules is None:
            return None

        # Check if value is valid
        if value is None or pd.isna(value):
            return None

        # Handle Anticoagulant type (categorical)
        if feature_rules.get("type") == "Anticoagulant":
            if value == "given":
                return feature_rules.get('high_text', '')
            else:
                return feature_rules.get('low_text', '')

        # Handle numeric thresholds
        try:
            numeric_value = float(value)
            mid_min = feature_rules.get('mid_min_th')
            mid_max = feature_rules.get('mid_max_th')

            if mid_min is not None and mid_max is not None:
                if numeric_value < mid_min:
                    template = feature_rules.get('low_text', '')
                elif numeric_value > mid_max:
                    template = feature_rules.get('high_text', '')
                else:
                    template = feature_rules.get('mid_text', '')

                # Replace <VALUE> placeholder
                return template.replace("<VALUE>", str(value))
        except (ValueError, TypeError):
            pass

        return None

    def _format_simple(self, feature_name: str, value) -> str:
        """
        Fallback to simple formatting when no rule exists.

        Args:
            feature_name: Name of the feature
            value: Value of the feature

        Returns:
            Simple formatted text
        """
        readable_name = self.get_feature_name(feature_name)

        # Handle special cases
        if feature_name == "Age" or feature_name == "CumulativeCost":
            value = round(float(value), 3) if not pd.isna(value) else value

        # Check if value exists
        if value is not None and not pd.isna(value):
            return f"{readable_name} is {value}."
        else:
            return f"{readable_name} is not measured yet."

    def format_row(self, row: pd.Series) -> str:
        """
        Format a row using narrative rules.

        Args:
            row: A single row from the DataFrame

        Returns:
            Narrative text description with one sentence per feature
        """
        text_lines = []

        for feature_name, value in row.items():
            # Skip time column if not including time
            if feature_name == "TimeFromHospFeat" and not self.include_time:
                continue

            # Check if we should skip this feature
            if self.should_skip_feature(feature_name, value):
                continue

            # Try rule-based formatting first
            narrative_text = self._apply_rule(feature_name, value)

            # Fall back to simple formatting if no rule exists
            if narrative_text is None:
                narrative_text = self._format_simple(feature_name, value)

            text_lines.append(narrative_text)

        # Join with newlines (one feature per line)
        return "\n".join(text_lines) if text_lines else "No data available."

