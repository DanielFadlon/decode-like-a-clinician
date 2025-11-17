"""
Per-Indicator Verbalizer

Creates natural language descriptions that show the temporal history of each feature/indicator.
"""

from typing import Optional, Dict, List
import pandas as pd
import os

from .base_verbalizer import BaseVerbalizer
from .history_builder import HistoryBuilder
from .time_encoding import TimeApproach, get_current_time_text
from .formatters import BaseFormatter


class PerIndicatorVerbalizer(BaseVerbalizer):
    """
    Verbalizer that describes the temporal history of each indicator/feature.

    Supports two formatting approaches:
    1. Simple list: "Feature1: value1 (t=0), value2 (t=3), value3 (t=6)."
    2. Narrative: "Feature1 was measured 3 times from t=0 to t=6, with values..."
    """

    def __init__(
        self,
        table_data_path: str,
        is_history_format: bool,
        time_approach: TimeApproach,
        feature_names_config: Optional[Dict[str, str]] = None,
        feature_names_config_path: Optional[str] = None,
        formatter: Optional[BaseFormatter] = None,
        num_changes_per_indicator: Optional[int] = 5,
        num_report_hours: Optional[int] = None,
        case_index_columns: Optional[List[str]] = None,
        case_identifier_column: Optional[str] = None,
        auto_save_history: bool = True
    ):
        """
        Initialize the per-indicator verbalizer.

        Args:
            table_data_path: Path to the data file (parquet or JSON)
            is_history_format: If True, data is already in history format; if False, will create history
            time_approach: Approach for encoding time information
            feature_names_config: Dictionary mapping feature keys to human-readable names
            feature_names_config_path: Path to JSON file containing feature names mapping
            formatter: Formatter to use (IndicatorSeriesFormatter or IndicatorNarrativeFormatter)
            num_changes_per_indicator: Number of historical changes to show per indicator
            num_report_hours: If set, only report changes within this time window
            case_index_columns: List of columns used as multi-index
            case_identifier_column: Column that uniquely identifies a case
            auto_save_history: If True and creating history from raw data, save the history to JSON
        """
        self.time_approach = time_approach
        self.num_changes_per_indicator = num_changes_per_indicator
        self.num_report_hours = num_report_hours

        # Set default index columns if not provided
        self.case_index_columns = case_index_columns or ['Encrypted_PatientNum', 'Encrypted_CaseNum']
        self.case_identifier_column = case_identifier_column or 'Encrypted_CaseNum'

        # Load or create history data
        if is_history_format:
            data = self._load_history_data(table_data_path)
        else:
            data = self._create_history_data(table_data_path, auto_save_history)

        super().__init__(data, feature_names_config, feature_names_config_path)

        # Set formatter
        if formatter is None:
            # Import here to avoid circular dependency
            from .formatters import IndicatorSeriesFormatter
            formatter = IndicatorSeriesFormatter(
                feature_names=self.feature_names,
                num_values_to_show=num_changes_per_indicator,
                time_approach=time_approach,
                num_report_hours=num_report_hours
            )

        self.formatter = formatter

    def _load_history_data(self, data_path: str) -> pd.DataFrame:
        """Load data that is already in history format."""
        if data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            return HistoryBuilder.load_history_from_json(data_path, self.case_index_columns)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

    def _create_history_data(self, data_path: str, auto_save: bool) -> pd.DataFrame:
        """Create history format from raw tabular data."""
        # Load raw data
        raw_data = pd.read_parquet(data_path)

        # Create history
        history_data = HistoryBuilder.create_full_history(
            raw_data,
            timestamp_column="TimeFromHospFeat",
            case_identifier_column=self.case_identifier_column
        )

        # Auto-save if requested
        if auto_save:
            directory = os.path.dirname(data_path)
            history_path = os.path.join(directory, 'history_data.json')
            HistoryBuilder.save_history_to_json(history_data, history_path)
            print(f"History data saved to: {history_path}")

        return history_data

    def get_text(self, row: pd.Series) -> str:
        """
        Convert a single row with history data into text.

        Args:
            row: A single row from the DataFrame (with history format)

        Returns:
            Natural language description of all indicator histories
        """
        # Get current time
        current_time = 0
        if 'TimeFromHospFeat' in row and isinstance(row['TimeFromHospFeat'], list):
            if len(row['TimeFromHospFeat']) > 0:
                current_time = row['TimeFromHospFeat'][-1].get('time', 0)

        # Add time context header
        time_context = get_current_time_text(self.time_approach, current_time)

        # Use formatter to generate feature descriptions
        feature_descriptions = self.formatter.format_row(row)

        return time_context + feature_descriptions

