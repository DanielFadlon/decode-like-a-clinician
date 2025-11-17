"""
History Builder

Creates temporal history of feature changes from tabular data.
"""

from typing import Optional
import pandas as pd
from tqdm import tqdm


class HistoryBuilder:
    """
    Converts tabular data into a history-based format where each feature contains
    a list of {time, value} dictionaries representing changes over time.
    """

    @staticmethod
    def create_feature_history(
        data: pd.DataFrame,
        timestamp_column: str,
        value_column: str,
        case_identifier_column: str
    ) -> pd.DataFrame:
        """
        Convert a single feature column to history format.

        For each sample at time t, creates a list of all changes (with timestamps)
        that occurred from time 0 to t for that feature.

        Args:
            data: DataFrame with multi-index including case identifier
            timestamp_column: Name of the column containing timestamps
            value_column: Name of the column to convert to history format
            case_identifier_column: Name of the column that identifies unique cases

        Returns:
            DataFrame with the value column converted to history list format
        """
        history_records = []

        # Process each case separately
        for case_id in data.index.get_level_values(case_identifier_column).unique():
            case_data = data.loc[data.index.get_level_values(case_identifier_column) == case_id]
            previous_value = None

            for i in range(len(case_data)):
                instance = case_data.iloc[i]
                current_value = instance[value_column]
                current_time = instance[timestamp_column]

                # Initialize history for first instance
                if i == 0:
                    history = [{"time": current_time, "value": current_value}]
                # If value changed, add to history
                elif not pd.isna(current_value) and current_value != previous_value:
                    history = history_records[-1] + [{"time": current_time, "value": current_value}]
                # If value unchanged, keep previous history
                else:
                    history = history_records[-1]

                previous_value = current_value
                history_records.append(history)

        data[value_column] = history_records
        return data

    @staticmethod
    def create_full_history(
        data: pd.DataFrame,
        timestamp_column: str = "TimeFromHospFeat",
        case_identifier_column: str = "Encrypted_CaseNum",
        exclude_columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Convert all feature columns to history format.

        Args:
            data: DataFrame containing the data
            timestamp_column: Name of the timestamp column
            case_identifier_column: Name of the case identifier column
            exclude_columns: Columns to exclude from history conversion (e.g., ['set_type', 'label'])

        Returns:
            DataFrame with all feature columns in history format
        """
        if exclude_columns is None:
            exclude_columns = [timestamp_column, 'set_type', 'label']
        else:
            exclude_columns = list(set(exclude_columns + [timestamp_column, 'set_type', 'label']))

        # Convert each feature column to history format
        for column in tqdm(data.columns, desc="Creating feature histories"):
            if column not in exclude_columns:
                data = HistoryBuilder.create_feature_history(
                    data,
                    timestamp_column,
                    column,
                    case_identifier_column
                )

        return data

    @staticmethod
    def save_history_to_json(data: pd.DataFrame, output_path: str) -> None:
        """
        Save history DataFrame to JSON format.

        Args:
            data: DataFrame with history data
            output_path: Path where the JSON file should be saved
        """
        # Convert to dictionary format suitable for JSON
        data_dict = data.to_dict(orient='index')

        import json
        with open(output_path, 'w') as f:
            json.dump(data_dict, f, indent=2)

    @staticmethod
    def load_history_from_json(input_path: str, case_index_columns: list) -> pd.DataFrame:
        """
        Load history DataFrame from JSON format.

        Args:
            input_path: Path to the JSON file
            case_index_columns: List of columns to use as multi-index

        Returns:
            DataFrame with history data
        """
        import json
        with open(input_path, 'r') as f:
            data_dict = json.load(f)

        # Convert back to DataFrame
        data = pd.DataFrame.from_dict(data_dict, orient='index')

        # Restore multi-index if needed
        if case_index_columns:
            # The index might be a string representation of tuple, need to parse it
            pass  # Implementation depends on how the index was serialized

        return data

