"""
Event Processor

Handles processing of event-wise data by identifying changes in values over time.
"""

from typing import List, Optional
import pandas as pd


class EventProcessor:
    """
    Processes tabular data to identify and extract events (changes in values).

    An event is defined as any change in a feature's value from one time point to the next.
    """

    def __init__(
        self,
        table_data: pd.DataFrame,
        case_index_columns: List[str],
        case_identifier_column: str,
        columns_to_ignore: Optional[List[str]] = None
    ):
        """
        Initialize the event processor.

        Args:
            table_data: DataFrame with multi-index including patient/case identifiers
            case_index_columns: List of column names used as the multi-index
            case_identifier_column: The column name that uniquely identifies a case
            columns_to_ignore: Columns to ignore when detecting events (e.g., cumulative values)
        """
        self.case_index_columns = case_index_columns
        self.case_identifier_column = case_identifier_column
        self.columns_to_ignore = columns_to_ignore or []

        # Extract events from the data
        self.event_data = self._extract_events(table_data)

    def _extract_events(self, table_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract events by identifying changes in values.

        Args:
            table_data: Original DataFrame

        Returns:
            DataFrame containing only rows with changed values
        """
        # Detect changes: keep only values that differ from the previous row
        event_data = table_data.apply(lambda col: col.where(col != col.shift()))

        # Drop ignored columns
        if self.columns_to_ignore:
            event_data = event_data.drop(columns=self.columns_to_ignore, errors='ignore')

        # Identify which columns to check for null values (all except TimeFromHospFeat)
        all_columns = set(event_data.columns)
        ignore_for_null_check = {'TimeFromHospFeat'}
        columns_to_check = list(all_columns - ignore_for_null_check)

        # Drop rows where all values (except time) are null (no events occurred)
        event_data = event_data.dropna(how='all', subset=columns_to_check)

        return event_data

    def get_data(self) -> pd.DataFrame:
        """Get the processed event data."""
        return self.event_data

    def get_last_x_events(
        self,
        current_row: pd.Series,
        num_events: int,
        reverse_chronological: bool = True
    ) -> pd.DataFrame:
        """
        Get the last X events for the case containing the current row.

        Args:
            current_row: The current row being processed
            num_events: Number of previous events to retrieve
            reverse_chronological: If True, return events in reverse chronological order

        Returns:
            DataFrame containing the last X events
        """
        try:
            # Find the position of the case identifier in the index
            case_index_position = self.case_index_columns.index(self.case_identifier_column)
            case_id = current_row.name[case_index_position]

            # Extract all events for this case
            case_events = self.event_data.xs(case_id, level=self.case_identifier_column)

            # Get events that occurred before the current time
            current_time = current_row['TimeFromHospFeat']
            previous_events = case_events[case_events['TimeFromHospFeat'] < current_time]

            # Limit to the last X events
            if len(previous_events) > num_events:
                previous_events = previous_events.tail(num_events)

            # Return in requested order
            if reverse_chronological:
                return previous_events.iloc[::-1]
            else:
                return previous_events

        except Exception as e:
            # Return empty DataFrame if there's an error
            print(f"Error retrieving events: {e}")
            print(f"Current row index: {current_row.name}")
            return pd.DataFrame()

