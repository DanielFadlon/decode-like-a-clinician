"""
Per-Event Verbalizer

Creates natural language descriptions that include the current state and recent events.
"""

from typing import Optional, Dict, List
import pandas as pd

from .base_verbalizer import BaseVerbalizer
from .event_processor import EventProcessor
from .time_encoding import TimeApproach, get_current_time_text, get_event_time_encoding, extract_time_from_data
from .formatters import BaseFormatter, TechnicalFormatter


class PerEventVerbalizer(BaseVerbalizer):
    """
    Verbalizer that describes the current patient state along with recent events.

    Output includes:
    1. Current time context
    2. Latest aggregated measurements (current state)
    3. Recent events (changes that occurred)
    """

    def __init__(
        self,
        table_data: pd.DataFrame,
        num_events: int,
        time_approach: TimeApproach,
        feature_names_config: Optional[Dict[str, str]] = None,
        feature_names_config_path: Optional[str] = None,
        case_index_columns: Optional[List[str]] = None,
        case_identifier_column: Optional[str] = None,
        columns_to_ignore_from_events: Optional[List[str]] = None,
        formatter: Optional[BaseFormatter] = None
    ):
        """
        Initialize the per-event verbalizer.

        Args:
            table_data: DataFrame containing the data to verbalize
            num_events: Number of recent events to include in the description
            time_approach: Approach for encoding time information
            feature_names_config: Dictionary mapping feature keys to human-readable names
            feature_names_config_path: Path to JSON file containing feature names mapping
            case_index_columns: List of columns used as multi-index (e.g., ['Encrypted_PatientNum', 'Encrypted_CaseNum'])
            case_identifier_column: Column that uniquely identifies a case (e.g., 'Encrypted_CaseNum')
            columns_to_ignore_from_events: Columns to exclude from event detection (e.g., ['CumulativeCost'])
            formatter: Formatter to use for text generation (defaults to SimpleFormatter)
        """
        super().__init__(table_data, feature_names_config, feature_names_config_path)

        self.num_events = num_events
        self.time_approach = time_approach

        # Set default index columns if not provided
        self.case_index_columns = case_index_columns or ['Encrypted_PatientNum', 'Encrypted_CaseNum']
        self.case_identifier_column = case_identifier_column or 'Encrypted_CaseNum'
        columns_to_ignore = columns_to_ignore_from_events or ['CumulativeCost']

        # Use provided formatter or create default TechnicalFormatter
        if formatter is None:
            formatter = TechnicalFormatter(
                feature_names=self.feature_names,
                skip_null_values=False,
                include_time=False
            )

        self.current_state_formatter = formatter

        # Create formatter for events (skip nulls, no time)
        if isinstance(formatter, TechnicalFormatter):
            self.event_formatter = TechnicalFormatter(
                feature_names=self.feature_names,
                skip_null_values=True,
                include_time=False
            )
        else:
            # For other formatters, use same formatter with skip_nulls
            self.event_formatter = formatter

        # Initialize event processor
        self.event_processor = EventProcessor(
            table_data=table_data,
            case_index_columns=self.case_index_columns,
            case_identifier_column=self.case_identifier_column,
            columns_to_ignore=columns_to_ignore
        )

    def get_text(self, row: pd.Series) -> str:
        """
        Convert a single row into a comprehensive event-based description.

        Args:
            row: A single row from the DataFrame

        Returns:
            Natural language description including current state and recent events
        """
        # Extract current time from row
        current_time = extract_time_from_data(row)

        # Build the text description
        text_parts = []

        # 1. Add time context
        text_parts.append(get_current_time_text(self.time_approach, current_time))

        # 2. Add current aggregated state
        current_state_text = self.current_state_formatter.format_row(row)
        text_parts.append(f"**Latest Aggregated Measurement:** {current_state_text}\n\n")

        # 3. Add recent events
        recent_events = self.event_processor.get_last_x_events(
            row,
            self.num_events,
            reverse_chronological=True
        )

        if not recent_events.empty:
            for event_index, event_data in recent_events.iterrows():
                # Extract event time from event data
                event_time = extract_time_from_data(event_data, event_index)

                # Get time encoding for this event
                time_encoding = get_event_time_encoding(
                    self.time_approach,
                    event_time,
                    current_time
                )

                # Get the event description (only changed values)
                event_description = self.event_formatter.format_row(event_data)

                text_parts.append(f"**{time_encoding}:** {event_description}\n\n")

        return "".join(text_parts)

