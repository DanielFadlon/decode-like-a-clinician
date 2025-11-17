"""
Tests for PerEventVerbalizer with exact expected outputs

TimeFromHosp (Timedelta) is in the index
TimeFromHospFeat (integer hours) is a column in the output
"""

import pytest
import pandas as pd
from src.verbalizer import PerEventVerbalizer, TechnicalFormatter, NarrativeFormatter, TimeApproach


class TestPerEventVerbalizerBasic:
    """Test basic PerEventVerbalizer functionality."""

    def test_technical_format_default(self, sample_row_data, feature_names):
        """
        Test PerEventVerbalizer with TechnicalFormatter at t=24.

        Expected: Multi-line output with time context, current state, and 3 medical updates
        """
        verbalizer = PerEventVerbalizer(
            table_data=sample_row_data,
            num_events=3,
            time_approach=TimeApproach.ABSOLUTE_POSITION,
            feature_names_config=feature_names
        )

        row_at_24 = sample_row_data.loc[(1, 1, pd.Timedelta(hours=24))]
        text = verbalizer.get_text(row_at_24)

        expected = ("The patient's hospital journey began at time t=0 hours and has now reached t=24 hours.\n\n"
                   "**Latest Aggregated Measurement:** Patient Age: 65, Heart Rate: 82, Body Temperature: 99.1, Blood Pressure: 130, Cumulative Cost: 5000.\n\n"
                   "**Medical updates from t=12:** Heart Rate: 78, Body Temperature: 98.8, Blood Pressure: 125.0.\n\n"
                   "**Medical updates from t=6:** Heart Rate: 75.\n\n"
                   "**Medical updates from t=0:** Patient Age: 65.0, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120.0.\n\n")

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output at t=24:\n{text}")

    def test_includes_current_state(self, sample_row_data, feature_names):
        """
        Test output at middle timepoint (t=12).

        Expected: Current state at t=12 with 2 historical events
        """
        verbalizer = PerEventVerbalizer(
            table_data=sample_row_data,
            num_events=2,
            time_approach=TimeApproach.ABSOLUTE_POSITION,
            feature_names_config=feature_names
        )

        row_at_12 = sample_row_data.loc[(1, 1, pd.Timedelta(hours=12))]
        text = verbalizer.get_text(row_at_12)

        expected = ("The patient's hospital journey began at time t=0 hours and has now reached t=12 hours.\n\n"
                   "**Latest Aggregated Measurement:** Patient Age: 65, Heart Rate: 78, Body Temperature: 98.8, Blood Pressure: 125, Cumulative Cost: 3500.\n\n"
                   "**Medical updates from t=6:** Heart Rate: 75.\n\n"
                   "**Medical updates from t=0:** Patient Age: 65.0, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120.0.\n\n")

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output at t=12:\n{text}")


class TestPerEventVerbalizerTimeEncoding:
    """Test different time encoding approaches."""

    def test_absolute_position(self, sample_row_data, feature_names):
        """
        Test ABSOLUTE_POSITION shows "t=X hours" format.

        Expected: Time header with "t=0 hours" and "t=24 hours", events marked with "t=X"
        """
        verbalizer = PerEventVerbalizer(
            table_data=sample_row_data,
            num_events=2,
            time_approach=TimeApproach.ABSOLUTE_POSITION,
            feature_names_config=feature_names
        )

        row_at_24 = sample_row_data.loc[(1, 1, pd.Timedelta(hours=24))]
        text = verbalizer.get_text(row_at_24)

        expected = ("The patient's hospital journey began at time t=0 hours and has now reached t=24 hours.\n\n"
                   "**Latest Aggregated Measurement:** Patient Age: 65, Heart Rate: 82, Body Temperature: 99.1, Blood Pressure: 130, Cumulative Cost: 5000.\n\n"
                   "**Medical updates from t=12:** Heart Rate: 78, Body Temperature: 98.8, Blood Pressure: 125.0.\n\n"
                   "**Medical updates from t=6:** Heart Rate: 75.\n\n")

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Absolute position format:\n{text}")

    def test_relative_position(self, sample_row_data, feature_names):
        """
        Test RELATIVE_POSITION shows "X hours ago" format.

        Expected: "The patient time from hospitalization is 24 hours."
                  "**Medical updates from 12 hours ago:**"
        """
        verbalizer = PerEventVerbalizer(
            table_data=sample_row_data,
            num_events=2,
            time_approach=TimeApproach.RELATIVE_POSITION,
            feature_names_config=feature_names
        )

        row_at_24 = sample_row_data.loc[(1, 1, pd.Timedelta(hours=24))]
        text = verbalizer.get_text(row_at_24)

        expected = ('The patient time from hospitalization is 24 hours.\n\n'
                   '**Latest Aggregated Measurement:** Patient Age: 65, Heart Rate: 82, Body Temperature: 99.1, Blood Pressure: 130, Cumulative Cost: 5000.\n\n'
                   '**Medical updates from 12 hours ago:** Heart Rate: 78, Body Temperature: 98.8, Blood Pressure: 125.0.\n\n'
                   '**Medical updates from 18 hours ago:** Heart Rate: 75.\n\n')

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Relative position format:\n{text}")

    def test_no_time_encoding(self, sample_row_data, feature_names):
        """
        Test NONE approach focuses on medical data without time preamble.

        Expected: Starts with "**Latest Aggregated Measurement:**", no time context
        """
        verbalizer = PerEventVerbalizer(
            table_data=sample_row_data,
            num_events=2,
            time_approach=TimeApproach.NONE,
            feature_names_config=feature_names
        )

        row_at_24 = sample_row_data.loc[(1, 1, pd.Timedelta(hours=24))]
        text = verbalizer.get_text(row_at_24)

        expected = ('**Latest Aggregated Measurement:** Patient Age: 65, Heart Rate: 82, Body Temperature: 99.1, Blood Pressure: 130, Cumulative Cost: 5000.\n\n'
                   '**Medical updates:** Heart Rate: 78, Body Temperature: 98.8, Blood Pressure: 125.0.\n\n'
                   '**Medical updates:** Heart Rate: 75.\n\n')

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ No time encoding format:\n{text}")


class TestPerEventVerbalizerNarrativeFormat:
    """Test narrative formatting."""

    def test_narrative_formatter_injection(self, sample_row_data, feature_names, narrative_rules):
        """
        Test NarrativeFormatter produces sentence-based output with rules applied.

        Expected: Narrative sentences with rules like "Heart rate is normal at 82 bpm."
        """
        formatter = NarrativeFormatter(
            rules_config=narrative_rules,
            feature_names=feature_names
        )

        verbalizer = PerEventVerbalizer(
            table_data=sample_row_data,
            num_events=2,
            time_approach=TimeApproach.RELATIVE_POSITION,
            formatter=formatter
        )

        row_at_24 = sample_row_data.loc[(1, 1, pd.Timedelta(hours=24))]
        text = verbalizer.get_text(row_at_24)

        expected = ('The patient time from hospitalization is 24 hours.\n\n'
                   '**Latest Aggregated Measurement:** time from hospitalization is 24.\n'
                   'Patient Age is 65.0.\n'
                   'Heart rate is normal at 82 bpm.\n'
                   'Body temperature is elevated at 99.1°F, indicating fever.\n'
                   'Blood pressure is elevated at 130 mmHg.\n'
                   'Cumulative Cost is 5000.0.\n\n'
                   '**Medical updates from 12 hours ago:** time from hospitalization is 12.\n'
                   'Patient Age is not measured yet.\n'
                   'Heart rate is normal at 78 bpm.\n'
                   'Body temperature is normal at 98.8°F.\n'
                   'Blood pressure is elevated at 125.0 mmHg.\n\n'
                   '**Medical updates from 18 hours ago:** time from hospitalization is 6.\n'
                   'Patient Age is not measured yet.\n'
                   'Heart rate is normal at 75 bpm.\n'
                   'Body Temperature is not measured yet.\n'
                   'Blood Pressure is not measured yet.\n\n')

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Narrative format with rules:\n{text}")


class TestPerEventVerbalizerEventDetection:
    """Test event detection."""

    def test_columns_ignored_from_events(self, sample_row_data, feature_names):
        """
        Test CumulativeCost is excluded from event detection.

        Expected: CumulativeCost not in event processor columns
        """
        verbalizer = PerEventVerbalizer(
            table_data=sample_row_data,
            num_events=5,
            time_approach=TimeApproach.ABSOLUTE_POSITION,
            feature_names_config=feature_names,
            columns_to_ignore_from_events=['CumulativeCost']
        )

        # Verify CumulativeCost is excluded from event processing
        event_columns = list(verbalizer.event_processor.get_data().columns)
        assert 'CumulativeCost' not in event_columns, \
            f"CumulativeCost should be excluded from events, but found in: {event_columns}"

        print(f"\n✓ CumulativeCost excluded from events")
        print(f"  Event processor columns: {event_columns}")


class TestPerEventVerbalizerEdgeCases:
    """Test edge cases."""

    def test_first_row_no_history(self, sample_row_data, feature_names):
        """
        Test first row (t=0) with no history.

        Expected: Current state shown without errors, no historical events
        """
        verbalizer = PerEventVerbalizer(
            table_data=sample_row_data,
            num_events=3,
            time_approach=TimeApproach.ABSOLUTE_POSITION,
            feature_names_config=feature_names
        )

        first_row = sample_row_data.loc[(1, 1, pd.Timedelta(hours=0))]
        text = verbalizer.get_text(first_row)

        expected = ("The patient's hospital journey began at time t=0 hours and has now reached t=0 hours.\n\n"
                   "**Latest Aggregated Measurement:** Patient Age: 65, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 1000.\n\n")

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ First row (t=0) with no history:\n{text}")
