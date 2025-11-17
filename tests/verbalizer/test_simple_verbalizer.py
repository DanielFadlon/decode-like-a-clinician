"""
Tests for SimpleVerbalizer with clear expected outputs

Expected output format: "time from hospitalization: X, Feature1: value1, Feature2: value2."
"""

import pytest
import pandas as pd
from src.verbalizer import SimpleVerbalizer


class TestSimpleVerbalizerBasic:
    """Test basic SimpleVerbalizer functionality."""

    def test_basic_verbalization(self, sample_row_data, feature_names):
        """
        Test basic technical format output.

        Expected: "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 1000."
        """
        verbalizer = SimpleVerbalizer(
            table_data=sample_row_data,
            feature_names_config=feature_names
        )

        first_row = sample_row_data.loc[(1, 1, pd.Timedelta(hours=0))]
        text = verbalizer.get_text(first_row)

        expected = "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 1000."

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}\n")

    def test_get_textual_data(self, sample_row_data, feature_names):
        """
        Test get_textual_data returns proper DataFrame.

        Expected first row: "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 1000."
        """
        verbalizer = SimpleVerbalizer(
            table_data=sample_row_data,
            feature_names_config=feature_names
        )

        result = verbalizer.get_textual_data()

        # Check structure
        assert list(result.columns) == ['text', 'set_type', 'label'], \
            f"Expected columns ['text', 'set_type', 'label'], got {list(result.columns)}"

        assert len(result) == 7, f"Expected 7 rows, got {len(result)}"

        # Check first row
        expected_first = "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 1000."
        assert result['text'].iloc[0] == expected_first, \
            f"\nExpected first row:\n{expected_first}\n\nGot:\n{result['text'].iloc[0]}"

        print(f"\n✓ DataFrame shape: {result.shape}")
        print(f"✓ First row: {result['text'].iloc[0]}\n")


class TestSimpleVerbalizerNullHandling:
    """Test null value handling."""

    def test_skip_null_values(self, sample_row_data, feature_names):
        """
        Test skip_null_values=True excludes null features.

        Expected: "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Blood Pressure: 120, Cumulative Cost: 1000."
        (Temperature excluded)
        """
        data = sample_row_data.copy()
        data.loc[(1, 1, pd.Timedelta(hours=0)), 'Temperature'] = None

        verbalizer = SimpleVerbalizer(
            table_data=data,
            feature_names_config=feature_names,
            skip_null_values=True
        )

        first_row = data.loc[(1, 1, pd.Timedelta(hours=0))]
        text = verbalizer.get_text(first_row)

        expected = "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Blood Pressure: 120, Cumulative Cost: 1000."

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output (null excluded): {text}\n")

    def test_include_null_values(self, sample_row_data, feature_names):
        """
        Test skip_null_values=False includes null features.

        Expected: "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Body Temperature: nan, Blood Pressure: 120, Cumulative Cost: 1000."
        """
        data = sample_row_data.copy()
        data.loc[(1, 1, pd.Timedelta(hours=0)), 'Temperature'] = None

        verbalizer = SimpleVerbalizer(
            table_data=data,
            feature_names_config=feature_names,
            skip_null_values=False
        )

        first_row = data.loc[(1, 1, pd.Timedelta(hours=0))]
        text = verbalizer.get_text(first_row)

        expected = "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Body Temperature: nan, Blood Pressure: 120, Cumulative Cost: 1000."

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output (null included): {text}\n")


class TestSimpleVerbalizerTimeHandling:
    """Test time feature inclusion/exclusion."""

    def test_exclude_time(self, sample_row_data, feature_names):
        """
        Test include_time=False excludes TimeFromHospFeat.

        Expected: "Patient Age: 65, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 1000."
        (time from hospitalization excluded)
        """
        verbalizer = SimpleVerbalizer(
            table_data=sample_row_data,
            feature_names_config=feature_names,
            include_time=False
        )

        first_row = sample_row_data.loc[(1, 1, pd.Timedelta(hours=0))]
        text = verbalizer.get_text(first_row)

        expected = "Patient Age: 65, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 1000."

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}\n")


class TestSimpleVerbalizerForwardFill:
    """Test forward fill functionality."""

    def test_forward_fill_applied(self, sample_row_data, feature_names):
        """
        Test apply_forward_fill=True fills missing values.

        Expected at t=6 with HR filled from t=0: "time from hospitalization: 6, Patient Age: 65, Heart Rate: 72.0, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 2000."
        """
        data = sample_row_data.copy()
        data.loc[(1, 1, pd.Timedelta(hours=6)), 'HeartRate'] = None
        data.loc[(1, 1, pd.Timedelta(hours=12)), 'HeartRate'] = None

        verbalizer = SimpleVerbalizer(
            table_data=data,
            feature_names_config=feature_names,
            apply_forward_fill=True
        )

        # Check that t=6 has forward-filled value from t=0
        row_at_6 = verbalizer.data.loc[(1, 1, pd.Timedelta(hours=6))]
        text = verbalizer.get_text(row_at_6)

        expected = "time from hospitalization: 6, Patient Age: 65, Heart Rate: 72.0, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 2000."

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output at t=6 (HeartRate forward-filled from 72): {text}\n")

    def test_forward_fill_preserves_metadata(self, sample_row_data, feature_names):
        """
        Test forward fill doesn't affect metadata columns.

        Expected: set_type='train' and label=[0,0,0,1,0,0,0] preserved
        """
        verbalizer = SimpleVerbalizer(
            table_data=sample_row_data,
            feature_names_config=feature_names,
            apply_forward_fill=True
        )

        # Check metadata preserved
        assert all(verbalizer.data['set_type'] == 'train'), "set_type should remain 'train'"
        assert list(verbalizer.data['label'].values) == [0, 0, 0, 1, 0, 0, 0], \
            f"labels should be preserved: {list(verbalizer.data['label'].values)}"

        print(f"\n✓ Metadata preserved: set_type=train, labels=[0,0,0,1,0,0,0]\n")


class TestSimpleVerbalizerFeatureNames:
    """Test feature name mapping."""

    def test_feature_name_mapping(self, sample_row_data, feature_names):
        """
        Test feature keys are mapped to human-readable names.

        Expected: "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 1000."
        """
        verbalizer = SimpleVerbalizer(
            table_data=sample_row_data,
            feature_names_config=feature_names
        )

        first_row = sample_row_data.loc[(1, 1, pd.Timedelta(hours=0))]
        text = verbalizer.get_text(first_row)

        expected = "time from hospitalization: 0, Patient Age: 65, Heart Rate: 72, Body Temperature: 98.6, Blood Pressure: 120, Cumulative Cost: 1000."

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output with readable names: {text}\n")

    def test_no_feature_name_mapping(self, sample_row_data):
        """
        Test without feature names, technical keys are used.

        Expected: "TimeFromHospFeat: 0, Age: 65, HeartRate: 72, Temperature: 98.6, BloodPressure: 120, CumulativeCost: 1000."
        """
        verbalizer = SimpleVerbalizer(
            table_data=sample_row_data
        )

        first_row = sample_row_data.loc[(1, 1, pd.Timedelta(hours=0))]
        text = verbalizer.get_text(first_row)

        expected = "TimeFromHospFeat: 0, Age: 65, HeartRate: 72, Temperature: 98.6, BloodPressure: 120, CumulativeCost: 1000."

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output with technical names: {text}\n")
