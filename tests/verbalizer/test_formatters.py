"""
Tests for Formatters with exact expected outputs

Uses pytest with exact string matching to verify formatter functionality.
"""

import pytest
import pandas as pd
from src.verbalizer.formatters import (
    TechnicalFormatter,
    NarrativeFormatter,
    IndicatorSeriesFormatter,
    IndicatorNarrativeFormatter
)
from src.verbalizer import TimeApproach


class TestTechnicalFormatter:
    """Test TechnicalFormatter functionality."""

    def test_basic_formatting(self, feature_names):
        """
        Test basic technical formatting produces expected output.

        Expected: "Heart Rate: 82.0, Body Temperature: 99.1, Blood Pressure: 130.0."
        """
        row = pd.Series({
            'HeartRate': 82,
            'Temperature': 99.1,
            'BloodPressure': 130
        })

        formatter = TechnicalFormatter(
            feature_names=feature_names
        )

        text = formatter.format_row(row)

        expected = 'Heart Rate: 82.0, Body Temperature: 99.1, Blood Pressure: 130.0.'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")

    def test_skip_null_values(self, feature_names):
        """
        Test skip_null_values=True excludes null features.

        Expected: "Heart Rate: 82.0, Blood Pressure: 130.0." (Temperature excluded)
        """
        row = pd.Series({
            'HeartRate': 82,
            'Temperature': None,
            'BloodPressure': 130
        })

        formatter = TechnicalFormatter(
            feature_names=feature_names,
            skip_null_values=True
        )

        text = formatter.format_row(row)

        expected = 'Heart Rate: 82.0, Blood Pressure: 130.0.'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")

    def test_metadata_columns_skipped(self, feature_names):
        """
        Test metadata columns are automatically skipped.

        Expected: "Heart Rate: 82." (set_type and label excluded)
        """
        row = pd.Series({
            'HeartRate': 82,
            'set_type': 'train',
            'label': 1
        })

        formatter = TechnicalFormatter(feature_names=feature_names)
        text = formatter.format_row(row)

        expected = 'Heart Rate: 82.'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")


class TestNarrativeFormatter:
    """Test NarrativeFormatter functionality."""

    def test_rule_based_formatting(self, feature_names, narrative_rules):
        """
        Test narrative rules produce expected output.

        Expected: "Heart rate is normal at 82 bpm."
        """
        row = pd.Series({'HeartRate': 82})

        formatter = NarrativeFormatter(
            rules_config=narrative_rules,
            feature_names=feature_names
        )

        text = formatter.format_row(row)

        expected = 'Heart rate is normal at 82 bpm.'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")

    def test_high_threshold_narrative(self, feature_names, narrative_rules):
        """
        Test high threshold produces elevated narrative.

        Expected: "Body temperature is elevated at 99.1°F, indicating fever."
        """
        row = pd.Series({'Temperature': 99.1})

        formatter = NarrativeFormatter(
            rules_config=narrative_rules,
            feature_names=feature_names
        )

        text = formatter.format_row(row)

        expected = 'Body temperature is elevated at 99.1°F, indicating fever.'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")

    def test_low_threshold_narrative(self, feature_names, narrative_rules):
        """
        Test low threshold produces low narrative.

        Expected: "Heart rate is low at 50 bpm, indicating bradycardia."
        """
        row = pd.Series({'HeartRate': 50})

        formatter = NarrativeFormatter(
            rules_config=narrative_rules,
            feature_names=feature_names
        )

        text = formatter.format_row(row)

        expected = 'Heart rate is low at 50 bpm, indicating bradycardia.'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")

    def test_fallback_to_simple_format(self, feature_names, narrative_rules):
        """
        Test features without rules fall back to simple format.

        Expected: "UnknownFeature is 42."
        """
        row = pd.Series({'UnknownFeature': 42})

        formatter = NarrativeFormatter(
            rules_config=narrative_rules,
            feature_names=feature_names
        )

        text = formatter.format_row(row)

        expected = 'UnknownFeature is 42.'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")


class TestIndicatorSeriesFormatter:
    """Test IndicatorSeriesFormatter functionality."""

    def test_series_formatting(self, feature_names):
        """
        Test series format produces time-stamped list.

        Expected: "Patient Age: 65.0.\nHeart Rate: 72 (t=0), 75 (t=6), 78 (t=12)."
        """
        row = pd.Series({
            'Age': [{'time': 0, 'value': 65}],
            'HeartRate': [
                {'time': 0, 'value': 72},
                {'time': 6, 'value': 75},
                {'time': 12, 'value': 78}
            ],
            'TimeFromHospFeat': [{'time': 12, 'value': 12}]
        })

        formatter = IndicatorSeriesFormatter(
            feature_names=feature_names,
            time_approach=TimeApproach.ABSOLUTE_POSITION
        )

        text = formatter.format_row(row)

        expected = 'Patient Age: 65.0.\nHeart Rate: 72 (t=0), 75 (t=6), 78 (t=12).'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output:\n{text}")

    def test_num_values_limit(self, feature_names):
        """
        Test num_values_to_show limits displayed values.

        Expected: "Heart Rate: 78 (t=12), 82 (t=18)." (only last 2 values)
        """
        row = pd.Series({
            'HeartRate': [
                {'time': 0, 'value': 72},
                {'time': 6, 'value': 75},
                {'time': 12, 'value': 78},
                {'time': 18, 'value': 82}
            ],
            'TimeFromHospFeat': [{'time': 18, 'value': 18}]
        })

        formatter = IndicatorSeriesFormatter(
            feature_names=feature_names,
            num_values_to_show=2,
            time_approach=TimeApproach.ABSOLUTE_POSITION
        )

        text = formatter.format_row(row)

        expected = 'Heart Rate: 78 (t=12), 82 (t=18).'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")


class TestIndicatorNarrativeFormatter:
    """Test IndicatorNarrativeFormatter functionality."""

    def test_temporal_narrative(self, feature_names):
        """
        Test temporal narrative produces expected format.

        Expected: "Patient Age: 65.\nHeart Rate was measured 3 times from t=0 to t=12, with values 72, 75 and 78 (final: 78 at t=12)."
        """
        row = pd.Series({
            'Age': [{'time': 0, 'value': 65}],
            'HeartRate': [
                {'time': 0, 'value': 72},
                {'time': 6, 'value': 75},
                {'time': 12, 'value': 78}
            ],
            'TimeFromHospFeat': [{'time': 12, 'value': 12}]
        })

        formatter = IndicatorNarrativeFormatter(
            feature_names=feature_names
        )

        text = formatter.format_row(row)

        expected = 'Patient Age: 65.\nHeart Rate was measured 3 times from t=0 to t=12, with values 72, 75 and 78 (final: 78 at t=12).'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output:\n{text}")

    def test_single_measurement_narrative(self, feature_names):
        """
        Test single measurement gets appropriate narrative.

        Expected: "Heart Rate was measured once from t=0 to t=0, with value 72 (final: 72 at t=0)."
        """
        row = pd.Series({
            'HeartRate': [{'time': 0, 'value': 72}],
            'TimeFromHospFeat': [{'time': 0, 'value': 0}]
        })

        formatter = IndicatorNarrativeFormatter(feature_names=feature_names)
        text = formatter.format_row(row)

        expected = 'Heart Rate was measured once from t=0 to t=0, with value 72 (final: 72 at t=0).'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")


class TestFormatterEdgeCases:
    """Test edge cases for all formatters."""

    def test_empty_row_technical(self):
        """
        Test TechnicalFormatter handles empty row gracefully.

        Expected: "No data available."
        """
        row = pd.Series({})
        formatter = TechnicalFormatter()
        text = formatter.format_row(row)

        expected = 'No data available.'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")

    def test_all_null_values(self, feature_names):
        """
        Test formatters handle all-null row.

        Expected: "No data available."
        """
        row = pd.Series({
            'HeartRate': None,
            'Temperature': None,
            'BloodPressure': None
        })

        formatter = TechnicalFormatter(
            feature_names=feature_names,
            skip_null_values=True
        )

        text = formatter.format_row(row)

        expected = 'No data available.'

        assert text == expected, f"\nExpected:\n{expected}\n\nGot:\n{text}"
        print(f"\n✓ Output: {text}")

