"""
Pytest configuration and fixtures for verbalizer tests.
"""

import pytest
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


@pytest.fixture
def sample_row_data():
    """
    Fixture providing sample row-based data for testing.

    TimeFromHospFeat represents hours passed since hospitalization.
    It is part of the multi-level index (patient, case, time).

    Returns:
        DataFrame with patient measurements over time
    """
    data = {
        'Encrypted_PatientNum': [1, 1, 1, 1, 2, 2, 2],
        'Encrypted_CaseNum': [1, 1, 1, 1, 2, 2, 2],
        'TimeFromHosp': [pd.Timedelta(hours=0), pd.Timedelta(hours=6), pd.Timedelta(hours=12), pd.Timedelta(hours=24), pd.Timedelta(hours=30), pd.Timedelta(hours=36), pd.Timedelta(hours=42)],# Hours since hospitalization
        'TimeFromHospFeat': [0, 6, 12, 24, 30, 36, 42],  # Hours since hospitalization
        'Age': [65, 65, 65, 65, 45, 45, 45],
        'HeartRate': [72, 75, 78, 82, 68, 70, 72],
        'Temperature': [98.6, 98.6, 98.8, 99.1, 98.4, 98.5, 98.7],
        'BloodPressure': [120, 120, 125, 130, 110, 115, 115],
        'CumulativeCost': [1000, 2000, 3500, 5000, 800, 1500, 2200],
        'set_type': ['train', 'train', 'train', 'train', 'train', 'train', 'train'],
        'label': [0, 0, 0, 1, 0, 0, 0]
    }

    df = pd.DataFrame(data)
    # Set multi-level index: patient -> case -> time (hours)
    df = df.set_index(['Encrypted_PatientNum', 'Encrypted_CaseNum', 'TimeFromHosp'])

    return df


@pytest.fixture
def sample_history_data():
    """
    Fixture providing sample history-based data for testing.

    Returns:
        DataFrame where each feature contains a list of {time, value} dictionaries
    """
    data = {
        'Encrypted_PatientNum': [1],
        'Encrypted_CaseNum': [1],
        'TimeFromHospFeat': [[
            {'time': 0, 'value': 0},
            {'time': 6, 'value': 6},
            {'time': 12, 'value': 12},
            {'time': 24, 'value': 24}
        ]],
        'Age': [[{'time': 0, 'value': 65}]],
        'HeartRate': [[
            {'time': 0, 'value': 72},
            {'time': 6, 'value': 75},
            {'time': 12, 'value': 78},
            {'time': 24, 'value': 82}
        ]],
        'Temperature': [[
            {'time': 0, 'value': 98.6},
            {'time': 12, 'value': 98.8},
            {'time': 24, 'value': 99.1}
        ]],
        'BloodPressure': [[
            {'time': 0, 'value': 120},
            {'time': 12, 'value': 125},
            {'time': 24, 'value': 130}
        ]],
        'CumulativeCost': [[
            {'time': 0, 'value': 1000},
            {'time': 6, 'value': 2000},
            {'time': 12, 'value': 3500},
            {'time': 24, 'value': 5000}
        ]],
        'set_type': ['train'],
        'label': [1]
    }

    df = pd.DataFrame(data)
    df = df.set_index(['Encrypted_PatientNum', 'Encrypted_CaseNum'])

    return df


@pytest.fixture
def feature_names():
    """Fixture providing feature name mappings."""
    return {
        'TimeFromHospFeat': 'time from hospitalization',
        'HeartRate': 'Heart Rate',
        'Temperature': 'Body Temperature',
        'BloodPressure': 'Blood Pressure',
        'Age': 'Patient Age',
        'CumulativeCost': 'Cumulative Cost'
    }


@pytest.fixture
def narrative_rules():
    """Fixture providing sample narrative rules for testing."""
    return {
        'HeartRate': {
            'type': 'numeric',
            'mid_min_th': 60,
            'mid_max_th': 100,
            'low_text': 'Heart rate is low at <VALUE> bpm, indicating bradycardia.',
            'mid_text': 'Heart rate is normal at <VALUE> bpm.',
            'high_text': 'Heart rate is elevated at <VALUE> bpm, indicating tachycardia.'
        },
        'Temperature': {
            'type': 'numeric',
            'mid_min_th': 97.0,
            'mid_max_th': 99.0,
            'low_text': 'Body temperature is low at <VALUE>°F.',
            'mid_text': 'Body temperature is normal at <VALUE>°F.',
            'high_text': 'Body temperature is elevated at <VALUE>°F, indicating fever.'
        },
        'BloodPressure': {
            'type': 'numeric',
            'mid_min_th': 90,
            'mid_max_th': 120,
            'low_text': 'Blood pressure is low at <VALUE> mmHg.',
            'mid_text': 'Blood pressure is normal at <VALUE> mmHg.',
            'high_text': 'Blood pressure is elevated at <VALUE> mmHg.'
        }
    }

