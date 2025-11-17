# Verbalizer Module

Clean and professional implementation for converting structured medical data into natural language descriptions.

## Overview

The verbalizer module provides three main verbalizers with two formatting approaches each:

### Verbalizers (What to include)
1. **SimpleVerbalizer** - Basic feature-value pair format
2. **PerEventVerbalizer** - Current state with recent events/changes
3. **PerIndicatorVerbalizer** - Temporal history for each indicator

### Formatters (How to format)
- **Technical Format** - Structured, precise data representation
- **Narrative Format** - Natural language with clinical context

## Format Comparison

### Technical Format (Default)
**Structured, precise data representation**
```
Heart Rate: 82, Blood Pressure: 130/85, Temperature: 99.1.
```

### Narrative Format
**Natural language with clinical context**
```
Heart rate is elevated at 82 bpm, indicating tachycardia.
Blood pressure is elevated at 130/85 mmHg.
Body temperature is elevated at 99.1°F, indicating fever.
```

---

## Usage Examples

### 1. SimpleVerbalizer

Converts each row into feature-value pairs.

```python
from src.verbalizer import SimpleVerbalizer
import pandas as pd

# Basic usage (technical format by default)
verbalizer = SimpleVerbalizer(
    table_data=data,
    feature_names_config_path="config/feature_names.json"
)

# With forward fill
verbalizer = SimpleVerbalizer(
    table_data=data,
    apply_forward_fill=True,  # Fill missing values forward
    skip_null_values=True,    # Don't show null values
    include_time=True
)

# Get textual data
textual_data = verbalizer.get_textual_data()
```

**Output:**
```
Age: 65, Blood Pressure: 120/80, Heart Rate: 75, Temperature: 98.6.
```

---

### 2. PerEventVerbalizer

Describes current state along with recent events (changes).

#### Technical Format (Default)
```python
from src.verbalizer import PerEventVerbalizer, TimeApproach

verbalizer = PerEventVerbalizer(
    table_data=data,
    num_events=5,
    time_approach=TimeApproach.RELATIVE_POSITION,
    feature_names_config_path="config/feature_names.json",
    case_index_columns=['Encrypted_PatientNum', 'Encrypted_CaseNum'],
    case_identifier_column='Encrypted_CaseNum',
    columns_to_ignore_from_events=['CumulativeCost']
)

textual_data = verbalizer.get_textual_data()
```

**Output:**
```
The patient time from hospitalization is 24 hours.

**Latest Aggregated Measurement:** Age: 65, Blood Pressure: 130/85, Heart Rate: 82.

**Medical updates from 2 hours ago:** Blood Pressure: 130/85.
**Medical updates from 6 hours ago:** Heart Rate: 82, Temperature: 99.1.
```

#### Narrative Format
```python
from src.verbalizer import PerEventVerbalizer, NarrativeFormatter, TimeApproach

# Define clinical rules
rules = {
    "HeartRate": {
        "type": "numeric",
        "mid_min_th": 60,
        "mid_max_th": 100,
        "low_text": "Heart rate is low at <VALUE> bpm.",
        "mid_text": "Heart rate is normal at <VALUE> bpm.",
        "high_text": "Heart rate is elevated at <VALUE> bpm."
    }
}

# Create narrative formatter
formatter = NarrativeFormatter(
    rules_config=rules,
    feature_names_config_path="config/feature_names.json"
)

# Use with verbalizer
verbalizer = PerEventVerbalizer(
    table_data=data,
    num_events=5,
    time_approach=TimeApproach.RELATIVE_POSITION,
    formatter=formatter  # Inject narrative formatter
)

textual_data = verbalizer.get_textual_data()
```

**Output:**
```
The patient time from hospitalization is 24 hours.

**Latest Aggregated Measurement:**
Heart rate is elevated at 82 bpm.
Blood pressure is elevated at 130/85 mmHg.

**Medical updates from 2 hours ago:**
Blood pressure is elevated at 130/85 mmHg.
```

---

### 3. PerIndicatorVerbalizer

Shows temporal history of each indicator with changes over time.

#### Technical Series Format (Default)
```python
from src.verbalizer import PerIndicatorVerbalizer, TimeApproach

# From raw data (will create history automatically)
verbalizer = PerIndicatorVerbalizer(
    table_data_path="data/patient_data.parquet",
    is_history_format=False,  # Will create history
    time_approach=TimeApproach.ABSOLUTE_POSITION,
    feature_names_config_path="config/feature_names.json",
    num_report_hours=48,  # Only show changes in last 48 hours
    auto_save_history=True  # Save created history to JSON
)

textual_data = verbalizer.get_textual_data()
```

**Output:**
```
The patient's hospital journey began at time t=0 hours and has now reached t=24 hours.

Age: 65.
CumulativeCost: 15000.
Blood Pressure: 120/80 (t=0), 125/80 (t=6), 130/85 (t=12), 135/85 (t=18).
Heart Rate: 72 (t=0), 75 (t=6), 78 (t=12), 82 (t=18).
Temperature: 98.6 (t=0), 98.8 (t=6), 99.1 (t=12).
```

#### Narrative Format
```python
from src.verbalizer import PerIndicatorVerbalizer, IndicatorNarrativeFormatter, TimeApproach

# Create narrative formatter
formatter = IndicatorNarrativeFormatter(
    feature_names_config_path="config/feature_names.json",
    num_values_to_show=5
)

# Use with verbalizer
verbalizer = PerIndicatorVerbalizer(
    table_data_path="data/patient_data.parquet",
    is_history_format=False,
    time_approach=TimeApproach.ABSOLUTE_POSITION,
    formatter=formatter
)

textual_data = verbalizer.get_textual_data()
```

**Output:**
```
The patient's hospital journey began at time t=0 hours and has now reached t=24 hours.

Age: 65.
CumulativeCost: 15000.
Blood Pressure was measured 4 times from t=0 to t=18, with values 120/80, 125/80, 130/85, and 135/85 (final: 135/85 at t=18).
Heart Rate was measured 4 times from t=0 to t=18, with values 72, 75, 78, and 82 (final: 82 at t=18).
Temperature was measured 3 times from t=0 to t=12, with values 98.6, 98.8, and 99.1 (final: 99.1 at t=12).
```

---

## Time Encoding Approaches

The module supports multiple ways to encode time information:

```python
from src.verbalizer import TimeApproach

TimeApproach.ABSOLUTE_POSITION          # "t=24 hours"
TimeApproach.RELATIVE_POSITION          # "3 hours ago"
TimeApproach.ABSOLUTE_POSITION_INTERVALS # "t=4 intervals" (6-hour windows)
TimeApproach.RELATIVE_POSITION_INTERVALS # "2 intervals ago"
TimeApproach.REAL_DATE                  # "2022-12-03 at 12:00:00"
TimeApproach.REAL_DAY_AND_HOUR         # "Day 1, 12:00"
TimeApproach.NONE                       # No time encoding
```

---

## Formatters

### Available Formatters

#### For Per-Event (Row-based data)
- **TechnicalFormatter** - `"HR: 120, BP: 130/85"`
- **NarrativeFormatter** - `"Heart rate is elevated at 120 bpm, indicating tachycardia."`

#### For Per-Indicator (History-based data)
- **IndicatorSeriesFormatter** - `"HR: 72 (t=0), 75 (t=6), 78 (t=12)"`
- **IndicatorNarrativeFormatter** - `"HR was measured 3 times from t=0 to t=12..."`

### Creating Custom Formatters

```python
from src.verbalizer.formatters import BaseFormatter
import pandas as pd

class MyCustomFormatter(BaseFormatter):
    def format_row(self, row: pd.Series) -> str:
        # Your custom formatting logic
        parts = []
        for feature, value in row.items():
            if self.should_skip_feature(feature, value):
                continue
            parts.append(f"{feature}={value}")
        return " | ".join(parts)

# Use it
from src.verbalizer import PerEventVerbalizer

verbalizer = PerEventVerbalizer(
    table_data=data,
    num_events=5,
    time_approach=TimeApproach.RELATIVE_POSITION,
    formatter=MyCustomFormatter()
)
```

---

## Configuration Files

### Feature Names Configuration
Map technical keys to human-readable names:

```json
{
  "BP_Systolic": "Systolic Blood Pressure",
  "BP_Diastolic": "Diastolic Blood Pressure",
  "HR": "Heart Rate",
  "Temp": "Temperature"
}
```

### Narrative Rules Configuration
Define clinical thresholds and narratives:

```json
{
  "HeartRate": {
    "type": "numeric",
    "mid_min_th": 60,
    "mid_max_th": 100,
    "low_text": "Heart rate is low at <VALUE> bpm, indicating bradycardia.",
    "mid_text": "Heart rate is normal at <VALUE> bpm.",
    "high_text": "Heart rate is elevated at <VALUE> bpm, indicating tachycardia."
  },
  "Temperature": {
    "type": "numeric",
    "mid_min_th": 97.0,
    "mid_max_th": 99.0,
    "low_text": "Body temperature is low at <VALUE>°F.",
    "mid_text": "Body temperature is normal at <VALUE>°F.",
    "high_text": "Body temperature is elevated at <VALUE>°F, indicating fever."
  }
}
```

---

## Testing

Comprehensive test suite with 18 tests covering all scenarios:

```bash
# Run individual test files
python tests/verbalizer/test_simple_verbalizer.py
python tests/verbalizer/test_per_event_verbalizer.py
python tests/verbalizer/test_per_indicator_verbalizer.py
python tests/verbalizer/test_formatters.py
```

See `tests/verbalizer/README.md` for detailed test documentation.
