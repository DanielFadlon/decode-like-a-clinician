"""
Verbalizer Module

This module provides tools for converting structured medical data into natural language descriptions.

Key Components:
- BaseVerbalizer: Base class for all verbalizers
- SimpleVerbalizer: Simple feature-value pair format
- PerEventVerbalizer: Current state + recent events/changes
- PerIndicatorVerbalizer: Temporal history of each indicator
- TimeApproach: Enum for different time encoding approaches

Formatters:
- TechnicalFormatter: "Feature: value" format (technical/structured)
- NarrativeFormatter: Rule-based narrative format (natural language)
- IndicatorSeriesFormatter: Technical series list format (for per-indicator)
- IndicatorNarrativeFormatter: Temporal narrative format (for per-indicator)
"""

from .base_verbalizer import BaseVerbalizer
from .simple_verbalizer import SimpleVerbalizer
from .per_event_verbalizer import PerEventVerbalizer
from .per_indicator_verbalizer import PerIndicatorVerbalizer
from .time_encoding import TimeApproach, get_current_time_text, get_event_time_encoding
from .event_processor import EventProcessor
from .history_builder import HistoryBuilder
from .formatters import (
    BaseFormatter,
    TechnicalFormatter,
    NarrativeFormatter,
    IndicatorSeriesFormatter,
    IndicatorNarrativeFormatter,
    SimpleFormatter  # Deprecated alias for TechnicalFormatter
)

__all__ = [
    'BaseVerbalizer',
    'SimpleVerbalizer',
    'PerEventVerbalizer',
    'PerIndicatorVerbalizer',
    'TimeApproach',
    'get_current_time_text',
    'get_event_time_encoding',
    'EventProcessor',
    'HistoryBuilder',
    'BaseFormatter',
    'TechnicalFormatter',
    'NarrativeFormatter',
    'IndicatorSeriesFormatter',
    'IndicatorNarrativeFormatter',
    'SimpleFormatter',  # Deprecated, kept for backward compatibility
]

