"""
Formatters Module

Provides different formatting strategies for converting feature values to text.

Available Formatters:
- TechnicalFormatter: "Feature: value" format (technical/structured)
- NarrativeFormatter: Rule-based narrative format (natural language)
- IndicatorSeriesFormatter: Technical series list format (for per-indicator)
- IndicatorNarrativeFormatter: Temporal narrative format (for per-indicator)
"""

from .base_formatter import BaseFormatter
from .technical_formatter import TechnicalFormatter
from .narrative_formatter import NarrativeFormatter
from .indicator_series_formatter import IndicatorSeriesFormatter
from .indicator_narrative_formatter import IndicatorNarrativeFormatter

# Backward compatibility alias
SimpleFormatter = TechnicalFormatter

__all__ = [
    'BaseFormatter',
    'TechnicalFormatter',
    'NarrativeFormatter',
    'IndicatorSeriesFormatter',
    'IndicatorNarrativeFormatter',
    'SimpleFormatter',  # Deprecated, use TechnicalFormatter
]

