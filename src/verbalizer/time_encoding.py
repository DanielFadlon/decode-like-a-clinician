"""
Time Encoding Utilities

Provides various approaches for encoding temporal information in natural language.
"""

from enum import Enum
from datetime import datetime, timedelta


class TimeApproach(Enum):
    """Different approaches for encoding time in natural language."""
    ABSOLUTE_POSITION = "absolute_position"
    RELATIVE_POSITION = "relative_position"
    ABSOLUTE_POSITION_INTERVALS = "absolute_position_intervals"
    RELATIVE_POSITION_INTERVALS = "relative_position_intervals"
    REAL_DATE = "real_date"
    REAL_DAY_AND_HOUR = "real_day_and_hour"
    NONE = "none"


# Constants
INTERVAL_SIZE = 6  # hours
START_DATE = "2022-12-02 at 12:00:00"


def add_time(date_iso_format_str: str, time_to_add_in_hours: float) -> str:
    """
    Add hours to a date string.

    Args:
        date_iso_format_str: Date string in format "%Y-%m-%d at %H:%M:%S"
        time_to_add_in_hours: Number of hours to add

    Returns:
        New date string in the same format
    """
    format_str = "%Y-%m-%d at %H:%M:%S"
    original_date = datetime.strptime(date_iso_format_str, format_str)
    new_date = original_date + timedelta(hours=time_to_add_in_hours)
    return new_date.strftime(format_str)


def get_date_by_time_from_hosp(time_from_hosp_in_hours: float) -> str:
    """Get absolute date based on hours from hospitalization start."""
    return add_time(START_DATE, time_from_hosp_in_hours)


def get_day_and_hour_text(time_in_hours: float) -> str:
    """
    Convert hours to "Day X, HH:00" format.

    Args:
        time_in_hours: Time in hours

    Returns:
        String in format "Day X, HH:00"
    """
    day = int(time_in_hours // 24)
    hour = int(time_in_hours % 24)
    hour_text = f"{hour:02d}"
    return f"Day {day}, {hour_text}:00"


def get_relative_position_encoded_text(
    time_from_hosp_in_hours: float,
    value_time_in_hours: float,
    should_count_intervals: bool = False
) -> str:
    """
    Encode relative time position as natural language.

    Args:
        time_from_hosp_in_hours: Current time from hospitalization
        value_time_in_hours: Time when the value was recorded
        should_count_intervals: Whether to count in intervals instead of hours

    Returns:
        Natural language description like "3 hours ago" or "2 intervals ago"
    """
    time_passed = int(time_from_hosp_in_hours - value_time_in_hours)

    if time_passed == 0:
        return "just now"

    if should_count_intervals:
        num_intervals = time_passed // INTERVAL_SIZE
        plural = "s" if num_intervals != 1 else ""
        return f"{num_intervals} interval{plural} ago"

    plural = "s" if time_passed != 1 else ""
    return f"{time_passed} hour{plural} ago"


def get_current_time_text(
    time_approach: TimeApproach,
    current_time_passed_in_hours: float
) -> str:
    """
    Generate the current time context text based on the time approach.

    Args:
        time_approach: The encoding approach to use
        current_time_passed_in_hours: Current time from hospitalization

    Returns:
        Natural language description of the current time
    """
    if time_approach == TimeApproach.ABSOLUTE_POSITION:
        return (
            f"The patient's hospital journey began at time t=0 hours "
            f"and has now reached t={int(current_time_passed_in_hours)} hours.\n\n"
        )
    elif time_approach == TimeApproach.RELATIVE_POSITION:
        return (
            f"The patient time from hospitalization is "
            f"{int(current_time_passed_in_hours)} hours.\n\n"
        )
    elif time_approach == TimeApproach.ABSOLUTE_POSITION_INTERVALS:
        intervals = int(current_time_passed_in_hours // INTERVAL_SIZE)
        return (
            f"The patient's hospital journey began at time t=0 interval windows "
            f"and has now reached t={intervals} intervals (window interval = 6 hours).\n\n"
        )
    elif time_approach == TimeApproach.RELATIVE_POSITION_INTERVALS:
        intervals = int(current_time_passed_in_hours // INTERVAL_SIZE)
        return (
            f"The patient time from hospitalization is {intervals} interval windows "
            f"(interval window = 6 hours).\n\n"
        )
    elif time_approach == TimeApproach.REAL_DATE:
        current_date = get_date_by_time_from_hosp(current_time_passed_in_hours)
        return (
            f"The patient's hospital journey began on {START_DATE} "
            f"and has now reached {current_date}.\n\n"
        )
    elif time_approach == TimeApproach.REAL_DAY_AND_HOUR:
        current_time_text = get_day_and_hour_text(current_time_passed_in_hours)
        return (
            f"The patient's hospital journey began on Day 0, 00:00 "
            f"and has now reached {current_time_text}.\n\n"
        )
    elif time_approach == TimeApproach.NONE:
        return ""
    else:
        raise ValueError(f"Unknown time approach: {time_approach}")


def get_event_time_encoding(
    time_approach: TimeApproach,
    event_time_in_hours: float,
    current_time_passed_in_hours: float
) -> str:
    """
    Generate the time encoding text for a specific event.

    Args:
        time_approach: The encoding approach to use
        event_time_in_hours: Time when the event occurred
        current_time_passed_in_hours: Current time from hospitalization

    Returns:
        Natural language description of the event time
    """
    prefix = "Medical updates"

    if time_approach == TimeApproach.ABSOLUTE_POSITION:
        return f"{prefix} from t={int(event_time_in_hours)}"
    elif time_approach == TimeApproach.RELATIVE_POSITION:
        relative_text = get_relative_position_encoded_text(
            current_time_passed_in_hours,
            event_time_in_hours
        )
        return f"{prefix} from {relative_text}"
    elif time_approach == TimeApproach.ABSOLUTE_POSITION_INTERVALS:
        intervals = int(event_time_in_hours // INTERVAL_SIZE)
        return f"{prefix} from t={intervals}"
    elif time_approach == TimeApproach.RELATIVE_POSITION_INTERVALS:
        relative_text = get_relative_position_encoded_text(
            current_time_passed_in_hours,
            event_time_in_hours,
            should_count_intervals=True
        )
        return f"{prefix} from {relative_text}"
    elif time_approach == TimeApproach.REAL_DATE:
        event_date = get_date_by_time_from_hosp(event_time_in_hours)
        return f"{prefix} from {event_date}"
    elif time_approach == TimeApproach.REAL_DAY_AND_HOUR:
        event_time_text = get_day_and_hour_text(event_time_in_hours)
        return f"{prefix} from {event_time_text}"
    elif time_approach == TimeApproach.NONE:
        return prefix
    else:
        raise ValueError(f"Unknown time approach: {time_approach}")

