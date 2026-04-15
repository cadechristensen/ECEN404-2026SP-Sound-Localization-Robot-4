"""
Utility functions for audio filtering.
"""

from typing import List, Tuple


def merge_overlapping_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Merge overlapping time segments to avoid double-counting duration.

    Args:
        segments: List of (start_time, end_time) tuples

    Returns:
        List of non-overlapping merged (start_time, end_time) tuples
    """
    if not segments:
        return []

    # Validate that every segment has start <= end before sorting.
    for start, end in segments:
        assert start <= end, f"Invalid segment: start ({start:.3f}) > end ({end:.3f})"

    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])

    merged = [sorted_segments[0]]

    for current_start, current_end in sorted_segments[1:]:
        last_start, last_end = merged[-1]

        # Check if current segment overlaps with the last merged segment
        if current_start <= last_end:
            # Merge by extending the end time if needed
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap, add as new segment
            merged.append((current_start, current_end))

    return merged
