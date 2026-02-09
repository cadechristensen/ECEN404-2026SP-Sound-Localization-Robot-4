"""
Temporal smoothing for baby cry detection to reduce false positives.

Maintains a sliding window of recent predictions and requires multiple
consecutive high-confidence predictions before triggering an alert.
Filters out transient sounds (door slams, brief noises) that would
otherwise cause false positive detections.
"""

import time
import logging
import threading
from collections import deque
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class TemporalPrediction:
    """Single prediction with timestamp for temporal tracking."""
    confidence: float
    timestamp: float
    is_positive: bool


@dataclass
class SmoothedDetectionResult:
    """Result from temporal smoothing analysis."""
    should_alert: bool
    smoothed_confidence: float
    raw_confidence: float
    consecutive_high_count: int
    window_size: int
    predictions_in_window: int
    confidence_history: List[float] = field(default_factory=list)


class TemporalSmoothedDetector:
    """
    Temporal smoothing for baby cry detection to reduce false positives.

    This class maintains a sliding window of recent predictions and requires
    multiple consecutive high-confidence predictions before triggering an alert.
    This helps filter out transient sounds (door slams, brief noises) that might
    cause false positive detections.

    The smoothing works by:
    1. Maintaining a deque of recent predictions (configurable window size)
    2. Counting consecutive high-confidence predictions
    3. Computing smoothed probability as average of recent predictions
    4. Only alerting when min_consecutive high-confidence predictions are met

    Usage:
        smoother = TemporalSmoothedDetector(
            window_size=5,
            min_consecutive=3,
            confidence_threshold=0.5
        )

        for audio_chunk in audio_stream:
            is_cry, confidence = detector.detect_cry(audio_chunk)
            result = smoother.update(confidence)
            if result.should_alert:
                # Trigger actual alert
                pass
    """

    def __init__(
        self,
        window_size: int = 5,
        min_consecutive: int = 3,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize temporal smoothing detector.

        Args:
            window_size: Number of recent predictions to keep in sliding window.
                         Default 5 means we track the last 5 predictions.
            min_consecutive: Minimum number of consecutive high-confidence
                             predictions required before alerting. Default 3.
            confidence_threshold: Threshold above which a prediction is
                                  considered high-confidence. Default 0.5.

        Raises:
            ValueError: If min_consecutive > window_size or invalid parameters.
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if min_consecutive < 1:
            raise ValueError("min_consecutive must be at least 1")
        if min_consecutive > window_size:
            raise ValueError(
                f"min_consecutive ({min_consecutive}) cannot exceed "
                f"window_size ({window_size})"
            )
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self.window_size = window_size
        self.min_consecutive = min_consecutive
        self.confidence_threshold = confidence_threshold

        # Sliding window of predictions using deque for O(1) append/pop
        self._predictions: deque[TemporalPrediction] = deque(maxlen=window_size)

        # Lock for thread-safe access
        self._lock = threading.Lock()

        logging.info(
            f"TemporalSmoothedDetector initialized: "
            f"window_size={window_size}, min_consecutive={min_consecutive}, "
            f"threshold={confidence_threshold}"
        )

    def update(self, confidence: float, timestamp: Optional[float] = None) -> SmoothedDetectionResult:
        """
        Update with a new prediction and return smoothed detection result.

        This is the main method to call after each raw prediction from the
        baby cry detection model.

        Args:
            confidence: Raw confidence score from the model (0.0 to 1.0).
            timestamp: Optional timestamp for the prediction. If None,
                       current time is used.

        Returns:
            SmoothedDetectionResult containing:
            - should_alert: True if temporal criteria are met for alerting
            - smoothed_confidence: Average confidence over the window
            - raw_confidence: The confidence value passed to this call
            - consecutive_high_count: Current streak of high-confidence predictions
            - window_size: Configured window size
            - predictions_in_window: Number of predictions currently in window
            - confidence_history: List of recent confidence values
        """
        if timestamp is None:
            timestamp = time.time()

        is_positive = confidence >= self.confidence_threshold

        prediction = TemporalPrediction(
            confidence=confidence,
            timestamp=timestamp,
            is_positive=is_positive
        )

        with self._lock:
            self._predictions.append(prediction)

            # Calculate consecutive high-confidence count (from most recent)
            consecutive_count = self._count_consecutive_high()

            # Calculate smoothed confidence
            smoothed_confidence = self._calculate_smoothed_confidence()

            # Determine if we should alert
            should_alert = consecutive_count >= self.min_consecutive

            # Get confidence history for debugging
            history = [p.confidence for p in self._predictions]

        result = SmoothedDetectionResult(
            should_alert=should_alert,
            smoothed_confidence=smoothed_confidence,
            raw_confidence=confidence,
            consecutive_high_count=consecutive_count,
            window_size=self.window_size,
            predictions_in_window=len(self._predictions),
            confidence_history=history
        )

        if should_alert:
            logging.debug(
                f"Temporal smoothing: ALERT triggered - "
                f"{consecutive_count} consecutive high predictions "
                f"(smoothed: {smoothed_confidence:.2%})"
            )

        return result

    def _count_consecutive_high(self) -> int:
        """
        Count consecutive high-confidence predictions from most recent.

        Returns:
            Number of consecutive predictions at or above threshold,
            counting backward from the most recent.
        """
        count = 0
        # Iterate from most recent to oldest
        for pred in reversed(self._predictions):
            if pred.is_positive:
                count += 1
            else:
                break
        return count

    def _calculate_smoothed_confidence(self) -> float:
        """
        Calculate smoothed confidence as average of predictions in window.

        Returns:
            Average confidence value, or 0.0 if no predictions yet.
        """
        if not self._predictions:
            return 0.0
        return sum(p.confidence for p in self._predictions) / len(self._predictions)

    def get_consecutive_count(self) -> int:
        """
        Get current consecutive high-confidence prediction count.

        Thread-safe accessor for the current streak count.

        Returns:
            Number of consecutive high-confidence predictions.
        """
        with self._lock:
            return self._count_consecutive_high()

    def get_smoothed_confidence(self) -> float:
        """
        Get current smoothed confidence value.

        Thread-safe accessor for the smoothed confidence.

        Returns:
            Average confidence over current window.
        """
        with self._lock:
            return self._calculate_smoothed_confidence()

    def get_prediction_count(self) -> int:
        """
        Get number of predictions currently in the window.

        Returns:
            Number of predictions stored (0 to window_size).
        """
        with self._lock:
            return len(self._predictions)

    def is_window_filled(self) -> bool:
        """
        Check if the prediction window is fully populated.

        Returns:
            True if window has reached window_size predictions.
        """
        with self._lock:
            return len(self._predictions) >= self.window_size

    def reset(self) -> None:
        """
        Clear all predictions and reset state.

        Call this when starting a new detection session or after
        the robot returns to low-power mode.
        """
        with self._lock:
            self._predictions.clear()
        logging.debug("TemporalSmoothedDetector reset")

    def get_state_summary(self) -> Dict:
        """
        Get a summary of current detector state for debugging.

        Returns:
            Dictionary with current state information.
        """
        with self._lock:
            return {
                'window_size': self.window_size,
                'min_consecutive': self.min_consecutive,
                'confidence_threshold': self.confidence_threshold,
                'predictions_count': len(self._predictions),
                'consecutive_high': self._count_consecutive_high(),
                'smoothed_confidence': self._calculate_smoothed_confidence(),
                'is_window_filled': len(self._predictions) >= self.window_size,
                'recent_confidences': [p.confidence for p in self._predictions]
            }
