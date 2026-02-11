"""
Shared data types for the baby cry detection deployment pipeline.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Container for detection results."""
    is_cry: bool
    confidence: float
    timestamp: float
    audio_buffer: np.ndarray
    filtered_audio: Optional[np.ndarray] = None
