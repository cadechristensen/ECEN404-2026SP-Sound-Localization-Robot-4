"""
Raspberry Pi optimized configuration for baby cry detection.
Lightweight filtering settings for real-time performance.

Imports the canonical Config from src.config and overrides only Pi-specific values,
eliminating the maintenance risk of duplicated configuration.
"""

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.config can be imported.
# This mirrors the pattern used in realtime_baby_cry_detector.py.
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import Config


class ConfigPi(Config):
    """
    Raspberry Pi optimized configuration.

    Inherits all values from the canonical src.config.Config and overrides
    only what is necessary for Pi deployment.  This eliminates the duplicate
    Config class that previously drifted out of sync with src/config.py.

    Key differences from base Config:
    - Forced CPU device (Pi has no GPU)
    - Reduced filtering complexity for speed
    - Optimized for real-time, single-threaded performance
    - Lower resource usage
    """

    # Force CPU — Pi has no GPU
    DEVICE = 'cpu'

    # Model path — change this to switch deployed models
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_best_2026-03-14_run16_d384.pth")

    # Detection thresholds — centralized reference for deployment code.
    # Bimodal distribution: 94.3% of predictions are <0.1 or >0.9.
    # Raising from 0.85 to 0.92 eliminates FPs in [0.85-0.92] with negligible recall loss.
    DETECTION_THRESHOLD = 0.92

    # TTA confirmation uses a dual-channel AND gate (both channels must agree),
    # which already provides strong FP filtering. The per-channel threshold can
    # be lower than 0.92 because room acoustics, speaker playback, and SNR
    # weighting degrade per-channel confidence vs direct-file evaluation.
    #! SL & App gets confidence above 0.70
    CONFIRMATION_THRESHOLD = 0.70

    # Temporal smoothing parameters — centralized for deployment code.
    # The temporal smoother sits in the quick-detection (low-power) stage,
    # filtering transient false positives before expensive TTA confirmation.
    # Its threshold should match the quick-detection gate (0.5), NOT the
    # final confirmation threshold (0.92) — TTA handles that separately.
    TEMPORAL_WINDOW_SIZE = 5
    TEMPORAL_MIN_CONSECUTIVE = 2   # Requires ~1s sustained cry (2 × 0.5s chunks)
    TEMPORAL_CONFIDENCE_THRESHOLD = 0.50

    # Context and cooldown
    CONTEXT_DURATION = 10.0        # Seconds of audio context for cry region extraction
    DETECTION_COOLDOWN = 5.0      # Seconds before re-detection (post-cry babbling filter)

    # Pi_Integration orchestrator uses a lower "quick-detection" threshold
    # for the two-stage pathway (quick detect at 0.5 → TTA confirm at 0.70).
    PI_ORCHESTRATOR_LISTEN_THRESHOLD = 0.5

    # Spectral flatness gate — rejects broadband noise before neural network inference.
    # Baby cries are tonal (flatness ~0.01-0.15); broadband noise (crickets, water) > 0.5.
    # Note: 0.5s chunks from the mic array produce flatness 0.25-0.40 even for genuine
    # cries, so the threshold must be higher than the idealized single-speaker value.
    SPECTRAL_FLATNESS_THRESHOLD = 0.50

    # Advanced Filtering Configuration (Pi-Optimized)
    USE_ADVANCED_FILTERING = True  # Keep enabled — filters are fast

    # Voice Activity Detection (tighter range for Pi speed)
    VAD_ENERGY_THRESHOLD = 0.25  # Slightly higher than base 0.20 for Pi
    VAD_FREQ_MIN = 250           # Baby cry starts around 250 Hz (tighter range)
    VAD_FREQ_MAX = 800           # Reduced from 1000 for speed

    # Deep spectrum features (DISABLED for Pi — too slow)
    USE_DEEP_SPECTRUM = False
    EXTRACT_MFCC_DELTAS = False
    EXTRACT_SPECTRAL_CONTRAST = False
    EXTRACT_CHROMA = False

    # Model optimization
    QUANTIZE_MODEL = False  # FP32 — calibrated thresholds are tuned to FP32 probabilities
    OPTIMIZE_FOR_MOBILE = True

    #! Multi-channel voting strategy — change here to switch globally
    # "weighted"   : SNR-weighted confidence average across channels (recommended)
    # "logical_or" : any channel vote triggers a detection (more sensitive, higher FP rate)
    PI_MULTICHANNEL_VOTING = "weighted"

    # Pi-specific performance settings
    PI_USE_VAD_GATING = True   # Only run model when VAD detects activity
    PI_VAD_BUFFER_SIZE = 3     # Number of VAD frames to buffer before detection
    PI_BATCH_SIZE = 1          # Process one sample at a time for low latency
    PI_NUM_WORKERS = 0         # Single-threaded for Pi (avoid overhead)

    # Memory optimization — override base training values
    BATCH_SIZE = 1      # Process one at a time to reduce memory
    NUM_WORKERS = 0     # No parallel workers on Pi

    # Audio buffer settings for real-time
    PI_BUFFER_SIZE = 2048      # Larger buffer for stability (was 1024)
    PI_SAMPLE_RATE = 16000     # Standard rate
    PI_CHANNELS = 4            # 4-channel mic array (TI PCM6260-Q1)
    PI_AUDIO_CHUNK_DURATION = 1.0  # Process 1.0s chunks

    # Capture at 48 kHz to preserve phase for DOAnet localization.
    # Audio is resampled to SAMPLE_RATE (16 kHz) before BCD inference.
    CAPTURE_SAMPLE_RATE = 48000

    # Duration of fresh recording for RELISTEN re-localization (seconds)
    LOCALIZATION_RECORD_DURATION = 5.0

    # Microphone array geometry (parabolic dish configuration)
    # Channel-to-angle mapping: ch0=135°, ch1=315°, ch2=45°, ch3=225°
    MIC_CHANNEL_ANGLES = {0: 135, 1: 315, 2: 45, 3: 225}
    MIC_GAIN_DB = 31  # On-axis gain of parabolic dish
    MIC_NUM_CHANNELS = 4

    def __init__(self):
        """Initialize Pi configuration with CPU-only overrides."""
        super().__init__()

        # Instance-level override ensures DEVICE is always 'cpu'
        self.DEVICE = 'cpu'

        self._validate_pi_settings()

    def _validate_pi_settings(self):
        """Validate Pi-specific settings."""
        if self.USE_DEEP_SPECTRUM:
            raise ValueError("Deep spectrum must be disabled on Pi (USE_DEEP_SPECTRUM=True)")

        if self.NUM_WORKERS != 0:
            raise ValueError(
                f"Pi should use single-threaded processing (NUM_WORKERS={self.NUM_WORKERS}, expected 0)"
            )
