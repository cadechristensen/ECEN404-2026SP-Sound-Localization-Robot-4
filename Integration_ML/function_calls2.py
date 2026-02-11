"""
Baby cry detection helpers for wrap.py.

Wraps the RealtimeBabyCryDetector from the local deployment folder so that
wrap.py can call simple functions without managing paths, config, or resampling.
"""

import logging
import numpy as np
import torch
import torchaudio
import soundfile as sf
import function_calls
from Sound_Characterization.realtime_baby_cry_detector import RealtimeBabyCryDetector
from Sound_Characterization.config_pi import ConfigPi

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
MODEL_PATH = "Sound_Characterization/calibrated_model.pth"  # Path to the trained baby cry detection model
MIC_SAMPLE_RATE = 48000                                     # TI PCM6260-Q1 mic array rate

_config = ConfigPi()
_detector = None


def _get_detector():
    """Lazy-init singleton detector (avoids reloading the model every call)."""
    global _detector
    if _detector is None:
        logging.info(f"Loading baby cry model from {MODEL_PATH}...")
        _detector = RealtimeBabyCryDetector(
            model_path=MODEL_PATH,
            config=_config,
            detection_threshold=_config.CONFIDENCE_THRESHOLD,
            confirmation_threshold=0.85,
            use_tta=False,
            num_channels=_config.PI_CHANNELS,
            enable_multichannel=True,
            enable_temporal_smoothing=False,
        )
    return _detector


def record_and_resample():
    """
    Record audio from the mic array and return both the 48 kHz original
    and the 16 kHz version needed by the cry detection model.

    Returns:
        (audio_48k, audio_16k) as numpy arrays with shape (samples, channels),
        or (None, None) if recording failed.
    """
    raw_file = function_calls.record_audio(
        filename="live_input.wav",
        duration=_config.DURATION,
        sample_rate=MIC_SAMPLE_RATE,
        channels=_config.PI_CHANNELS,
    )
    if raw_file is None:
        return None, None

    audio_48k, _ = sf.read(raw_file)
    audio_48k = audio_48k.astype(np.float32)

    # Resample 48 kHz → 16 kHz using torchaudio (proper sinc interpolation)
    tensor_48k = torch.from_numpy(audio_48k.T)          # (channels, samples)
    tensor_16k = torchaudio.functional.resample(
        tensor_48k, orig_freq=MIC_SAMPLE_RATE, new_freq=_config.SAMPLE_RATE
    )
    audio_16k = tensor_16k.T.numpy()                     # (samples, channels)

    return audio_48k, audio_16k


def detect_cry(audio_16k):
    """
    Quick, low-power cry screening (no TTA).

    Returns:
        (is_cry: bool, confidence: float)
    """
    detector = _get_detector()
    return detector.detect_cry(audio_16k, use_tta=False)


def confirm_and_filter(audio_16k):
    """
    Confirm a potential cry with TTA and isolate cry segments.

    Returns:
        DetectionResult with .is_cry, .confidence, .filtered_audio
    """
    detector = _get_detector()
    return detector.confirm_and_filter(audio_16k)


def save_filtered_audio(filtered_audio, output_path="filtered_cry.wav"):
    """
    Resample filtered 16 kHz audio back to 48 kHz and save as WAV
    for DOAnet localization.

    Args:
        filtered_audio: numpy array from DetectionResult.filtered_audio
        output_path: destination WAV path

    Returns:
        output_path on success, None on failure.
    """
    if filtered_audio is None:
        return None

    tensor = torch.from_numpy(filtered_audio)
    if tensor.ndim == 2:
        tensor = tensor.T                                # (channels, samples)
    else:
        tensor = tensor.unsqueeze(0)

    tensor_48k = torchaudio.functional.resample(
        tensor, orig_freq=_config.SAMPLE_RATE, new_freq=MIC_SAMPLE_RATE
    )
    sf.write(output_path, tensor_48k.T.numpy(), MIC_SAMPLE_RATE)
    return output_path
