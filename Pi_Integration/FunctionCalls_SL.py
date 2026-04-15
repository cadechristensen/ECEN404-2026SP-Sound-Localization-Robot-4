# Function Calls for Sound Localization model

import os
import sys
import re
import tempfile
import logging
import numpy as np

_PI_INTEGRATION_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PI_INTEGRATION_DIR)
_SL_DIR = os.path.join(_PROJECT_ROOT, 'SoundLocalization')

if _SL_DIR not in sys.path:
    sys.path.insert(0, _SL_DIR)

from function_calls import Infer

logger = logging.getLogger(__name__)


class SoundLocalization:
    """
    Wrapper around DOAnet Infer class.

    Saves multichannel audio to a temporary WAV file, runs inference via
    Infer.process_file(), parses the result string, and cleans up.
    """

    # DOAnet expects 48 kHz input
    DOANET_SAMPLE_RATE = 48000

    def __init__(self, models_dir: str = None, task_id: str = '6', single_model: str = None):
        # Use absolute path to models directory to avoid thread-unsafe os.chdir()
        if models_dir is None:
            models_dir = os.path.join(_SL_DIR, 'models')

        # Infer needs SL_DIR on sys.path (already added above)
        self._infer = Infer(task_id=task_id, models_dir=models_dir, single_model=single_model)

        logger.info("SoundLocalization initialized")

    def localize(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_channels: int,
    ) -> dict:
        """
        Run DOA + distance inference on multichannel audio.

        Args:
            audio_data: NumPy array of shape (num_samples, num_channels) or (num_samples,).
            sample_rate: Sample rate of audio_data.
            num_channels: Number of channels in audio_data.

        Returns:
            dict with keys:
                direction_deg (float): Estimated direction in degrees (0-360).
                distance_ft (float): Estimated distance in feet.
                sources (str): Raw result string from DOAnet.
        """
        import soundfile as sf

        # Ensure shape is (num_channels, num_samples) for soundfile write
        if audio_data.ndim == 1:
            write_data = audio_data  # mono
        elif audio_data.shape[0] == num_channels and audio_data.shape[1] != num_channels:
            # Already (channels, samples) — transpose for soundfile (samples, channels)
            write_data = audio_data.T
        else:
            # (samples, channels) — soundfile expects this
            write_data = audio_data

        # Resample to 48 kHz if needed
        if sample_rate != self.DOANET_SAMPLE_RATE:
            import librosa
            # librosa.resample expects (channels, samples) for multi-channel
            if write_data.ndim == 2:
                resampled_channels = []
                for ch in range(write_data.shape[1]):
                    resampled_channels.append(
                        librosa.resample(
                            write_data[:, ch].astype(np.float32),
                            orig_sr=sample_rate,
                            target_sr=self.DOANET_SAMPLE_RATE,
                        )
                    )
                write_data = np.column_stack(resampled_channels)
            else:
                write_data = librosa.resample(
                    write_data.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=self.DOANET_SAMPLE_RATE,
                )

        # Write to temp WAV file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.wav')
        os.close(tmp_fd)

        try:
            sf.write(tmp_path, write_data, self.DOANET_SAMPLE_RATE)
            logger.debug(f"Temp audio written to {tmp_path}")

            # Run inference (uses absolute paths, no chdir needed)
            result_str = self._infer.process_file(tmp_path)
            # Note: main.py's _localize_and_navigate logs the full
            # "Localization (relisten N/M): ... — <sources>" line with
            # more context, so we don't duplicate it here.
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return self._parse_result(result_str)

    @staticmethod
    def _parse_result(result_str: str) -> dict:
        """
        Parse DOAnet result string.

        Expected format examples:
            "Source 0: 123.4° (Loudness: 0.85) | Distance: 5.2 ft"
            "Source 0: 123.4° (Loudness: 0.85) | Source 1: 300.0° (Loudness: 0.60) | Distance: 5.2 ft"
            "No active sources detected."
        """
        result = {
            'direction_deg': 0.0,
            'distance_ft': 0.0,
            'sources': result_str,
        }

        # Source 0 is already the best source (sorted by closest to
        # loudest mic in function_calls.py), so just pick the first match
        source_match = re.search(
            r'Source\s*0:\s*([\d.]+)°\s*\(Loudness:\s*([\d.]+)\)', result_str
        )
        if source_match:
            result['direction_deg'] = float(source_match.group(1))

        # Extract distance
        dist_match = re.search(r'Distance:\s*([\d.]+)\s*ft', result_str)
        if dist_match:
            result['distance_ft'] = float(dist_match.group(1))

        return result
