# Function Call for Baby Cry Detection model

import contextlib
import os
import queue  # for detection_queue return-type annotation
import sys
import time
import logging
import threading
import numpy as np
import torch
from typing import Optional, Callable, Tuple

# Add baby_cry_detection deployment directory to path
_PI_INTEGRATION_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PI_INTEGRATION_DIR)
_BCD_DEPLOY_DIR = os.path.join(_PROJECT_ROOT, "baby_cry_detection", "deployment")

if _BCD_DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _BCD_DEPLOY_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# baby_cry_detection root needed for src.* imports inside the detector
_BCD_ROOT = os.path.join(_PROJECT_ROOT, "baby_cry_detection")
if _BCD_ROOT not in sys.path:
    sys.path.insert(0, _BCD_ROOT)

from detector import BabyCryDetector, DetectionResult
from audio_pipeline import AudioPipeline
from audio_filtering import NoiseFilter

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _silence_stderr_fd():
    """Temporarily redirect stderr (fd 2) to /dev/null.

    Hides libjack/libasound C-level stderr noise (e.g. "Cannot connect to
    server socket") during PyAudio/sounddevice stream initialization. These
    messages bypass Python logging so only fd-level redirection catches them.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def _score_and_extract_cry_regions(all_segments, sr, threshold, total_samples):
    """
    Compute per-sample cry score from overlapping window predictions using
    max pooling, then threshold to find contiguous cry regions.

    Max pooling means each sample gets the highest score from any window that
    covers it. This preserves more of the cry signal for sound localization
    compared to averaging, which can dilute high-confidence windows.

    Args:
        all_segments: List of (start_time, end_time, score) from isolate_baby_cry.
        sr: Sample rate.
        threshold: Score threshold for cry classification.
        total_samples: Total number of audio samples.

    Returns:
        List of (start_time, end_time) cry regions (non-overlapping).
    """
    if not all_segments:
        return []

    # Max-pool scores per sample
    max_score = np.zeros(total_samples, dtype=np.float32)

    for start, end, score in all_segments:
        s_idx = int(start * sr)
        e_idx = min(int(end * sr), total_samples)
        np.maximum(max_score[s_idx:e_idx], score, out=max_score[s_idx:e_idx])

    # Threshold to binary cry mask
    cry_mask = max_score >= threshold

    # Vectorized edge detection — ~100x faster than a pure-Python sample-level
    # loop (16 kHz * 30 s = 480 000 iterations saved on Pi CPU).
    mask_int = cry_mask.view(np.int8)
    edges = np.diff(mask_int, prepend=np.int8(0), append=np.int8(0))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(s) / sr, int(e) / sr) for s, e in zip(starts, ends)]


class BabyCryDetection:
    """
    Wrapper around BabyCryDetector + AudioPipeline exposing a clean interface
    for the Pi_Integration orchestrator.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device_index: Optional[int] = None,
        num_channels: int = 4,
        detection_threshold: Optional[float] = None,
        confirmation_threshold: Optional[float] = None,
        device: str = "cpu",
        verbose: bool = False,  # default False — verbose=True prints 14 lines/call on Pi
        enable_multichannel: bool = True,
        multichannel_voting: Optional[str] = None,
        enable_temporal_smoothing: bool = True,
        use_tta: bool = False,
    ):
        # Import ConfigPi here to avoid module-level import issues
        from config_pi import ConfigPi

        config = ConfigPi()
        detection_threshold = (
            detection_threshold
            if detection_threshold is not None
            else config.PI_ORCHESTRATOR_LISTEN_THRESHOLD
        )
        confirmation_threshold = (
            confirmation_threshold
            if confirmation_threshold is not None
            else config.CONFIRMATION_THRESHOLD
        )

        self.device_index = device_index
        self.num_channels = num_channels
        self._lock = threading.Lock()  # guards _user_callback across threads
        self._user_callback: Optional[Callable] = None
        self._running = False

        self._detector = BabyCryDetector(
            model_path=model_path,
            config=config,
            detection_threshold=detection_threshold,
            confirmation_threshold=confirmation_threshold,
            device=device,
            audio_device_index=device_index,
            num_channels=num_channels,
            verbose=verbose,
            enable_multichannel=enable_multichannel,
            multichannel_voting=multichannel_voting,
            enable_temporal_smoothing=enable_temporal_smoothing,
            use_tta=use_tta,
        )
        self._pipeline: Optional[AudioPipeline] = None
        self._sl_filter: Optional[NoiseFilter] = None
        logger.info("BabyCryDetection initialized")

    @property
    def sample_rate(self) -> int:
        """Expose the detector's configured sample rate."""
        return self._detector.config.SAMPLE_RATE

    @property
    def capture_sample_rate(self) -> int:
        """Expose the capture sample rate (48 kHz for DOAnet phase accuracy)."""
        return self._detector.capture_sample_rate

    def start(self, on_cry_callback: Callable[[DetectionResult], None]) -> None:
        """
        Start real-time cry detection via AudioPipeline.

        Args:
            on_cry_callback: Called with a DetectionResult when a cry is confirmed.
        """
        if self._running:
            raise RuntimeError("BabyCryDetection is already running")
        # Verify attribute exists before monkey-patching — protects against renames
        if not hasattr(self._detector, "on_cry_detected"):
            raise AttributeError(
                "BabyCryDetector has no 'on_cry_detected' attribute — "
                "was it renamed? Check detector.py."
            )
        with self._lock:  # protect callback write
            self._user_callback = on_cry_callback
        self._detector.on_cry_detected = self._on_detection
        # Set _running=True BEFORE spawning the thread so that stop() called
        # concurrently during startup does not no-op and leak the processing thread.
        self._running = True
        try:
            self._pipeline = AudioPipeline(self._detector)
            # Wrap pipeline start in stderr suppression to hide libjack's
            # "Cannot connect to server socket" noise when PyAudio opens
            # the input stream.  Logging is unaffected (it uses stdout or
            # Python-level handlers, not fd 2 directly).
            with _silence_stderr_fd():
                self._pipeline.start(stream_audio=True)
        except Exception:
            # On failure the internal processing thread may already be running;
            # call stop() to join it so we don't leave a zombie thread.
            self._running = False
            self._detector.on_cry_detected = None  # undo monkey-patch
            try:
                if self._pipeline is not None:
                    self._pipeline.stop()
            except Exception as e:
                logger.warning(f"Error stopping pipeline during cleanup: {e}")
            self._pipeline = None
            raise
        logger.info("BabyCryDetection started")

    def _on_detection(self, detection: DetectionResult) -> None:
        """Internal callback bridging the detector to the orchestrator."""
        # DEBUG (not INFO) because main.py's _on_cry_detected already logs
        # "Cry confirmed (confidence=X%)" immediately after this — no need
        # to print the same number twice under -q.
        logger.debug(
            f"Cry forwarded to orchestrator (confidence={detection.confidence:.2%})"
        )
        with self._lock:  # safe read of callback from detector thread
            cb = self._user_callback
        if cb:
            cb(detection)

    def stop(self) -> None:
        """Stop detection and release audio resources."""
        if self._running:
            if self._pipeline is not None:
                self._pipeline.stop()
                self._pipeline = None
            self._running = False
            with self._lock:
                self._user_callback = None  # clear stale callback after stop
            logger.info("BabyCryDetection stopped")

    def reset(self) -> None:
        """Return detector to low-power listening mode."""
        self._detector.reset_to_low_power()
        # DEBUG (not INFO) because reset() is called on every _return_to_listening
        # and every _enter_relisten — it would spam the log under -q.
        logger.debug("BabyCryDetection reset to low-power mode")

    def detect_from_audio(self, audio_data: np.ndarray) -> DetectionResult:
        """
        Run cry detection on a full audio array (no mic needed).

        Uses isolate_baby_cry_multichannel() for phase-preserving multi-channel
        filtering, or isolate_baby_cry() for mono input.

        Args:
            audio_data: Float32/Float64 NumPy array at 16kHz, shape (num_samples, num_channels)
                        or (num_samples,). MUST be resampled to 16kHz before calling.
                        Values should be in [-1, 1]; if the peak exceeds 1.0 the array
                        is normalized before detection. audio_buffer in the returned
                        DetectionResult holds the original (pre-normalization) array.

        Returns:
            DetectionResult with is_cry, confidence, filtered_audio, etc.
        """
        if not isinstance(audio_data, np.ndarray):
            raise TypeError(
                f"audio_data must be np.ndarray, got {type(audio_data).__name__}"
            )
        if audio_data.size == 0:
            raise ValueError("audio_data is empty")
        if not np.isfinite(audio_data).all():
            raise ValueError("audio_data contains NaN or Inf values")
        if audio_data.dtype not in (np.float32, np.float64):
            audio_data = audio_data.astype(np.float32)

        # Keep the raw (original-scale) array for the DetectionResult so
        # downstream DOA/localization code receives unmodified amplitude values.
        raw_audio_data = audio_data

        # Normalize to [-1, 1] for detection only
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            logger.info(f"Normalizing audio (peak={max_val:.3f} -> 1.0)")
            audio_data = audio_data / max_val

        audio_filter = self._detector.audio_filter
        sr = self._detector.config.SAMPLE_RATE
        threshold = self._detector.detection_threshold
        num_channels = audio_data.shape[1] if audio_data.ndim > 1 else 1

        audio_tensor = torch.from_numpy(audio_data).float()

        if audio_data.ndim > 1 and num_channels >= 2:
            # --- Multi-channel: phase-preserving filtering ---
            isolated_audio, cry_segs, all_segs = (
                audio_filter.isolate_baby_cry_multichannel(
                    audio_tensor,
                    cry_threshold=threshold,
                    use_acoustic_features=False,
                )
            )
        else:
            # --- Single-channel fallback ---
            mono = audio_tensor[:, 0] if audio_tensor.ndim > 1 else audio_tensor
            isolated_audio, cry_segs, all_segs = audio_filter.isolate_baby_cry(
                mono,
                cry_threshold=threshold,
                use_acoustic_features=False,
            )

        final_confidence = max((prob for _, _, prob in all_segs), default=0.0)
        is_cry = final_confidence >= threshold

        logger.info(
            f"Detection: {len(all_segs)} segments ({len(cry_segs)} cry), "
            f"confidence={final_confidence:.2%}, is_cry={is_cry}"
        )

        # Extract cry regions from segment scores for filtered audio
        filtered_audio = None
        cry_regions = None
        if is_cry and all_segs:
            total_samples = audio_data.shape[0]
            cry_regions = _score_and_extract_cry_regions(
                all_segs, sr, threshold, total_samples
            )
            logger.info(
                f"Cry regions: {[(f'{s:.2f}s', f'{e:.2f}s') for s, e in cry_regions]}"
            )
            chunks = []
            for start, end in cry_regions:
                s_idx = int(start * sr)
                e_idx = min(int(end * sr), total_samples)
                chunks.append(audio_data[s_idx:e_idx])
            if chunks:
                filtered_audio = np.concatenate(chunks, axis=0)
                if filtered_audio.ndim > 1:
                    logger.info(
                        f"Filtered audio: {filtered_audio.shape[0] / sr:.2f}s, "
                        f"{filtered_audio.shape[1]}-ch"
                    )
                else:
                    logger.info(
                        f"Filtered audio: {filtered_audio.shape[0] / sr:.2f}s, mono"
                    )

        return DetectionResult(
            is_cry=is_cry,
            confidence=final_confidence,
            timestamp=time.time(),
            audio_buffer=raw_audio_data,
            filtered_audio=filtered_audio,
            cry_regions=cry_regions,
        )

    def detect_with_confirmation(
        self,
        audio_data_16k: np.ndarray,
        audio_data_48k: np.ndarray,
    ) -> DetectionResult:
        """
        Run cry detection matching the live pipeline: initial screening at
        detection_threshold (0.50), then TTA confirmation at
        confirmation_threshold (0.70) with acoustic validation.

        Use this instead of detect_from_audio() when generating SL training
        data or running test mode, so the audio that reaches SoundLocalization
        matches what the live pipeline produces.

        Args:
            audio_data_16k: 16kHz resampled audio for BCD inference.
                Shape (num_samples,) or (num_samples, num_channels).
            audio_data_48k: Original 48kHz audio for phase-preserved
                audio_buffer and cry region extraction.
                Shape (num_samples,) or (num_samples, num_channels).

        Returns:
            DetectionResult with TTA-confirmed confidence, cry_regions from
            the confirmation path, and audio_buffer set to the 48kHz original.
        """
        # Stage 1: Quick screening at detection_threshold (0.50)
        initial = self.detect_from_audio(audio_data_16k)

        if not initial.is_cry:
            logger.info(
                f"Initial screening: no cry (confidence={initial.confidence:.2%})"
            )
            # Return with audio_buffer pointing to 48kHz original
            return DetectionResult(
                is_cry=False,
                confidence=initial.confidence,
                timestamp=initial.timestamp,
                audio_buffer=audio_data_48k,
                filtered_audio=None,
                cry_regions=None,
            )

        # Stage 2: TTA confirmation at confirmation_threshold (0.70)
        # confirm_and_filter expects the 48kHz audio (it resamples to 16kHz
        # internally and returns audio_buffer = 48kHz original)
        logger.info(
            f"Initial screening passed (confidence={initial.confidence:.2%}), "
            f"running TTA confirmation..."
        )
        confirmed = self._detector.confirm_and_filter(audio_data_48k)

        if not confirmed.is_cry:
            logger.info(
                f"TTA confirmation rejected " f"(confidence={confirmed.confidence:.2%})"
            )
        else:
            logger.info(
                f"TTA confirmation passed "
                f"(confidence={confirmed.confidence:.2%}, "
                f"cry_regions={len(confirmed.cry_regions or [])})"
            )

        return confirmed

    # ------------------------------------------------------------------
    # Cry audio extraction for sound localization
    # ------------------------------------------------------------------
    @staticmethod
    def _slice_audio_by_regions(
        audio: np.ndarray,
        regions,
        sample_rate: int,
    ) -> Optional[np.ndarray]:
        """Concatenate audio slices defined by (start_sec, end_sec) regions.

        Args:
            audio: Source audio, shape (num_samples,) or (num_samples, num_channels).
            regions: Iterable of (start_sec, end_sec) tuples.
            sample_rate: Sample rate of `audio` in Hz.

        Returns:
            Concatenated audio covering only the requested regions, or None
            if `regions` is empty or produces no valid slices.
        """
        if not regions:
            return None
        total_samples = audio.shape[0]
        chunks = []
        for start, end in regions:
            s_idx = int(start * sample_rate)
            e_idx = min(int(end * sample_rate), total_samples)
            if e_idx > s_idx:
                chunks.append(audio[s_idx:e_idx])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    def extract_cry_audio_from_detection(
        self,
        detection: DetectionResult,
        min_duration: float = 5.0,
    ) -> Tuple[Optional[np.ndarray], int, np.ndarray]:
        """Extract cry regions from an existing DetectionResult's audio buffer.

        Used for first-pass localization where the orchestrator already has
        a detection from the live audio buffer and just needs to pull the
        cry portions out for the direction model while keeping the full raw
        audio for the distance model.

        Args:
            detection: The DetectionResult captured when the cry was confirmed.
                Must have a non-None `audio_buffer`.
            min_duration: Minimum total cry duration (seconds) required for
                localization. If the extracted cry audio is shorter than this,
                cry_audio will be None (caller should treat as false positive).

        Returns:
            Tuple of (cry_audio, sample_rate, raw_audio):
              - cry_audio: Concatenated cry regions for the direction model, or
                None if no cry regions exist or total duration is below threshold.
              - sample_rate: The capture sample rate (48 kHz on this system).
              - raw_audio: The full untouched capture buffer for the distance model.

        Raises:
            RuntimeError: If `detection` is None or has no audio_buffer.
        """
        if detection is None or detection.audio_buffer is None:
            raise RuntimeError("No audio buffer available from detection")

        raw_audio = detection.audio_buffer
        sr = self.capture_sample_rate

        cry_audio = self._slice_audio_by_regions(raw_audio, detection.cry_regions, sr)
        if cry_audio is None:
            # No cry regions found — likely a false positive
            logger.warning("No cry regions in buffer — returning to LISTENING")
            return None, sr, raw_audio

        duration = cry_audio.shape[0] / sr
        if duration < min_duration:
            logger.warning(
                f"Cry regions too short ({duration:.2f}s < {min_duration}s) "
                "— returning to LISTENING"
            )
            return None, sr, raw_audio

        # DEBUG (not INFO) because main.py logs the final "Localization
        # (relisten N/M): X deg at Y ft" line with all the useful numbers.
        # The cry-duration info is only diagnostic — surface it with --debug.
        logger.debug(f"Using cry regions from buffer ({duration:.2f}s, {sr}Hz)")
        return cry_audio, sr, raw_audio

    def filter_for_localization(self, audio_48k: np.ndarray) -> np.ndarray:
        """
        Apply phase-preserving noise filtering to 48kHz audio for SoundLocalization.

        Applies highpass (100Hz) -> bandpass (100-3000Hz) -> spectral subtraction
        with identical coefficients per channel to preserve inter-channel phase.

        Args:
            audio_48k: 48kHz audio, shape (num_samples,) or (num_samples, num_channels).

        Returns:
            Filtered audio with same shape, phase preserved across channels.
        """
        if self._sl_filter is None:
            self._sl_filter = NoiseFilter(
                sample_rate=48000,
                highpass_cutoff=100,
                bandpass_low=100,
                bandpass_high=3000,
            )
        filtered = self._sl_filter.filter_audio(audio_48k)
        if isinstance(filtered, np.ndarray):
            return filtered
        return filtered.numpy()

    @property
    def detection_queue(self) -> "queue.Queue":
        """Expose the detector's detection queue."""
        return self._detector.detection_queue
