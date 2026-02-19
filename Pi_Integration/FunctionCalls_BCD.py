# Function Call for Baby Cry Detection model

import os
import sys
import logging
import threading
import numpy as np
from typing import Optional, Callable

# Add baby_cry_detection deployment directory to path
_PI_INTEGRATION_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PI_INTEGRATION_DIR)
_BCD_DEPLOY_DIR = os.path.join(_PROJECT_ROOT, 'baby_cry_detection', 'deployment', 'raspberry_pi')

if _BCD_DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _BCD_DEPLOY_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# baby_cry_detection root needed for src.* imports inside the detector
_BCD_ROOT = os.path.join(_PROJECT_ROOT, 'baby_cry_detection')
if _BCD_ROOT not in sys.path:
    sys.path.insert(0, _BCD_ROOT)

from realtime_baby_cry_detector import RealtimeBabyCryDetector
from detection_types import DetectionResult

logger = logging.getLogger(__name__)

# Hardcoded model path (relative to project root)
DEFAULT_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, 'baby_cry_detection', 'deployment', 'calibrated_model.pth'
)


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

    # Extract contiguous cry regions
    regions = []
    in_cry = False
    start_idx = 0
    for i in range(len(cry_mask)):
        if cry_mask[i] and not in_cry:
            start_idx = i
            in_cry = True
        elif not cry_mask[i] and in_cry:
            regions.append((start_idx / sr, i / sr))
            in_cry = False
    if in_cry:
        regions.append((start_idx / sr, len(cry_mask) / sr))

    return regions


class BabyCryDetection:
    """
    Wrapper around RealtimeBabyCryDetector exposing a clean interface
    for the Pi_Integration orchestrator.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device_index: Optional[int] = None,
        num_channels: int = 4,
        detection_threshold: float = 0.5,
        confirmation_threshold: float = 0.85,
        device: str = 'cpu',
        verbose: bool = True,
    ):
        self.model_path = model_path
        self.device_index = device_index
        self.num_channels = num_channels
        self._user_callback: Optional[Callable] = None
        self._running = False

        self._detector = RealtimeBabyCryDetector(
            model_path=model_path,
            detection_threshold=detection_threshold,
            confirmation_threshold=confirmation_threshold,
            device=device,
            audio_device_index=device_index,
            num_channels=num_channels,
            verbose=verbose,
        )
        logger.info("BabyCryDetection initialized")

    def start(self, on_cry_callback: Callable[[DetectionResult], None]) -> None:
        """
        Start real-time cry detection.

        Args:
            on_cry_callback: Called with a DetectionResult when a cry is confirmed.
        """
        self._user_callback = on_cry_callback
        self._detector.on_cry_detected = self._on_detection
        self._detector.start(stream_audio=True)
        self._running = True
        logger.info("BabyCryDetection started")

    def _on_detection(self, detection: DetectionResult) -> None:
        """Internal callback bridging the detector to the orchestrator."""
        logger.info(
            f"Cry forwarded to orchestrator (confidence={detection.confidence:.2%})"
        )
        if self._user_callback:
            self._user_callback(detection)

    def stop(self) -> None:
        """Stop detection and release audio resources."""
        if self._running:
            self._detector.stop()
            self._running = False
            logger.info("BabyCryDetection stopped")

    def reset(self) -> None:
        """Return detector to low-power listening mode."""
        self._detector.reset_to_low_power()
        logger.info("BabyCryDetection reset to low-power mode")

    def detect_from_audio(self, audio_data) -> DetectionResult:
        """
        Run dual-channel voting cry detection on a full audio array (no mic needed).

        Uses the MultichannelDetector's channel selection and voting strategy,
        with BabyCryAudioFilter.isolate_baby_cry() to scan the entire file in
        overlapping segments.

        Args:
            audio_data: NumPy array, shape (num_samples, num_channels) or (num_samples,).

        Returns:
            DetectionResult with is_cry, confidence, filtered_audio, etc.
        """
        import time
        import torch

        # Normalize audio to [-1, 1] to prevent clipping issues
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            logger.info(f"Normalizing audio (peak={max_val:.3f} -> 1.0)")
            audio_data = audio_data / max_val

        audio_filter = self._detector.audio_filter
        sr = self._detector.config.SAMPLE_RATE
        threshold = self._detector.detection_threshold
        mc_detector = self._detector.multichannel_detector
        num_channels = audio_data.shape[1] if audio_data.ndim > 1 else 1

        if audio_data.ndim > 1 and num_channels >= 2 and mc_detector is not None:
            # --- Dual-channel voting via MultichannelDetector ---
            selected, snr_scores = mc_detector.select_best_channels(audio_data, n_channels=2)
            voting_strategy = mc_detector.voting_strategy

            logger.info(
                f"Channel SNRs: {', '.join(f'Ch{i}: {snr_scores[i]:.1f}dB' for i in range(num_channels))}"
            )
            logger.info(f"Selected channels: primary={selected[0]}, secondary={selected[1]}")

            channel_confidences = []
            channel_all_segments = []

            for ch in selected:
                ch_audio = torch.from_numpy(audio_data[:, ch]).float()
                _, cry_segs, all_segs = audio_filter.isolate_baby_cry(
                    ch_audio, cry_threshold=threshold, use_acoustic_features=False,
                )
                conf = max((prob for _, _, prob in all_segs), default=0.0)
                channel_confidences.append(conf)
                channel_all_segments.append(all_segs)
                logger.info(
                    f"  Ch{ch}: {len(all_segs)} segments ({len(cry_segs)} cry), max conf: {conf:.4f}"
                )

            # Apply configured voting strategy
            if voting_strategy == "weighted":
                weights = np.exp(np.array([snr_scores[ch] for ch in selected]) / 10.0)
                weights = weights / weights.sum()
                final_confidence = float(np.sum(np.array(channel_confidences) * weights))
                is_cry = final_confidence >= threshold
                best_ch_idx = int(np.argmax(channel_confidences))
                logger.info(
                    f"Voting result ({voting_strategy}): confidence={final_confidence:.2%} "
                    f"(weights: {', '.join(f'Ch{selected[i]}={weights[i]:.2f}' for i in range(len(selected)))}), "
                    f"is_cry={is_cry}"
                )
            else:
                # logical_or: cry if EITHER channel detects
                final_confidence = float(max(channel_confidences))
                best_ch_idx = int(np.argmax(channel_confidences))
                is_cry = final_confidence >= threshold
                logger.info(
                    f"Voting result ({voting_strategy}): confidence={final_confidence:.2%} "
                    f"(best: Ch{selected[best_ch_idx]}), is_cry={is_cry}"
                )

            # Extract cry regions from the best-performing channel
            filtered_audio = None
            if is_cry and channel_all_segments[best_ch_idx]:
                total_samples = audio_data.shape[0]
                cry_regions = _score_and_extract_cry_regions(
                    channel_all_segments[best_ch_idx], sr, threshold, total_samples
                )
                logger.info(f"Cry regions: {[(f'{s:.2f}s', f'{e:.2f}s') for s, e in cry_regions]}")
                chunks = []
                for start, end in cry_regions:
                    s_idx = int(start * sr)
                    e_idx = min(int(end * sr), total_samples)
                    chunks.append(audio_data[s_idx:e_idx])
                if chunks:
                    filtered_audio = np.concatenate(chunks, axis=0)
                    logger.info(
                        f"Filtered audio: {filtered_audio.shape[0] / sr:.2f}s, "
                        f"{filtered_audio.shape[1]}-ch"
                    )
        else:
            # --- Single-channel fallback ---
            ch_audio = torch.from_numpy(
                audio_data[:, 0] if audio_data.ndim > 1 else audio_data
            ).float()
            _, cry_segs, all_segs = audio_filter.isolate_baby_cry(
                ch_audio, cry_threshold=threshold, use_acoustic_features=False,
            )
            final_confidence = max((prob for _, _, prob in all_segs), default=0.0)
            is_cry = final_confidence >= threshold

            filtered_audio = None
            if is_cry and all_segs:
                total_samples = audio_data.shape[0]
                cry_regions = _score_and_extract_cry_regions(
                    all_segs, sr, threshold, total_samples
                )
                chunks = []
                for start, end in cry_regions:
                    s_idx = int(start * sr)
                    e_idx = min(int(end * sr), total_samples)
                    chunks.append(audio_data[s_idx:e_idx])
                if chunks:
                    filtered_audio = np.concatenate(chunks, axis=0)

        return DetectionResult(
            is_cry=is_cry,
            confidence=final_confidence,
            timestamp=time.time(),
            audio_buffer=audio_data,
            filtered_audio=filtered_audio,
        )

    @property
    def detection_queue(self):
        """Expose the detector's multiprocessing detection queue."""
        return self._detector.detection_queue
