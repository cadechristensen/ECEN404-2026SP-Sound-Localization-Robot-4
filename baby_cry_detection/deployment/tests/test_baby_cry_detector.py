#!/usr/bin/env python3
"""
Offline Baby Cry Detection Test Script for Prerecorded 4-Channel Audio.

Runs the full detection pipeline (identical to realtime_baby_cry_detector.py) on
a prerecorded WAV file instead of a live audio stream.  Exercises:

  1. Multi-channel audio loading (4-ch WAV from TI PCM6260-Q1 array)
  2. Sliding-window chunking with configurable overlap
  3. Per-chunk multichannel detection (SNR-based channel selection + dual-channel voting)
  4. Temporal smoothing (consecutive high-confidence gate)
  5. TTA confirmation on context window
  6. Phase-preserving audio filtering (bandpass + spectral subtraction)
  7. Summary report with per-chunk timeline

Usage:
    python test_baby_cry_detector.py --audio <multichannel.wav> --model <model.pth>
    python test_baby_cry_detector.py --audio ../../../Pi_Integration/7ft_set1.wav --model ../models/model_best_copy2.pth
    python baby_cry_detection/deployment/tests/test_baby_cry_detector.py --audio Pi_Integration/BABYCRY.wav --model baby_cry_detection/deployment/models/model_best_copy3.pth
"""

import os
import sys
import torch
import numpy as np
import time
import logging
import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field, asdict

# Insert project root so src.* imports resolve
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Insert parent deployment/ directory so local modules resolve
_DEPLOY_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)

# Local deployment modules
from config_pi import ConfigPi
from detector import BabyCryDetector, DetectionResult

# Backward-compat alias used in test docstrings / comments
RealtimeBabyCryDetector = BabyCryDetector


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """Result from processing a single audio chunk."""
    chunk_idx: int
    time_start: float
    time_end: float
    is_cry: bool
    confidence: float
    primary_channel: Optional[int] = None
    secondary_channel: Optional[int] = None
    channel_snr_scores: Optional[List[float]] = None
    channel_confidences: Optional[List[float]] = None
    smoothed_confidence: Optional[float] = None
    consecutive_high: int = 0
    temporal_alert: bool = False


@dataclass
class ConfirmationResult:
    """Result from a TTA confirmation pass."""
    chunk_idx: int
    time_start: float
    time_end: float
    context_start: float
    context_end: float
    is_cry: bool
    confidence: float
    passed_confirmation: bool
    filtered_audio_shape: Optional[Tuple[int, ...]] = None


@dataclass
class TestRunSummary:
    """Full test run summary."""
    audio_file: str
    model_path: str
    duration_seconds: float
    sample_rate: int
    num_channels: int
    chunk_duration: float
    overlap_ratio: float
    total_chunks: int
    detection_threshold: float
    confirmation_threshold: float
    temporal_smoothing: bool
    multichannel_enabled: bool
    voting_strategy: str
    chunk_results: List[ChunkResult] = field(default_factory=list)
    confirmations: List[ConfirmationResult] = field(default_factory=list)
    cry_regions: List[Tuple[float, float, float]] = field(default_factory=list)
    elapsed_time: float = 0.0


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_multichannel_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int, int]:
    """
    Load a multichannel WAV file and resample to target_sr if needed.

    Returns:
        (audio_data, sample_rate, num_channels)
        audio_data has shape (num_samples, num_channels)
    """
    import soundfile as sf

    audio, sr = sf.read(audio_path, always_2d=True)  # (samples, channels)
    num_channels = audio.shape[1]

    if sr != target_sr:
        # Resample each channel independently
        import librosa
        resampled_channels = []
        for ch in range(num_channels):
            resampled = librosa.resample(audio[:, ch], orig_sr=sr, target_sr=target_sr)
            resampled_channels.append(resampled)
        audio = np.stack(resampled_channels, axis=1)
        sr = target_sr

    return audio.astype(np.float32), sr, num_channels


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

class OfflineBabyCryTester:
    """
    Runs the full real-time detection pipeline on a prerecorded audio file.

    Mirrors RealtimeBabyCryDetector.process_audio_stream() but replaces the
    live PyAudio callback with deterministic sliding-window iteration over the
    file.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[ConfigPi] = None,
        detection_threshold: Optional[float] = None,
        confirmation_threshold: Optional[float] = None,
        device: str = 'cpu',
        num_channels: int = 4,
        enable_multichannel: bool = True,
        multichannel_voting: Optional[str] = None,
        enable_temporal_smoothing: bool = True,
        temporal_window_size: Optional[int] = None,
        temporal_min_consecutive: Optional[int] = None,
        temporal_confidence_threshold: Optional[float] = None,
        chunk_duration: float = 1.0,
        overlap_ratio: float = 0.5,
        context_duration: Optional[float] = None,
        verbose: bool = True,
    ):
        self.config = config or ConfigPi()
        self.detection_threshold = detection_threshold if detection_threshold is not None else self.config.DETECTION_THRESHOLD
        self.confirmation_threshold = confirmation_threshold if confirmation_threshold is not None else self.config.CONFIRMATION_THRESHOLD
        self.chunk_duration = chunk_duration
        self.overlap_ratio = overlap_ratio
        self.context_duration = context_duration if context_duration is not None else self.config.CONTEXT_DURATION
        self.verbose = verbose
        self.num_channels = num_channels

        voting = multichannel_voting or getattr(
            self.config, 'PI_MULTICHANNEL_VOTING', 'weighted'
        )
        self.voting_strategy = voting

        # Resolve temporal params: explicit arg > config value
        temporal_window_size = temporal_window_size if temporal_window_size is not None else self.config.TEMPORAL_WINDOW_SIZE
        temporal_min_consecutive = temporal_min_consecutive if temporal_min_consecutive is not None else self.config.TEMPORAL_MIN_CONSECUTIVE
        temporal_confidence_threshold = temporal_confidence_threshold if temporal_confidence_threshold is not None else self.config.TEMPORAL_CONFIDENCE_THRESHOLD

        # Build the real detector (reuses all model loading, mel-transform, etc.)
        # stream_audio=False so it never opens PyAudio.
        self.detector = BabyCryDetector(
            model_path=model_path,
            config=self.config,
            use_tta=False,
            detection_threshold=self.detection_threshold,
            confirmation_threshold=self.confirmation_threshold,
            device=device,
            audio_device_index=None,
            num_channels=num_channels,
            enable_multichannel=enable_multichannel,
            multichannel_voting=voting,
            enable_temporal_smoothing=enable_temporal_smoothing,
            temporal_window_size=temporal_window_size,
            temporal_min_consecutive=temporal_min_consecutive,
            temporal_confidence_threshold=temporal_confidence_threshold,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, audio_path: str) -> TestRunSummary:
        """Run detection pipeline over an entire prerecorded file."""
        t0 = time.perf_counter()

        # Load audio
        audio, sr, file_channels = load_multichannel_audio(
            audio_path, target_sr=self.config.SAMPLE_RATE
        )
        total_samples, actual_channels = audio.shape
        duration = total_samples / sr

        logging.info(f"Loaded {audio_path}: {duration:.2f}s, {actual_channels} channels, {sr} Hz")

        if actual_channels < self.num_channels:
            logging.warning(
                f"File has {actual_channels} channels but detector expects {self.num_channels}. "
                f"Padding missing channels with zeros."
            )
            pad = np.zeros((total_samples, self.num_channels - actual_channels), dtype=np.float32)
            audio = np.hstack([audio, pad])
        elif actual_channels > self.num_channels:
            logging.warning(
                f"File has {actual_channels} channels, using first {self.num_channels}."
            )
            audio = audio[:, :self.num_channels]

        # Prepare summary
        summary = TestRunSummary(
            audio_file=str(audio_path),
            model_path=str(self.detector.config.MODEL_PATH) if hasattr(self.detector.config, 'MODEL_PATH') else 'unknown',
            duration_seconds=duration,
            sample_rate=sr,
            num_channels=self.num_channels,
            chunk_duration=self.chunk_duration,
            overlap_ratio=self.overlap_ratio,
            total_chunks=0,
            detection_threshold=self.detection_threshold,
            confirmation_threshold=self.confirmation_threshold,
            temporal_smoothing=self.detector.enable_temporal_smoothing,
            multichannel_enabled=self.detector.multichannel_detector is not None,
            voting_strategy=self.voting_strategy,
        )

        # Sliding window iteration
        chunk_samples = int(self.chunk_duration * sr)
        hop_samples = max(1, int(chunk_samples * (1.0 - self.overlap_ratio)))
        chunk_idx = 0

        # Detection cooldown tracking (mirrors realtime detector)
        last_detection_time_s = -999.0
        detection_cooldown = 2.0

        starts = list(range(0, total_samples - chunk_samples + 1, hop_samples))
        summary.total_chunks = len(starts)

        logging.info(
            f"Processing {summary.total_chunks} chunks "
            f"(chunk={self.chunk_duration}s, overlap={self.overlap_ratio:.0%}, hop={hop_samples/sr:.2f}s)"
        )

        for start in starts:
            end = start + chunk_samples
            chunk = audio[start:end]  # (chunk_samples, num_channels)
            time_start = start / sr
            time_end = end / sr

            # Feed chunk into the circular buffer (same as realtime)
            self.detector.audio_buffer.add(chunk)

            # Cooldown check
            if time_start - last_detection_time_s < detection_cooldown:
                chunk_idx += 1
                continue

            # ----- Stage 1: Quick detection (no TTA) -----
            cr = self._detect_chunk(chunk, chunk_idx, time_start, time_end)
            summary.chunk_results.append(cr)

            # ----- Stage 2: Temporal smoothing -----
            should_confirm = False
            if self.detector.temporal_smoother is not None:
                smoothed = self.detector.temporal_smoother.update(cr.confidence, timestamp=time_start)
                cr.smoothed_confidence = smoothed.smoothed_confidence
                cr.consecutive_high = smoothed.consecutive_high_count
                cr.temporal_alert = smoothed.should_alert
                should_confirm = smoothed.should_alert
            else:
                should_confirm = cr.is_cry

            # ----- Stage 3: TTA confirmation + filtering -----
            if should_confirm:
                conf_result = self._confirm_and_filter(chunk_idx, time_start, time_end, sr)
                summary.confirmations.append(conf_result)

                if conf_result.passed_confirmation:
                    last_detection_time_s = time_start
                    # Reset temporal smoother after confirmed detection
                    if self.detector.temporal_smoother is not None:
                        self.detector.temporal_smoother.reset()

            chunk_idx += 1

        summary.elapsed_time = time.perf_counter() - t0
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_chunk(
        self, chunk: np.ndarray, idx: int, t_start: float, t_end: float
    ) -> ChunkResult:
        """Run quick detection on a single chunk (no TTA)."""
        mc = self.detector.multichannel_detector

        if mc is not None and chunk.ndim > 1:
            mc_result = mc.detect_cry_dual_channel(
                chunk, use_tta=False, confidence_threshold=self.detection_threshold
            )
            return ChunkResult(
                chunk_idx=idx,
                time_start=round(t_start, 3),
                time_end=round(t_end, 3),
                is_cry=mc_result.is_cry,
                confidence=round(mc_result.confidence, 4),
                primary_channel=mc_result.primary_channel,
                secondary_channel=mc_result.secondary_channel,
                channel_snr_scores=[round(s, 2) for s in mc_result.channel_snr_scores],
                channel_confidences=[round(c, 4) for c in mc_result.channel_confidences],
            )
        else:
            is_cry, conf, _ = self.detector.detect_cry(chunk, use_tta=False)
            return ChunkResult(
                chunk_idx=idx,
                time_start=round(t_start, 3),
                time_end=round(t_end, 3),
                is_cry=is_cry,
                confidence=round(conf, 4),
            )

    def _confirm_and_filter(
        self, idx: int, t_start: float, t_end: float, sr: int
    ) -> ConfirmationResult:
        """Run TTA confirmation + audio filtering on the context window."""
        context_audio = self.detector.audio_buffer.get_last_n_seconds(self.context_duration)
        context_samples = context_audio.shape[0]
        context_start = max(0.0, t_end - context_samples / sr)

        detection = self.detector.confirm_and_filter(context_audio)

        passed = detection.is_cry and detection.confidence >= self.confirmation_threshold

        return ConfirmationResult(
            chunk_idx=idx,
            time_start=round(t_start, 3),
            time_end=round(t_end, 3),
            context_start=round(context_start, 3),
            context_end=round(t_end, 3),
            is_cry=detection.is_cry,
            confidence=round(detection.confidence, 4),
            passed_confirmation=passed,
            filtered_audio_shape=detection.filtered_audio.shape if detection.filtered_audio is not None else None,
        )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_report(summary: TestRunSummary):
    """Print a human-readable report of the test run."""
    W = 74

    print(f"\n{'=' * W}")
    print("  OFFLINE BABY CRY DETECTION TEST REPORT")
    print(f"{'=' * W}")

    print(f"\n  Audio file:       {summary.audio_file}")
    print(f"  Duration:         {summary.duration_seconds:.2f}s")
    print(f"  Sample rate:      {summary.sample_rate} Hz")
    print(f"  Channels:         {summary.num_channels}")
    print(f"  Chunk duration:   {summary.chunk_duration}s (overlap {summary.overlap_ratio:.0%})")
    print(f"  Total chunks:     {summary.total_chunks}")

    print(f"\n  Detection thresh: {summary.detection_threshold:.0%}")
    print(f"  Confirm thresh:   {summary.confirmation_threshold:.0%}")
    print(f"  Multichannel:     {'ON (' + summary.voting_strategy + ')' if summary.multichannel_enabled else 'OFF'}")
    print(f"  Temporal smooth:  {'ON' if summary.temporal_smoothing else 'OFF'}")

    # ---- Per-chunk timeline ----
    print(f"\n{'─' * W}")
    print("  CHUNK-BY-CHUNK TIMELINE")
    print(f"{'─' * W}")
    print(f"  {'Chunk':>5}  {'Time':>11}  {'Cry?':>5}  {'Conf':>7}  {'Smooth':>7}  {'Consec':>6}  {'Alert':>5}  {'Ch':>4}")

    for cr in summary.chunk_results:
        time_str = f"{cr.time_start:5.1f}-{cr.time_end:5.1f}s"
        cry_str = "YES" if cr.is_cry else "no"
        conf_str = f"{cr.confidence:.2%}"
        smooth_str = f"{cr.smoothed_confidence:.2%}" if cr.smoothed_confidence is not None else "  -"
        consec_str = str(cr.consecutive_high) if cr.smoothed_confidence is not None else " -"
        alert_str = ">>>" if cr.temporal_alert else ""
        ch_str = f"{cr.primary_channel}/{cr.secondary_channel}" if cr.primary_channel is not None else " -"

        marker = " *" if cr.is_cry else "  "
        print(f" {marker}{cr.chunk_idx:>4}  {time_str:>11}  {cry_str:>5}  {conf_str:>7}  {smooth_str:>7}  {consec_str:>6}  {alert_str:>5}  {ch_str:>4}")

    # ---- Confirmations ----
    if summary.confirmations:
        print(f"\n{'─' * W}")
        print("  TTA CONFIRMATION RESULTS")
        print(f"{'─' * W}")
        print(f"  {'Chunk':>5}  {'Context':>15}  {'Cry?':>5}  {'Conf':>7}  {'Passed':>7}  {'Filtered Shape'}")

        for conf in summary.confirmations:
            ctx = f"{conf.context_start:5.1f}-{conf.context_end:5.1f}s"
            cry_str = "YES" if conf.is_cry else "no"
            conf_str = f"{conf.confidence:.2%}"
            passed_str = "PASS" if conf.passed_confirmation else "FAIL"
            shape_str = str(conf.filtered_audio_shape) if conf.filtered_audio_shape else "  -"
            print(f"  {conf.chunk_idx:>5}  {ctx:>15}  {cry_str:>5}  {conf_str:>7}  {passed_str:>7}  {shape_str}")
    else:
        print(f"\n  No chunks triggered TTA confirmation.")

    # ---- Summary stats ----
    print(f"\n{'─' * W}")
    print("  SUMMARY")
    print(f"{'─' * W}")

    total = len(summary.chunk_results)
    cry_chunks = sum(1 for r in summary.chunk_results if r.is_cry)
    confirm_passed = sum(1 for c in summary.confirmations if c.passed_confirmation)
    confirm_total = len(summary.confirmations)
    avg_conf = np.mean([r.confidence for r in summary.chunk_results]) if summary.chunk_results else 0
    max_conf = max((r.confidence for r in summary.chunk_results), default=0)

    # Find cry time ranges from chunk results
    cry_ranges = []
    for cr in summary.chunk_results:
        if cr.is_cry:
            if cry_ranges and cr.time_start <= cry_ranges[-1][1] + 0.01:
                cry_ranges[-1] = (cry_ranges[-1][0], cr.time_end)
            else:
                cry_ranges.append((cr.time_start, cr.time_end))

    print(f"  Chunks with cry:       {cry_chunks}/{total} ({cry_chunks/total:.0%})" if total else "")
    print(f"  Avg confidence:        {avg_conf:.2%}")
    print(f"  Max confidence:        {max_conf:.2%}")
    print(f"  Confirmations tried:   {confirm_total}")
    print(f"  Confirmations passed:  {confirm_passed}")

    if cry_ranges:
        print(f"\n  Detected cry regions (quick detection):")
        for i, (s, e) in enumerate(cry_ranges, 1):
            print(f"    {i}. {s:.1f}s - {e:.1f}s  ({e-s:.1f}s)")
    else:
        print(f"\n  No cry detected in any chunk.")

    print(f"\n  Processing time:  {summary.elapsed_time:.2f}s ({summary.elapsed_time/summary.duration_seconds:.1f}x realtime)")
    print(f"{'=' * W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _config = ConfigPi()

    parser = argparse.ArgumentParser(
        description='Offline Baby Cry Detection Test — full pipeline on prerecorded multichannel audio'
    )
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to prerecorded WAV file (mono or multichannel)')
    parser.add_argument('--model', type=str, default=None,
                        help=f'Path to trained model checkpoint (default: from config: {_config.MODEL_PATH})')
    parser.add_argument('--channels', type=int, default=_config.PI_CHANNELS,
                        help=f'Expected number of channels (default: {_config.PI_CHANNELS})')
    parser.add_argument('--threshold', type=float, default=_config.DETECTION_THRESHOLD,
                        help=f'Quick detection threshold (default: {_config.DETECTION_THRESHOLD})')
    parser.add_argument('--confirm-threshold', type=float, default=_config.CONFIRMATION_THRESHOLD,
                        help=f'TTA confirmation threshold (default: {_config.CONFIRMATION_THRESHOLD})')
    parser.add_argument('--chunk-duration', type=float, default=1.0,
                        help='Chunk duration in seconds (default: 1.0)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Chunk overlap ratio 0.0-0.9 (default: 0.5)')
    parser.add_argument('--context-duration', type=float, default=_config.CONTEXT_DURATION,
                        help=f'Context window for TTA confirmation (default: {_config.CONTEXT_DURATION}s)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda (default: cpu)')

    # Multichannel options
    parser.add_argument('--disable-multichannel', action='store_true',
                        help='Disable multichannel detection (use channel 0 only)')
    parser.add_argument('--multichannel-voting', type=str, default=None,
                        choices=['weighted', 'logical_or'],
                        help='Voting strategy (default: from config)')

    # Temporal smoothing options
    parser.add_argument('--disable-temporal-smoothing', action='store_true',
                        help='Disable temporal smoothing')
    parser.add_argument('--temporal-window', type=int, default=_config.TEMPORAL_WINDOW_SIZE,
                        help=f'Temporal smoothing window size (default: {_config.TEMPORAL_WINDOW_SIZE})')
    parser.add_argument('--temporal-consecutive', type=int, default=_config.TEMPORAL_MIN_CONSECUTIVE,
                        help=f'Min consecutive high-confidence predictions (default: {_config.TEMPORAL_MIN_CONSECUTIVE})')
    parser.add_argument('--temporal-threshold', type=float, default=_config.TEMPORAL_CONFIDENCE_THRESHOLD,
                        help=f'Temporal confidence threshold (default: {_config.TEMPORAL_CONFIDENCE_THRESHOLD})')

    # Output options
    parser.add_argument('--save-json', type=str, default=None,
                        help='Save full results to JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress logging output (show only final report)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug-level logging')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    if not 0.0 <= args.overlap < 1.0:
        print(f"Error: overlap must be in [0.0, 1.0), got {args.overlap}")
        sys.exit(1)

    # Setup logging
    if args.quiet:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    elif args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Build tester
    tester = OfflineBabyCryTester(
        model_path=args.model,
        detection_threshold=args.threshold,
        confirmation_threshold=args.confirm_threshold,
        device=args.device,
        num_channels=args.channels,
        enable_multichannel=not args.disable_multichannel,
        multichannel_voting=args.multichannel_voting,
        enable_temporal_smoothing=not args.disable_temporal_smoothing,
        temporal_window_size=args.temporal_window,
        temporal_min_consecutive=args.temporal_consecutive,
        temporal_confidence_threshold=args.temporal_threshold,
        chunk_duration=args.chunk_duration,
        overlap_ratio=args.overlap,
        context_duration=args.context_duration,
        verbose=not args.quiet,
    )

    # Run
    summary = tester.run(args.audio)

    # Print report
    print_report(summary)

    # Save JSON if requested
    if args.save_json:
        out = asdict(summary)
        with open(args.save_json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Results saved to {args.save_json}")


if __name__ == '__main__':
    main()
