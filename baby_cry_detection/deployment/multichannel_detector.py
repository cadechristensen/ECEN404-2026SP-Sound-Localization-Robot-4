"""
Multi-channel baby cry detection with adaptive channel selection and dual-channel voting.

This module implements advanced multi-channel detection strategies to improve
reliability and robustness over single-channel detection. Key features:
- Enhanced SNR computation optimized for cry detection (300-900 Hz)
- Adaptive channel selection based on signal quality
- Dual-channel voting for improved accuracy
- Channel health monitoring
- Phase preservation for sound localization
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from dataclasses import dataclass
from scipy import signal
from scipy.signal import coherence


@dataclass
class ChannelQualityMetrics:
    """Metrics for assessing channel quality."""
    channel_idx: int
    snr_db: float
    rms: float
    clipping: bool
    coherence_scores: Optional[List[float]] = None


@dataclass
class MultiChannelDetectionResult:
    """Results from multi-channel detection."""
    is_cry: bool
    confidence: float
    primary_channel: int
    secondary_channel: Optional[int]
    channel_snr_scores: np.ndarray
    multi_channel_agreement: Optional[float]
    channel_confidences: List[float]
    metadata: Dict


class EnhancedSNRComputation:
    """
    Enhanced SNR computation optimized for infant cry detection.

    Uses frequency bands specifically tuned for cry characteristics:
    - Signal band: 300-900 Hz (cry fundamental + first harmonic)
    - Noise bands: 50-200 Hz (rumble) + 1500-4000 Hz (environmental)
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize SNR computation.

        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT size for frequency analysis (must match config.N_FFT = 2048)
            hop_length: Hop length for STFT (must match config.HOP_LENGTH = 512)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Frequency band definitions (optimized for cry detection)
        self.signal_band_low = 300   # Cry fundamental starts here
        self.signal_band_high = 900  # Includes first harmonic
        self.noise_band_low1 = 50    # Low-frequency rumble
        self.noise_band_high1 = 200
        self.noise_band_low2 = 1500  # High-frequency environmental
        self.noise_band_high2 = 4000

    def compute_cry_snr(self, channel_audio: np.ndarray) -> float:
        """
        Compute SNR optimized for cry detection.

        Args:
            channel_audio: Single-channel audio array

        Returns:
            SNR in dB
        """
        # Handle torch tensors
        if isinstance(channel_audio, torch.Tensor):
            channel_audio = channel_audio.cpu().numpy()

        # Ensure 1D array
        if channel_audio.ndim > 1:
            channel_audio = channel_audio.flatten()

        # Compute STFT
        f, t, Zxx = signal.stft(
            channel_audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )

        # Magnitude spectrum
        magnitude = np.abs(Zxx)

        # Signal band energy (cry fundamental + first harmonic)
        signal_mask = (f >= self.signal_band_low) & (f <= self.signal_band_high)
        signal_energy = np.mean(magnitude[signal_mask, :] ** 2)

        # Noise band energy (low-frequency rumble + high-frequency environmental)
        noise_mask_low = (f >= self.noise_band_low1) & (f < self.noise_band_high1)
        noise_mask_high = (f > self.noise_band_low2) & (f < self.noise_band_high2)

        noise_energy_low = np.mean(magnitude[noise_mask_low, :] ** 2)
        noise_energy_high = np.mean(magnitude[noise_mask_high, :] ** 2)
        noise_energy = (noise_energy_low + noise_energy_high) / 2

        # Compute SNR in dB
        snr_db = 10 * np.log10((signal_energy + 1e-10) / (noise_energy + 1e-10))

        return float(snr_db)

    def compute_snr_all_channels(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute SNR for all channels.

        Args:
            audio: Multi-channel audio (num_samples, num_channels)

        Returns:
            SNR scores for each channel (num_channels,)
        """
        if audio.ndim == 1:
            return np.array([self.compute_cry_snr(audio)])

        num_channels = audio.shape[1]
        snr_scores = np.zeros(num_channels)

        for ch in range(num_channels):
            snr_scores[ch] = self.compute_cry_snr(audio[:, ch])

        # Replace NaN scores (from corrupted audio) with very low SNR
        # so channel selection never picks a NaN channel
        np.nan_to_num(snr_scores, copy=False, nan=-100.0)

        return snr_scores


class ChannelHealthMonitor:
    """Monitor channel health to detect hardware failures and signal issues."""

    def __init__(self, sample_rate: int = 16000, num_channels: int = 4,
                 snr_computer: Optional["EnhancedSNRComputation"] = None):
        """
        Initialize channel health monitor.

        Args:
            sample_rate: Audio sample rate
            num_channels: Number of microphone channels
            snr_computer: Optional pre-constructed EnhancedSNRComputation to reuse.
                          If None a new instance is created once and cached.
        """
        self.sample_rate = sample_rate
        self.num_channels = num_channels

        # Cache the SNR computer so get_channel_health() doesn't allocate a new
        # EnhancedSNRComputation object on every call (hot path on Pi).
        self._snr_computer = snr_computer or EnhancedSNRComputation(sample_rate=sample_rate)

        # Health thresholds
        self.rms_min_threshold = 0.001   # Minimum RMS (detect silence/disconnection)
        self.clipping_threshold = 0.95   # Maximum amplitude (detect clipping)
        self.coherence_threshold = 0.7   # Minimum coherence between channels

        # Periodic coherence cache — check_coherence is O(N^2) channel pairs
        # and costs ~200ms for 4 channels. Only run it every 60 seconds.
        self._coherence_check_interval = 60  # seconds
        self._last_coherence_check: float = 0.0
        self._cached_coherence: Dict[str, float] = {}

    def check_channel_rms(self, audio: np.ndarray) -> List[float]:
        """
        Check RMS level for each channel.

        Args:
            audio: Multi-channel audio (num_samples, num_channels)

        Returns:
            RMS values for each channel
        """
        if audio.ndim == 1:
            return [float(np.sqrt(np.mean(audio ** 2)))]

        rms_values = []
        for ch in range(audio.shape[1]):
            rms = np.sqrt(np.mean(audio[:, ch] ** 2))
            rms_values.append(float(rms))

            if rms < self.rms_min_threshold:
                logging.warning(f"Channel {ch} has low RMS ({rms:.6f}), possible mic failure")

        return rms_values

    def check_clipping(self, audio: np.ndarray) -> List[bool]:
        """
        Check for clipping in each channel.

        Args:
            audio: Multi-channel audio (num_samples, num_channels)

        Returns:
            List of boolean flags indicating clipping per channel
        """
        if audio.ndim == 1:
            max_amp = np.max(np.abs(audio))
            return [bool(max_amp > self.clipping_threshold)]

        clipping_flags = []
        for ch in range(audio.shape[1]):
            max_amp = np.max(np.abs(audio[:, ch]))
            is_clipping = max_amp > self.clipping_threshold

            if is_clipping:
                logging.warning(f"Channel {ch} clipping detected (max: {max_amp:.3f})")

            clipping_flags.append(bool(is_clipping))

        return clipping_flags

    def check_coherence(self, audio: np.ndarray, freq_range: Tuple[int, int] = (300, 1000)) -> Dict[str, float]:
        """
        Check inter-channel coherence in cry frequency range.

        Args:
            audio: Multi-channel audio (num_samples, num_channels)
            freq_range: Frequency range for coherence analysis (Hz)

        Returns:
            Dictionary of coherence scores between channel pairs
        """
        if audio.ndim == 1 or audio.shape[1] < 2:
            return {}

        coherence_scores = {}

        # scipy.signal.coherence requires at least nperseg samples.
        # Clamp nperseg to half the audio length to avoid a ValueError during
        # buffer warm-up (first few seconds after startup on Pi).
        nperseg = min(1024, audio.shape[0] // 2)
        if nperseg < 16:
            # Audio chunk too short to compute a meaningful coherence estimate.
            return {}

        for ch1 in range(audio.shape[1]):
            for ch2 in range(ch1 + 1, audio.shape[1]):
                # Compute coherence
                f, Cxy = coherence(
                    audio[:, ch1],
                    audio[:, ch2],
                    fs=self.sample_rate,
                    nperseg=nperseg
                )

                # Average coherence in cry frequency range
                freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
                avg_coherence = float(np.mean(Cxy[freq_mask]))

                pair_key = f"ch{ch1}-ch{ch2}"
                coherence_scores[pair_key] = avg_coherence

                if avg_coherence < self.coherence_threshold:
                    logging.warning(
                        f"Low coherence between channels {ch1} and {ch2}: {avg_coherence:.2f}"
                    )

        return coherence_scores

    def get_channel_health(self, audio: np.ndarray) -> List[ChannelQualityMetrics]:
        """
        Get comprehensive health metrics for all channels.

        Args:
            audio: Multi-channel audio (num_samples, num_channels)

        Returns:
            List of ChannelQualityMetrics for each channel
        """
        rms_values = self.check_channel_rms(audio)
        clipping_flags = self.check_clipping(audio)

        # Only re-run the slow coherence check when the cache has expired
        import time as _time
        now = _time.time()
        if now - self._last_coherence_check >= self._coherence_check_interval:
            self._cached_coherence = self.check_coherence(audio)
            self._last_coherence_check = now
        coherence_dict = self._cached_coherence

        # Compute SNR using the cached instance (avoids repeated allocation on Pi).
        snr_scores = self._snr_computer.compute_snr_all_channels(audio)

        metrics = []
        for ch in range(len(rms_values)):
            # Get coherence scores for this channel.
            # Keys have the form "ch{a}-ch{b}" with a < b; match on exact prefix/suffix
            # so that ch1 does not accidentally match ch10, ch11, etc.
            ch_coherence = [
                score for key, score in coherence_dict.items()
                if key.startswith(f"ch{ch}-") or key.endswith(f"-ch{ch}")
            ]

            metrics.append(ChannelQualityMetrics(
                channel_idx=ch,
                snr_db=float(snr_scores[ch]),
                rms=rms_values[ch],
                clipping=clipping_flags[ch],
                coherence_scores=ch_coherence if ch_coherence else None
            ))

        return metrics


class DualChannelVotingDetector:
    """
    Dual-channel voting detector for improved reliability.

    Implements the Tier 2 strategy recommended by audio quality experts:
    - Select best 2 channels by SNR
    - Run detection on both channels
    - Use SNR-weighted voting or logical OR
    - Preserves phase for sound localization
    """

    def __init__(
        self,
        detector,
        num_channels: int = 4,
        voting_strategy: str = "weighted",
        sample_rate: int = 16000
    ):
        """
        Initialize dual-channel voting detector.

        Args:
            detector: RealtimeBabyCryDetector instance
            num_channels: Number of microphone channels
            voting_strategy: "weighted" (SNR-weighted) or "logical_or" (any channel)
            sample_rate: Audio sample rate
        """
        _VALID_STRATEGIES = frozenset({"weighted", "logical_or"})
        if voting_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown voting_strategy '{voting_strategy}'. "
                f"Must be one of {sorted(_VALID_STRATEGIES)}."
            )

        self.detector = detector
        self.num_channels = num_channels
        self.voting_strategy = voting_strategy
        self.sample_rate = sample_rate

        # Initialize components — share the snr_computer instance so both this
        # class and ChannelHealthMonitor use the same FFT parameters and avoid
        # redundant allocations on the Pi hot path.
        self.snr_computer = EnhancedSNRComputation(sample_rate=sample_rate)
        self.health_monitor = ChannelHealthMonitor(
            sample_rate=sample_rate,
            num_channels=num_channels,
            snr_computer=self.snr_computer,
        )

        logging.info(f"Initialized dual-channel voting detector with {voting_strategy} strategy")

    def select_best_channels(self, audio: np.ndarray, n_channels: int = 2) -> Tuple[List[int], np.ndarray]:
        """
        Select the n best channels based on SNR.

        Args:
            audio: Multi-channel audio (num_samples, num_channels)
            n_channels: Number of channels to select

        Returns:
            Tuple of (selected_channel_indices, snr_scores)
        """
        # Compute SNR for all channels
        snr_scores = self.snr_computer.compute_snr_all_channels(audio)

        # Sort channels by SNR (best first)
        sorted_indices = np.argsort(snr_scores)[::-1]

        # Select top n channels (clamp to available channel count)
        n_select = min(n_channels, len(sorted_indices))
        best_channels = sorted_indices[:n_select].tolist()

        logging.debug(f"Channel SNRs: {snr_scores}")
        logging.debug(f"Selected channels: {best_channels} (SNRs: {snr_scores[best_channels]})")

        return best_channels, snr_scores

    def detect_cry_dual_channel(
        self,
        audio: np.ndarray,
        use_tta: bool = False,
        confidence_threshold: float = 0.5
    ) -> MultiChannelDetectionResult:
        """
        Detect cry using dual-channel voting strategy.

        Args:
            audio: Multi-channel audio (num_samples, num_channels)
            use_tta: Whether to use test-time augmentation
            confidence_threshold: Detection confidence threshold

        Returns:
            MultiChannelDetectionResult with detection decision and metadata
        """
        # Ensure multi-channel input
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        # Guard: dual-channel voting requires at least 2 channels.
        # Fall back to single-channel detection if the input is mono so that the
        # caller always gets a well-formed MultiChannelDetectionResult.
        if audio.shape[1] < 2:
            logging.warning(
                "detect_cry_dual_channel called with fewer than 2 channels "
                f"(got {audio.shape[1]}). Falling back to single-channel detection."
            )
            channel_audio = audio[:, 0]
            is_cry, confidence, _ = self.detector.detect_cry(channel_audio, use_tta=use_tta)
            snr_scores = self.snr_computer.compute_snr_all_channels(audio)
            return MultiChannelDetectionResult(
                is_cry=is_cry,
                confidence=confidence,
                primary_channel=0,
                secondary_channel=None,
                channel_snr_scores=snr_scores,
                multi_channel_agreement=None,
                channel_confidences=[confidence],
                metadata={
                    'voting_strategy': 'single_channel_fallback',
                    'selected_channels': [0],
                    'use_tta': use_tta,
                }
            )

        # Select best 2 channels
        best_channels, snr_scores = self.select_best_channels(audio, n_channels=2)

        # Run detection on both selected channels
        channel_confidences = []
        for ch in best_channels:
            # Extract channel audio
            channel_audio = audio[:, ch]

            # Run detection
            is_cry_ch, confidence_ch, _ = self.detector.detect_cry(channel_audio, use_tta=use_tta)
            channel_confidences.append(confidence_ch)

            logging.debug(f"Channel {ch} detection: {is_cry_ch} (conf: {confidence_ch:.2%})")

        # Apply voting strategy
        if self.voting_strategy == "weighted":
            # SNR-weighted average
            weights = np.array([snr_scores[ch] for ch in best_channels])
            weights = np.exp(weights / 10.0)  # Exponential weighting
            weights = weights / weights.sum()  # Normalize

            weighted_confidence = np.sum(np.array(channel_confidences) * weights)
            is_cry = weighted_confidence >= confidence_threshold
            final_confidence = float(weighted_confidence)

            logging.debug(f"SNR weights: {weights}, Weighted confidence: {weighted_confidence:.2%}")

        elif self.voting_strategy == "logical_or":
            # Logical OR (cry if EITHER channel detects)
            is_cry = any(conf >= confidence_threshold for conf in channel_confidences)
            final_confidence = float(max(channel_confidences))

            logging.debug(f"Logical OR result: {is_cry}, Max confidence: {final_confidence:.2%}")

        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

        # Compute multi-channel agreement
        if len(channel_confidences) == 2:
            agreement = 1.0 - abs(channel_confidences[0] - channel_confidences[1])
        else:
            agreement = None

        # Create result
        result = MultiChannelDetectionResult(
            is_cry=is_cry,
            confidence=final_confidence,
            primary_channel=best_channels[0],
            secondary_channel=best_channels[1] if len(best_channels) > 1 else None,
            channel_snr_scores=snr_scores,
            multi_channel_agreement=agreement,
            channel_confidences=channel_confidences,
            metadata={
                'voting_strategy': self.voting_strategy,
                'selected_channels': best_channels,
                'use_tta': use_tta
            }
        )

        return result

    def detect_with_health_check(
        self,
        audio: np.ndarray,
        use_tta: bool = False,
        confidence_threshold: float = 0.75
    ) -> Tuple[MultiChannelDetectionResult, List[ChannelQualityMetrics]]:
        """
        Detect cry with channel health monitoring.

        Args:
            audio: Multi-channel audio (num_samples, num_channels)
            use_tta: Whether to use test-time augmentation
            confidence_threshold: Dual-channel voting threshold (default: 0.75 per spec).
                                  Callers performing final confirmation (e.g.
                                  confirm_and_filter) should pass
                                  self.confirmation_threshold (0.92) explicitly.

        Returns:
            Tuple of (detection_result, channel_health_metrics)
        """
        # Check channel health
        health_metrics = self.health_monitor.get_channel_health(audio)

        # Log any health issues
        for metric in health_metrics:
            if metric.rms < self.health_monitor.rms_min_threshold:
                logging.warning(
                    f"Channel {metric.channel_idx} health issue: Low RMS ({metric.rms:.6f})"
                )
            if metric.clipping:
                logging.warning(
                    f"Channel {metric.channel_idx} health issue: Clipping detected"
                )

        # Run detection
        result = self.detect_cry_dual_channel(audio, use_tta, confidence_threshold)

        return result, health_metrics


def create_multichannel_detector(
    detector,
    num_channels: int = 4,
    voting_strategy: str = "weighted",
    sample_rate: int = 16000
) -> DualChannelVotingDetector:
    """
    Factory function to create a multi-channel detector.

    Args:
        detector: Existing RealtimeBabyCryDetector instance
        num_channels: Number of microphone channels
        voting_strategy: "weighted" or "logical_or"
        sample_rate: Audio sample rate

    Returns:
        DualChannelVotingDetector instance
    """
    return DualChannelVotingDetector(
        detector=detector,
        num_channels=num_channels,
        voting_strategy=voting_strategy,
        sample_rate=sample_rate
    )
