"""
Enhanced Acoustic Features for Baby Cry Detection.

Implements advanced signal processing features:
- Pitch tracking (F0 extraction)
- Harmonic-to-Noise Ratio (HNR)
- Zero-Crossing Rate (ZCR)
- Temporal regularity analysis

These features help distinguish baby cries from other sounds based on
acoustic properties rather than learned patterns.
"""

import warnings
import torch
import numpy as np
from typing import Any, Tuple, Dict, Optional
import librosa

try:
    from .config import Config
except ImportError:
    import sys
    from pathlib import Path as _Path
    _src_dir = _Path(__file__).parent
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    from config import Config  # type: ignore


class AcousticFeatureExtractor:
    """Extract acoustic features for baby cry validation."""

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize acoustic feature extractor.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        assert sample_rate == Config.SAMPLE_RATE, (
            f"sample_rate {sample_rate} does not match Config.SAMPLE_RATE "
            f"{Config.SAMPLE_RATE}. Pass the correct rate or update Config."
        )
        self.sample_rate = sample_rate

        # Baby cry F0 range — kept in sync with Config to avoid three-way divergence
        # (previously diverged: __init__=250/700, validate_cry_binary=250/800, config=300/600)
        self.f0_min = Config.CRY_F0_MIN
        self.f0_max = Config.CRY_F0_MAX

        # Typical baby cry characteristics
        self.cry_hnr_min = 0.4  # Minimum harmonic-to-noise ratio
        self.cry_duration_min = 0.5  # Minimum cry segment duration (seconds)
        self.cry_duration_max = 5.0  # Maximum cry segment duration (seconds)

    def extract_pitch_librosa(self, audio: torch.Tensor,
                             frame_length: int = 2048,
                             hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch (F0) contour using librosa's pyin algorithm.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Tuple of (f0_contour, voiced_flag) where:
            - f0_contour: Fundamental frequency values (Hz), NaN for unvoiced
            - voiced_flag: Boolean array indicating voiced segments
        """
        # Ensure mono — pyin does not support multichannel input
        if isinstance(audio, torch.Tensor):
            if audio.ndim > 1:
                raise ValueError(
                    f"extract_pitch_librosa expects mono audio, got shape {audio.shape}. "
                    "Average or select a channel before calling."
                )
            audio_np = audio.detach().cpu().numpy()
        else:
            if audio.ndim > 1:
                raise ValueError(
                    f"extract_pitch_librosa expects mono audio, got shape {audio.shape}."
                )
            audio_np = audio

        # Use librosa's pyin for robust pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_np,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate,
            frame_length=frame_length,
            hop_length=hop_length
        )

        return f0, voiced_flag

    def extract_pitch_autocorrelation(self, audio: torch.Tensor,
                                     frame_length: int = 2048,
                                     hop_length: int = 512) -> torch.Tensor:
        """
        Extract pitch using autocorrelation method (faster, pure PyTorch).

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Pitch track tensor (Hz), 0 for unvoiced frames
        """
        # Minimum and maximum lag for baby cry F0 range
        min_lag = int(self.sample_rate / self.f0_max)
        max_lag = int(self.sample_rate / self.f0_min)

        # Build frames matrix in one shot: (n_frames, frame_length).
        # audio.unfold avoids an explicit Python loop over frames.
        frames = audio.unfold(0, frame_length, hop_length)  # (n_frames, frame_length)
        n_frames = frames.shape[0]

        # FFT-based autocorrelation — zero-pad to 2*frame_length for aperiodic
        # (linear) correlation, which avoids circular wrap-around artifacts.
        # Complexity: O(N log N) per frame vs O(N²) for the conv1d approach.
        X = torch.fft.rfft(frames, n=2 * frame_length, dim=1)
        autocorr_all = torch.fft.irfft(X * X.conj(), n=2 * frame_length, dim=1)[:, :frame_length]

        # Normalize each frame by its zero-lag value
        autocorr_all = autocorr_all / (autocorr_all[:, 0:1] + 1e-8)

        if max_lag < frame_length:
            # Slice to baby-cry lag range for all frames at once
            range_autocorr = autocorr_all[:, min_lag:max_lag]  # (n_frames, n_lags)
            max_vals = range_autocorr.max(dim=1).values        # (n_frames,)
            peak_offsets = range_autocorr.argmax(dim=1)         # (n_frames,)
            peak_lags = peak_offsets + min_lag                  # absolute lag index

            # Voiced: autocorrelation peak above threshold AND non-zero lag (guards /0)
            voiced = (max_vals > 0.3) & (peak_lags > 0)
            pitch_track = torch.where(
                voiced,
                torch.tensor(float(self.sample_rate), device=audio.device) / peak_lags.float(),
                torch.zeros(n_frames, device=audio.device)
            )
        else:
            pitch_track = torch.zeros(n_frames, device=audio.device)

        return pitch_track

    def compute_harmonic_to_noise_ratio(self, audio: torch.Tensor,
                                       f0_track: Optional[torch.Tensor] = None,
                                       frame_length: int = 2048,
                                       hop_length: int = 512) -> torch.Tensor:
        """
        Compute Harmonic-to-Noise Ratio (HNR) for each frame.

        HNR measures the ratio of harmonic (periodic) energy to noise energy.
        Baby cries have strong harmonic content (HNR > 0.4).

        Args:
            audio: Input audio tensor
            f0_track: Optional pre-computed pitch track
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            HNR values per frame (0-1 scale, higher = more harmonic)
        """
        # Extract pitch if not provided
        if f0_track is None:
            f0_track = self.extract_pitch_autocorrelation(audio, frame_length, hop_length)

        n_frames = f0_track.shape[0]

        # Batch-compute autocorrelation for all frames — avoids per-frame conv1d loop
        frames = audio.unfold(0, frame_length, hop_length)[:n_frames]  # (n_frames, frame_length)
        X = torch.fft.rfft(frames, n=2 * frame_length, dim=1)
        autocorr_all = torch.fft.irfft(X * X.conj(), n=2 * frame_length, dim=1)[:, :frame_length]
        autocorr_all = autocorr_all / (autocorr_all[:, 0:1] + 1e-8)

        hnr_values = []

        for frame_idx, f0 in enumerate(f0_track):
            f0_val = f0.item()  # 0-dim tensor → Python float (avoids implicit bool conversion)

            if f0_val == 0.0:
                # Unvoiced frame — pitch extractor found no periodic content
                hnr_values.append(0.0)
                continue

            if f0_val < self.f0_min or f0_val > self.f0_max:
                # F0 detected but outside baby-cry range — meaningless for HNR here
                hnr_values.append(0.0)
                continue

            pitch_period_samples = int(self.sample_rate / f0_val)

            if 0 < pitch_period_samples < frame_length:
                harmonic_strength = autocorr_all[frame_idx, pitch_period_samples].item()
                hnr = max(0.0, min(1.0, harmonic_strength))
            else:
                hnr = 0.0

            hnr_values.append(hnr)

        return torch.tensor(hnr_values)

    def compute_zero_crossing_rate(self, audio: torch.Tensor,
                                  frame_length: int = 512,
                                  hop_length: int = 256) -> torch.Tensor:
        """
        Compute Zero-Crossing Rate (ZCR) for each frame.

        ZCR counts how often the signal changes sign.
        Useful for distinguishing cry from noise and other sounds.
        Baby cries typically have moderate ZCR (not too low, not too high).

        Note: ZCR uses shorter frames (default 512/256) than pitch/HNR features
        (2048/512). This is intentional — ZCR captures fine-grained temporal
        changes better at shorter scales — but callers should be aware that the
        ZCR frame count will differ from the pitch/HNR frame count.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis (default 512, shorter than pitch frames)
            hop_length: Hop length between frames

        Returns:
            ZCR values per frame
        """
        zcr_values = []

        for start_idx in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start_idx:start_idx + frame_length]

            # Add a tiny offset before sign() so that exact-zero samples are assigned
            # a consistent sign; torch.sign(0.0) == 0.0 would otherwise produce
            # spurious zero-crossings when silence-padded frames are processed.
            signs = torch.sign(frame + 1e-10)
            sign_changes = torch.abs(torch.diff(signs))
            zcr = torch.sum(sign_changes) / (2.0 * len(frame))

            zcr_values.append(zcr.item())

        return torch.tensor(zcr_values)

    def compute_spectral_centroid(self, audio: torch.Tensor,
                                 frame_length: int = 2048,
                                 hop_length: int = 512) -> torch.Tensor:
        """
        Compute spectral centroid (center of mass of spectrum).

        Baby cries typically have centroid in 400-800 Hz range.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Spectral centroid values per frame (Hz)
        """
        # Hann window required for proper WOLA reconstruction and to reduce
        # spectral leakage. Without it, rectangular windowing inflates centroid values.
        window = torch.hann_window(frame_length, device=audio.device)

        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )

        magnitude = torch.abs(stft)  # (n_bins, n_frames)
        n_bins = frame_length // 2 + 1
        # torch.linspace is device-aware; torch.fft.fftfreq always returns a CPU tensor
        # regardless of input device, which would crash on CUDA audio.
        freqs = torch.linspace(0, self.sample_rate / 2, n_bins, device=audio.device)

        # Vectorized weighted average over the frequency axis — avoids per-frame loop
        total_mag = magnitude.sum(dim=0)             # (n_frames,)
        weighted = (freqs.unsqueeze(1) * magnitude).sum(dim=0)  # (n_frames,)
        centroids = torch.where(
            total_mag > 1e-8,
            weighted / total_mag,
            torch.zeros_like(total_mag)
        )

        return centroids

    def analyze_temporal_regularity(self, audio: torch.Tensor,
                                   frame_length: int = 2048,
                                   hop_length: int = 512) -> float:
        """
        Analyze temporal regularity of the audio.

        Baby cries have quasi-periodic structure with regular cry bursts.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Regularity score in [-1, 1]. Positive values indicate periodic energy
            patterns (typical of cry bursts); negative values indicate anti-periodic
            patterns (e.g. alternating loud/quiet frames).
        """
        # Vectorized short-term energy — avoids a Python loop over frames
        frames = audio.unfold(0, frame_length, hop_length)  # (n_frames, frame_length)
        energy = frames.pow(2).mean(dim=1)  # (n_frames,)

        if len(energy) < 10:
            return 0.0

        energy_norm = (energy - energy.mean()) / (energy.std() + 1e-8)

        # FFT-based autocorrelation of the energy envelope — O(N log N) vs O(N²)
        n = len(energy_norm)
        X = torch.fft.rfft(energy_norm, n=2 * n)
        autocorr = torch.fft.irfft(X * X.conj(), n=2 * n)[:n]
        autocorr = autocorr / (autocorr[0] + 1e-8)

        # Look for periodicity in 0.5-3 second range (typical cry burst rate)
        min_period_frames = int(0.5 * self.sample_rate / hop_length)
        max_period_frames = int(3.0 * self.sample_rate / hop_length)

        if max_period_frames < len(autocorr):
            periodic_range = autocorr[min_period_frames:max_period_frames]
            regularity = periodic_range.max().item()
        else:
            regularity = 0.0

        # Return raw value in [-1, 1] — negative autocorrelation (anti-periodic signal)
        # is meaningful and should not be silently clipped to 0.
        return regularity

    def extract_all_features(self, audio: torch.Tensor,
                           frame_length: int = 2048,
                           hop_length: int = 512) -> Dict[str, Any]:
        """
        Extract all acoustic features in one pass.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Dictionary containing all acoustic features
        """
        # Extract pitch
        f0_track = self.extract_pitch_autocorrelation(audio, frame_length, hop_length)

        # Extract HNR
        hnr_track = self.compute_harmonic_to_noise_ratio(audio, f0_track, frame_length, hop_length)

        # Extract ZCR
        zcr_track = self.compute_zero_crossing_rate(audio, frame_length=512, hop_length=256)

        # Extract spectral centroid
        centroid_track = self.compute_spectral_centroid(audio, frame_length, hop_length)

        # Temporal regularity (single value)
        regularity = self.analyze_temporal_regularity(audio, frame_length, hop_length)

        # Compute statistics for pitch
        voiced_frames = f0_track > 0
        if voiced_frames.any():
            pitch_mean = f0_track[voiced_frames].mean().item()
            pitch_std = f0_track[voiced_frames].std().item()
            pitch_min = f0_track[voiced_frames].min().item()
            pitch_max = f0_track[voiced_frames].max().item()
        else:
            pitch_mean = pitch_std = pitch_min = pitch_max = 0.0

        # Compute statistics for HNR
        hnr_mean = hnr_track.mean().item()
        hnr_std = hnr_track.std().item()

        # Compute statistics for ZCR
        zcr_mean = zcr_track.mean().item()
        zcr_std = zcr_track.std().item()

        # Compute statistics for spectral centroid
        centroid_mean = centroid_track.mean().item()
        centroid_std = centroid_track.std().item()

        return {
            # Raw tracks
            'f0_track': f0_track,
            'hnr_track': hnr_track,
            'zcr_track': zcr_track,
            'centroid_track': centroid_track,

            # Pitch statistics
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_min': pitch_min,
            'pitch_max': pitch_max,
            'pitch_range': pitch_max - pitch_min,

            # HNR statistics
            'hnr_mean': hnr_mean,
            'hnr_std': hnr_std,

            # ZCR statistics
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,

            # Spectral centroid statistics
            'centroid_mean': centroid_mean,
            'centroid_std': centroid_std,

            # Temporal features
            'regularity': regularity,
            'duration': len(audio) / self.sample_rate
        }


def validate_cry_binary(features: Dict[str, float],
                       cry_prob: float,
                       threshold: float = 0.5) -> Tuple[bool, str]:
    """
    Binary validation: accept or reject prediction based on acoustic features.

    DOES NOT modify probability - only returns accept/reject decision.
    This implements true binary classification without adjusting probabilities.

    Args:
        features: Dictionary of acoustic features
        cry_prob: Model's predicted cry probability (NOT MODIFIED)
        threshold: Minimum threshold for cry prediction

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    # Rule 1: Verify pitch is in typical baby cry range (Config.CRY_F0_MIN/MAX)
    if features['pitch_mean'] > 0:
        if features['pitch_mean'] < Config.CRY_F0_MIN:
            return False, f"Pitch too low (< {Config.CRY_F0_MIN} Hz) - likely adult or environmental"
        if features['pitch_mean'] > Config.CRY_F0_MAX:
            return False, f"Pitch too high (> {Config.CRY_F0_MAX} Hz) - likely not baby cry"

    # Rule 2: Check duration
    if features['duration'] < 0.5:
        return False, "Segment too short (< 0.5s) - likely noise"

    # Rule 3: Verify HNR (Harmonic-to-Noise Ratio)
    if features['hnr_mean'] < 0.3:
        return False, "HNR too low (< 0.3) - likely noise"

    # All checks passed
    return True, "Passed acoustic validation"


def validate_cry_with_acoustic_features(features: Dict[str, float],
                                       cry_prob: float,
                                       threshold: float = 0.5) -> Tuple[bool, float, str]:
    """
    Post-processing heuristics to validate if a prediction is truly a baby cry.

    DEPRECATED: Use validate_cry_binary for true binary classification.
    This function modifies probabilities and should only be used for hybrid scoring.

    Args:
        features: Dictionary of acoustic features from extract_all_features()
        cry_prob: Model's predicted cry probability
        threshold: Minimum threshold for cry prediction

    Returns:
        Tuple of (is_cry, adjusted_probability, rejection_reason)
    """
    warnings.warn(
        "validate_cry_with_acoustic_features is deprecated. "
        "Use validate_cry_binary for true binary classification.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Start with model prediction
    if cry_prob < threshold:
        return False, cry_prob, "Model confidence too low"

    # Rule 1: Verify pitch is in typical baby cry range (Config.CRY_F0_MIN/MAX)
    if features['pitch_mean'] > 0:  # Only check if pitch was detected
        if features['pitch_mean'] < Config.CRY_F0_MIN:
            return False, 0.0, f"Pitch too low (< {Config.CRY_F0_MIN} Hz) - likely adult or environmental"
        if features['pitch_mean'] > Config.CRY_F0_MAX:
            return False, 0.0, f"Pitch too high (> {Config.CRY_F0_MAX} Hz) - likely noise or artifact"

    # Rule 2: Check duration pattern (cries are typically 0.5-5 seconds)
    if features['duration'] < 0.5:
        return False, 0.0, "Segment too short (< 0.5s)"
    if features['duration'] > 5.0:
        # Don't reject, but reduce confidence for very long segments
        cry_prob *= 0.7

    # Rule 3: Verify harmonic-to-noise ratio (cries have strong harmonic content)
    if features['hnr_mean'] < 0.3:
        return False, 0.0, "HNR too low (< 0.3) - likely noise or environmental sound"

    # Rule 4: Check spectral centroid (baby cries typically 400-800 Hz)
    if features['centroid_mean'] > 0:
        if features['centroid_mean'] < 300:
            cry_prob *= 0.5  # Reduce confidence for low centroid
        elif features['centroid_mean'] > 1500:
            cry_prob *= 0.6  # Reduce confidence for very high centroid

    # Rule 5: Verify pitch variation (cries have moderate variation)
    if features['pitch_range'] > 0:
        if features['pitch_range'] < 20:
            # Very stable pitch - might be music or sustained tone
            cry_prob *= 0.7
        elif features['pitch_range'] > 300:
            # Too much variation - might be noise or multiple speakers
            cry_prob *= 0.8

    # Rule 6: Check ZCR (moderate values expected)
    if features['zcr_mean'] > 0.3:
        # Very high ZCR suggests noise
        cry_prob *= 0.6
    elif features['zcr_mean'] < 0.02:
        # Very low ZCR suggests low-frequency rumble
        cry_prob *= 0.7

    # Rule 7: Boost confidence for strong harmonic content
    if features['hnr_mean'] > 0.6:
        cry_prob *= 1.2  # Boost for strong harmonics

    # Rule 8: Check temporal regularity (check stricter bound first to avoid dead branch)
    if features['regularity'] > 0.8:
        # Too regular — likely music or periodic noise
        cry_prob *= 0.8
    elif features['regularity'] > 0.5:
        # Some regularity is good (cry bursts)
        cry_prob *= 1.1

    # Clamp final probability
    cry_prob = max(0.0, min(1.0, cry_prob))

    # Final decision
    is_cry = cry_prob >= threshold
    reason = "Passed all acoustic validation checks" if is_cry else "Adjusted probability below threshold"

    return is_cry, cry_prob, reason
