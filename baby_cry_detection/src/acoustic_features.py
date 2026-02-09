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

import torch
import numpy as np
from typing import Tuple, Dict, Optional
import librosa


class AcousticFeatureExtractor:
    """Extract acoustic features for baby cry validation."""

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize acoustic feature extractor.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Baby cry F0 range (Hz)
        self.f0_min = 250
        self.f0_max = 700

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
        # Convert to numpy if torch tensor
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
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
        pitch_track = []

        # Minimum and maximum lag for baby cry F0 range
        min_lag = int(self.sample_rate / self.f0_max)
        max_lag = int(self.sample_rate / self.f0_min)

        for start_idx in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start_idx:start_idx + frame_length]

            # Autocorrelation using convolution
            autocorr = torch.nn.functional.conv1d(
                frame.unsqueeze(0).unsqueeze(0),
                frame.flip(0).unsqueeze(0).unsqueeze(0),
                padding=frame_length - 1
            )[0, 0, frame_length-1:]

            # Normalize autocorrelation
            autocorr = autocorr / (autocorr[0] + 1e-8)

            # Find peak in baby cry F0 range
            if max_lag < len(autocorr):
                autocorr_range = autocorr[min_lag:max_lag]

                # Require minimum autocorrelation strength
                if autocorr_range.max() > 0.3:  # Voiced threshold
                    peak_lag = min_lag + torch.argmax(autocorr_range).item()
                    pitch = self.sample_rate / peak_lag
                else:
                    pitch = 0.0  # Unvoiced
            else:
                pitch = 0.0

            pitch_track.append(pitch)

        return torch.tensor(pitch_track)

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

        hnr_values = []

        for frame_idx, f0 in enumerate(f0_track):
            start_idx = frame_idx * hop_length
            end_idx = start_idx + frame_length

            if end_idx > len(audio):
                break

            frame = audio[start_idx:end_idx]

            # If unvoiced (f0 == 0), HNR is 0
            if f0 < self.f0_min or f0 > self.f0_max:
                hnr_values.append(0.0)
                continue

            # Compute autocorrelation
            autocorr = torch.nn.functional.conv1d(
                frame.unsqueeze(0).unsqueeze(0),
                frame.flip(0).unsqueeze(0).unsqueeze(0),
                padding=frame_length - 1
            )[0, 0, frame_length-1:]

            # Normalize
            autocorr = autocorr / (autocorr[0] + 1e-8)

            # HNR is estimated from autocorrelation at the pitch period
            pitch_period_samples = int(self.sample_rate / f0)

            if pitch_period_samples < len(autocorr):
                # Harmonic strength is autocorrelation at pitch period
                harmonic_strength = autocorr[pitch_period_samples].item()

                # Convert to HNR (0-1 scale)
                # High autocorrelation at pitch period = strong harmonics
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

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            ZCR values per frame
        """
        zcr_values = []

        for start_idx in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start_idx:start_idx + frame_length]

            # Count sign changes
            signs = torch.sign(frame)
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
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            return_complex=True
        )

        magnitude = torch.abs(stft)
        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]

        # Compute weighted average frequency
        centroids = []
        for frame_idx in range(magnitude.shape[1]):
            frame_mag = magnitude[:, frame_idx]

            # Weighted average
            total_mag = torch.sum(frame_mag)
            if total_mag > 1e-8:
                centroid = torch.sum(freqs * frame_mag) / total_mag
            else:
                centroid = 0.0

            centroids.append(centroid.item())

        return torch.tensor(centroids)

    def analyze_temporal_regularity(self, audio: torch.Tensor,
                                   frame_length: int = 2048,
                                   hop_length: int = 512) -> float:
        """
        Analyze temporal regularity of the audio.

        Baby cries have quasi-periodic structure with regular cry bursts.
        Returns regularity score (0-1, higher = more regular).

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Regularity score (0-1)
        """
        # Compute short-term energy
        energy = []
        for start_idx in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start_idx:start_idx + frame_length]
            frame_energy = torch.mean(frame ** 2).item()
            energy.append(frame_energy)

        energy = torch.tensor(energy)

        if len(energy) < 10:
            return 0.0

        # Autocorrelation of energy envelope
        energy_norm = (energy - energy.mean()) / (energy.std() + 1e-8)

        # Compute autocorrelation
        autocorr = torch.nn.functional.conv1d(
            energy_norm.unsqueeze(0).unsqueeze(0),
            energy_norm.flip(0).unsqueeze(0).unsqueeze(0),
            padding=len(energy_norm) - 1
        )[0, 0, len(energy_norm)-1:]

        autocorr = autocorr / (autocorr[0] + 1e-8)

        # Look for periodicity in 0.5-3 second range (typical cry burst rate)
        min_period_frames = int(0.5 * self.sample_rate / hop_length)
        max_period_frames = int(3.0 * self.sample_rate / hop_length)

        if max_period_frames < len(autocorr):
            periodic_range = autocorr[min_period_frames:max_period_frames]
            regularity = periodic_range.max().item()
        else:
            regularity = 0.0

        return max(0.0, min(1.0, regularity))

    def extract_all_features(self, audio: torch.Tensor,
                           frame_length: int = 2048,
                           hop_length: int = 512) -> Dict[str, torch.Tensor]:
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
    # Rule 1: Verify pitch is in typical baby cry range
    if features['pitch_mean'] > 0:
        if features['pitch_mean'] < 250:
            return False, "Pitch too low (< 250 Hz) - likely adult or environmental"
        if features['pitch_mean'] > 800:
            return False, "Pitch too high (> 800 Hz) - likely not baby cry"

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
    # Start with model prediction
    if cry_prob < threshold:
        return False, cry_prob, "Model confidence too low"

    # Rule 1: Verify pitch is in typical baby cry range (250-700 Hz)
    if features['pitch_mean'] > 0:  # Only check if pitch was detected
        if features['pitch_mean'] < 250:
            return False, 0.0, "Pitch too low (< 250 Hz) - likely adult or environmental"
        if features['pitch_mean'] > 800:
            return False, 0.0, "Pitch too high (> 800 Hz) - likely noise or artifact"

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

    # Rule 8: Check temporal regularity
    if features['regularity'] > 0.5:
        # Some regularity is good (cry bursts)
        cry_prob *= 1.1
    elif features['regularity'] > 0.8:
        # Too regular might be music
        cry_prob *= 0.8

    # Clamp final probability
    cry_prob = max(0.0, min(1.0, cry_prob))

    # Final decision
    is_cry = cry_prob >= threshold
    reason = "Passed all acoustic validation checks" if is_cry else "Adjusted probability below threshold"

    return is_cry, cry_prob, reason
