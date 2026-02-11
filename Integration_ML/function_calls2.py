"""
Sound characterization and noise filtering functions for baby cry detection and localization.

This module contains:
1. Sound characterization: Extract acoustic features (RMS, spectral centroid, MFCCs)
2. Noise filtering: High-pass, band-pass, and spectral subtraction filters
3. VAD (Voice Activity Detection): Detect cry activity in audio
"""

import numpy as np
import librosa
import torch
from scipy import signal
from typing import Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")


# ========================================
# SOUND CHARACTERIZATION FUNCTIONS
# ========================================

def extract_distance_features(y: np.ndarray, sr: int = 48000,
                              frame_length: int = 2048,
                              hop_length: int = 256) -> dict:
    """
    Extract acoustic features for distance estimation from audio signal.

    Computes statistical features from RMS energy, spectral centroid, and MFCCs
    that correlate with sound source distance.

    Args:
        y: Audio signal (mono)
        sr: Sample rate
        frame_length: FFT window size
        hop_length: Hop length for STFT

    Returns:
        Dictionary containing:
            - rms_mean, rms_std: RMS energy statistics
            - spec_cent_mean, spec_cent_std: Spectral centroid statistics
            - mfcc_mean_1..13, mfcc_std_1..13: MFCC coefficient statistics
    """
    # Root Mean Square energy (correlates with loudness/distance)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    # Spectral centroid (brightness of sound - affected by distance)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)

    # MFCCs (Mel-Frequency Cepstral Coefficients - timbral features)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    features = {
        'rms_mean': np.mean(rms),
        'rms_std': np.std(rms),
        'spec_cent_mean': np.mean(spec_cent),
        'spec_cent_std': np.std(spec_cent),
    }

    # Add MFCC features
    features.update({f'mfcc_mean_{i+1}': m for i, m in enumerate(mfccs_mean)})
    features.update({f'mfcc_std_{i+1}': s for i, s in enumerate(mfccs_std)})

    return features


def compute_snr(signal_data: np.ndarray, noise_start: int = 0,
                noise_duration: float = 0.5, sr: int = 16000) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) of audio signal.

    Args:
        signal_data: Audio signal
        noise_start: Start sample for noise estimation
        noise_duration: Duration of noise segment in seconds
        sr: Sample rate

    Returns:
        SNR in decibels
    """
    noise_samples = int(noise_duration * sr)
    noise_end = noise_start + noise_samples

    if noise_end >= len(signal_data):
        noise_end = len(signal_data) // 2
        noise_start = 0

    noise_segment = signal_data[noise_start:noise_end]
    signal_segment = signal_data[noise_end:]

    noise_power = np.mean(noise_segment ** 2)
    signal_power = np.mean(signal_segment ** 2)

    if noise_power == 0:
        return float('inf')

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def compute_spectral_rolloff(y: np.ndarray, sr: int = 16000,
                             roll_percent: float = 0.85) -> float:
    """
    Compute spectral rolloff frequency (frequency below which X% of energy is contained).

    Args:
        y: Audio signal
        sr: Sample rate
        roll_percent: Percentage of total spectral energy (default 85%)

    Returns:
        Mean spectral rolloff frequency in Hz
    """
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)
    return np.mean(rolloff)


# ========================================
# NOISE FILTERING FUNCTIONS
# ========================================

class NoiseFilter:
    """
    Noise filtering using high-pass, band-pass, and spectral subtraction.

    All filters apply identical coefficients to every channel so that
    inter-channel phase relationships are preserved for sound localization.
    Accepts 1D (single channel) or 2D (num_samples, num_channels) input
    as either torch tensors or numpy arrays.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 highpass_cutoff: int = 100,
                 bandpass_low: int = 100,
                 bandpass_high: int = 3000,
                 noise_reduce_strength: float = 0.3):
        """
        Initialize noise filter and pre-compute Butterworth coefficients.

        Args:
            sample_rate: Audio sample rate
            highpass_cutoff: High-pass filter cutoff frequency (Hz)
            bandpass_low: Band-pass filter low cutoff (Hz)
            bandpass_high: Band-pass filter high cutoff (Hz)
            noise_reduce_strength: Spectral subtraction strength (0-1)
        """
        self.sample_rate = sample_rate
        self.noise_reduce_strength = noise_reduce_strength

        # Pre-compute Butterworth coefficients once.
        # Using the same (b, a) across all channels preserves phase coherence.
        nyquist = sample_rate / 2.0
        self._hp_b, self._hp_a = signal.butter(5, highpass_cutoff / nyquist, btype='high')
        self._bp_b, self._bp_a = signal.butter(4, [bandpass_low / nyquist, bandpass_high / nyquist], btype='band')

    def _to_numpy(self, audio):
        """Convert tensor to numpy if needed."""
        return audio.numpy() if isinstance(audio, torch.Tensor) else audio

    def _apply_per_channel(self, audio_np: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Apply filtfilt with identical (b, a) coefficients to each channel.

        Args:
            audio_np: 1D or 2D numpy array
            b: Filter numerator coefficients
            a: Filter denominator coefficients

        Returns:
            Filtered array with same shape as input
        """
        if audio_np.ndim == 1:
            return signal.filtfilt(b, a, audio_np).copy()

        filtered = np.empty_like(audio_np)
        for ch in range(audio_np.shape[1]):
            filtered[:, ch] = signal.filtfilt(b, a, audio_np[:, ch])
        return filtered

    def apply_highpass_filter(self, audio):
        """
        Apply 5th-order Butterworth high-pass filter.

        Removes low-frequency rumble and background noise below cutoff frequency.

        Args:
            audio: 1D or 2D input (tensor or numpy)

        Returns:
            Filtered audio with same type and shape
        """
        audio_np = self._to_numpy(audio)
        filtered = self._apply_per_channel(audio_np, self._hp_b, self._hp_a)
        return torch.tensor(filtered, dtype=torch.float32) if isinstance(audio, torch.Tensor) else filtered

    def apply_bandpass_filter(self, audio):
        """
        Apply 4th-order Butterworth band-pass filter.

        Isolates baby cry frequency range (typically 100-3000 Hz).

        Args:
            audio: 1D or 2D input (tensor or numpy)

        Returns:
            Filtered audio with same type and shape
        """
        audio_np = self._to_numpy(audio)
        filtered = self._apply_per_channel(audio_np, self._bp_b, self._bp_a)
        return torch.tensor(filtered, dtype=torch.float32) if isinstance(audio, torch.Tensor) else filtered

    def _spectral_sub_single(self, waveform: np.ndarray, n_fft: int, hop: int, alpha: float) -> np.ndarray:
        """
        Spectral subtraction on a single channel with phase preservation.

        Noise floor is estimated from the first 10 STFT frames (typically the
        quietest region). Only the magnitude spectrum is modified; phase is
        kept intact to preserve inter-channel timing for localization.

        Args:
            waveform: 1D numpy array
            n_fft: FFT size
            hop: Hop length
            alpha: Subtraction strength

        Returns:
            Denoised 1D numpy array
        """
        f, t, Zxx = signal.stft(waveform, fs=self.sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Noise estimate from first 10 frames
        noise_frames = min(10, magnitude.shape[1])
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # Subtract with spectral floor (10% of original magnitude)
        subtracted = np.maximum(magnitude - alpha * noise_spectrum, 0.1 * magnitude)

        # Reconstruct preserving original phase
        Zxx_denoised = subtracted * np.exp(1j * phase)
        _, denoised = signal.istft(Zxx_denoised, fs=self.sample_rate, nperseg=n_fft, noverlap=n_fft - hop)

        return denoised

    def spectral_subtraction(self, audio, noise_profile=None):
        """
        Per-channel spectral subtraction with phase preservation.

        For multi-channel input, noise is estimated independently per channel.
        Output channels are length-aligned to the shortest istft result to
        maintain sample-aligned phase across channels.

        Args:
            audio: 1D or 2D audio (tensor or numpy)
            noise_profile: Accepted for API compatibility; unused in multi-channel mode

        Returns:
            Denoised audio with same type as input
        """
        audio_np = self._to_numpy(audio)
        n_fft, hop, alpha = 1024, 256, self.noise_reduce_strength

        if audio_np.ndim == 1:
            denoised = self._spectral_sub_single(audio_np, n_fft, hop, alpha)
        else:
            channels = [
                self._spectral_sub_single(audio_np[:, ch], n_fft, hop, alpha)
                for ch in range(audio_np.shape[1])
            ]
            # Align channels to shortest output (istft can differ by 1 sample)
            min_len = min(len(ch) for ch in channels)
            denoised = np.column_stack([ch[:min_len] for ch in channels])

        return torch.tensor(denoised, dtype=torch.float32) if isinstance(audio, torch.Tensor) else denoised

    def filter_audio(self, audio, use_spectral_sub: bool = True):
        """
        Full noise filtering pipeline: high-pass -> band-pass -> spectral subtraction.

        Accepts 1D (single channel) or 2D (num_samples, num_channels) input
        as either torch tensors or numpy arrays.

        Args:
            audio: Input audio
            use_spectral_sub: Whether to apply spectral subtraction

        Returns:
            Filtered audio with same type and shape as input
        """
        filtered = self.apply_highpass_filter(audio)
        filtered = self.apply_bandpass_filter(filtered)
        if use_spectral_sub:
            filtered = self.spectral_subtraction(filtered)
        return filtered


class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD) for cry detection and segmentation.
    Detects and segments baby cry sounds from silent/background periods.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 frame_length: int = 400,  # 25ms at 16kHz
                 hop_length: int = 160,    # 10ms at 16kHz
                 energy_threshold: float = 0.01,
                 freq_min: int = 200,       # Baby cry range starts around 200 Hz
                 freq_max: int = 1000):     # Baby cry harmonics up to ~1000 Hz
        """
        Initialize VAD with baby cry-specific parameters.

        Args:
            sample_rate: Audio sample rate
            frame_length: Frame length in samples (25ms default)
            hop_length: Hop length in samples (10ms default)
            energy_threshold: Energy threshold for activity detection
            freq_min: Minimum frequency for baby cry detection
            freq_max: Maximum frequency for baby cry detection
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.freq_min = freq_min
        self.freq_max = freq_max

    def compute_energy(self, waveform: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Compute short-time energy of the waveform.

        Args:
            waveform: Input audio waveform

        Returns:
            Energy values per frame
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # Compute energy in overlapping frames
        frames = librosa.util.frame(waveform_np,
                                    frame_length=self.frame_length,
                                    hop_length=self.hop_length)
        energy = np.sum(frames ** 2, axis=0) / self.frame_length

        return energy

    def compute_zero_crossing_rate(self, waveform: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Compute zero-crossing rate (useful for distinguishing voiced/unvoiced).

        Args:
            waveform: Input audio waveform

        Returns:
            Zero-crossing rate per frame
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        zcr = librosa.feature.zero_crossing_rate(
            waveform_np,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )

        return zcr[0]

    def compute_spectral_energy_in_band(self, waveform: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Compute energy in baby cry frequency band (200-1000 Hz).

        Args:
            waveform: Input audio waveform

        Returns:
            Energy in cry band per frame
        """
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        # Compute STFT
        stft = librosa.stft(waveform_np,
                           n_fft=self.frame_length * 2,
                           hop_length=self.hop_length)

        # Get frequencies
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length * 2)

        # Find indices in cry frequency band
        band_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)

        # Compute energy in band
        band_energy = np.sum(np.abs(stft[band_mask, :]) ** 2, axis=0)

        return band_energy

    def detect_activity(self, waveform: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect voice activity in the waveform using multiple features.

        Combines energy, zero-crossing rate, and cry-band energy to detect
        baby cry activity with confidence scores.

        Args:
            waveform: Input audio waveform

        Returns:
            Tuple of (activity_mask, confidence_scores)
        """
        # Compute multiple features
        energy = self.compute_energy(waveform)
        zcr = self.compute_zero_crossing_rate(waveform)
        band_energy = self.compute_spectral_energy_in_band(waveform)

        # Find minimum length to align all features
        min_len = min(len(energy), len(zcr), len(band_energy))

        # Truncate all features to same length
        energy = energy[:min_len]
        zcr = zcr[:min_len]
        band_energy = band_energy[:min_len]

        # Normalize features
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        zcr_norm = (zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-8)
        band_energy_norm = (band_energy - band_energy.min()) / (band_energy.max() - band_energy.min() + 1e-8)

        # Combine features (weighted)
        # High energy + moderate ZCR + high band energy = cry
        confidence = 0.4 * energy_norm + 0.2 * (1 - zcr_norm) + 0.4 * band_energy_norm

        # Threshold
        activity_mask = confidence > self.energy_threshold

        # Apply morphological operations to remove small gaps
        activity_mask = self._smooth_mask(activity_mask, min_duration_frames=5)

        return activity_mask, confidence

    def _smooth_mask(self, mask: np.ndarray, min_duration_frames: int = 5) -> np.ndarray:
        """
        Smooth activity mask to remove short gaps and spurious detections.

        Args:
            mask: Binary activity mask
            min_duration_frames: Minimum duration for activity segments

        Returns:
            Smoothed mask
        """
        from scipy.ndimage import binary_closing, binary_opening

        # Close small gaps
        mask = binary_closing(mask, structure=np.ones(min_duration_frames))

        # Remove small detections
        mask = binary_opening(mask, structure=np.ones(min_duration_frames))

        return mask

    def segment_audio(self, waveform: Union[torch.Tensor, np.ndarray]) -> list:
        """
        Segment audio into cry and non-cry regions.

        Args:
            waveform: Input audio waveform

        Returns:
            List of (start_sample, end_sample, is_cry) tuples
        """
        activity_mask, _ = self.detect_activity(waveform)

        # Convert frame indices to sample indices
        segments = []
        in_segment = False
        start_frame = 0

        for i, is_active in enumerate(activity_mask):
            if is_active and not in_segment:
                start_frame = i
                in_segment = True
            elif not is_active and in_segment:
                start_sample = start_frame * self.hop_length
                end_sample = i * self.hop_length
                segments.append((start_sample, end_sample, True))
                in_segment = False

        # Handle last segment
        if in_segment:
            start_sample = start_frame * self.hop_length
            end_sample = len(waveform)
            segments.append((start_sample, end_sample, True))

        return segments


# ========================================
# USAGE EXAMPLES
# ========================================

if __name__ == "__main__":
    # Example 1: Extract distance features from audio
    print("Example 1: Sound Characterization")
    y, sr = librosa.load('example.wav', sr=48000, mono=True)
    features = extract_distance_features(y, sr)
    print(f"RMS Mean: {features['rms_mean']:.4f}")
    print(f"Spectral Centroid: {features['spec_cent_mean']:.2f} Hz")
    print(f"First MFCC: {features['mfcc_mean_1']:.4f}")

    # Example 2: Apply noise filtering
    print("\nExample 2: Noise Filtering")
    y_noisy, sr = librosa.load('noisy_audio.wav', sr=16000, mono=False)
    noise_filter = NoiseFilter(sample_rate=sr)
    y_filtered = noise_filter.filter_audio(y_noisy)
    print(f"Input shape: {y_noisy.shape}, Output shape: {y_filtered.shape}")

    # Example 3: Voice activity detection
    print("\nExample 3: VAD")
    vad = VoiceActivityDetector(sample_rate=sr)
    activity_mask, confidence = vad.detect_activity(torch.tensor(y_noisy[:, 0]))
    segments = vad.segment_audio(torch.tensor(y_noisy[:, 0]))
    print(f"Detected {len(segments)} cry segments")
    for i, (start, end, is_cry) in enumerate(segments):
        print(f"  Segment {i+1}: {start/sr:.2f}s - {end/sr:.2f}s")
