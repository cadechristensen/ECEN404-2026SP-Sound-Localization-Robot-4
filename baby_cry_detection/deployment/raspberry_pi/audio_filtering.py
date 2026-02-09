"""
Audio filtering for baby cry detection on Raspberry Pi.

Implements VAD-gated noise filtering with phase-preserving 4-channel support.
All filters use identical coefficients per channel so inter-channel phase
relationships (required for sound localization) are never disturbed.

Pipeline:
    1. Voice Activity Detection  - energy-based, best channel by cry-band SNR for multi-channel
    2. High-pass filter          - Butterworth, removes rumble below cutoff
    3. Band-pass filter          - Butterworth, isolates cry frequency range
    4. Spectral subtraction      - per-channel noise estimation, phase preserved
"""

import torch
import numpy as np
import librosa
from scipy import signal
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from .config_pi import Config
except ImportError:
    try:
        from config_pi import Config  # type: ignore
    except ImportError:
        try:
            from .config import Config
        except ImportError:
            from config import Config  # type: ignore

try:
    from .multichannel_detector import EnhancedSNRComputation
except ImportError:
    from multichannel_detector import EnhancedSNRComputation  # type: ignore


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

    def compute_energy(self, waveform: torch.Tensor) -> np.ndarray:
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

    def compute_zero_crossing_rate(self, waveform: torch.Tensor) -> np.ndarray:
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

    def compute_spectral_energy_in_band(self, waveform: torch.Tensor) -> np.ndarray:
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

    def detect_activity(self, waveform: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect voice activity in the waveform using multiple features.

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

    def segment_audio(self, waveform: torch.Tensor) -> list:
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
        kept intact.

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

    # Wiener filter omitted: it introduces phase distortion incompatible with
    # inter-channel time-of-arrival estimation for sound localization.

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


class AudioFilteringPipeline:
    """
    VAD-gated noise filtering pipeline.

    For multi-channel input, the channel with the highest cry-band SNR is
    selected as the VAD activity gate; noise filtering applies to all
    channels with phase preserved.
    """

    def __init__(self, config: Config = Config()):
        """
        Initialize pipeline from config.

        Args:
            config: Configuration object (Config or ConfigPi)
        """
        self.config = config

        self.vad = VoiceActivityDetector(
            sample_rate=config.SAMPLE_RATE,
            freq_min=config.VAD_FREQ_MIN if hasattr(config, 'VAD_FREQ_MIN') else 200,
            freq_max=config.VAD_FREQ_MAX if hasattr(config, 'VAD_FREQ_MAX') else 1000
        )

        self.noise_filter = NoiseFilter(
            sample_rate=config.SAMPLE_RATE,
            highpass_cutoff=config.HIGHPASS_CUTOFF,
            bandpass_low=config.BANDPASS_LOW,
            bandpass_high=config.BANDPASS_HIGH,
            noise_reduce_strength=config.NOISE_REDUCE_STRENGTH
        )

        self.snr_computer = EnhancedSNRComputation(sample_rate=config.SAMPLE_RATE)

    def preprocess_audio(self,
                        waveform,
                        apply_vad: bool = False,
                        apply_filtering: bool = True) -> dict:
        """
        Preprocessing with optional VAD segmentation and noise filtering.

        For 2D input (num_samples, num_channels):
        - VAD runs on the channel with the highest cry-band SNR (activity gating)
        - Noise filtering applies to all channels, phase preserved

        Args:
            waveform: 1D or 2D audio (tensor or numpy)
            apply_vad: Whether to run VAD segmentation
            apply_filtering: Whether to apply noise filtering

        Returns:
            Dictionary with 'original', 'filtered', and optional VAD outputs
        """
        results = {'original': waveform}

        # Select best channel by cry-band SNR for VAD activity gating
        if isinstance(waveform, torch.Tensor):
            if waveform.ndim == 2:
                best_ch = int(np.argmax(self.snr_computer.compute_snr_all_channels(waveform.numpy())))
                vad_input = waveform[:, best_ch]
            else:
                vad_input = waveform
        elif isinstance(waveform, np.ndarray):
            if waveform.ndim == 2:
                best_ch = int(np.argmax(self.snr_computer.compute_snr_all_channels(waveform)))
                vad_input = torch.from_numpy(waveform[:, best_ch].copy()).float()
            else:
                vad_input = torch.from_numpy(waveform.copy()).float()
        else:
            vad_input = waveform

        if apply_vad:
            activity_mask, confidence = self.vad.detect_activity(vad_input)
            segments = self.vad.segment_audio(vad_input)
            results['vad_mask'] = activity_mask
            results['vad_confidence'] = confidence
            results['vad_segments'] = segments

        # Noise filtering on full input (all channels if multi-channel)
        if apply_filtering:
            results['filtered'] = self.noise_filter.filter_audio(waveform)
        else:
            results['filtered'] = waveform

        return results
