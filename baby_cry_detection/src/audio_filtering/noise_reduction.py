"""
Noise reduction and voice activity detection.

Includes spectral subtraction and VAD for baby cry isolation.
"""

import torch
import numpy as np
from scipy.ndimage import median_filter


class NoiseReducer:
    """
    Noise reduction using spectral subtraction and voice activity detection.

    Optimized for Raspberry Pi 5 performance.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 160):
        """
        Initialize noise reducer.

        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def spectral_subtraction(self, audio: torch.Tensor,
                           noise_duration: float = 0.5) -> torch.Tensor:
        """
        Apply spectral subtraction to reduce background noise.

        Uses gentle noise reduction to preserve audio quality.

        Args:
            audio: Input audio tensor
            noise_duration: Duration of noise estimation period (seconds)

        Returns:
            Denoised audio tensor
        """
        # Estimate noise from beginning of audio
        noise_samples = int(noise_duration * self.sample_rate)
        noise_segment = audio[:noise_samples] if len(audio) > noise_samples else audio[:len(audio)//4]

        # Compute STFT of full audio and noise
        stft_audio = torch.stft(audio, n_fft=self.n_fft,
                               hop_length=self.hop_length, return_complex=True)
        stft_noise = torch.stft(noise_segment, n_fft=self.n_fft,
                               hop_length=self.hop_length, return_complex=True)

        # Estimate noise power spectrum
        noise_power = torch.mean(torch.abs(stft_noise) ** 2, dim=-1, keepdim=True)

        # Apply spectral subtraction
        audio_magnitude = torch.abs(stft_audio)
        audio_phase = torch.angle(stft_audio)

        # Subtract noise estimate (gentler for better audio quality)
        alpha = 1.0  # Reduced from 2.0 - less aggressive, better quality
        clean_magnitude = audio_magnitude - alpha * torch.sqrt(noise_power)

        # Apply higher spectral floor to preserve more audio content
        spectral_floor = 0.3 * audio_magnitude  # Increased from 0.1
        clean_magnitude = torch.maximum(clean_magnitude, spectral_floor)

        # Reconstruct complex spectrum
        clean_stft = clean_magnitude * torch.exp(1j * audio_phase)

        # Inverse STFT
        clean_audio = torch.istft(clean_stft, n_fft=self.n_fft,
                                 hop_length=self.hop_length, length=len(audio))

        return clean_audio

    def voice_activity_detection(self, audio: torch.Tensor,
                                frame_length: int = 1024,
                                threshold: float = 0.01) -> torch.Tensor:
        """
        Detect voice activity using energy and spectral features.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            threshold: Energy threshold for voice detection

        Returns:
            Binary mask indicating voice activity
        """
        # Frame the audio
        frames = audio.unfold(-1, frame_length, frame_length // 2)

        # Energy-based detection
        energy = torch.mean(frames ** 2, dim=-1)
        energy_thresh = torch.quantile(energy, 0.3) + threshold
        energy_mask = energy > energy_thresh

        # Spectral centroid for voice characteristics
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=frame_length // 2,
            return_complex=True
        )

        magnitude = torch.abs(stft)
        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]

        # Compute spectral centroid
        spectral_centroid = torch.sum(magnitude * freqs.unsqueeze(-1), dim=0) / (torch.sum(magnitude, dim=0) + 1e-8)

        # Voice typically has centroid in 200-2000 Hz range
        centroid_mask = (spectral_centroid >= 200) & (spectral_centroid <= 2000)

        # Ensure masks have the same length by taking minimum
        min_length = min(len(energy_mask), len(centroid_mask))
        energy_mask = energy_mask[:min_length]
        centroid_mask = centroid_mask[:min_length]

        # Combine masks
        voice_mask = energy_mask & centroid_mask

        # Smooth the mask
        voice_mask_smooth = median_filter(voice_mask.numpy().astype(float), size=5) > 0.5

        return torch.from_numpy(voice_mask_smooth)
