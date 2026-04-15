"""
Multi-channel audio processing with phase preservation for sound localization.

CRITICAL: All processing preserves phase relationships across 4 channels for sound localization.
Designed for Raspberry Pi 5 8GB deployment.
"""

import torch
import numpy as np
import torchaudio.transforms as T
from scipy import signal as scipy_signal
from typing import Tuple, List


class MultichannelProcessor:
    """
    Multi-channel (4-channel) audio processor with phase preservation.

    All filtering operations maintain phase coherence across channels
    to enable accurate sound localization using microphone array data.
    """

    def __init__(self, sample_rate: int = 16000, num_channels: int = 4):
        """
        Initialize multichannel processor.

        Args:
            sample_rate: Audio sample rate in Hz
            num_channels: Number of audio channels (default: 4 for mic array)
        """
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        # Cached resamplers keyed by (orig_rate, target_rate).
        # T.Resample pre-computes a polyphase filter kernel on construction;
        # recreating it on every call wastes CPU recomputing the same kernel.
        self._resamplers: dict = {}

    def select_best_channel_by_snr(
        self,
        audio: torch.Tensor,
        signal_band: Tuple[int, int] = (300, 900),
        noise_bands: List[Tuple[int, int]] = [(50, 200), (1500, 4000)]
    ) -> Tuple[int, np.ndarray]:
        """
        Select the channel with the highest cry-band SNR.

        Signal band: 300-900 Hz covers the infant cry fundamental and first
        harmonic.  Noise bands: 50-200 Hz (low-frequency rumble) and
        1500-4000 Hz (environmental noise).  Uses the same band definitions
        as EnhancedSNRComputation in the real-time detector.

        Args:
            audio: Multi-channel audio tensor (num_samples, num_channels)
            signal_band: (low, high) Hz for cry signal band
            noise_bands: List of (low, high) Hz tuples for noise reference bands

        Returns:
            Tuple of (best_channel_index, snr_scores_per_channel)
        """
        if audio.dim() == 1:
            return 0, np.array([0.0])

        audio_np = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else audio
        num_channels = audio_np.shape[1]
        snr_scores = np.zeros(num_channels)

        for ch in range(num_channels):
            f, _, Zxx = scipy_signal.stft(audio_np[:, ch], fs=self.sample_rate, nperseg=1024)
            magnitude = np.abs(Zxx)

            signal_mask = (f >= signal_band[0]) & (f <= signal_band[1])
            signal_energy = np.mean(magnitude[signal_mask, :] ** 2)

            noise_energy = 0.0
            for low, high in noise_bands:
                noise_mask = (f >= low) & (f <= high)
                noise_energy += np.mean(magnitude[noise_mask, :] ** 2)
            noise_energy /= len(noise_bands)

            snr_scores[ch] = 10 * np.log10((signal_energy + 1e-10) / (noise_energy + 1e-10))

        return int(np.argmax(snr_scores)), snr_scores

    def extract_cry_segments_multichannel(self, audio: torch.Tensor,
                                         cry_segments: List[Tuple[float, float]],
                                         merge_segments_fn) -> torch.Tensor:
        """
        Extract and concatenate cry segments from multi-channel audio.

        CRITICAL: Preserves all channels and phase relationships.

        Args:
            audio: Multi-channel input audio tensor (num_samples, num_channels)
            cry_segments: List of (start_time, end_time) tuples in seconds
            merge_segments_fn: Function to merge overlapping segments
                (injected to avoid a circular import with utils.py)

        Returns:
            Concatenated cry-only multi-channel audio tensor
            Shape: (num_samples, num_channels)
        """
        if len(cry_segments) == 0:
            return torch.zeros((0, audio.shape[1]), dtype=audio.dtype)

        # Merge overlapping segments to avoid extracting duplicate audio
        merged_segments = merge_segments_fn(cry_segments)

        extracted_segments = []

        for start_time, end_time in merged_segments:
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)

            start_idx = max(0, start_idx)
            end_idx = min(len(audio), end_idx)

            if start_idx < end_idx:
                # Extract all channels, preserving phase
                segment = audio[start_idx:end_idx, :]
                extracted_segments.append(segment)

        if not extracted_segments:
            return torch.zeros((0, audio.shape[1]), dtype=audio.dtype)

        # Concatenate segments while preserving all channels
        concatenated = torch.cat(extracted_segments, dim=0)

        return concatenated

    def apply_mask_multichannel(self, audio: torch.Tensor,
                               cry_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply cry detection mask to all channels while preserving phase.

        Args:
            audio: Multi-channel audio tensor (num_samples, num_channels)
            cry_mask: Boolean mask indicating cry regions (num_samples,)

        Returns:
            Masked multi-channel audio tensor with same shape
        """
        num_channels = audio.shape[1] if audio.dim() > 1 else 1
        isolated_audio = torch.zeros_like(audio)

        for ch in range(num_channels):
            # Apply same mask to each channel (preserves phase relationships)
            isolated_audio[cry_mask, ch] = audio[cry_mask, ch]

        return isolated_audio

    def resample_multichannel(self, audio: torch.Tensor,
                             orig_sample_rate: int,
                             target_sample_rate: int) -> torch.Tensor:
        """
        Resample multi-channel audio while preserving phase relationships.

        Args:
            audio: Multi-channel audio tensor (num_samples, num_channels)
            orig_sample_rate: Original sample rate
            target_sample_rate: Target sample rate

        Returns:
            Resampled multi-channel audio tensor
        """
        if orig_sample_rate == target_sample_rate:
            return audio

        key = (orig_sample_rate, target_sample_rate)
        if key not in self._resamplers:
            self._resamplers[key] = T.Resample(orig_sample_rate, target_sample_rate)
        resampler = self._resamplers[key]

        if audio.dim() > 1:
            # Multi-channel resample - process each channel separately
            resampled_channels = []
            for ch in range(audio.shape[1]):
                resampled_channels.append(resampler(audio[:, ch]))
            return torch.stack(resampled_channels, dim=1)
        else:
            # Single channel
            return resampler(audio)

    def save_multichannel(self, audio: torch.Tensor, filepath: str,
                         sample_rate: int):
        """
        Save multi-channel audio preserving all channels.

        Args:
            audio: Multi-channel audio tensor (num_samples, num_channels)
            filepath: Output file path
            sample_rate: Sample rate for output file
        """
        import torchaudio

        if audio.dim() > 1:
            # Transpose to (num_channels, num_samples) for torchaudio
            torchaudio.save(filepath, audio.transpose(0, 1), sample_rate)
        else:
            # Single channel
            torchaudio.save(filepath, audio.unsqueeze(0), sample_rate)

    def validate_multichannel_audio(self, audio: torch.Tensor) -> Tuple[bool, str]:
        """
        Validate multi-channel audio format.

        Args:
            audio: Input audio tensor

        Returns:
            Tuple of (is_valid, message)
        """
        if audio.dim() == 1:
            return True, "Single channel audio"

        if audio.dim() != 2:
            return False, f"Invalid audio dimensions: {audio.shape}. Expected (samples, channels)"

        num_channels = audio.shape[1]
        if num_channels != self.num_channels:
            return False, f"Expected {self.num_channels} channels, got {num_channels}"

        return True, f"Valid multi-channel audio: {num_channels} channels"
