"""
Spectral filtering and frequency domain processing.

Optimized for Raspberry Pi 5 with phase preservation for sound localization.
"""

import torch
import numpy as np
from typing import Tuple


class SpectralFilters:
    """
    Spectral filtering operations for baby cry isolation.

    All methods preserve phase relationships for multi-channel sound localization.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 160):
        """
        Initialize spectral filters.

        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Baby cry frequency characteristics (Hz)
        self.cry_freq_min = 100    # Fundamental frequency minimum
        self.cry_freq_max = 3000   # Harmonic content maximum
        self.cry_peak_freq = 400   # Typical crying peak frequency

    def bandpass_filter(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain bandpass filtering to emphasize baby cry frequencies.

        Uses smooth transitions to minimize artifacts while preserving phase.

        Args:
            audio: Input audio tensor (mono or single channel)

        Returns:
            Filtered audio tensor with same shape
        """
        # Compute STFT (preserves phase)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )

        # Create frequency mask for baby cry range
        freqs = torch.fft.fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2 + 1]

        # Frequency-based filter with smooth transitions (better audio quality)
        freq_mask = torch.ones_like(freqs) * 0.3  # Keep some background for naturalness

        # Smooth bandpass filter for cry frequencies
        cry_band = (freqs >= self.cry_freq_min) & (freqs <= self.cry_freq_max)
        freq_mask[cry_band] = 1.0

        # Gentle emphasis on typical cry frequencies (avoid over-amplification)
        peak_band = (freqs >= 300) & (freqs <= 600)
        freq_mask[peak_band] = 1.3  # Reduced from 2.0 to avoid distortion

        # Apply smooth roll-off at edges (reduces artifacts)
        transition_width = 50  # Hz
        for i, f in enumerate(freqs):
            if self.cry_freq_min - transition_width < f < self.cry_freq_min:
                # Smooth transition at low end
                alpha = (f - (self.cry_freq_min - transition_width)) / transition_width
                freq_mask[i] = 0.3 + alpha * 0.7
            elif self.cry_freq_max < f < self.cry_freq_max + transition_width:
                # Smooth transition at high end
                alpha = 1 - (f - self.cry_freq_max) / transition_width
                freq_mask[i] = 0.3 + alpha * 0.7

        # Apply filter (preserves phase)
        stft_filtered = stft * freq_mask.unsqueeze(-1)

        # Reconstruct audio (phase-preserving inverse STFT)
        filtered_audio = torch.istft(
            stft_filtered,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            length=audio.shape[-1]
        )

        return filtered_audio

    def analyze_energy_distribution(self, audio: torch.Tensor,
                                   frame_length: int = 2048,
                                   hop_length: int = 512) -> torch.Tensor:
        """
        Analyze energy distribution to detect high concentration in 300-600 Hz.

        Args:
            audio: Input audio tensor
            frame_length: Frame size
            hop_length: Hop length

        Returns:
            Energy concentration scores per frame
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft) ** 2  # Power spectrum

        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]

        # Define frequency bands
        cry_band = (freqs >= 300) & (freqs <= 600)
        total_band = (freqs >= 100) & (freqs <= 4000)

        concentration_scores = []

        for frame_idx in range(magnitude.shape[1]):
            frame_power = magnitude[:, frame_idx]

            cry_energy = frame_power[cry_band].sum().item()
            total_energy = frame_power[total_band].sum().item()

            if total_energy > 1e-8:
                # Ratio of energy in cry band vs total
                concentration = cry_energy / total_energy
                # Baby cries should have >30% of energy in 300-600 Hz
                score = min(1.0, concentration / 0.4)
            else:
                score = 0.0

            concentration_scores.append(score)

        return torch.tensor(concentration_scores)

    def filter_adult_speech(self, audio: torch.Tensor,
                          frame_length: int = 2048,
                          hop_length: int = 512) -> torch.Tensor:
        """
        Detect and filter out adult speech based on lower F0 (80-250 Hz).
        Adult speech has fundamentally different pitch range than baby cries.

        Args:
            audio: Input audio tensor
            frame_length: Frame size
            hop_length: Hop length

        Returns:
            Confidence scores (0=adult speech, 1=not adult speech) per frame
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

        # Define frequency bands
        adult_speech_band = (freqs >= 80) & (freqs <= 250)  # Adult F0 range
        baby_cry_band = (freqs >= 300) & (freqs <= 600)     # Baby cry F0 range

        rejection_scores = []

        for frame_idx in range(magnitude.shape[1]):
            frame_power = magnitude[:, frame_idx] ** 2

            adult_energy = frame_power[adult_speech_band].sum().item()
            baby_energy = frame_power[baby_cry_band].sum().item()
            total_energy = frame_power.sum().item()

            if total_energy > 1e-8:
                adult_ratio = adult_energy / total_energy
                baby_ratio = baby_energy / total_energy

                # If more energy in adult range than baby range, likely adult speech
                if adult_ratio > baby_ratio and adult_ratio > 0.2:
                    # Strong adult speech indicator
                    score = max(0.0, 1.0 - (adult_ratio / 0.4))
                else:
                    score = 1.0
            else:
                score = 1.0

            rejection_scores.append(score)

        return torch.tensor(rejection_scores)

    def filter_music(self, audio: torch.Tensor,
                   frame_length: int = 2048,
                   hop_length: int = 512,
                   stability_window: int = 20) -> torch.Tensor:
        """
        Detect and filter out music based on stable pitch patterns.
        Music has more stable pitch compared to cry's varying pitch contours.

        Args:
            audio: Input audio tensor
            frame_length: Frame size
            hop_length: Hop length
            stability_window: Number of frames to analyze for stability

        Returns:
            Confidence scores (0=music, 1=not music) per frame
        """
        # Track pitch over time using autocorrelation
        pitch_track = []

        for start_idx in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start_idx:start_idx + frame_length]

            # Autocorrelation for pitch detection
            autocorr = torch.nn.functional.conv1d(
                frame.unsqueeze(0).unsqueeze(0),
                frame.flip(0).unsqueeze(0).unsqueeze(0),
                padding=frame_length - 1
            )[0, 0, frame_length-1:]

            # Search for pitch in wide range (100-2000 Hz)
            min_lag = int(self.sample_rate / 2000)
            max_lag = int(self.sample_rate / 100)

            if max_lag < len(autocorr):
                autocorr_range = autocorr[min_lag:max_lag]
                if autocorr_range.max() > 0:
                    peak_lag = min_lag + torch.argmax(autocorr_range).item()
                    pitch = self.sample_rate / peak_lag
                else:
                    pitch = 0.0
            else:
                pitch = 0.0

            pitch_track.append(pitch)

        pitch_track = torch.tensor(pitch_track)

        # Analyze pitch stability
        rejection_scores = []

        for i in range(len(pitch_track)):
            start_win = max(0, i - stability_window // 2)
            end_win = min(len(pitch_track), i + stability_window // 2)
            window = pitch_track[start_win:end_win]

            valid_pitches = window[window > 0]

            if len(valid_pitches) > stability_window // 2:
                # Compute pitch stability (coefficient of variation)
                pitch_mean = valid_pitches.mean().item()
                pitch_std = valid_pitches.std().item()

                if pitch_mean > 0:
                    cv = pitch_std / pitch_mean  # Coefficient of variation

                    # Music has very stable pitch (low CV < 0.05)
                    # Baby cries have varying pitch (CV > 0.1)
                    if cv < 0.05:
                        # Very stable = likely music
                        score = 0.2
                    elif cv < 0.1:
                        # Somewhat stable = possibly music
                        score = 0.5
                    else:
                        # Varying pitch = not music
                        score = 1.0
                else:
                    score = 1.0
            else:
                score = 1.0

            rejection_scores.append(score)

        return torch.tensor(rejection_scores)

    def filter_environmental_sounds(self, audio: torch.Tensor,
                                   frame_length: int = 2048,
                                   hop_length: int = 512) -> torch.Tensor:
        """
        Detect and filter out environmental sounds based on lack of harmonic structure.
        Environmental sounds (fan, traffic, white noise) lack clear harmonics.

        Args:
            audio: Input audio tensor
            frame_length: Frame size
            hop_length: Hop length

        Returns:
            Confidence scores (0=environmental, 1=not environmental) per frame
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

        rejection_scores = []

        for frame_idx in range(magnitude.shape[1]):
            frame_mag = magnitude[:, frame_idx]

            # Measure spectral flatness (Wiener entropy)
            # High flatness = noise-like = environmental sound
            # Low flatness = tonal/harmonic = voice/cry
            geometric_mean = torch.exp(torch.mean(torch.log(frame_mag + 1e-10)))
            arithmetic_mean = torch.mean(frame_mag)

            if arithmetic_mean > 1e-8:
                spectral_flatness = geometric_mean / arithmetic_mean

                # Also check for harmonic structure
                # Look for peaks in spectrum (harmonics create peaks)
                # Smooth the spectrum and find peaks
                smoothed = torch.nn.functional.avg_pool1d(
                    frame_mag.unsqueeze(0).unsqueeze(0),
                    kernel_size=5,
                    stride=1,
                    padding=2
                ).squeeze()

                # Count significant peaks
                is_peak = (frame_mag[1:-1] > frame_mag[:-2]) & (frame_mag[1:-1] > frame_mag[2:])
                peak_heights = frame_mag[1:-1][is_peak]
                significant_peaks = (peak_heights > 0.3 * frame_mag.max()).sum().item()

                # Environmental sounds: high flatness (>0.5) and few peaks
                # Harmonic sounds (cries): low flatness (<0.3) and multiple peaks
                if spectral_flatness > 0.5 and significant_peaks < 3:
                    # Likely environmental noise
                    score = 0.3
                elif spectral_flatness < 0.3 and significant_peaks >= 4:
                    # Likely harmonic (cry or voice)
                    score = 1.0
                else:
                    # Ambiguous
                    score = 0.6
            else:
                score = 1.0

            rejection_scores.append(score)

        return torch.tensor(rejection_scores)
