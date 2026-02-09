"""
Acoustic feature detection for baby cry identification.

Includes harmonic structure, pitch contours, temporal patterns, and frequency modulation detection.
"""

import torch
import numpy as np
from typing import Tuple


class AcousticDetector:
    """
    Acoustic feature detection for baby cry characteristics.

    Analyzes harmonics, pitch contours, temporal patterns, and frequency modulation.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize acoustic detector.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def detect_harmonic_structure(self, audio: torch.Tensor,
                                 frame_length: int = 2048,
                                 hop_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect harmonic structure characteristic of baby cries.
        Baby cries have clear fundamental frequency + harmonic overtones.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Tuple of (harmonic_confidence_per_frame, fundamental_frequencies)
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)

        # Frequency bins
        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]
        max_freq = freqs.max().item()

        # Find fundamental frequency in baby cry range (300-600 Hz)
        cry_f0_min_idx = torch.argmin(torch.abs(freqs - 300))
        cry_f0_max_idx = torch.argmin(torch.abs(freqs - 600))

        # For each time frame, find peaks
        harmonic_scores = []
        f0_estimates = []

        for frame_idx in range(magnitude.shape[1]):
            frame_mag = magnitude[:, frame_idx]

            # Find fundamental frequency (strongest peak in 300-600 Hz)
            cry_range_mag = frame_mag[cry_f0_min_idx:cry_f0_max_idx]
            if cry_range_mag.max() < 1e-6:
                harmonic_scores.append(0.0)
                f0_estimates.append(0.0)
                continue

            local_peak_idx = torch.argmax(cry_range_mag)
            f0_idx = cry_f0_min_idx + local_peak_idx
            f0 = freqs[f0_idx].item()

            # Check for harmonics at 2*f0, 3*f0, 4*f0
            harmonic_strength = 0.0
            num_harmonics_found = 0

            for harmonic_num in [2, 3, 4]:
                expected_freq = f0 * harmonic_num
                if expected_freq > max_freq:
                    break

                # Find nearest frequency bin
                harmonic_idx = torch.argmin(torch.abs(freqs - expected_freq))

                # Check if there's a peak within +/-50 Hz tolerance
                tolerance_bins = int(50 / (self.sample_rate / frame_length))
                start_idx = max(0, harmonic_idx - tolerance_bins)
                end_idx = min(len(frame_mag), harmonic_idx + tolerance_bins)

                local_max = torch.max(frame_mag[start_idx:end_idx])
                f0_magnitude = frame_mag[f0_idx]

                # Harmonic should be weaker than fundamental but still present
                if local_max > 0.1 * f0_magnitude:
                    harmonic_strength += local_max / f0_magnitude
                    num_harmonics_found += 1

            # Score based on number and strength of harmonics
            if num_harmonics_found >= 2:
                score = min(1.0, harmonic_strength / 3.0)
            else:
                score = 0.0

            harmonic_scores.append(score)
            f0_estimates.append(f0)

        return torch.tensor(harmonic_scores), torch.tensor(f0_estimates)

    def detect_temporal_patterns(self, audio: torch.Tensor,
                                cry_burst_min: float = 0.3,
                                cry_burst_max: float = 2.0,
                                pause_min: float = 0.1,
                                pause_max: float = 0.8) -> torch.Tensor:
        """
        Detect temporal patterns: cry bursts followed by inhalation pauses.
        Baby cries have characteristic rhythm of vocalization + brief silence.

        Args:
            audio: Input audio tensor
            cry_burst_min: Minimum cry burst duration (seconds)
            cry_burst_max: Maximum cry burst duration (seconds)
            pause_min: Minimum pause duration (seconds)
            pause_max: Maximum pause duration (seconds)

        Returns:
            Temporal pattern confidence score per frame
        """
        # Compute short-term energy
        frame_length = int(0.05 * self.sample_rate)  # 50ms frames
        hop_length = int(0.025 * self.sample_rate)   # 25ms hop

        frames = audio.unfold(-1, frame_length, hop_length)
        energy = torch.mean(frames ** 2, dim=-1)

        # Normalize energy
        energy = (energy - energy.mean()) / (energy.std() + 1e-8)

        # Detect active vs pause segments
        energy_threshold = 0.0  # Above mean
        is_active = energy > energy_threshold

        # Find transitions
        transitions = torch.diff(is_active.float())
        burst_starts = torch.where(transitions > 0)[0]
        burst_ends = torch.where(transitions < 0)[0]

        # Align starts and ends
        if len(burst_starts) == 0 or len(burst_ends) == 0:
            return torch.zeros(len(energy))

        if burst_ends[0] < burst_starts[0]:
            burst_ends = burst_ends[1:]
        if len(burst_starts) > len(burst_ends):
            burst_starts = burst_starts[:len(burst_ends)]

        # Analyze burst durations
        pattern_scores = torch.zeros(len(energy))

        for i in range(len(burst_starts) - 1):
            burst_duration = (burst_ends[i] - burst_starts[i]) * hop_length / self.sample_rate
            pause_duration = (burst_starts[i+1] - burst_ends[i]) * hop_length / self.sample_rate

            # Check if matches cry pattern
            burst_match = (cry_burst_min <= burst_duration <= cry_burst_max)
            pause_match = (pause_min <= pause_duration <= pause_max)

            if burst_match and pause_match:
                # Mark this region with high score
                pattern_scores[burst_starts[i]:burst_ends[i]] = 1.0

        return pattern_scores

    def track_pitch_contours(self, audio: torch.Tensor,
                           frame_length: int = 2048,
                           hop_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track pitch contours to identify rising/falling patterns unique to infant distress.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length between frames

        Returns:
            Tuple of (contour_scores, pitch_track)
        """
        # Use autocorrelation for pitch tracking
        pitch_track = []

        for start_idx in range(0, len(audio) - frame_length, hop_length):
            frame = audio[start_idx:start_idx + frame_length]

            # Autocorrelation method
            autocorr = torch.nn.functional.conv1d(
                frame.unsqueeze(0).unsqueeze(0),
                frame.flip(0).unsqueeze(0).unsqueeze(0),
                padding=frame_length - 1
            )[0, 0, frame_length-1:]

            # Look for peaks in baby cry F0 range (300-600 Hz)
            min_lag = int(self.sample_rate / 600)  # Max freq -> min lag
            max_lag = int(self.sample_rate / 300)  # Min freq -> max lag

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

        # Analyze pitch contours for cry-like patterns
        contour_scores = torch.zeros(len(pitch_track))

        # Look for rising/falling patterns typical of cries
        window_size = int(0.3 * self.sample_rate / hop_length)  # 300ms windows

        for i in range(window_size, len(pitch_track) - window_size):
            window = pitch_track[i-window_size:i+window_size]
            valid_pitches = window[window > 0]

            if len(valid_pitches) > window_size:
                # Calculate pitch variation
                pitch_std = valid_pitches.std().item()
                pitch_range = (valid_pitches.max() - valid_pitches.min()).item()

                # Baby cries have moderate pitch variation (not flat, not erratic)
                if 20 < pitch_range < 200 and pitch_std > 10:
                    contour_scores[i] = min(1.0, pitch_range / 150)

        return contour_scores, pitch_track

    def detect_frequency_modulation(self, audio: torch.Tensor,
                                   frame_length: int = 2048,
                                   hop_length: int = 512,
                                   modulation_rate_min: float = 3.0,
                                   modulation_rate_max: float = 12.0) -> torch.Tensor:
        """
        Detect rapid vibrato-like frequency modulation characteristic of baby cries.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length
            modulation_rate_min: Minimum FM rate in Hz
            modulation_rate_max: Maximum FM rate in Hz

        Returns:
            FM detection scores per frame
        """
        # Compute instantaneous frequency using STFT phase
        stft = torch.stft(
            audio,
            n_fft=frame_length,
            hop_length=hop_length,
            return_complex=True
        )

        # Phase unwrapping and differentiation for instantaneous frequency
        phase = torch.angle(stft)
        phase_diff = torch.diff(phase, dim=1)

        # Instantaneous frequency
        inst_freq = phase_diff * self.sample_rate / (2 * np.pi * hop_length)

        # For each frequency bin in cry range, detect modulation
        freqs = torch.fft.fftfreq(frame_length, 1/self.sample_rate)[:frame_length//2 + 1]
        cry_bins = (freqs >= 300) & (freqs <= 600)

        fm_scores = []

        for frame_idx in range(inst_freq.shape[1]):
            # Get instantaneous frequencies in cry range
            cry_inst_freqs = inst_freq[cry_bins, frame_idx]

            if len(cry_inst_freqs) > 10:
                # Measure variation (vibrato creates periodic variation)
                variation = cry_inst_freqs.std().item()

                # Baby cry FM typically has 5-15 Hz variation
                if 5 < variation < 20:
                    score = min(1.0, variation / 15)
                else:
                    score = 0.0
            else:
                score = 0.0

            fm_scores.append(score)

        return torch.tensor(fm_scores)

    def compute_all_features(self, audio: torch.Tensor,
                           frame_length: int = 2048,
                           hop_length: int = 512) -> dict:
        """
        Compute all acoustic features for baby cry detection.

        Args:
            audio: Input audio tensor
            frame_length: Frame size for analysis
            hop_length: Hop length

        Returns:
            Dictionary containing all acoustic feature scores
        """
        harmonic_scores, f0_track = self.detect_harmonic_structure(audio, frame_length, hop_length)
        temporal_scores = self.detect_temporal_patterns(audio)
        contour_scores, pitch_track = self.track_pitch_contours(audio, frame_length, hop_length)
        fm_scores = self.detect_frequency_modulation(audio, frame_length, hop_length)

        return {
            'harmonic_scores': harmonic_scores,
            'f0_track': f0_track,
            'temporal_scores': temporal_scores,
            'contour_scores': contour_scores,
            'pitch_track': pitch_track,
            'fm_scores': fm_scores
        }

    def combine_scores(self, features: dict, segment_duration: float = 3.0,
                      rejection_filters: dict = None) -> list:
        """
        Combine all acoustic features into final cry confidence scores.

        Args:
            features: Dictionary of acoustic features
            segment_duration: Duration of each segment in seconds
            rejection_filters: Optional dict with adult_rejection, music_rejection, env_rejection

        Returns:
            List of (start_time, end_time, cry_confidence) tuples
        """
        # Get the shortest feature length for alignment
        min_len = min(
            len(features['harmonic_scores']),
            len(features['contour_scores']),
            len(features['fm_scores'])
        )

        # Truncate all to same length
        harmonic = features['harmonic_scores'][:min_len]
        contour = features['contour_scores'][:min_len]
        fm = features['fm_scores'][:min_len]

        # Weighted combination of features
        # Baby cry indicators (positive weights)
        cry_indicators = (
            0.40 * harmonic +      # Harmonics are strong indicator
            0.30 * contour +       # Pitch contours
            0.30 * fm              # Frequency modulation
        )

        # Apply rejection filters if provided
        if rejection_filters:
            min_len_reject = min(
                min_len,
                len(rejection_filters.get('adult_rejection', [])),
                len(rejection_filters.get('music_rejection', [])),
                len(rejection_filters.get('env_rejection', []))
            )

            adult_rej = rejection_filters['adult_rejection'][:min_len_reject]
            music_rej = rejection_filters['music_rejection'][:min_len_reject]
            env_rej = rejection_filters['env_rejection'][:min_len_reject]

            cry_indicators = cry_indicators[:min_len_reject]
            rejection_multiplier = adult_rej * music_rej * env_rej
            combined_scores = cry_indicators * rejection_multiplier
        else:
            combined_scores = cry_indicators

        # Convert frame-level scores to segments
        hop_length = 512
        frame_duration = hop_length / self.sample_rate
        segment_frames = int(segment_duration / frame_duration)

        segments = []
        for i in range(0, len(combined_scores) - segment_frames, segment_frames // 2):
            segment_score = combined_scores[i:i+segment_frames].mean().item()
            start_time = i * frame_duration
            end_time = (i + segment_frames) * frame_duration

            segments.append((start_time, end_time, segment_score))

        return segments
