"""
Circular audio buffer for continuous multi-channel audio capture.
Preserves inter-channel phase relationships for sound localization.
"""

import numpy as np
import threading


class CircularAudioBuffer:
    """Circular buffer for continuous multi-channel audio capture with context."""

    def __init__(self, max_duration: float, sample_rate: int, num_channels: int = 4):
        """
        Initialize circular buffer for multi-channel audio.

        Args:
            max_duration: Maximum duration to store (seconds)
            sample_rate: Audio sample rate
            num_channels: Number of audio channels to preserve
        """
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        # Store as (num_samples, num_channels) to preserve phase relationships
        self.buffer = np.zeros((self.max_samples, num_channels), dtype=np.float32)
        self.write_idx = 0
        self.is_full = False
        self.lock = threading.Lock()

    def add(self, audio_chunk: np.ndarray):
        """
        Add multi-channel audio chunk to buffer.

        Args:
            audio_chunk: Audio data with shape (num_samples, num_channels)
        """
        with self.lock:
            chunk_len = len(audio_chunk)
            remaining_space = self.max_samples - self.write_idx

            if chunk_len <= remaining_space:
                # Fits without wrapping
                self.buffer[self.write_idx:self.write_idx + chunk_len] = audio_chunk
                self.write_idx += chunk_len
                if self.write_idx >= self.max_samples:
                    self.write_idx = 0
                    self.is_full = True
            else:
                # Need to wrap around
                self.buffer[self.write_idx:] = audio_chunk[:remaining_space]
                overflow = chunk_len - remaining_space
                self.buffer[:overflow] = audio_chunk[remaining_space:]
                self.write_idx = overflow
                self.is_full = True

    def get_last_n_seconds(self, duration: float) -> np.ndarray:
        """
        Get last N seconds of multi-channel audio.

        Args:
            duration: Duration in seconds

        Returns:
            Audio array with shape (num_samples, num_channels)
        """
        with self.lock:
            n_samples = int(duration * self.sample_rate)
            n_samples = min(n_samples, self.max_samples)

            if not self.is_full:
                # Buffer not full yet, return what we have
                return self.buffer[:self.write_idx].copy()
            else:
                # Buffer is full, get last n_samples in circular order
                start_idx = (self.write_idx - n_samples) % self.max_samples

                if start_idx < self.write_idx:
                    # No wrap around
                    return self.buffer[start_idx:self.write_idx].copy()
                else:
                    # Wrap around case
                    return np.vstack([
                        self.buffer[start_idx:],
                        self.buffer[:self.write_idx]
                    ])

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer = np.zeros((self.max_samples, self.num_channels), dtype=np.float32)
            self.write_idx = 0
            self.is_full = False
