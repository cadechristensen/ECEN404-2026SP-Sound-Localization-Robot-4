"""
Audio filtering package for baby cry detection and isolation.

This package provides modular components for:
- Spectral filtering and frequency domain processing
- Acoustic feature detection (harmonics, pitch contours, temporal patterns)
- Noise reduction and spectral subtraction
- ML model-based classification
- Multi-channel audio processing with phase preservation
"""

from .core import BabyCryAudioFilter, create_audio_filter

__all__ = ['BabyCryAudioFilter', 'create_audio_filter']
