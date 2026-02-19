"""
Baby Cry Audio Isolation and Filtering Module.

Implements advanced audio processing techniques to isolate baby cries from background noise.

This file now serves as a clean interface to the modular audio_filtering package.
All functionality has been refactored into specialized modules for professional maintenance.

REFACTORED ARCHITECTURE:
- audio_filtering/core.py: Main BabyCryAudioFilter class orchestration
- audio_filtering/filters.py: Spectral filtering and frequency domain processing
- audio_filtering/detection.py: Acoustic feature detection (harmonics, pitch, temporal)
- audio_filtering/noise_reduction.py: Spectral subtraction and VAD
- audio_filtering/classification.py: ML model integration
- audio_filtering/multichannel.py: 4-channel processing with phase preservation
- audio_filtering/utils.py: Helper functions

CRITICAL REQUIREMENTS:
- 4-channel microphone array support
- Phase preservation across all channels for sound localization
- Optimized for Raspberry Pi 5 8GB deployment
- Binary classification for baby cry detection
"""

import os
from typing import Optional

# Import main interface from refactored module
from src.audio_filtering import BabyCryAudioFilter, create_audio_filter

# Re-export for backward compatibility
__all__ = ['BabyCryAudioFilter', 'create_audio_filter']


if __name__ == "__main__":
    # Example usage
    from src.config import Config

    config = Config()

    # Initialize filter (replace with your model path)
    model_path = "results/latest/model_best.pth"
    audio_filter = create_audio_filter(config, model_path)

    # Process an audio file
    input_file = "test_audio.wav"
    output_file = "filtered_cry.wav"

    if os.path.exists(input_file):
        results = audio_filter.process_audio_file(input_file, output_file)
        print(f"Results: {results}")
    else:
        print(f"Test file {input_file} not found")
