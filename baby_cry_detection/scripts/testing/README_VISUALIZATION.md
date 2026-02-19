# test_my_audio.py - Now with Visualization!

## Overview

The `test_my_audio.py` script now supports **comprehensive visualization** of all filtering stages, similar to `test_filtering.py` but with the complete filtering pipeline.

## What's New?

### 12-Panel Visualization Grid

When you use the `--plot` flag, the script generates a comprehensive visualization showing:

1. **Original Waveform** - Raw input audio
2. **Spectral Filtered** - After 100-3000 Hz bandpass filter
3. **Denoised** - After spectral subtraction
4. **Original Spectrogram** - Mel-spectrogram of raw audio
5. **Filtered Spectrogram** - After spectral filtering
6. **Denoised Spectrogram** - After noise reduction
7. **Voice Activity Detection** - VAD mask showing detected activity
8. **ML Predictions** - Model confidence scores over time
9. **Acoustic Features** - Harmonic and energy scores (if enabled)
10. **Rejection Filters** - Adult speech, music, environmental filters (if enabled)
11. **Detected Cry Segments** - Final cry regions highlighted
12. **Summary Statistics** - Processing results and applied filters

## Usage

```bash
# Basic usage (no plots)
python scripts/testing/test_my_audio.py your_audio.wav

# With visualization plots
python scripts/testing/test_my_audio.py your_audio.wav --plot

# With model and plots
python scripts/testing/test_my_audio.py your_audio.wav results/model_best.pth 0.5 --plot

# Short flag version
python scripts/testing/test_my_audio.py your_audio.wav -p
```

## Output Files

### Audio Output
- `your_audio_filtered.wav` - Isolated cry segments (non-cry parts zeroed/removed)

### Visualization Output (when using `--plot`)
- `filtering_visualizations/your_audio_filtering_analysis.png` - 12-panel visualization grid

## Key Differences from test_filtering.py

| Feature | test_filtering.py | test_my_audio.py (with --plot) |
|---------|-------------------|--------------------------------|
| Audio duration | 3 seconds (truncated) | Full length |
| Filtering stages | Basic (highpass, bandpass, spectral sub) | Complete (all filters + ML + acoustic) |
| ML model | Not used | Used for classification |
| Acoustic features | Display only | Used for validation |
| Rejection filters | Not shown | Adult speech, music, environmental |
| Output audio | All audio (filtered) | Only cry segments |
| Multi-channel support | No | Yes (preserves phase) |
| Purpose | Educational/demo | Production + visualization |

## Best Practices for Sound Localization

For sound localization models, use `test_my_audio.py` because:

1. **Preserves multi-channel audio** - Keeps stereo/multi-mic phase relationships
2. **Removes competing sounds** - Eliminates adult speech, music, environmental noise
3. **Full-length processing** - Doesn't truncate to 3 seconds
4. **Better filtering** - Uses all available filters (8 stages total)

```bash
# Example for localization preprocessing
python scripts/testing/test_my_audio.py stereo_recording.wav results/model_best.pth 0.3 --plot
# This will:
# - Process full audio length
# - Preserve both channels with phase info
# - Remove non-cry sounds
# - Generate plots to verify filtering quality
# - Output: stereo_recording_filtered.wav (ready for localization)
```

## Visualization Insights

### Plot 8: ML Predictions
- **Red regions**: High confidence cries (above threshold)
- **Orange regions**: Low confidence (below threshold)
- Use this to tune your threshold

### Plot 10: Rejection Filters
- **Values near 1**: Audio passes filter (kept)
- **Values near 0**: Audio rejected (suppressed)
- Shows how adult speech, music, environmental sounds are filtered

### Plot 11: Final Cry Segments
- **Red highlights**: Detected cry regions
- These are the only parts kept in the output audio

## Tips

1. **Start with plots** - Use `--plot` to understand how filtering affects your audio
2. **Adjust threshold** - If too many/few segments detected, change the threshold parameter
3. **Check rejection filters (Plot 10)** - Verify non-cry sounds are being suppressed
4. **Compare spectrograms (Plots 4-6)** - See noise reduction visually

## Example Workflow

```bash
# Step 1: Test with plots to verify filtering
python scripts/testing/test_my_audio.py baby_crying.wav results/model_best.pth 0.5 --plot

# Step 2: Review the visualization
# Check if cry segments look correct

# Step 3: Adjust threshold if needed
python scripts/testing/test_my_audio.py baby_crying.wav results/model_best.pth 0.3 --plot

# Step 4: Process without plots for production
python scripts/testing/test_my_audio.py baby_crying.wav results/model_best.pth 0.3

# Step 5: Use filtered audio for localization
# The output *_filtered.wav is ready for your localization model
```

## Performance Note

- Generating plots adds ~2-5 seconds of processing time
- For batch processing, omit `--plot` flag
- Plots are most useful for debugging and threshold tuning
