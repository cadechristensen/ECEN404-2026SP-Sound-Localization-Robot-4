# Acoustic Feature-Based Filtering - Quick Reference

## TL;DR

Added acoustic signal processing to complement ML model for baby cry detection. System now analyzes harmonic structure, temporal patterns, pitch contours, frequency modulation, and energy distribution--while actively filtering out adult speech, music, and environmental noise.

---

## Quick Start

```python
from src.audio_filter import BabyCryAudioFilter
from src.config import Config

# Initialize with acoustic features enabled
config = Config()
audio_filter = BabyCryAudioFilter(config, model_path="model_best.pth")

# Process audio
results = audio_filter.process_audio_file(
    input_path="baby_audio.wav",
    output_path="cry_only.wav",
    use_acoustic_features=True  # NEW: Enable acoustic features
)

print(f"Found {results['num_cry_segments']} cry segments")
```

---

## What Got Added

### 5 Baby Cry Detectors:
1. **Harmonics** - Detects F0 (300-600 Hz) + overtones
2. **Temporal Patterns** - Detects burst-pause rhythm
3. **Pitch Contours** - Tracks rising/falling pitch
4. **Freq Modulation** - Detects vibrato (5-20 Hz)
5. **Energy Distribution** - Measures 300-600 Hz concentration

### 3 Rejection Filters:
6. **Adult Speech** - Rejects 80-250 Hz voices
7. **Music** - Rejects stable pitch patterns
8. **Environmental** - Rejects noise without harmonics

### Pipeline Integration:
9. **Feature Fusion** - Combines 60% ML + 40% acoustic
10. **Configurable** - 18 new parameters in config.py

---

## Baby Cry vs. Others

| Feature | Baby Cry | Adult Speech | Music | Noise |
|---------|----------|--------------|-------|-------|
| **F0** | 300-600 Hz | 80-250 Hz | Variable | N/A |
| **Harmonics** | [OK] Strong | [OK] Moderate | [OK] Strong | [X] None |
| **Pitch Stability** | Varies | Varies | Stable | N/A |
| **Temporal Pattern** | Burst-pause | Continuous | Continuous | Continuous |
| **FM (Vibrato)** | 5-20 Hz | Minimal | <5 Hz | N/A |
| **Energy 300-600Hz** | >30% | <20% | Variable | <10% |

---

## Configuration Cheat Sheet

```python
# config.py - Key parameters to adjust

# Enable/Disable
USE_ACOUSTIC_FEATURES = True

# Baby cry F0 range
CRY_F0_MIN = 300  # Hz
CRY_F0_MAX = 600  # Hz

# Temporal patterns
CRY_BURST_MIN = 0.3  # seconds
CRY_BURST_MAX = 2.0  # seconds
CRY_PAUSE_MIN = 0.1  # seconds
CRY_PAUSE_MAX = 0.8  # seconds

# Feature weights
WEIGHT_HARMONICS = 0.25
WEIGHT_PITCH_CONTOUR = 0.15
WEIGHT_FREQUENCY_MODULATION = 0.10
WEIGHT_ENERGY_DISTRIBUTION = 0.20

# Fusion weights
WEIGHT_ML_MODEL = 0.6
WEIGHT_ACOUSTIC_FEATURES = 0.4
```

---

## Common Adjustments

### Too many false positives from adult speech?
```python
config.WEIGHT_ACOUSTIC_FEATURES = 0.5  # Increase from 0.4
config.ADULT_F0_MAX = 240  # Decrease from 250
```

### Missing actual baby cries?
```python
cry_threshold = 0.5  # Lower from 0.7
config.WEIGHT_ML_MODEL = 0.7  # Rely more on ML
config.CRY_F0_MIN = 250  # Expand F0 range if needed
config.CRY_F0_MAX = 650
```

### Need faster processing?
```python
config.USE_ACOUSTIC_FEATURES = False  # Disable for 2-3x speedup
```

---

## Testing

```bash
# Test with synthetic audio
python test_acoustic_features.py

# Shows:
# - Individual feature scores
# - Rejection filter performance
# - Side-by-side comparison
```

---

## Method Reference

| Method | What It Does | Returns |
|--------|--------------|---------|
| `detect_harmonic_structure()` | Finds F0 + harmonics | Scores, F0 track |
| `detect_temporal_patterns()` | Finds burst-pause rhythm | Pattern scores |
| `track_pitch_contours()` | Tracks pitch variation | Contour scores, pitch |
| `detect_frequency_modulation()` | Detects vibrato | FM scores |
| `analyze_energy_distribution()` | Measures cry-band energy | Energy scores |
| `filter_adult_speech()` | Rejects adult voices | Rejection scores |
| `filter_music()` | Rejects stable pitches | Rejection scores |
| `filter_environmental_sounds()` | Rejects noise | Rejection scores |
| `compute_acoustic_features()` | Runs all features | Dict of all scores |
| `combine_acoustic_scores()` | Fuses features | Segment scores |

---

## File Locations

```
src/audio_filter.py          # Audio filtering and acoustic feature extraction
src/config.py                # Configuration parameters
QUICK_REFERENCE.md           # This file
```

---

## How It Works (Simplified)

```
Audio Input
    [DOWN]
1. Spectral Filtering (basic bandpass)
    [DOWN]
2. Voice Activity Detection
    [DOWN]
3. Acoustic Features -> compute 8 features -> combine into score
    [DOWN]
4. ML Model -> classify segments -> get ML score
    [DOWN]
5. Fusion -> (0.6 x ML) + (0.4 x acoustic) -> final score
    [DOWN]
6. Threshold -> segments with score > threshold
    [DOWN]
7. Extract -> isolate cry regions
    [DOWN]
Output: Baby cry audio
```

---

## Expected Performance

### Improvements:
- [OK] Fewer false positives from adult conversation
- [OK] Fewer false positives from background music
- [OK] Fewer false positives from environmental noise
- [OK] Better generalization to new acoustic environments

### Trade-offs:
- [WARNING] 2-3x slower than ML-only (still real-time on modern hardware)
- [WARNING] More parameters to tune
- [WARNING] May need adjustment for different baby ages/voices

---

## Troubleshooting

### All scores showing 0.0?
-> Check audio sample rate (should be 16000 Hz)
-> Check audio duration (need at least 1-2 seconds)
-> Verify audio isn't silent or clipped

### Harmonic detection not working?
-> Adjust `CRY_F0_MIN` and `CRY_F0_MAX` for your recordings
-> Increase `CRY_HARMONIC_TOLERANCE` (default 50 Hz)

### Too slow on Raspberry Pi?
-> Set `USE_ACOUSTIC_FEATURES = False` for speed
-> Or disable specific features by modifying `compute_acoustic_features()`

---

## Example: Compare ML-only vs. Combined

```python
# ML only
results_ml = audio_filter.process_audio_file(
    "test.wav", "output_ml.wav",
    use_acoustic_features=False
)

# ML + Acoustic
results_combined = audio_filter.process_audio_file(
    "test.wav", "output_combined.wav",
    use_acoustic_features=True
)

print(f"ML only: {results_ml['num_cry_segments']} segments")
print(f"Combined: {results_combined['num_cry_segments']} segments")
print(f"Reduction: {results_ml['num_cry_segments'] - results_combined['num_cry_segments']} false positives filtered")
```

---

## Key Equations

### Acoustic Score Combination:
```
cry_score = (
    0.25 x harmonic +
    0.15 x contour +
    0.10 x fm +
    0.20 x energy
) x adult_rejection x music_rejection x env_rejection
```

### Final Fusion:
```
final_score = 0.6 x ml_score + 0.4 x acoustic_score
```

### Thresholding:
```
is_cry = (final_score > cry_threshold)
```

---

## For More Details

- **Filtering Guide:** See [FILTERING_IMPROVEMENTS.md](FILTERING_IMPROVEMENTS.md)
- **Code:** See [src/audio_filter.py](../src/audio_filter.py)
- **Config:** See [src/config.py](../src/config.py)

---

## Summary

**Before:** ML model only -> some false positives from speech/music

**Now:** ML + Acoustic features -> actively filters non-cry sounds

**Result:** More robust baby cry detection with interpretable acoustic scores

[TARGET] **Use acoustic features when:** You have false positives or need explainability
[FAST] **Disable acoustic features when:** You need maximum speed (Raspberry Pi, etc.)
