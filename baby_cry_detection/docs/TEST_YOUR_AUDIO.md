# Testing Your Own Audio Files

## Quick Start - 2 Ways to Test

### Method 1: Extract Baby Cries (Saves Filtered Audio)
```bash
python test_my_audio.py your_audio.wav
```

This:
- Detects cry segments
- Saves filtered audio (only baby cries)
- Shows segment timestamps
- Provides statistics

**Example:**
```bash
python test_my_audio.py mixed_audio.wav
python test_my_audio.py mixed_audio.wav results/model_best.pth 0.85  # With model + threshold
python test_my_audio.py mixed_audio.wav None 0.5  # Acoustic features only
```

**Arguments:**
- `audio_file` - Your audio file (required)
- `model_path` - Path to trained model (optional, use "None" for acoustic-only)
- `threshold` - Detection threshold 0.0-1.0 (optional, default 0.5)

---

### Method 2: Python Script (Full Control)
```python
from src.audio_filtering import BabyCryAudioFilter
from src.config import Config

# Initialize
config = Config()
audio_filter = BabyCryAudioFilter(
    config=config,
    model_path="results/model_best.pth"  # or None
)

# Process your audio
results = audio_filter.process_audio_file(
    input_path="your_audio.wav",
    output_path="baby_cries_only.wav",
    cry_threshold=0.5,  # Adjust sensitivity
    use_acoustic_features=True
)

# View results
print(f"Cry segments: {results['num_cry_segments']}")
print(f"Cry duration: {results['cry_duration']:.2f}s")
for start, end in results['cry_segments']:
    print(f"  {start:.2f}s - {end:.2f}s")
```

---

## Visualization (--plot flag)

When you use `--plot`, the script generates a 12-panel visualization grid:

1. **Original Waveform** — Raw input audio
2. **Spectral Filtered** — After 100-3000 Hz bandpass
3. **Denoised** — After spectral subtraction
4. **Original Spectrogram** — Mel-spectrogram of raw audio
5. **Filtered Spectrogram** — After spectral filtering
6. **Denoised Spectrogram** — After noise reduction
7. **Voice Activity Detection** — VAD mask
8. **ML Predictions** — Model confidence over time (red = above threshold)
9. **Acoustic Features** — Harmonic and energy scores
10. **Rejection Filters** — Adult speech, music, environmental (values near 1 = passes)
11. **Detected Cry Segments** — Final cry regions highlighted
12. **Summary Statistics** — Processing results

Output saved to `filtering_visualizations/<filename>_filtering_analysis.png`.

---

## Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`)
- OGG (`.ogg`)
- FLAC (`.flac`)
- M4A (`.m4a`)
- 3GP (`.3gp`)
- WebM (`.webm`)
- MP4 (`.mp4`)

---

## Understanding the Results

### Acoustic Feature Scores (0.0 - 1.0)

| Feature | Baby Cry Range | Meaning |
|---------|----------------|---------|
| **Harmonic Structure** | >0.5 | Has clear F0 + harmonics |
| **F0 (Hz)** | 300-600 | Fundamental frequency |
| **Temporal Patterns** | >0.3 | Has burst-pause rhythm |
| **Pitch Contours** | >0.3 | Has pitch variation |
| **Freq Modulation** | >0.3 | Has vibrato/FM |
| **Energy (300-600Hz)** | >0.5 | Energy in cry band |

### Rejection Scores (0.0 - 1.0)

| Filter | Interpretation |
|--------|----------------|
| **Adult Speech** | <0.3 = adult speech, >0.7 = not adult |
| **Music** | <0.3 = music, >0.7 = not music |
| **Environmental** | <0.3 = noise, >0.7 = not noise |

### Combined Score

- **>0.7** - Strong baby cry detection
- **0.4-0.7** - Moderate/possible baby cry
- **<0.4** - Unlikely to be baby cry

---

## Adjusting Sensitivity

### Too Many False Positives?
```python
# Increase threshold (less sensitive)
cry_threshold=0.85  # Default is 0.5

# Or increase acoustic feature weight
config.WEIGHT_ACOUSTIC_FEATURES = 0.5  # Default 0.4
```

### Missing Real Cries?
```python
# Decrease threshold (more sensitive)
cry_threshold=0.3

# Or rely more on ML model
config.WEIGHT_ML_MODEL = 0.7  # Default 0.6
config.WEIGHT_ACOUSTIC_FEATURES = 0.3
```

### Adjust for Different Baby Ages
```python
# Younger babies (higher pitch)
config.CRY_F0_MIN = 350
config.CRY_F0_MAX = 650

# Older babies (lower pitch)
config.CRY_F0_MIN = 250
config.CRY_F0_MAX = 550
```

---

## Example Workflows

### Workflow 1: Analyze Unknown Audio
```bash
# Step 1: Test the audio with the model
python scripts/testing/test_my_audio.py unknown.wav --threshold 0.85 --plot

# Step 2: Review the visualization output and filtered audio
# unknown_filtered.wav will contain only baby cries
```

### Workflow 2: Process Multiple Files
```python
# test_batch.py
from pathlib import Path
from src.audio_filtering import BabyCryAudioFilter
from src.config import Config

audio_filter = BabyCryAudioFilter(Config(), "results/model_best.pth")

# Process all WAV files in a folder
for audio_file in Path("my_recordings").glob("*.wav"):
    output_file = f"filtered/{audio_file.stem}_filtered.wav"

    results = audio_filter.process_audio_file(
        str(audio_file),
        output_file,
        cry_threshold=0.5
    )

    print(f"{audio_file.name}: {results['num_cry_segments']} cry segments")
```

### Workflow 3: Real-time Analysis
```python
import torch
import torchaudio
from src.audio_filtering import BabyCryAudioFilter
from src.config import Config

# Initialize
audio_filter = BabyCryAudioFilter(Config(), "results/model_best.pth")

# Load audio
audio, sr = torchaudio.load("live_recording.wav")
audio = audio.mean(dim=0)

# Analyze features
features = audio_filter.compute_acoustic_features(audio)

# Get instant assessment
harmonic = features['harmonic_scores'].mean().item()
energy = features['energy_scores'].mean().item()

if harmonic > 0.5 and energy > 0.5:
    print("BABY CRY DETECTED!")
else:
    print("No baby cry")
```

---

## Troubleshooting

> [!warning]- "No cry segments found" but I can hear cries
> 1. Lower the threshold: `cry_threshold=0.3`
> 2. Check if audio is too short (need >3 seconds)
> 3. Adjust F0 range if baby has unusual pitch
> 4. Check audio quality (should be clear, not heavily compressed)

> [!warning]- Too many false positives
> 1. Raise the threshold: `cry_threshold=0.85`
> 2. Increase acoustic feature weight
> 3. Check if adult speech/music is being detected (use `test_my_audio.py --plot --acoustic`)
> 4. Adjust rejection filters

> [!warning]- Audio file won't load
> 1. Check file exists and path is correct
> 2. Try converting to WAV format first
> 3. Check file isn't corrupted
> 4. Ensure file is readable (not locked by another program)

> [!warning]- Scores all showing 0.0
> 1. Check audio isn't silent
> 2. Verify sample rate (should be 16000 Hz or will be resampled)
> 3. Check audio duration (need at least 1-2 seconds)
> 4. Verify audio amplitude (shouldn't be clipped or too quiet)

---

## Example Output

### test_my_audio.py output (with --acoustic flag):
```
================================================================================
Analyzing: baby_recording.wav
================================================================================

Loading audio...
Duration: 10.50 seconds

================================================================================
ACOUSTIC FEATURE ANALYSIS
================================================================================

1. Baby Cry Indicators (higher = more likely baby cry)
--------------------------------------------------------------------------------
  Harmonic Structure:     0.723
    -> Strong harmonic structure detected [OK]
  Fundamental Frequency:  425.3 Hz
    -> In baby cry range (300-600 Hz) [OK]
  Temporal Patterns:      0.541
    -> Burst-pause pattern detected [OK]
  Pitch Contours:         0.612
    -> Dynamic pitch variation [OK]
  Frequency Modulation:   0.458
    -> Vibrato detected [OK]
  Energy (300-600 Hz):    0.687
    -> High energy in baby cry band [OK]

2. Rejection Filters (higher = NOT that type)
--------------------------------------------------------------------------------
  Adult Speech Filter:    0.892
    -> Not adult speech [OK]
  Music Filter:           0.945
    -> Not music [OK]
  Environmental Filter:   0.812
    -> Not environmental noise [OK]

3. Overall Assessment
--------------------------------------------------------------------------------
  Combined Acoustic Score: 0.743
  Assessment: LIKELY BABY CRY [OK][OK][OK]

4. Machine Learning Prediction
--------------------------------------------------------------------------------
  Average ML Confidence:   0.856
  Final Combined Score:    0.811
  Final Assessment: STRONG baby cry detection [OK][OK][OK]
```

---

## Tips for Best Results

1. **Use clear audio** - Minimize background noise when recording
2. **Adequate length** - At least 3-5 seconds per segment
3. **Proper sample rate** - 16000 Hz recommended (auto-resampled if different)
4. **Mono audio** - Stereo is converted to mono automatically
5. **Use visualization** - Run test_my_audio.py with `--plot --acoustic` for full analysis
6. **Tune threshold** - Start at 0.5, adjust based on results
7. **Use ML model** - Better results with trained model vs acoustic-only
8. **Check formats** - WAV/FLAC work best, MP3 may have compression artifacts

---

## Need More Help?

- **Quick reference:** [[Sound-Localization-Claude/baby_cry_detection/docs/QUICK_REFERENCE|QUICK_REFERENCE]]
- **Configuration options:** [[Sound-Localization-Claude/baby_cry_detection/src/config.py|src/config.py]]
