# Testing Your Own Audio Files

## Quick Start - 3 Ways to Test

### Method 1: Analyze Audio (Shows Detailed Features)
```bash
python analyze_audio.py your_audio.wav
```

This shows:
- Harmonic structure score
- Fundamental frequency (F0)
- Temporal patterns
- Pitch contours
- Frequency modulation
- Energy distribution
- Adult/music/environmental rejection scores
- Overall assessment

**Example:**
```bash
python analyze_audio.py baby_recording.wav
python analyze_audio.py baby_recording.wav results/model_best.pth  # With ML model
```

---

### Method 2: Extract Baby Cries (Saves Filtered Audio)
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
python test_my_audio.py mixed_audio.wav results/model_best.pth 0.7  # With model + threshold
python test_my_audio.py mixed_audio.wav None 0.5  # Acoustic features only
```

**Arguments:**
- `audio_file` - Your audio file (required)
- `model_path` - Path to trained model (optional, use "None" for acoustic-only)
- `threshold` - Detection threshold 0.0-1.0 (optional, default 0.5)

---

### Method 3: Python Script (Full Control)
```python
from src.audio_filter import BabyCryAudioFilter
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
cry_threshold=0.7  # Default is 0.5

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
# Step 1: Analyze to see what's in the audio
python analyze_audio.py unknown.wav

# Step 2: If cries detected, extract them
python test_my_audio.py unknown.wav results/model_best.pth 0.6

# Step 3: Listen to output file
# unknown_filtered.wav will contain only baby cries
```

### Workflow 2: Process Multiple Files
```python
# test_batch.py
from pathlib import Path
from src.audio_filter import BabyCryAudioFilter
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
from src.audio_filter import BabyCryAudioFilter
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

### "No cry segments found" but I can hear cries
**Solutions:**
1. Lower the threshold: `cry_threshold=0.3`
2. Check if audio is too short (need >3 seconds)
3. Adjust F0 range if baby has unusual pitch
4. Check audio quality (should be clear, not heavily compressed)

### Too many false positives
**Solutions:**
1. Raise the threshold: `cry_threshold=0.7`
2. Increase acoustic feature weight
3. Check if adult speech/music is being detected (use analyze_audio.py)
4. Adjust rejection filters

### Audio file won't load
**Solutions:**
1. Check file exists and path is correct
2. Try converting to WAV format first
3. Check file isn't corrupted
4. Ensure file is readable (not locked by another program)

### Scores all showing 0.0
**Solutions:**
1. Check audio isn't silent
2. Verify sample rate (should be 16000 Hz or will be resampled)
3. Check audio duration (need at least 1-2 seconds)
4. Verify audio amplitude (shouldn't be clipped or too quiet)

---

## Example Output

### analyze_audio.py output:
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
5. **Combine methods** - Use analyze_audio.py first, then test_my_audio.py
6. **Tune threshold** - Start at 0.5, adjust based on results
7. **Use ML model** - Better results with trained model vs acoustic-only
8. **Check formats** - WAV/FLAC work best, MP3 may have compression artifacts

---

## Need More Help?

- **Full documentation:** [ACOUSTIC_FEATURES.md](ACOUSTIC_FEATURES.md)
- **Quick reference:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Implementation details:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Configuration options:** [src/config.py](src/config.py)
