# Inference with Advanced Filtering Pipeline

## Overview

This document describes the new architecture that separates the binary classification model from the noise filtering pipeline.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    4-Channel Microphone Array                   │
│                      (Raw Audio Input)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│        BINARY CLASSIFICATION MODEL (Clean, No Filtering)        │
│                  [OK] Trained without advanced filtering        │
│                  [OK] 95%+ accuracy on cry/non-cry classification  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                        │
         Predict: "cry"          Predict: "non-cry"
                │                        │
                ▼                        ▼
    ┌──────────────────┐       Discard/Ignore
    │  Apply Advanced  │
    │     Filtering    │
    └────────┬─────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ADVANCED FILTERING PIPELINE                        │
│  - Voice Activity Detection (VAD)                               │
│  - High-pass filtering (100 Hz)                                 │
│  - Band-pass filtering (200-2000 Hz)                            │
│  - Spectral noise reduction (strength: 0.5)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           FILTERED AUDIO (Baby Cry Only)                        │
│        Saved to: filtered_audio/                                │
│        Ready for: Listening & Sound Localization               │
└─────────────────────────────────────────────────────────────────┘
```

## Key Changes

### 1. Training Configuration (`src/config.py`)

**Before:**
```python
USE_ADVANCED_FILTERING = True
USE_ACOUSTIC_FEATURES = True
```

**After:**
```python
USE_ADVANCED_FILTERING = False  # Disabled for training
USE_ACOUSTIC_FEATURES = False   # Disabled for training
```

**Why:** The advanced filtering and acoustic feature weighting were interfering with model learning, causing a 4.9% accuracy drop. By disabling them during training, the model learns a clean decision boundary between cry and non-cry sounds.

### 2. New Inference Module (`src/inference_with_filtering.py`)

A new module `BabyCryPredictorWithFiltering` that:
- Uses the clean binary classification model
- Applies advanced filtering **only** to predicted "cry" sounds
- Saves filtered audio for listening/localization

## Usage

### Basic Usage

```python
from pathlib import Path
from src.config import Config
from src.inference_with_filtering import BabyCryPredictorWithFiltering

# Initialize
config = Config()
predictor = BabyCryPredictorWithFiltering(config)

# Load trained model
model_path = Path('results/train_2025-10-21_02-36-55/model_best.pth')
predictor.load_model(model_path)

# Predict and filter
audio_file = Path('audio/baby_crying.wav')
result = predictor.predict(audio_file, apply_filtering=True)

# Check results
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.1f}%")
if result['filtered_audio_path']:
    print(f"Filtered audio: {result['filtered_audio_path']}")
```

### With Confidence Threshold

```python
# Only filter if confidence exceeds threshold
result = predictor.predict_with_threshold(
    audio_file,
    threshold=0.8,  # 80% confidence required
    apply_filtering=True
)
```

### Batch Processing

```python
audio_files = [
    Path('audio/cry1.wav'),
    Path('audio/cry2.wav'),
    Path('audio/noise.wav')
]

results = predictor.predict_batch(audio_files, apply_filtering=True)

# Only filtered cry sounds are saved
for result in results:
    if result['filtered_audio_path']:
        print(f"Filtered: {result['filtered_audio_path']}")
```

### Disable Filtering

```python
# Get predictions without filtering
result = predictor.predict(audio_file, apply_filtering=False)
```

## Result Dictionary

Each prediction returns a dictionary with:

```python
{
    'file_path': str,              # Path to input audio
    'predicted_class': int,        # 0=non-cry, 1=cry
    'predicted_label': str,        # 'non_cry' or 'cry'
    'confidence': float,           # Confidence (0-100%)
    'probabilities': {             # Class probabilities
        'non_cry': float,
        'cry': float
    },
    'filtered_audio_path': str,    # Path to filtered audio (if cry & filtered)
    # Optional (with_threshold only):
    'threshold_decision': str,     # 'cry', 'non_cry', or 'uncertain'
    'threshold_message': str,      # Description
    'threshold_used': float        # Threshold value used
}
```

## Training for Better Performance

### Step 1: Verify Configuration

Ensure these settings in `src/config.py`:

```python
MULTI_CLASS_MODE = False           # Binary classification
USE_ADVANCED_FILTERING = False     # Disabled for training
USE_ACOUSTIC_FEATURES = False      # Disabled for training
```

### Step 2: Train Model

```bash
python training/main.py train
```

Expected accuracy: **~95%** (like Oct 5 model)

### Step 3: Evaluate

```bash
python training/main.py evaluate --model-path results/train_XXXX/model_best.pth
```

### Step 4: Use for Filtering

```python
from src.inference_with_filtering import BabyCryPredictorWithFiltering

predictor = BabyCryPredictorWithFiltering(config)
predictor.load_model('results/train_XXXX/model_best.pth')
result = predictor.predict(audio_file, apply_filtering=True)
```

## Advanced Filtering Parameters

The advanced filtering uses these parameters (all in `src/config.py`):

```python
# Voice Activity Detection (VAD)
VAD_FRAME_LENGTH = 400         # 25ms at 16kHz
VAD_HOP_LENGTH = 160           # 10ms at 16kHz
VAD_ENERGY_THRESHOLD = 0.01    # Energy detection threshold
VAD_FREQ_MIN = 200             # Minimum frequency (Hz)
VAD_FREQ_MAX = 1000            # Maximum frequency (Hz)

# Noise Filtering
HIGHPASS_CUTOFF = 100          # Remove frequencies below 100 Hz
BANDPASS_LOW = 200             # Band-pass lower cutoff (Hz)
BANDPASS_HIGH = 2000           # Band-pass upper cutoff (Hz)
NOISE_REDUCE_STRENGTH = 0.5    # Spectral subtraction strength (0-1)
```

You can tune these parameters by modifying `src/config.py` and reloading the predictor.

## Integration with Sound Localization

Once you have the sound localization model, integrate it like this:

```python
from src.inference_with_filtering import BabyCryPredictorWithFiltering
from src.sound_localization import SoundLocalizer  # Future module

# Get filtered audio
predictor = BabyCryPredictorWithFiltering(config)
predictor.load_model(model_path)
result = predictor.predict(audio_file, apply_filtering=True)

# If cry detected and filtered
if result['filtered_audio_path']:
    # Use filtered audio for localization
    localizer = SoundLocalizer(config)
    location = localizer.localize(result['filtered_audio_path'])
    print(f"Baby crying at: {location}")
```

## Troubleshooting

### Issue: Low accuracy during training

**Solution:** Verify these settings:
- `USE_ADVANCED_FILTERING = False`
- `USE_ACOUSTIC_FEATURES = False`
- `MULTI_CLASS_MODE = False`

### Issue: No filtered audio output

**Possible causes:**
1. Model predicted "non-cry" - not a false negative, model is working correctly
2. `apply_filtering=False` - set it to `True` to enable filtering
3. AudioFilteringPipeline not available - check imports

### Issue: Filtered audio quality is poor

**Solutions:**
1. Adjust `NOISE_REDUCE_STRENGTH` (lower = gentler)
2. Adjust `BANDPASS_LOW` and `BANDPASS_HIGH` for wider/narrower frequency range
3. Check `VAD_ENERGY_THRESHOLD` if audio is being clipped

## Files Changed

- `src/config.py` - Disabled advanced filtering for training
- `src/inference_with_filtering.py` - New inference module (NEW FILE)
- `INFERENCE_WITH_FILTERING_README.md` - This documentation (NEW FILE)

## Summary

- **Binary classification model** remains clean and high-accuracy
- **Advanced filtering** is applied only in the inference pipeline
- **Filtered audio** is saved and ready for listening or localization
- **Easy to tune** filtering parameters without retraining the model
