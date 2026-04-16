# Baby Cry Detection Dataset Summary

## Current Dataset Composition

**Total raw samples: ~8,695**
**Total used after filtering (MIN_AUDIO_DURATION=0.8s): ~8,680**
**Class distribution ratio: ~0.83:1 (cry:non-cry) after Run 16 filtering**

### Cry Samples: ~3,928

#### 1. Cry Baby Dataset (~3,928 samples)
- **Source**: Multiple curated sources
- **Location**: `data/cry_baby/`
- **Format**: .wav, .mp3, .flac, .ogg, .m4a, .3gp, .webm, .mp4 files
- **Content**: Comprehensive collection of baby cry recordings
- **Composition**:
  - Original cry samples from Donate-a-Cry corpus
  - Hugging Face baby_crying_sound dataset
  - Baby Cry Sense Dataset (Kaggle)
  - ICSD real recordings (strong and weak labels)
  - CryCeleb2023 dataset samples
- **Categories**: Various cry types (hunger, pain, discomfort, tired, belly pain, burping, scared, lonely, cold/hot)
- **Purpose**: Primary training data for binary classification and sound localization

### Non-Cry Samples: ~4,767

#### 1. Hard Negatives Dataset (~4,767 samples)
- **Location**: `data/hard_negatives/`
- **Format**: .wav, .mp3, .flac, .ogg, .m4a, .3gp, .webm, .mp4 files
- **Content**: Comprehensive collection of non-cry sounds
- **Composition**:
  - Baby non-cry sounds (babbling, laughing, cooing, silence)
  - Adult speech recordings from LibriSpeech
  - Environmental household sounds from ESC-50
  - Other challenging acoustic scenarios
- **Purpose**: Reduce false positives and improve model robustness in home environment

### Background Noise for Augmentation: 1,960 samples
- **Location**: `data/noise/`
- **Source**: ESC-50 environmental sounds
- **Format**: .wav files
- **Purpose**: Mixed into cry samples during training for robustness

> [!note] The noise directory is NOT used as training labels, only for augmentation.

## Class Imbalance

**Current ratio: ~0.83:1 (cry:non-cry)** - Well-balanced after Run 16 filtering (MIN_AUDIO_DURATION=0.8s)

This well-balanced dataset is further optimized through:
1. WeightedRandomSampler during training (optional, may provide marginal improvement)
2. Class weights in loss function
3. On-the-fly data augmentation for cry samples
4. Stratified train/val/test splitting

### Balancing Options

**Current Status: Well-balanced (~0.83:1 ratio after Run 16 filtering)**

With the consolidated cry_baby and hard_negatives datasets after duration filtering, the dataset is well-balanced. Options:

**Option 1: Train with current balance (Recommended)**
- Use current ~3,928 cry vs ~4,767 non-cry ratio
- ~0.83:1 class balance
- Pro: Near-perfect balance, all real recordings
- Pro: Optimal for sound localization (no synthetic audio)
- Pro: Realistic household acoustic properties
- Pro: Maximum dataset diversity
- Pro: Filtered for minimum duration (0.8s) ensuring quality samples

**Option 2: Reduce dataset size for faster training**
```bash
python scripts/balance_dataset.py --ratio 1.0 --max-samples 3000
```
Creates: 3,000 cry vs 3,000 non-cry (1:1 ratio)
- Pro: Faster training
- Con: Discards ~16% of samples
- Con: Less exposure to cry and environmental sound diversity

**Option 3: Collect more samples for specific scenarios**
- Record cry samples in YOUR specific home environment
- Add more household sounds from your setting
- Pro: Optimizes for your deployment environment
- Con: Time investment

## Data Splits

The dataset is split using stratified sampling:
- **Training**: 60% (5,208 samples: 2,357 cry + 2,851 non-cry)
- **Validation**: 20% (1,736 samples: 785 cry + 951 non-cry)
- **Test**: 20% (1,736 samples: 786 cry + 950 non-cry)

Stratified splitting ensures each split maintains the class ratio (~0.83:1).

## Data Augmentation

### On-the-Fly Augmentation (Training Only)

Applied randomly during training to cry samples:
- **Gaussian Noise**: Adds random noise (50% probability)
- **Time Stretching**: 0.8x to 1.2x speed (30% probability)
- **Pitch Shifting**: -2 to +2 semitones (30% probability)
- **Background Noise Mixing**: Mixes with noise/ files (70% probability)

### Benefits of On-the-Fly Augmentation:
1. No data leakage (applied after train/val/test split)
2. Different augmentation each epoch (more variety)
3. No storage overhead
4. Validation/test sets remain unaugmented (true performance)

### Why Pre-Augmented Data Was Removed:
The original `cry_aug/` directory (1,968 files) was deleted because:
- Pre-augmented versions could leak into validation/test sets
- Original cry + augmented version might appear in different splits
- Inflated evaluation metrics artificially
- On-the-fly augmentation is the industry standard

## Audio Preprocessing Pipeline

All audio files undergo:
1. **Loading**: torchaudio (primary) or librosa (fallback)
2. **Resampling**: Convert to 16kHz sample rate
3. **Mono Conversion**: Stereo files converted to mono
4. **Duration Normalization**: Pad or truncate to 3 seconds
5. **Feature Extraction**: Convert to log-mel spectrogram
   - 128 mel-frequency bins
   - 0-8000 Hz frequency range
   - 2048 FFT window size
   - 512 hop length
6. **Normalization**: Z-score normalization

## Supported Audio Formats

- .wav (primary format)
- .ogg
- .mp3
- .flac
- .m4a
- .3gp
- .webm
- .mp4

All formats are automatically converted to the standard preprocessing pipeline.

## Data Collection History

### Successful Additions:
1. **Original cry samples**: 696 files from Donate-a-Cry corpus
2. **Original non-cry**: Baby sounds from Freesound.org community
3. **LibriSpeech speech**: 4,481 adult speech samples (January 2025)
4. **ESC-50 environmental**: 1,960 environmental sounds (January 2025)
5. **Hugging Face baby cries**: 69 unique files from mahmudulhasan01/baby_crying_sound (October 2025)
6. **Kaggle Baby Cry Sense**: 57 unique files from mennaahmed23 dataset (October 2025)
7. **ICSD real recordings**: 2,312 files (424 strong + 1,888 weak) from ICSD dataset (October 2025)
8. **Phase 2 baby_noncry expansion**: +302 files from Freesound.org (babbling, laughing, cooing, toddler speech) (February 2026)
9. **Phase 2 adult_scream expansion**: +50 files from Freesound.org (female/children screams) (February 2026)

### Failed/Filtered Attempts:
1. **AudioSet baby cries**: Downloaded 75 samples, but manual inspection revealed they were NOT baby cries (deleted)
2. **FSD50K baby cries**: Dataset search found 0 baby cry samples
3. **Hugging Face duplicates**: Removed 696 duplicates and 216 mislabeled files (baby laughs/silence)
4. **Kaggle duplicates**: Removed 1,047 duplicates from 1,054 downloaded files

## Dataset Quality Notes

### Cry Samples (~3,928)
- **Quality**: High - manually verified and professionally annotated
- **Diversity**: Multiple babies, ages, cry types, and sources
- **Categories**: Hunger, pain, discomfort, tired, belly pain, burping, scared, lonely, cold/hot
- **Sources**: Consolidated from Donate-a-Cry, ICSD, CryCeleb, Hugging Face, Kaggle datasets
- **Improvement**: +464% more cry samples than original (696 -> ~3,928, after filtering)
- **Acoustic Quality**: All real recordings, no synthetic audio
- **Filtering Applied**: Removed duplicates, mislabeled files, snoring, synthetic files, and short samples (<0.8s)
- **Label Types**: Mix of strong labels (precise timing) and weak labels (clip-level) - all are real recordings
- **Duration**: Minimum 0.8 seconds per sample after filtering

### Non-Cry Samples (~4,767)
- **Quality**: High - professional datasets (LibriSpeech, ESC-50, Freesound.org)
- **Diversity**: Excellent - baby sounds, speech, household, environmental
- **Coverage**: Well-represents home environment sounds
- **Composition**: Consolidated hard negatives for improved false positive reduction
- **Duration**: Minimum 0.8 seconds per sample after filtering

## Recommendations

### Short-term:
1. Train with current well-balanced dataset (~0.83:1 ratio) - optimal for all use cases
2. Minimal need for WeightedRandomSampler with near-perfect balance
3. Monitor precision/recall to ensure both classes perform well
4. All real recordings make this optimal for sound localization applications
5. Duration filtering (MIN_AUDIO_DURATION=0.8s) removes low-quality short samples

### Long-term:
1. Record cry samples in YOUR specific home environment for best localization accuracy
2. Test model performance with real-world audio from your deployment setting
3. Consider adding more household sounds specific to your environment
4. Validate sound localization performance with spatial audio recordings

## Verification Commands

Count all training data:
```bash
python scripts/count_training_data.py
```

Create balanced subset:
```bash
python scripts/balance_dataset.py --ratio 5.0
```

Analyze dataset:
```bash
python main.py analyze
```

## File Structure

```
data/
├── cry_baby/              # ~3,928 baby cry samples (TRAINING)
│                          # - Consolidated from multiple sources
│                          # - Includes Donate-a-Cry, ICSD, CryCeleb, HF, Kaggle
│                          # - After MIN_AUDIO_DURATION=0.8s filtering
├── hard_negatives/        # ~4,767 non-cry samples (TRAINING)
│                          # - Baby non-cry sounds (babbling, laughing, etc.)
│                          # - Adult speech from LibriSpeech
│                          # - Environmental sounds from ESC-50
│                          # - After MIN_AUDIO_DURATION=0.8s filtering
├── noise/                 # 1,960 sounds (AUGMENTATION ONLY)
├── cry/                   # Original cry samples (archived/legacy)
├── cry_ICSD/              # Original ICSD dataset (archived/legacy)
├── cry_crycaleb/          # Original CryCeleb dataset (archived/legacy)
├── baby_noncry/           # Original non-cry baby sounds (archived/legacy)
├── adult_speech/          # Original adult speech (archived/legacy)
└── environmental/         # Original environmental sounds (archived/legacy)
```

> [!important] Only `cry_baby/` and `hard_negatives/` are used as training labels. The `noise/` directory is exclusively for augmentation. Other directories contain original dataset structures (archived/legacy).

## Known Issues

1. **Class balance**: ~0.83:1 ratio (well-balanced after duration filtering)
2. **Dataset consolidation**: cry_baby and hard_negatives directories consolidate multiple source datasets
3. **Multiple sources**: Cry samples from 5 different sources may have quality variations
4. **Format variety**: .wav, .mp3, .flac, .ogg, .m4a, .3gp, .webm, .mp4 files
5. **Legacy directories**: Original dataset directories kept for reference but not used in training
6. **Mixed label types**: Combines strong and weak labels (all are real recordings, just different annotation detail)
7. **Duration filtering**: MIN_AUDIO_DURATION=0.8s applied, removing very short samples that may reduce data quality

## Change Log

**March 2026 (Run 16 — Duration Filtering):**
- Applied MIN_AUDIO_DURATION=0.8s filtering to all samples
- Raw cry files: ~3,928 (consolidated from cry/, cry_crycaleb/, cry_ICSD/)
- Raw non-cry files: ~4,767 (hard_negatives/)
- Total raw: ~8,695; used in training: ~8,680 (15 samples filtered out)
- Final class balance: ~0.83:1 (cry:non-cry) — well-balanced
- Dataset splits (stratified): Train 5,208 (2,357 cry + 2,851 non-cry), Val 1,736 (785 cry + 951 non-cry), Test 1,736 (786 cry + 950 non-cry)
- Model trained: train_2026-03-14_20-11-56 (model_best.pth deployed)

**February 2026 (Phase 2 — Hard Negative Expansion):**
- Expanded baby_noncry: +302 samples via Freesound API
  - Baby babbling/proto-speech: ~100 new samples
  - Baby laughing: ~75 new samples
  - Baby cooing: ~75 new samples
  - Toddler speech: ~50 new samples
- Expanded adult_scream: +50 samples via Freesound API
  - High-pitched female screams: ~30 new samples
  - Children's screams: ~20 new samples
- Motivation: /assess graded model C with 59 FPs; baby_noncry (12 FPs) and adult_scream (4 FPs) were top false positive categories
- Download script: [[Sound-Localization-Claude/baby_cry_detection/scripts/download_freesound.py|scripts/download_freesound.py]] (requires Freesound API key)
- Total hard_negatives after expansion: ~5,059 samples (before filtering)
- Updated class balance: ~0.89:1 (cry:non-cry) — still well-balanced (before filtering)
- Requires re-preprocessing after download: `python scripts/preprocess_dataset.py --output data/processed/v2`

**November 2025:**
- Consolidated dataset into cry_baby (~4,505 samples) and hard_negatives (~4,707 samples) directories (before filtering)
- Created unified cry_baby directory combining all cry sources
- Created unified hard_negatives directory combining all non-cry sources
- Total dataset: ~9,212 samples (~4,505 cry + ~4,707 non-cry)
- Class balance: ~0.96:1 (cry:non-cry) - well-balanced (before filtering)
- Legacy directories preserved for reference but not used in training
- Simplified dataset structure for easier maintenance and training

**October 2025:**
- Added 2,312 real cry samples from ICSD (424 strong + 1,888 weak labels)
- Created cry_ICSD folder with real recordings only
- Filtered out 883 snoring files from ICSD (not baby cries)
- Filtered out 500 synthetic files from ICSD (maintain acoustic realism for sound localization)
- Included weak label files (real recordings, clip-level annotations work fine for binary classification)
- Added 69 unique cry samples from Hugging Face (mahmudulhasan01/baby_crying_sound)
- Added 57 unique cry samples from Kaggle (Baby Cry Sense Dataset - mennaahmed23)
- Added 1,423 cry samples from CryCeleb2023 dataset
- Added .3gp audio format support
- Removed 1,743 duplicate files and 216 mislabeled files during dataset expansion
- Updated total cry samples: 696 -> 4,557 (+554%; before consolidation)
- Reduced adult_speech from 4,481 to 2,241 files (50% reduction for better balance)
- Updated class imbalance: 1:9.77 -> 1:1 (perfectly balanced, before consolidation)
- Consolidated cry_1 and cry_2 folders into main cry folder
- Total dataset: 5,379 -> 9,264 samples (all real recordings, before consolidation)

**January 2025:**
- Added 4,481 LibriSpeech adult speech samples
- Added 1,960 ESC-50 household environmental sounds
- Removed 1,968 pre-augmented cry samples (prevent data leakage)
- Updated audio format support (.webm, .mp4)
- Attempted AudioSet/FSD50K downloads (failed - no valid baby cries)
- Organized project structure (scripts/, docs/ folders)
