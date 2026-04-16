# Baby Cry Detection & Sound Localization System

A complete AI-powered baby monitoring system combining real-time cry detection with sound localization for autonomous robot navigation.

---

## Overview

This system detects baby cries in real-time, filters the audio, and determines the baby's location using a 4-microphone array. Designed for deployment on Raspberry Pi 5 for robotic baby monitoring applications.

### Key Features

- **Real-time Baby Cry Detection** - CNN-Transformer hybrid model
- **Audio Filtering** - Isolates baby cry frequencies (100-3000 Hz)
- **Sound Localization Interface** - Ready for TDOA/beamforming integration
- **Low-Power Listening Mode** - Optimized for battery operation
- **Two-Stage Detection** - Fast multichannel voting + temporal smoothing
- **Raspberry Pi Optimized** - Designed for edge deployment

---

## System Architecture

```
LOW-POWER LISTENING (1-sec chunks)
        ↓
CRY DETECTED? (>0.92 confidence)
        ↓
TEMPORAL SMOOTHING (3+ consecutive frames)
        ↓
CONFIRMED CRY
        ↓
AUDIO FILTERING (noise removal, frequency isolation)
        ↓
SOUND LOCALIZATION (direction + distance)
        ↓
ROBOT NAVIGATION
```

---

## Project Structure

```
baby_cry_detection/
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── src/                                # Core source code
│   ├── __init__.py
│   ├── config.py                       # System configuration & hyperparameters
│   ├── model.py                        # CNN-Transformer hybrid architecture
│   ├── train.py                        # Training loop with AMP, early stopping
│   ├── evaluate.py                     # Evaluation interface
│   ├── inference.py                    # Inference utilities
│   ├── calibration.py                  # Temperature scaling calibration
│   ├── dataset.py                      # Dataset handling
│   ├── preprocessed_dataset.py         # Preprocessed spectrogram cache loader
│   ├── data_preprocessing.py           # Audio preprocessing & mel-spectrogram extraction
│   ├── acoustic_features.py            # Acoustic feature extraction & validation
│   ├── utils.py                        # Utilities
│   ├── audio_filtering/                # Audio filtering package (modular)
│   │   ├── __init__.py
│   │   ├── core.py                     # BabyCryAudioFilter main class
│   │   ├── filters.py                  # Spectral filtering & frequency processing
│   │   ├── detection.py                # Acoustic feature detection
│   │   ├── noise_reduction.py          # Spectral subtraction & VAD
│   │   ├── classification.py           # ML model integration
│   │   ├── multichannel.py             # 4-channel processing with phase preservation
│   │   └── utils.py                    # Helper functions
│   └── evaluation/                     # Evaluation package (modular)
│       ├── __init__.py
│       ├── core.py                     # ModelEvaluator class
│       ├── metrics.py                  # Metrics calculation
│       ├── analysis.py                 # Statistical analysis
│       ├── visualizations.py           # Plotting functions
│       └── utils.py                    # Ensemble, predictions
│
├── training/                           # Training entry point
│   └── main.py                         # CLI: train, evaluate, analyze, test, predict
│
├── deployment/                         # Raspberry Pi deployment
│   ├── realtime_baby_cry_detector.py   # CLI entry point
│   ├── detector.py                     # BabyCryDetector class + DetectionResult
│   ├── audio_pipeline.py               # AudioPipeline (PyAudio + processing loop)
│   ├── multichannel_detector.py        # SNR-based channel selection & voting
│   ├── temporal_smoother.py            # Sliding-window false-positive filter
│   ├── audio_buffer.py                 # Thread-safe circular buffer
│   ├── audio_filtering.py              # Phase-preserving filtering pipeline
│   ├── config_pi.py                    # Pi-optimized configuration
│   ├── requirements-pi.txt
│   ├── models/                         # Model files (.pth, git-ignored)
│   ├── docs/                           # Deployment documentation
│   │   ├── DEPLOYMENT_GUIDE.md
│   │   └── FILTERING_REFERENCE.md
│   ├── tests/                          # Deployment tests
│   │   ├── test_baby_cry_detector.py
│   │   ├── test_multichannel_detector.py
│   │   └── test_pi_filtering.py
│   └── tools/                          # Setup & diagnostics
│       ├── multichannel_recorder.py
│       ├── pi_setup.py
│       ├── pi_diagnostics.py
│       └── verify_deployment.sh
│
├── scripts/                            # Utility scripts
│   ├── preprocess_dataset.py           # Dataset preprocessing
│   └── testing/                        # Testing scripts
│       ├── test_my_audio.py
│       ├── record_and_filter.py
│       ├── test_multichannel_audio.py
│       ├── convert_8ch_to_4ch.py
│       └── verify_class_mapping.py
│
├── docs/                               # Documentation
│   ├── guides/
│   │   ├── EVALUATION_GUIDE.md
│   │   ├── INFERENCE_MODEL_GUIDE.md
│   │   └── PREPROCESSING_GUIDE.md
│   ├── DATASET_CREDITS.md
│   ├── DATASET_SUMMARY.md
│   └── TEST_YOUR_AUDIO.md
│
├── data/                               # Training data
│   ├── cry_baby/                       # Positive samples (~3,928 baby cry files)
│   │   ├── cry/
│   │   ├── cry_ICSD/
│   │   └── cry_crycaleb/
│   ├── hard_negatives/                 # Negative samples (~4,767 files across 59 categories)
│   ├── noise/                          # Background noise files (~1,960 for augmentation)
│   └── processed/                      # Preprocessed data cache
│
└── results/                            # Training results & models
    └── train_YYYY-MM-DD_HH-MM-SS/
        ├── model_best.pth
        ├── model_inference.pth
        ├── ensemble_checkpoints.json
        ├── experiment_config.json
        ├── training_history.json
        └── evaluations/
            └── eval_YYYY-MM-DD_HH-MM-SS/
                ├── calibrated_model.pth    # Temperature-scaled model (deployed)
                ├── plots/                  # Confusion matrices, ROC/PR curves, reliability diagrams
                └── logs/                   # Per-evaluation log files
```

---

## Quick Start

### 1. Training (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python training/main.py train

# Evaluate the model
python training/main.py evaluate --model results/train_*/model_best.pth
```

### 2. Deployment (Raspberry Pi)

See complete deployment guide: [Deployment Guide](deployment/docs/DEPLOYMENT_GUIDE.md)

**Quick deployment:**
```bash
# Transfer files to Raspberry Pi
scp deployment/*.py \
    src/{config,model,data_preprocessing}.py \
    results/train_*/model_best.pth \
    requirements.txt \
    pi@raspberrypi.local:~/baby_monitor/

# On Raspberry Pi
cd ~/baby_monitor
python3 realtime_baby_cry_detector.py --device-index 2
```

---

## Hardware Requirements

### Development
- Python 3.8+
- GPU (optional, recommended for training)
- 8GB+ RAM

### Deployment (Raspberry Pi)
- Raspberry Pi 5 (8GB RAM)
- TI PCM6260-Q1 microphone array (4 channels)
- 4x Electret Condenser Microphones (-24dB, Mouser #: 665-AOM-5024L-HD-R)

---

## Model Architecture

- **Input:** 3-second audio clips (16kHz, mono)
- **Features:** Log Mel spectrograms (128 mel bins)
- **Architecture:** CNN + Transformer hybrid
  - CNN layers: Extract spatial features from spectrograms
  - Transformer layers: Capture temporal dependencies
  - Attention pooling: Focus on important frames
- **Output:** Binary classification (cry / non-cry)

### Performance
- **Accuracy:** 97.93%
- **Precision:** 97.93%
- **Recall:** 97.93%
- **ROC-AUC:** 99.50%
- **Inference time:** 100-200ms (Raspberry Pi CPU)

---

## Dataset

### Data Organization

The dataset is organized into **4 main folders** with a clear separation between positive and negative samples:

**Active Training Data:**
- **cry_baby/** - Positive samples (baby cry) - ~3,928 files
  - cry/ - Cry dataset samples
  - cry_ICSD/ - ICSD dataset samples
  - cry_crycaleb/ - CryCeleb2023 dataset samples
- **hard_negatives/** - Negative samples (non-cry) - ~4,767 files
  - adult_speech: 2,241 files (.flac format)
  - baby_noncry: 248 files
  - silence: 108 files
  - adult_scream, adult_shout, child_tantrum: 50 files each
  - All other categories (46 categories): 40 files each
  - Total: 59 distinct categories
- **noise/** - Background noise files - 1,960 files
  - Contains audio files directly (no subfolders)

### Summary

| Category       | Files      | Purpose                                     |
| -------------- | ---------- | ------------------------------------------- |
| cry_baby       | ~3,928     | Positive samples (baby cry)                 |
| hard_negatives | ~4,767     | Negative samples (non-cry sounds)           |
| noise          | ~1,960     | Background noise (data augmentation only)   |
| **Total Raw**  | **~8,695** | **Training/evaluation data (used: ~8,680)** |

**Train/Val/Test Split:** Train: 5,208 | Val: 1,736 | Test: 1,736 | **Class Ratio:** ~0.83:1 (cry to non-cry)

> [!note] Noise Folder Usage
> The noise folder (1,960 files) is used exclusively for data augmentation during training and is NOT counted in the class distribution. Only cry_baby and hard_negatives folders contribute to the training labels.

---

## Sound Localization Integration

Sound localization is integrated directly in the detection pipeline via the DOAnet CRNN model in `SoundLocalization/` at the repository root. On confirmed cry detection, 4-channel phase-preserved audio is passed via `detection_queue` for TDOA/beamforming processing.

---

## Usage Examples

### Training
```bash
# Basic training
python training/main.py train

# Resume from checkpoint
python training/main.py train --resume results/train_*/model_best.pth

# Custom configuration
python training/main.py train --config custom_config.py
```

### Evaluation
```bash
# Evaluate on test set
python training/main.py evaluate --model results/train_*/model_best.pth

# Detailed analysis
python training/main.py analyze --model results/train_*/model_best.pth
```

### Deployment
```bash
# Real-time cry detection on Raspberry Pi (--model defaults to config_pi.MODEL_PATH)
python deployment/realtime_baby_cry_detector.py --device-index 2
```

### Testing Your Audio Files

The testing scripts make it easy to test your own audio files with the trained model.

```bash
# Test an audio file (uses default model)
python scripts/testing/test_my_audio.py my_audio.wav
python scripts/testing/test_my_audio.py my_audio.wav --threshold 0.85 --plot

# Record and filter audio in real-time
python scripts/testing/record_and_filter.py
python scripts/testing/record_and_filter.py --duration 10 --threshold 0.85 --plot

# Use a different model
python scripts/testing/test_my_audio.py my_audio.wav --model path/to/your/model.pth
python scripts/testing/record_and_filter.py --model path/to/your/model.pth
```

---

## Quick Reference

### Getting Started
- [Quick Reference](docs/QUICK_REFERENCE.md) - Quick reference guide

### Development & Training
- [Preprocessing Guide](docs/guides/PREPROCESSING_GUIDE.md) - Dataset preprocessing for faster training
- [Evaluation Guide](docs/guides/EVALUATION_GUIDE.md) - Model evaluation guide
- [Inference Model Guide](docs/guides/INFERENCE_MODEL_GUIDE.md) - Model inference guide

### Deployment
- [Deployment Guide](deployment/docs/DEPLOYMENT_GUIDE.md) - Comprehensive Pi deployment
- [Filtering Reference](deployment/docs/FILTERING_REFERENCE.md) - Filtering details for Pi

### Documentation & Resources
- [Dataset Credits](docs/DATASET_CREDITS.md) - Dataset attribution
- [Dataset Summary](docs/DATASET_SUMMARY.md) - Dataset overview
- [Test Your Audio](docs/TEST_YOUR_AUDIO.md) - Audio testing guide
- [Visualization Guide](docs/VISUALIZATION_GUIDE.md) - Visualization tools

---

## Dependencies

Main dependencies:
- PyTorch >= 2.0.0
- torchaudio >= 2.0.0
- numpy >= 1.24.0
- librosa >= 0.10.0
- pyaudio >= 0.2.13 (for real-time audio)
- scipy >= 1.10.0

See [requirements.txt](requirements.txt) for complete list.

---

## Configuration

Key configuration parameters in `src/config.py`:

```python
# Audio
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 128

# Model
CNN_CHANNELS = [32, 64, 128, 256]
D_MODEL = 384
N_HEADS = 8
N_LAYERS = 4

# Training
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
PATIENCE = 20
CRY_WEIGHT_MULTIPLIER = 1.15
```

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- CryCeleb2023 dataset contributors
- ICSD dataset contributors
- Open-source audio processing libraries

---

## Support

For deployment issues, see:
- [Deployment Guide](deployment/docs/DEPLOYMENT_GUIDE.md)

For training and evaluation issues, see:
- [Evaluation Guide](docs/guides/EVALUATION_GUIDE.md)
- [Preprocessing Guide](docs/guides/PREPROCESSING_GUIDE.md)

---

## Project Status

- [x] Model training and evaluation
- [x] Audio filtering implementation
- [x] Raspberry Pi deployment system
- [x] Sound localization interface
- [ ] Sound localization model integration (user-specific)
- [ ] Robot navigation integration (user-specific)

---

**Version:** 3.0 — Run 16
**Last Updated:** March 2026
