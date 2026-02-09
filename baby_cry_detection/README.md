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
- **Two-Stage Detection** - Fast initial detection + TTA confirmation
- **Raspberry Pi Optimized** - Designed for edge deployment

---

## System Architecture

```
LOW-POWER LISTENING (1-sec chunks)
    |
    v
CRY DETECTED? (>75%)
    |
    v
CAPTURE CONTEXT (3-5 seconds)
    |
    v
CONFIRM WITH TTA (>85%)
    |
    v
AUDIO FILTERING (noise removal, frequency isolation)
    |
    v
SOUND LOCALIZATION (direction + distance)
    |
    v
ROBOT NAVIGATION
```

---

## Project Structure

```
baby_cry_detection_2/
|
+-- README.md                    # This file
+-- requirements.txt             # Python dependencies
|
+-- src/                         # Core source code
|   +-- config.py                # System configuration
|   +-- model.py                 # Neural network architecture
|   +-- audio_filter.py          # Audio processing & filtering (refactored interface)
|   +-- audio_filtering/         # Audio filtering package (modular)
|   |   +-- __init__.py
|   |   +-- core.py              # BabyCryAudioFilter main class
|   |   +-- filters.py           # Spectral filtering & frequency processing
|   |   +-- detection.py         # Acoustic feature detection
|   |   +-- noise_reduction.py   # Spectral subtraction & VAD
|   |   +-- classification.py    # ML model integration
|   |   +-- multichannel.py      # 4-channel processing with phase preservation
|   |   +-- utils.py             # Helper functions
|   +-- data_preprocessing.py    # Audio preprocessing utilities
|   +-- train.py                 # Training module
|   +-- evaluate.py              # Evaluation module (refactored interface)
|   +-- evaluation/              # Evaluation package (modular)
|   |   +-- __init__.py
|   |   +-- core.py              # ModelEvaluator class
|   |   +-- metrics.py           # Metrics calculation
|   |   +-- analysis.py          # Statistical analysis
|   |   +-- visualizations.py    # Plotting functions
|   |   +-- utils.py             # TTA, ensemble, predictions
|   +-- dataset.py               # Dataset handling
|   +-- preprocessed_dataset.py  # Preprocessed dataset loader
|   +-- inference.py             # Inference utilities
|   +-- acoustic_features.py     # Acoustic feature extraction
|   +-- acoustic_monitoring.py   # Acoustic monitoring tools
|   +-- calibration.py           # Model calibration
|   +-- utils.py                 # Utilities
|
+-- training/                    # Training scripts
|   +-- main.py                  # Main training script
|
+-- deployment/                  # Deployment files
|   +-- PI_DEPLOYMENT_QUICKSTART.md           # Quick deployment guide
|   +-- RASPBERRY_PI_DEPLOYMENT_GUIDE.md      # Comprehensive Pi deployment
|   +-- FILTERING_DEPLOYMENT_GUIDE.md         # Audio filtering deployment
|   +-- SOUND_LOCALIZATION_INTEGRATION.md     # Sound localization guide
|   +-- deploy.bat, deploy_to_pi.ps1          # Deployment scripts
|   +-- monitoring_config.py                  # Monitoring configuration
|   +-- raspberry_pi/                         # Raspberry Pi deployment
|   |   +-- robot_baby_monitor.py
|   |   +-- realtime_baby_cry_detector.py
|   |   +-- multichannel_detector.py
|   |   +-- sound_localization_interface.py
|   |   +-- audio_filtering.py
|   |   +-- config_pi.py
|   |   +-- pi_setup.py
|   |   +-- test_model_and_filter.py
|   |   +-- test_multichannel_detector.py
|   |   +-- test_pi_filtering.py
|   |   +-- PI_DEPLOYMENT_STEPS.md
|   |   +-- README_FILTERING.md
|   |   +-- verify_raspberry_pi_files.sh
|   |   +-- requirements-pi.txt
|
+-- scripts/                     # Utility scripts
|   +-- preprocess_dataset.py   # Dataset preprocessing
|   +-- testing/                 # Testing scripts
|   |   +-- test_my_audio.py
|   |   +-- record_and_filter.py
|   |   +-- test_multichannel_audio.py
|   |   +-- analyze_icsd_misclassification.py
|   |   +-- audio_quality_analysis.py
|   |   +-- auto_test_audio.py
|   |   +-- convert_8ch_to_4ch.py
|   |   +-- debug_calibrated_model.py
|   |   +-- verify_class_mapping.py
|   |   +-- README_VISUALIZATION.md
|
+-- docs/                        # Documentation
|   +-- guides/                  # How-to guides
|   |   +-- EVALUATION_GUIDE.md
|   |   +-- INFERENCE_MODEL_GUIDE.md
|   |   +-- INFERENCE_WITH_FILTERING_README.md
|   |   +-- PREPROCESSING_GUIDE.md
|   +-- resources/               # Research papers and references
|   |   +-- Lightweight_deep_learning_infant_cry.pdf
|   |   +-- PCM6260Q1EVM-PDK_Evaluation_Model.pdf
|   +-- QUICK_REFERENCE.md       # Quick reference guide
|   +-- DATASET_CREDITS.md       # Dataset attribution
|   +-- DATASET_SUMMARY.md       # Dataset overview
|   +-- FILTERING_IMPROVEMENTS.md # Audio filtering research and technical details
|   +-- TEST_YOUR_AUDIO.md       # Audio testing guide
|   +-- VISUALIZATION_GUIDE.md   # Visualization tools
|
+-- data/                        # Training data
|   +-- cry_baby/                # Positive samples (baby cry)
|   |   +-- cry/                 # Cry dataset samples
|   |   +-- cry_ICSD/            # ICSD cry samples
|   |   +-- cry_crycaleb/        # CryCeleb dataset samples
|   +-- hard_negatives/          # Negative samples (60 categories)
|   |   +-- adult_scream/, adult_shout/, adult_speech/
|   |   +-- airplane/, baby_noncry/, breathing/
|   |   +-- brushing_teeth/, can_opening/, car_horn/
|   |   +-- cat/, chainsaw/, child_tantrum/, chirping_birds/
|   |   +-- church_bells/, clapping/, clock_alarm/, clock_tick/
|   |   +-- coughing/, cow/, crackling_fire/, crickets/
|   |   +-- crow/, dog/, door_wood_creaks/, door_wood_knock/
|   |   +-- drinking_sipping/, engine/, fireworks/, footsteps/
|   |   +-- frog/, glass_breaking/, hand_saw/, helicopter/
|   |   +-- hen/, insects/, keyboard_typing/, laughing/
|   |   +-- mouse_click/, music_vocal/, pig/, pouring_water/
|   |   +-- rain/, relabeled/, rooster/, sea_waves/, sheep/
|   |   +-- silence/, siren/, sneezing/, snoring/, thunderstorm/
|   |   +-- toilet_flush/, train/, vacuum_cleaner/
|   |   +-- washing_machine/, water_drops/, wind/
|   +-- noise/                   # Background noise files
|   +-- processed/               # Preprocessed data cache
|   |   +-- v1/, v1_new/         # Preprocessing versions
|   +-- quarantine/              # Quality control
|   |   +-- poor_quality_cry/    # Quarantined poor quality samples
|
+-- examples/                    # Example outputs and demos
|   +-- inputs/                  # Example input files
|   +-- outputs/                 # Example output files
|   +-- filtering_visualizations/ # Filtering examples
|   +-- multichannel_visualizations/ # Multi-channel examples
|   +-- pre_demo/                # Pre-demo files
|   +-- rec_demo*.wav            # 4-channel demo recordings
|
+-- demo/                        # Demo files and presentations
|
+-- results/                     # Training results & models
    +-- train_YYYY-MM-DD_HH-MM-SS/
        +-- model_best.pth       # Best model checkpoint
        +-- logs/                # Training logs
        +-- plots/               # Training plots
        +-- evaluations/         # Evaluation results
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

See complete deployment guide: [deployment/raspberry_pi/PI_DEPLOYMENT_STEPS.md](deployment/raspberry_pi/PI_DEPLOYMENT_STEPS.md)

**Quick deployment:**
```bash
# Transfer files to Raspberry Pi
scp deployment/raspberry_pi/*.py \
    src/{config,model,audio_filter,data_preprocessing}.py \
    results/train_*/model_best.pth \
    requirements.txt \
    pi@raspberrypi.local:~/baby_monitor/

# On Raspberry Pi
cd ~/baby_monitor
python3 robot_baby_monitor.py --model model_best.pth --device-index 2 --channels 4
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
- **Accuracy:** ~96%
- **Precision:** ~95%
- **Recall:** ~97%
- **Inference time:** 100-200ms (Raspberry Pi CPU)

---

## Dataset

### Data Organization

The dataset is organized into **4 main folders** with a clear separation between positive and negative samples:

**Active Training Data:**
- **cry_baby/** - Positive samples (baby cry) - 4,505 files
  - cry/ - Cry dataset samples
  - cry_ICSD/ - ICSD dataset samples
  - cry_crycaleb/ - CryCeleb2023 dataset samples
- **hard_negatives/** - Negative samples (non-cry) - 4,707 files
  - adult_speech: 2,241 files (.flac format)
  - baby_noncry: 248 files
  - silence: 108 files
  - adult_scream, adult_shout, child_tantrum: 50 files each
  - All other categories (46 categories): 40 files each
  - Total: 59 distinct categories
- **noise/** - Background noise files - 1,960 files
  - Contains audio files directly (no subfolders)

### Summary

| Category | Files | Purpose |
|----------|-------|---------|
| cry_baby | 4,505 | Positive samples (baby cry) |
| hard_negatives | 4,707 | Negative samples (non-cry sounds) |
| noise | 1,960 | Background noise (data augmentation only) |
| **Total Active** | **11,172** | **Training/evaluation data** |

**Class Ratio:** Cry samples: 4,505 | Non-cry samples: 4,707 | Ratio: 4,505:4,707 ≈ 0.96:1 (cry to non-cry) - essentially balanced

**Important Note:** The noise folder (1,960 files) is used exclusively for data augmentation during training and is NOT counted in the class distribution. Only cry_baby and hard_negatives folders contribute to the training labels.

---

## Sound Localization Integration

To integrate your sound localization model:

1. **Edit:** `deployment/raspberry_pi/sound_localization_interface.py`
2. **Find:** Line 63 - "INTEGRATE YOUR SOUND LOCALIZATION MODEL HERE"
3. **Replace** placeholder with your model

See complete guide: [deployment/SOUND_LOCALIZATION_INTEGRATION.md](deployment/SOUND_LOCALIZATION_INTEGRATION.md)

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
# Full system (cry detection + localization)
python deployment/raspberry_pi/robot_baby_monitor.py \
    --model model_best.pth \
    --device-index 2 \
    --channels 4

# Cry detection only
python deployment/raspberry_pi/realtime_baby_cry_detector.py \
    --model model_best.pth \
    --device-index 2
```

### Testing Your Audio Files

The testing scripts make it easy to test your own audio files with the trained model.

```bash
# Test an audio file (uses default model)
python scripts/testing/test_my_audio.py my_audio.wav
python scripts/testing/test_my_audio.py my_audio.wav --threshold 0.7 --plot

# Record and filter audio in real-time
python scripts/testing/record_and_filter.py
python scripts/testing/record_and_filter.py --duration 10 --threshold 0.7 --plot

# Use a different model
python scripts/testing/test_my_audio.py my_audio.wav --model path/to/your/model.pth
python scripts/testing/record_and_filter.py --model path/to/your/model.pth
```

---

## Quick Reference

### Getting Started
- [QUICK_START.md](QUICK_START.md) - Training commands and setup
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Quick reference guide

### Development & Training
- [docs/guides/PREPROCESSING_GUIDE.md](docs/guides/PREPROCESSING_GUIDE.md) - Dataset preprocessing for faster training
- [docs/guides/EVALUATION_GUIDE.md](docs/guides/EVALUATION_GUIDE.md) - Model evaluation guide
- [docs/guides/INFERENCE_MODEL_GUIDE.md](docs/guides/INFERENCE_MODEL_GUIDE.md) - Model inference guide
- [docs/guides/INFERENCE_WITH_FILTERING_README.md](docs/guides/INFERENCE_WITH_FILTERING_README.md) - Full inference pipeline with filtering

### Deployment
- [deployment/PI_DEPLOYMENT_QUICKSTART.md](deployment/PI_DEPLOYMENT_QUICKSTART.md) - Quick start deployment guide
- [deployment/RASPBERRY_PI_DEPLOYMENT_GUIDE.md](deployment/RASPBERRY_PI_DEPLOYMENT_GUIDE.md) - Comprehensive Pi deployment
- [deployment/FILTERING_DEPLOYMENT_GUIDE.md](deployment/FILTERING_DEPLOYMENT_GUIDE.md) - Audio filtering deployment
- [deployment/SOUND_LOCALIZATION_INTEGRATION.md](deployment/SOUND_LOCALIZATION_INTEGRATION.md) - Sound localization integration
- [deployment/raspberry_pi/PI_DEPLOYMENT_STEPS.md](deployment/raspberry_pi/PI_DEPLOYMENT_STEPS.md) - Step-by-step deployment
- [deployment/raspberry_pi/README_FILTERING.md](deployment/raspberry_pi/README_FILTERING.md) - Filtering details for Pi

### Documentation & Resources
- [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Quick reference guide
- [docs/DATASET_CREDITS.md](docs/DATASET_CREDITS.md) - Dataset attribution
- [docs/DATASET_SUMMARY.md](docs/DATASET_SUMMARY.md) - Dataset overview
- [docs/FILTERING_IMPROVEMENTS.md](docs/FILTERING_IMPROVEMENTS.md) - Audio filtering research and technical details
- [docs/TEST_YOUR_AUDIO.md](docs/TEST_YOUR_AUDIO.md) - Audio testing guide
- [docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md) - Visualization tools
- [docs/resources/](docs/resources/) - Research papers and references

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
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4

# Training
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50
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
- [Deployment Guide](deployment/raspberry_pi/PI_DEPLOYMENT_STEPS.md)
- [Audio Filtering Guide](deployment/FILTERING_DEPLOYMENT_GUIDE.md)

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

**Version:** 2.0
**Last Updated:** November 2025
