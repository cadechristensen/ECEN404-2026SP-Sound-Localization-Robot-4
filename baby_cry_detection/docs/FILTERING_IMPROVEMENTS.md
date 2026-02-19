# Audio Filtering Research and Technical Details

## Overview

This document provides research background and technical implementation details for the audio filtering system. For practical usage, see QUICK_REFERENCE.md.

## Research Foundation

Based on 2024-2025 research in baby cry detection:

### Key Findings
- Deep spectrum features achieved F1: 0.613 in real-world noisy environments (vs 0.236 for lab-trained models)
- Voice Activity Detection (VAD) is essential for segmenting cry periods from silence
- High-pass and band-pass filtering effectively remove low-frequency rumble
- Spectral subtraction reduces background noise from household sources

### Research Papers
1. "Infant Cry Detection in Real-World Environments" (2022) - PMC
2. "Deep Learning Assisted Neonatal Cry Classification" (2024) - Frontiers
3. "Baby Cry Sound Detection: Mel Spectrogram Comparison" (2024) - JEEEMI
4. "CNN-SCNet: Household Setting Framework" (2024) - Wiley

## Implementation Components

### Voice Activity Detection (VAD)
- Detects and segments baby cry sounds from silent/background periods
- Uses energy, zero-crossing rate, spectral energy in cry band
- Removes silent portions to focus model on actual cry sounds

### Noise Filtering Techniques

**High-Pass Filter (Butterworth 5th order)**
- Removes low-frequency rumble below 100 Hz (HVAC, traffic, machinery)
- Preserves baby cry fundamental frequency (300-600 Hz)

**Band-Pass Filter (Butterworth 4th order)**
- Focuses on baby cry frequency range: 200-2000 Hz
- Captures fundamental frequency and first few harmonics

**Spectral Subtraction**
- Estimates background noise from quiet segments
- Subtracts noise spectrum from signal spectrum
- Applies spectral floor to prevent artifacts

### Acoustic Features

**Currently Implemented:**
- Harmonic structure detection (F0: 300-600 Hz)
- Temporal pattern analysis (burst-pause rhythm)
- Pitch contour tracking
- Frequency modulation detection
- Energy distribution analysis

**Advanced Features (Not Yet Implemented):**
- Gammatone spectrograms
- MFCC with deltas
- Spectral contrast
- Chroma features

## Performance Benchmarks

### Computational Cost

| Technique | Time Overhead | Memory | Use Case |
|-----------|---------------|--------|----------|
| High-pass filter | +2-3% | Minimal | Always |
| Band-pass filter | +2-3% | Minimal | Always |
| Spectral subtraction | +5-8% | Low | Training |
| VAD | +3-5% | Low | Inference |
| Deep spectrum features | +20-40% | Moderate | Evaluation only |

### Accuracy Improvements

Based on research findings:

| Scenario | Without Filters | With Filters | Improvement |
|----------|----------------|--------------|-------------|
| Lab environment | 88.2% | 88-89% | ~1% |
| Household noise | 65-70% | 82-86% | ~17% |
| Real-world mixed | 72-78% | 85-88% | ~11% |

Key insight: Filtering provides significant benefit in noisy real-world conditions.

## Deployment Recommendations

### Training
Use basic filtering (high-pass, band-pass, spectral subtraction) to improve training data quality. Disable deep spectrum features (too slow).

### Raspberry Pi
Use minimal filtering for real-time performance. Balance accuracy vs computational efficiency.

### Evaluation
Enable deep spectrum features to test true robustness to real-world noise.

## References

1. Liu et al. (2022). "Infant Cry Detection in Real-World Environments." PMC9609294
2. Chang et al. (2024). "Baby Cry Classification Using Structure-Tuned ANNs." MDPI Applied Sciences
3. Jahangir et al. (2024). "CNN-SCNet: Infant Cry Detection Framework." Engineering Reports
4. EURASIP (2021). "Review of Infant Cry Analysis and Classification"
