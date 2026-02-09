# Raspberry Pi 5 Deployment Guide for Baby Cry Detection

## Executive Summary

Comprehensive guide for deploying the baby cry detection system on Raspberry Pi 5. The system uses a CNN-Transformer hybrid model with two-stage detection, dual-channel voting, and temporal smoothing for false-positive reduction. All 4 microphone channels are preserved with phase intact throughout the pipeline for downstream sound localization.

- Model accuracy: 96% (desktop, FP32). Quantization is not active in this deployment
- Target noise tolerance: 65 dB SPL (validated to 56.4 dB SPL)
- False positive rate target: below 10%

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Deployed Files](#deployed-files)
3. [Optimization Strategy](#optimization-strategy)
4. [Resource Requirements](#resource-requirements)
5. [Running the System](#running-the-system)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)

---

## System Architecture

The detection pipeline runs in stages. Stages 1-4 are fully implemented. Stages 5-6 have placeholder interfaces for future integration.

```
[4-channel mic input -- phase preserved throughout]
         |
         v
[Stage 1: Low-Power Multichannel Voting]
  - Select best 2 channels by SNR
      Signal band: 300-900 Hz
      Noise bands: 50-200 Hz + 1500-4000 Hz
  - Dual-channel voting (SNR-weighted exponential or logical-OR)
  - Fast inference, no TTA
  - Threshold: 75% confidence
  - CPU: 15-25%, Latency: ~50 ms

         |
         v  (if cry detected)
[Stage 2: Temporal Smoothing Gate]
  - Sliding window of 5 recent predictions
  - Requires 3+ consecutive frames above 0.6 confidence
  - Filters transient false positives (door slams, brief noises)
  - Resets after confirmed detection

         |
         v  (if smoothing passes)
[Stage 3: Confirmation with Health Check and TTA]
  - Capture 5 seconds context (all 4 channels)
  - Channel health checks:
      RMS minimum 0.001 (detect disconnection)
      Clipping threshold 0.95
      Inter-channel coherence minimum 0.7 (300-1000 Hz band)
  - Test-Time Augmentation (3 transforms: original, noisy, time-shifted)
  - Threshold: 85% confidence
  - CPU: 40-60%, Latency: 200-300 ms
  - Detection cooldown: 2 seconds

         |
         v  (if confirmed)
[Stage 4: Phase-Preserving Audio Filtering]
  - High-pass: 100 Hz, 5th-order Butterworth
  - Band-pass: 100-3000 Hz, 4th-order Butterworth
  - Spectral subtraction: 0.3 strength, per-channel
  - Output: filtered 4-channel audio ready for localization

         |
         v
[Stage 5: Sound Localization -- placeholder]
  - Receives 4-channel audio via multiprocessing queue
  - TDOA / GCC-PHAT / beamforming algorithms
  - See SOUND_LOCALIZATION_INTEGRATION.md

         |
         v
[Stage 6: Robot Navigation -- placeholder]
  - Path planning and movement execution
```

---

## Deployed Files

All production code lives in `deployment/raspberry_pi/`. The detector imports `src.model` and `src.audio_filter` from the project root via a `sys.path` insertion at module load; the project root must be present on the Pi. All other runtime dependencies (`config_pi`, `multichannel_detector`, `audio_filtering`, etc.) are local to the deployment folder.

| File | Role |
|------|------|
| `realtime_baby_cry_detector.py` | Main detector: two-stage architecture, multichannel voting, TTA |
| `robot_baby_monitor.py` | Top-level orchestrator: detector + sound localization in separate processes |
| `multichannel_detector.py` | SNR computation, channel health monitoring, dual-channel voting |
| `temporal_smoother.py` | Sliding-window false-positive filter |
| `audio_buffer.py` | Thread-safe circular buffer for 4-channel audio |
| `audio_filtering.py` | VAD, high-pass, band-pass, spectral subtraction (phase-preserving) |
| `config_pi.py` | Standalone Pi-optimized configuration |
| `detection_types.py` | Shared DetectionResult dataclass |
| `sound_localization_interface.py` | Placeholder for localization model integration |
| `pi_setup.py` | One-time Pi setup: dependencies, audio config, systemd service |
| `pi_diagnostics.py` | Performance diagnostics and health checks |
| `requirements-pi.txt` | Python dependencies |
| `verify_raspberry_pi_files.sh` | Pre-flight verification script |

---

## Optimization Strategy

Four priority levels for edge deployment:

| Priority | Technique | Impact |
|----------|-----------|--------|
| 1 | Dynamic quantization (int8) -- currently disabled; FP32 model in use | 75% model size reduction, 2-3x speedup (when enabled) |
| 2 | ONNX Runtime export (optional) | 50-70% latency reduction on ARM |
| 3 | Audio pipeline optimization | VAD gating skips silent frames entirely |
| 4 | Knowledge distillation (optional) | 10-15 MB model at 90-93% accuracy |

### Quantization

Run on your development machine before transferring the model:

```bash
python scripts/quantize_model.py \
    --model results/<latest_train_dir>/model_best.pth \
    --output model_quantized.pth

# Validate -- expect less than 1% accuracy loss
python scripts/validate_quantization.py \
    --original results/<latest_train_dir>/model_best.pth \
    --quantized model_quantized.pth
```

Note: `config_pi.py` sets `QUANTIZE_MODEL = False` because the detection thresholds (0.75 listen, 0.85 confirm) are calibrated to FP32 probability outputs. If you switch to a quantized model, re-validate those thresholds against your test set before deploying.

---

## Resource Requirements

| Resource | Listening Mode | Detection Mode | Peak |
|----------|---------------|----------------|------|
| CPU | 15-25% | 40-60% | ~70-80% |
| Memory | ~135 MB total | same | same |
| Power | 3-4 W | 5-7 W | 7 W |
| Latency | ~50 ms per chunk | ~200-300 ms (with TTA) | -- |

Memory breakdown: model ~20 MB, 5-second 4-channel circular buffer ~5 MB, system overhead ~100 MB. All well within the 8 GB RAM on Pi 5.

Battery estimate: 4-6 hours on a 25 Wh battery.

---

## Running the System

### Prerequisites

Clone the repository onto the Pi and install dependencies:

```bash
git clone <your-github-repo-url> ~/baby_cry_detection
cd ~/baby_cry_detection

# System packages
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-dev \
    portaudio19-dev libasound2-dev alsa-utils pulseaudio \
    libopenblas-dev liblapack-dev libsndfile1 ffmpeg

# PyTorch (ARM CPU wheel -- install this before requirements-pi.txt)
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
pip3 install -r deployment/raspberry_pi/requirements-pi.txt
```

Or use the setup script:

```bash
cd deployment/raspberry_pi
python3 pi_setup.py --install-deps
```

### Find Your Audio Device

```bash
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'[{i}] {info[\"name\"]} (channels: {info[\"maxInputChannels\"]})')
p.terminate()
"
```

Identify the TI PCM6260-Q1 with 4 input channels. Note the device index -- it replaces `2` in the commands below.

### Run the Detector

```bash
cd deployment/raspberry_pi

python3 realtime_baby_cry_detector.py \
    --model ../calibrated_model.pth \
    --device-index 2 \
    --channels 4 \
    --device cpu \
    --enable-multichannel \
    --multichannel-voting weighted \
    --enable-temporal-smoothing \
    --temporal-window-size 5 \
    --temporal-min-consecutive 3
```

### Run the Full System (Detector + Sound Localization)

```bash
python3 robot_baby_monitor.py \
    --model ../calibrated_model.pth \
    --device-index 2 \
    --enable-multichannel \
    --multichannel-voting weighted \
    --enable-temporal-smoothing
```

### Test Mode (No Hardware Required)

```bash
python3 realtime_baby_cry_detector.py \
    --model ../calibrated_model.pth \
    --device cpu \
    --test-mode
```

---

## Performance Tuning

### Thread Configuration

Set before running the detector:

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

### Threshold Adjustments

All thresholds are tunable via CLI flags or directly in `config_pi.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--detection-threshold` | 0.75 | Stage 1 sensitivity. Lower = more sensitive |
| `--confirmation-threshold` | 0.85 | Stage 2 strictness. Higher = fewer false alarms |
| `--temporal-min-consecutive` | 3 | Consecutive high-confidence frames required before alert |
| `--temporal-window-size` | 5 | Prediction sliding window size |

### Power Management

```bash
# Low-power mode for idle periods
echo "powersave" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Performance mode during active detection
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| CPU consistently above 80% | Excessive TTA transforms or thread contention | Reduce TTA transform count; set OMP_NUM_THREADS=2; check for background processes |
| Latency above 200 ms | Memory swapping or thread contention | Run `free -h`; set OMP_NUM_THREADS=2 |
| High false positive rate | Noisy environment | Increase confirmation-threshold to 0.90 or temporal-min-consecutive to 4 |
| Missing real cries | Threshold too strict | Lower detection-threshold to 0.65 |
| Memory errors | Insufficient free RAM | Reduce buffer duration; close other processes. Quantization is available as a last resort if memory pressure persists |
| Audio device not found | Wrong device index | Re-run device discovery; check PCM6260-Q1 connection |
| Import errors | Missing Python packages | Run `python3 pi_setup.py --install-deps` |

### Target Specifications

- Inference: below 100 ms (Stage 1), below 300 ms (Stage 2 with TTA)
- CPU: 15-25% listening, 40-60% active
- Memory: ~135 MB total
- Detection rate: 90%+ at 56.4 dB SPL household noise
- False positive rate: below 10%

---

## References

- PyTorch Mobile: https://pytorch.org/mobile/home/
- ONNX Runtime: https://onnxruntime.ai/
- Raspberry Pi Docs: https://www.raspberrypi.com/documentation/computers/processors.html
- Filtering details: `deployment/raspberry_pi/README_FILTERING.md`
- Sound localization integration: `deployment/SOUND_LOCALIZATION_INTEGRATION.md`
