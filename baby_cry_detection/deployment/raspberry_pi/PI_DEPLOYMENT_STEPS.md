# Raspberry Pi Deployment -- Quick Reference

This guide assumes you have cloned the full repository onto the Pi. All paths are relative to the repository root.

---

## Step 1: Install Dependencies

```bash
# System packages
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-dev \
    portaudio19-dev libasound2-dev alsa-utils pulseaudio \
    libopenblas-dev liblapack-dev libsndfile1 ffmpeg

# PyTorch ARM CPU wheel -- install this FIRST, before requirements-pi.txt
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
pip3 install -r deployment/raspberry_pi/requirements-pi.txt
```

Or use the setup script from the raspberry_pi folder:

```bash
cd deployment/raspberry_pi
python3 pi_setup.py --install-deps
```

---

## Step 2: Model -- Currently FP32 (Quantization Optional, Not Active)

The current deployment uses `calibrated_model.pth`, a full-precision (FP32)
model located at `deployment/calibrated_model.pth` in the repository root.
No quantization step is required. The model file must already be present
before running the detector.

Quantization is available but is currently disabled. The detection thresholds
(0.75 listen, 0.85 confirm) are calibrated to FP32 probability outputs. If
quantization is re-enabled in the future, those thresholds must be
re-validated against the test set before deploying. To quantize when needed:

```bash
# OPTIONAL -- not used in the current deployment
python3 scripts/quantize_model.py \
    --model results/<latest_train_dir>/model_best.pth \
    --output model_quantized.pth
```

---

## Step 3: Find Your Audio Device

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

Identify the TI PCM6260-Q1 with 4 input channels. Use that device index in Step 5.

---

## Step 4: Verify Deployment

```bash
cd deployment/raspberry_pi
bash verify_raspberry_pi_files.sh
```

All core scripts and imports must show as OK before proceeding.

---

## Step 5: Run the System

```bash
cd deployment/raspberry_pi

# Detector only (replace 2 with your device index)
python3 realtime_baby_cry_detector.py \
    --model ../calibrated_model.pth \
    --device-index 2 \
    --channels 4 \
    --device cpu \
    --enable-multichannel \
    --multichannel-voting weighted \
    --enable-temporal-smoothing

# Full system with sound localization
python3 robot_baby_monitor.py \
    --model ../calibrated_model.pth \
    --device-index 2 \
    --enable-multichannel \
    --multichannel-voting weighted \
    --enable-temporal-smoothing

# Test mode -- no audio hardware needed
python3 realtime_baby_cry_detector.py \
    --model ../calibrated_model.pth \
    --device cpu \
    --test-mode
```

---

## System Overview

### Two-Stage Detection

**Stage 1 -- Listening.** Processes 1-second audio chunks. Selects the 2 best microphone channels by SNR in the 300-900 Hz cry band. Runs dual-channel voting using either SNR-weighted exponential combination or logical-OR. Threshold: 75%. CPU usage: 15-25%.

**Stage 2 -- Confirmation.** Triggered by Stage 1. Captures 5 seconds of context across all 4 channels. Performs channel health checks: RMS minimum 0.001 (detects disconnection), clipping threshold 0.95, inter-channel coherence minimum 0.7 in the 300-1000 Hz band. Applies Test-Time Augmentation with 3 transforms. Threshold: 85%. CPU usage: 40-60%. Detection cooldown: 2 seconds between alerts.

### Temporal Smoothing

After Stage 2 confirmation, a sliding window of 5 recent predictions is evaluated. The alert triggers only when 3 or more consecutive frames exceed 0.6 confidence. This filters out transient false positives such as door slams or brief environmental noise. The window resets after each confirmed detection.

### Channel Selection

SNR is computed for each of the 4 microphone channels using cry-specific frequency bands:

- Signal band: 300-900 Hz (fundamental frequency and first harmonic)
- Noise bands: 50-200 Hz (low-frequency rumble) and 1500-4000 Hz (environmental noise)

The 2 highest-SNR channels are selected for dual-channel voting.

### Channel Health Monitoring

Before confirmation, each channel is validated:

- RMS below 0.001: channel is silent or disconnected
- Peak amplitude above 0.95: clipping detected
- Inter-channel coherence below 0.7 in 300-1000 Hz: microphone array integrity compromised

Detections are suppressed when channel health checks fail.

---

## Performance Reference

| Component | Latency | CPU |
|-----------|---------|-----|
| Audio capture (4 channels) | ~0.5 ms | minimal |
| Stage 1 inference | 50-100 ms | 15-25% |
| Stage 2 inference (TTA, 3 transforms) | 200-300 ms | 40-60% |
| Multichannel filtering (4 channels) | 150-200 ms | ~20% |
| Total cry-to-output | 500-700 ms | peak ~70% |

Memory footprint: 135-400 MB including the 5-second 4-channel circular buffer.

---

## Sound Localization Integration

On confirmed cry, the detector passes 4-channel audio to `detection_queue`:

```python
detection_data = {
    'raw_audio':      np.ndarray,   # shape (num_samples, 4) -- unfiltered
    'filtered_audio': np.ndarray,   # shape (num_samples, 4) -- cry regions only
    'sample_rate':    16000,
    'num_channels':   4,
    'confidence':     float,
    'timestamp':      float
}
```

Edit `sound_localization_interface.py` in the marked placeholder section to integrate your localization model. Reference algorithms (TDOA, GCC-PHAT, delay-and-sum beamforming) are documented in `../SOUND_LOCALIZATION_INTEGRATION.md`.

---

## Documentation

- Full deployment reference: `../RASPBERRY_PI_DEPLOYMENT_GUIDE.md`
- Filtering details: `README_FILTERING.md` (this folder)
- Sound localization algorithms: `../SOUND_LOCALIZATION_INTEGRATION.md`
- Setup script help: `python3 pi_setup.py --help`
