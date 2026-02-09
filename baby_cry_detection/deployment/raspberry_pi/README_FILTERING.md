# Audio Filtering -- Raspberry Pi

Quick reference for the phase-preserving audio filtering pipeline used in real-time baby cry detection on Pi.

---

## Filtering Pipeline

```
[Input: 4-channel audio -- shape (num_samples, 4)]
         |
         v
[Voice Activity Detection]
  - Energy-based frame thresholding
  - Zero-crossing rate analysis
  - Spectral energy check in cry band (250-800 Hz on Pi)
  - Only frames with detected activity are processed further

         |
         v
[High-Pass Filter -- 100 Hz, 5th-order Butterworth]
  - Removes low-frequency rumble
  - Applied independently per channel with identical coefficients

         |
         v
[Band-Pass Filter -- 100-3000 Hz, 4th-order Butterworth]
  - Isolates cry frequency range including upper harmonics
  - Applied independently per channel with identical coefficients

         |
         v
[Spectral Subtraction -- strength 0.3]
  - Noise profile estimated from initial silence
  - Spectral floor at 0.3x input magnitude to preserve content
  - Phase preserved per channel -- no cross-channel mixing

         |
         v
[Output: 4-channel filtered audio -- phase relationships intact]
  - Inter-channel time differences preserved for TDOA and beamforming
  - Ready for downstream sound localization
```

Using identical filter coefficients per channel and performing spectral subtraction independently per channel ensures that the inter-channel phase relationships required for sound localization are never disturbed.

---

## What Is Enabled on Pi

| Filter | Status | Overhead | Purpose |
|--------|--------|----------|---------|
| High-pass (100 Hz) | Enabled | 0.2% | Remove low-frequency rumble |
| Band-pass (100-3000 Hz) | Enabled | 0.2% | Isolate cry frequency range |
| Spectral subtraction (0.3) | Enabled | 0.8% | Reduce background noise |
| Voice Activity Detection | Enabled | 0.3% | Skip silent frames, save CPU |
| Deep spectrum features | Disabled | -- | Too slow for real-time |
| MFCC deltas | Disabled | -- | Not needed |
| Spectral contrast | Disabled | -- | Too slow |
| Chroma features | Disabled | -- | Too slow |
| **Total enabled overhead** | | **~1.5%** | |

---

## Key Files

- `config_pi.py` -- Pi-optimized configuration. Standalone, no `src/` imports required. Filtering parameters: `HIGHPASS_CUTOFF`, `BANDPASS_LOW`, `BANDPASS_HIGH`, `NOISE_REDUCE_STRENGTH`.
- `audio_filtering.py` -- Implementation: `VoiceActivityDetector`, `NoiseFilter`, `AudioFilteringPipeline`.

---

## Usage

### Automatic (Recommended)

Filtering is applied automatically when the detector uses `ConfigPi`. No additional configuration needed.

```python
from config_pi import ConfigPi

config = ConfigPi()
# AudioFilteringPipeline is used internally by the detector
```

### Manual

```python
from config_pi import ConfigPi
from audio_filtering import AudioFilteringPipeline
import numpy as np

config = ConfigPi()
pipeline = AudioFilteringPipeline(config)

# 1 second of 4-channel audio
audio = np.random.randn(16000, 4).astype(np.float32)

result = pipeline.preprocess_audio(audio, apply_vad=True, apply_filtering=True)
filtered = result['filtered']   # shape (16000, 4) -- phase preserved
vad_mask = result['vad_mask']   # activity mask
```

### VAD Gating (Save CPU)

```python
from audio_filtering import VoiceActivityDetector

vad = VoiceActivityDetector(sample_rate=16000)
activity_mask, confidence = vad.detect_activity(audio)

if activity_mask.mean() > 0.3:
    # Activity detected -- run model inference
    prediction = model.predict(audio)
```

---

## Benchmarking

Run the included performance test on your Pi:

```bash
python3 test_pi_filtering.py
```

Note: `test_pi_filtering.py` is currently undergoing a bug fix (invalid keyword
argument). Confirm the script runs cleanly before relying on its output.

Expected results on Pi 5:

- Filtering time: 15-20 ms per second of audio
- Real-time factor: 0.015-0.020
- Total with model inference: 200-250 ms per second of audio

---

## Troubleshooting

**Filtering is too slow:** Reduce spectral subtraction strength in `config_pi.py` -- set `NOISE_REDUCE_STRENGTH = 0.2`.

**Disable filtering temporarily:**

```python
config = ConfigPi()
config.USE_ADVANCED_FILTERING = False
```

**Import errors:** Ensure you are running from the `deployment/raspberry_pi/` directory, or add it to `sys.path`.
