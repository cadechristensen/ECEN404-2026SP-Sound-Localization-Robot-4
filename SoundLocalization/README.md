# Sound Localization

Estimates the bearing of a detected cry from a 4-channel microphone array using an angle-specific CRNN (DOAnet). Runs alongside the baby cry detection pipeline on the Raspberry Pi 5 and feeds navigation commands to the ESP32.

> **Scope note:** `Infer` returns direction of arrival only. Distance to the source is computed separately in `Pi_Integration/record_samtry.py::predict_ml_distance()`; the two are combined by the orchestrator before the NAV command is sent over UART.

---

## Overview

Instead of a single global model covering all 360°, this subsystem uses multiple CRNN checkpoints — each trained on a subset of angles — and selects the right one at inference time from the per-channel cry-band RMS pattern. The selected model then predicts a continuous bearing within its quadrant.

**Inference pipeline:**

1. Load 4-channel audio at 48 kHz (phase-preserved from the BCD filter)
2. Compute cry-band (300–3000 Hz) RMS for each channel
3. Rank channels → choose the angle-specific model via RMS ratio
4. Extract 10-channel features (4 log-mel + 6 GCC-PHAT pair features)
5. CRNN forward → DOA unit vector + per-source activity score
6. Select the active source whose bearing is closest to the estimated angle
7. Return the bearing as a formatted string

---

## Entry Points

| Use case | Class / module | Notes |
|----------|----------------|-------|
| File-based inference | `Infer.process_file(wav_path)` in `function_calls.py` | Loads all angle-specific models at construction |
| Live streaming from the mic array | `AudioInferenceEngine` in `RecordLive.py` | Opens PyAudio, buffers chunks, feeds `Infer` |
| Orchestrator integration | `Pi_Integration/FunctionCalls_SL.py::SoundLocalization` | Wraps `Infer` + external distance model for the state machine |

---

## Model Architecture (DOAnet CRNN)

Defined in `doanet_model.py`; hyperparameters in `doanet_parameters.py` under `task_id = '6'`.

- **CNN front-end:** 3 × 2-D conv blocks (128 filters) with `MaxPool2d` + `Dropout2d`
- **Recurrent core:** 3-layer bidirectional GRU (hidden size 256) with multiplicative gating
- **Attention:** Multi-head self-attention (4 heads, 256-dim)
- **DOA head:** Two 128-wide FNN layers → Linear → `tanh` → (x, y, z) per source
- **Activity head:** Two 128-wide FNN layers → Linear → sigmoid activity score
- **Output:** Up to 2 simultaneous sources (`unique_classes = 2`)

Bearing is recovered as `arctan2(y, x) % 360`.

---

## Input Features (10 channels)

| Channels | Feature | Notes |
|----------|---------|-------|
| 0 – 3 | Per-channel log-mel spectrogram | 64 mel bins, 48 kHz input, hop = 0.02 s |
| 4 – 9 | GCC-PHAT between all mic pairs | ⁴C₂ = 6 pairs |

Feature extraction lives in `cls_feature_class.FeatureClass`. Global normalization scaler: `models/mic_wts` (joblib).

---

## Angle-Specific Model Selection

The mic array places one microphone in each 45° quadrant:

| Channel | Mic bearing |
|---------|-------------|
| 0 | 135° |
| 1 | 315° |
| 2 | 45°  |
| 3 | 225° |

At inference, `Infer._select_model()`:

1. Computes cry-band RMS for each channel (bandpass 300–3000 Hz)
2. Ranks channels → `top1`, `top2`
3. If `top1_rms / top2_rms < 1.3` → sound is "between" the two mics → use the midpoint model
   - `{45°, 135°} → 90°`
   - `{135°, 225°} → 180°`
   - `{225°, 315°} → 270°`
   - `{315°, 45°}  → 0°`
   - Non-adjacent pairs fall back to the dominant channel's model
4. If `top1_rms / top2_rms ≥ 1.3` → one channel clearly dominates → use that channel's model directly

The specific checkpoint file for each angle bucket is configured in `Infer._MODEL_FILES` and can be swapped without changing the selection logic.

---

## File Layout

```
SoundLocalization/
├── function_calls.py        # Infer class — file-based inference entry point
├── RecordLive.py            # AudioInferenceEngine — live PyAudio → Infer
├── doanet_model.py          # CRNN architecture (CNN + BiGRU + self-attention)
├── doanet_parameters.py     # Model hyperparameters (task_id='6', fs=48000)
├── cls_feature_class.py     # 10-channel feature extraction (log-mel + GCC-PHAT)
├── train_sl.py              # Offline training script for the CRNN
├── distance_train.py        # Trainer for the external distance regressor
├── models/                  # Inference artifacts — CRNN checkpoints + mic_wts scaler
└── sl_training_data/        # Labeled training clips for the CRNN
```

---

## Usage

### File-based inference

```python
from function_calls import Infer

engine = Infer()                          # loads angle-specific models at construction
result = engine.process_file("clip.wav")
print(result)                             # "Source 0: 123.4° (Loudness: 0.85)"
```

### Through the Pi orchestrator

The production path is `Pi_Integration/FunctionCalls_SL.py::SoundLocalization`, which wraps `Infer`, pairs it with the external distance model, and returns a `{direction_deg, distance_ft}` dict that the state machine hands directly to the ESP32 over UART.

---

## Audio Requirements

- **Sample rate:** 48 kHz (do **not** downsample from 16 kHz — GCC-PHAT requires phase information that cannot be recovered after resampling)
- **Channels:** 4 (TI PCM6260-Q1 USB card, ALSA name "T20")
- **Mic spacing:** ≈ 5 cm (≈ 145 μs time-of-flight difference at 90°)
- **Clip length:** 3 s of cry-rich audio is the sweet spot

---

## Limitations

- Only eight discrete angle buckets. Bearing is returned as a continuous value but the model underlying each quadrant can bias the estimate toward bucket centers.
- `FeatureClass` is re-instantiated inside every `process_file()` call, so mel filter weights are recomputed. Caching this in `Infer.__init__` is a known optimization.
- Model files are untracked in git due to size — check the external release tarball or team storage for the latest weights.

---

## Related Documentation

- [Project Overview](../README.md) — top-level architecture and system flow
- [Audio Sample-Rate Flow](../Pi_Integration/AUDIO_SAMPLE_RATE_FLOW.md) — how 16 kHz BCD and 48 kHz SL pipelines coexist on the Pi
- [Baby Cry Detection README](../baby_cry_detection/README.md) — upstream subsystem that supplies the cry-isolated 48 kHz buffer
