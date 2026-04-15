# Audio Sample Rate Flow

End-to-end path from mic recording through baby cry detection and sound localization. The pipeline is **strictly sequential** — BCD's multi-stage detection must complete before SoundLocalization receives audio.

## Pipeline Overview (Live Mode)

```
Step 1: RECORD (48kHz, 4ch)
──────────────────────────────────────────────
Microphone (PCM6260-Q1, 4ch)
Records at 48kHz via AudioPipeline
    │
    ├─► 48kHz chunks go into circular audio_buffer
    │   (kept for DOAnet phase accuracy)
    │
    └─► Each chunk resampled to 16kHz for BCD
        (detector._resample_to_model_rate)
        │
        ▼

Step 2: BCD DETECTION (16kHz)
──────────────────────────────────────────────
Quick detection on 16kHz chunk:
  - Multichannel voting (SNR-based dual-channel)
    OR single-channel fallback
  - Model confidence threshold: 0.92
    │
    ▼
Temporal smoothing:
  - 3+ consecutive high-confidence predictions required
  - Filters transient false positives (door slams, brief noises)
    │
    │  If smoothing criteria NOT met → continue listening
    ▼
DetectionResult:
  - is_cry=True
  - confidence (model score)
  - audio_buffer (original 48kHz 4ch from circular buffer)
  - cry_regions (timestamps from segment scoring)
    │
    ▼

Step 3: CRY REGION EXTRACTION (48kHz)
──────────────────────────────────────────────
Cry timestamps mapped back to 48kHz ORIGINAL
from the audio_buffer (not the 16kHz resample)
    │
    │  e.g., cry at 1.2s-2.5s → extract those
    │  samples from the 48kHz 4ch buffer
    │
    │  Fallback: if cry_regions is empty,
    │  uses full 48kHz buffer instead
    ▼
Filtered 48kHz 4ch audio (cry windows only)
    │
    ▼

Step 4: SOUND LOCALIZATION (48kHz SELD) + DISTANCE (48kHz, separate module)
──────────────────────────────────────────────
Filtered 48kHz cry audio is handed to two independent consumers
that both run at 48kHz:

(a) SELD CRNN — Direction of Arrival
    FunctionCalls_SL.py → function_calls.py:Infer.process_file()
    │
    │  Audio written to a temp WAV at 48kHz
    │
    ▼
    cls_feature_class._load_audio() calls
    librosa.load(sr=self._fs)  ← self._fs = 48000 (from doanet_parameters.py)
    │
    ▼
    Features computed at 48kHz:
    - 4ch log-mel spectrograms (4 × 64 mel bins)
    - 6ch GCC-PHAT (6 mic pairs × 64 bins)
    → 10-channel input to CRNN
    → Output: bearing angle (degrees).  DOA only — no distance.

(b) Distance Regressor (sklearn) — OUTSIDE SoundLocalization/
    Pi_Integration/record_samtry.py::predict_ml_distance()
    │
    │  Mono downmix of the same 48kHz cry audio
    │
    ▼
    Features: RMS mean/std, spectral centroid mean/std,
    13 MFCC means + 13 MFCC stds = 30 features
    → sklearn regressor → Output: distance in feet.

The orchestrator combines (a) + (b) into {direction_deg, distance_ft}
before sending NAV over UART.
```

## Sample Rates by Component

| Step | Component               | File                                                     | Sample Rate | Why                                                               |
| ---- | ----------------------- | -------------------------------------------------------- | ----------- | ----------------------------------------------------------------- |
| 1    | Mic capture             | `audio_pipeline.py`                                      | **48kHz**   | Hardware native rate, 48kHz kept in circular buffer for phase     |
| 2    | BCD detection           | `detector.py:detect_cry()`                               | **16kHz**   | Resample per-chunk, 0.92 confidence threshold                     |
| 2    | Temporal smoothing      | `temporal_smoother.py`                                   | **16kHz**   | Operates on confidence scores, not audio                          |
| 2    | Multichannel voting     | `multichannel_detector.py`                               | **16kHz**   | SNR-based channel selection, dual-channel weighted voting         |
| 3    | Cry region extraction   | `main.py:_localize_from_buffer()`                        | **48kHz**   | Timestamps from BCD mapped to 48kHz circular buffer               |
| 4a   | SELD features (GCC+mel) | `cls_feature_class.py:_load_audio()`                     | **48kHz**   | `doanet_parameters.py` sets `fs=48000` — GCC-PHAT needs raw phase |
| 4b   | Distance features       | `Pi_Integration/record_samtry.py::predict_ml_distance()` | **48kHz**   | Mono downmix, run in parallel with `Infer.process_file()`         |

## BCD Detection Stages Detail

The BCD pipeline uses **multichannel voting + temporal smoothing**:

### Detection
- Runs on every incoming 16kHz chunk
- Multichannel voting (SNR-based dual-channel) or single-channel fallback
- Model confidence threshold: 0.92 (bimodal distribution — 94.3% of predictions are <0.1 or >0.9)
- **Purpose**: fast, accurate screening per chunk

### Temporal Smoothing
- Requires 3+ consecutive high-confidence frames above 0.6
- Sliding window of 5 recent predictions
- Filters transient false positives (door slams, brief noises)
- **Purpose**: eliminate false positives before waking the robot

## Key Details

- **The pipeline is sequential, not parallel.** BCD detection + temporal smoothing must complete before SL receives audio.
- **SL receives filtered 48kHz audio** — only the cry windows extracted from the 48kHz circular buffer, not the full recording.
- **SELD stays at 48kHz end-to-end.** Audio enters `FunctionCalls_SL.py` at 48kHz, `Infer` writes it to a temp WAV, and `FeatureClass.extract_features_for_file()` reads it back with `librosa.load(sr=self._fs)` where `self._fs = 48000`. No resampling.
- **GCC-PHAT phase resolution at 48kHz** is ~20.8μs per sample. At ~5cm mic spacing and 343 m/s sound speed, the maximum TDOA is ~145μs — about 7 samples at 48kHz. Resampling down to 24kHz would cut that to ~3.5 samples and lose phase information GCC-PHAT relies on.
- **BCD processes at 16kHz but the 48kHz original is preserved.** The circular buffer stores raw 48kHz chunks. Cry timestamps are mapped back by sample index math (`int(start * 48000)`).
- **Distance prediction lives outside SoundLocalization/.** `Infer.process_file()` is DOA-only; `Pi_Integration/record_samtry.py::predict_ml_distance()` consumes the same 48kHz cry audio and returns distance independently. The orchestrator combines the two.

## RELISTEN vs Initial Detection

RELISTEN no longer runs a separate recording. When the ESP32 returns `RELISTEN`, the orchestrator transitions into the `RELISTEN` state with BCD still streaming. The next confirmed cry flows through the *same* `_on_cry_detected` callback as the initial detection and spawns a fresh `_localize_and_navigate` thread — there is no `_localize_from_fresh_recording` code path.

| Scenario                                     | Audio Source                               | BCD Stage                                               | What SL Receives                              |
| -------------------------------------------- | ------------------------------------------ | ------------------------------------------------------- | --------------------------------------------- |
| First detection (`LISTENING` → `LOCALIZING`) | 48kHz circular buffer                      | Full pipeline: multichannel voting → temporal smoothing | Filtered 48kHz cry regions from buffer        |
| RELISTEN (robot reported dead end)           | 48kHz circular buffer of the next live cry | Same full pipeline via `_on_cry_detected`               | Same — filtered 48kHz cry regions from buffer |
| Fallback (cry detected but no regions)       | Full 48kHz buffer                          | Same as above                                           | Unfiltered full 48kHz audio                   |
| Test mode (`--test-audio`)                   | Audio file at original SR                  | `detect_from_audio()` only                              | Filtered cry regions at original SR           |

> [!note] RELISTEN has a silence timer
> On entering `RELISTEN`, the orchestrator arms a 60-second silence timer (`RELISTEN_TIMEOUT`). If no fresh cry arrives, the timer fires, the state returns to `LISTENING`, and the relisten counter resets. `MAX_RELISTEN=3` is enforced per cry event across thread boundaries via the persistent `_relisten_count`.
