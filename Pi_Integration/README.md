# Pi Integration

Top-level orchestrator that runs on the Raspberry Pi 5. It starts each subsystem (baby cry detection, sound localization, obstacle avoidance, mobile alerts), wires them together through thin wrapper classes, and drives the end-to-end state machine from *listening for a cry* through *navigate toward the source* to *caregiver notified*.

---

## Entry Points

| File                   | Purpose                                                                       |
| ---------------------- | ----------------------------------------------------------------------------- |
| `main.py`              | Orchestrator — defines `State`, `Orchestrator`, and `main()`                  |
| `FunctionCalls_BCD.py` | Wraps `BabyCryDetection` — exposes streaming + buffer APIs                    |
| `FunctionCalls_SL.py`  | Wraps DOAnet `Infer` + distance model; returns `{direction_deg, distance_ft}` |
| `FunctionCalls_OA.py`  | Wraps UART serial link to the ESP32                                           |
| `FunctionCalls_App.py` | Wraps the Flask alert server                                                  |
| `record_samtry.py`     | Live multichannel capture + ML distance prediction                            |

Launch the full system:

```bash
python3 Pi_Integration/main.py --device-name "TI USB Audio" -q
```

---

## State Machine

States are defined in `main.State` and transitions are managed by the `Orchestrator` class.

| State        | Behavior                                                                                             |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| `LISTENING`  | BCD stream running; `_on_cry_detected` callback fires on a confirmed cry                             |
| `LOCALIZING` | Daemon thread consumes the 48 kHz detection buffer and runs DOAnet + distance                        |
| `NAVIGATING` | `NAV angle=... dist_ft=...` sent to ESP32; waits for `READY` / `OBSTACLE` / `RELISTEN`               |
| `RELISTEN`   | ESP32 reported a dead end; orchestrator waits (up to 60 s) for a fresh cry via the live BCD callback |
| `ARRIVED`    | Robot at target; app alert fired; settles 30 s before returning to `LISTENING`                       |

**Key constants:** `ARRIVED_DURATION=30s`, `NAV_TIMEOUT=120s`, `MAX_RELISTEN=3`, `RELISTEN_TIMEOUT=60s`.

---

## Detection Buffer Flow

Two-pass detection avoids the blind window that a stop-and-re-record approach would create:

1. **Streaming pass** — `FunctionCalls_BCD` feeds 1 s chunks through the 6-stage detection pipeline. On a confirmed cry, `_on_cry_detected` is called with a `DetectionResult` holding the 48 kHz buffer and in-clip cry regions.
2. **Buffer pass** — `_localize_and_navigate` (daemon thread) consumes that same buffer and runs DOAnet on the cry regions. No audio is discarded; the mic stream keeps running.

`RELISTEN` reuses the exact same callback — the next detected cry populates a fresh `_last_detection` and spawns a new localize-and-navigate thread. The relisten counter survives across that thread boundary so `MAX_RELISTEN` is enforced per cry event, not per thread.

---

## Subsystem Wrappers (`FunctionCalls_*`)

Each wrapper keeps the orchestrator independent of its subsystem's internals.

| Wrapper                              | Exposes                                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `FunctionCalls_BCD.BabyCryDetection` | `start(on_cry_callback)`, `stop()`, `reset()`, `detect_from_audio()`, `filter_for_localization()` |
| `FunctionCalls_SL.SoundLocalization` | `localize(audio, sr)` → `{direction_deg, distance_ft, sources}`                                   |
| `FunctionCalls_OA`                   | `send_nav_command(direction_deg, distance_ft)`, `wait_for_response(timeout)`                      |
| `FunctionCalls_App.MobileApp`        | `start_server()`, `send_alert(confidence)`, `reset_notification()`, `stop()`                      |

---

## Audio Sample-Rate Strategy

Audio is captured at 48 kHz, 4 channels. See [AUDIO_SAMPLE_RATE_FLOW.md](AUDIO_SAMPLE_RATE_FLOW.md) for the full detail. Summary:

- Baby cry detection operates at 16 kHz (downsampled in-stream).
- Sound localization operates at 48 kHz (required for GCC-PHAT phase information).
- Audio filtering is phase-preserving so SL can run on the original 48 kHz buffer.

---

## Error Recovery

- **Nav timeout** → recovers to `LISTENING` and resets BCD rather than staying stuck in `NAVIGATING`.
- **App crash** → `FunctionCalls_App` returns `None`; the orchestrator continues without notifications.
- **Shutdown** → `threading.Event` is set; daemon threads drain; hard-exit after 3 s.
- **Relisten race** → the silence timer is armed under `_state_lock` so racing cry callbacks either see the prior state (ignored) or `RELISTEN` (handled).

---

## File Layout

```
Pi_Integration/
├── main.py                       # Orchestrator — State enum + Orchestrator class
├── FunctionCalls_BCD.py          # BCD streaming / buffer wrapper
├── FunctionCalls_SL.py           # DOAnet + distance wrapper
├── FunctionCalls_OA.py           # UART link to ESP32
├── FunctionCalls_App.py          # Flask alert server wrapper
├── record_samtry.py              # Live capture + ML distance predictor
├── generate_sl_training_data.py  # Helper — produces SL training clips from live audio
├── test_nav.py                   # Navigation harness (live + simulated modes)
├── test_live_orchestrator.py     # End-to-end orchestrator smoke test
├── test_mic_levels.py            # Per-channel RMS check for the mic array
└── AUDIO_SAMPLE_RATE_FLOW.md     # Sample-rate handling across subsystems
```

---

## Related Documentation

- [Project Overview](../README.md) — full system architecture and quick start
- [Audio Sample-Rate Flow](AUDIO_SAMPLE_RATE_FLOW.md) — where 16 kHz and 48 kHz audio live
- [Baby Cry Detection](../baby_cry_detection/README.md) — upstream detector that supplies the buffer
- [Sound Localization](../SoundLocalization/README.md) — DOAnet bearing estimation
- [Obstacle Avoidance](../ObstacleAvoidance/README.md) — ESP32 firmware that receives NAV commands
