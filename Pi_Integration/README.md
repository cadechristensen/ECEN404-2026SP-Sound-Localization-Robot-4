# Pi Integration

Top-level orchestrator that runs on the Raspberry Pi 5. It starts each subsystem (baby cry detection, sound localization, obstacle avoidance, mobile alerts), wires them together through thin wrapper classes, and drives the end-to-end state machine from *listening for a cry* through *navigate toward the source* to *caregiver notified*.

---

## Entry Points

| File | Purpose |
|------|---------|
| `main.py` | Orchestrator — defines `State`, `Orchestrator`, and `main()` |
| `FunctionCalls_BCD.py` | Wraps `BabyCryDetection` — exposes streaming + buffer APIs |
| `FunctionCalls_SL.py` | Wraps DOAnet `Infer` + distance model; returns `{direction_deg, distance_ft}` |
| `FunctionCalls_OA.py` | Wraps UART serial link to the ESP32 |
| `FunctionCalls_App.py` | Wraps the Flask alert server |
| `record_samtry.py` | Live multichannel capture + ML distance prediction |

Launch the full system:

```bash
python3 Pi_Integration/main.py --device-name "TI USB Audio" -q
```

---

## State Machine

States are defined in `main.State` and transitions are managed by the `Orchestrator` class.

| State | Behavior |
|-------|----------|
| `LISTENING` | BCD stream running; `_on_cry_detected` callback fires on a confirmed cry |
| `LOCALIZING` | Daemon thread consumes the 48 kHz detection buffer and runs DOAnet + distance |
| `NAVIGATING` | `NAV angle=... dist_ft=...` sent to ESP32; waits for `READY` / `OBSTACLE` / `RELISTEN` |
| `RELISTEN` | ESP32 reported a dead end; orchestrator waits (up to 60 s) for a fresh cry via the live BCD callback |
| `FINAL_TURN` | ESP32 reported `READY`; orchestrator relistens once more (up to 20 s) and sends a turn-only NAV (`dist_ft=0.01`) to face the refined bearing before transitioning to `ARRIVED` |
| `ARRIVED` | Robot at target; app alert fired; settles 30 s before returning to `LISTENING` |

**Key constants:** `ARRIVED_DURATION=30s`, `NAV_TIMEOUT=120s`, `MAX_RELISTEN=3`, `RELISTEN_TIMEOUT=60s`, `FINAL_TURN_TIMEOUT=20s`, `FINAL_TURN_NAV_TIMEOUT=10s`, `FINAL_TURN_ANGLE_DEADBAND=5°`, `FINAL_TURN_MAX_WORLD_DEVIATION=45°`.

### FINAL_TURN refinement flow

Between `NAVIGATING` (ESP32 replied `READY`) and `ARRIVED`, the orchestrator pauses in `FINAL_TURN` to catch one more cry and turn in place toward the refined bearing. Up to `MAX_RELISTEN=3` refinement attempts are allowed per event. A fresh cry triggers a second SL pass (angle only, no distance); if the refined angle is inside the ±5° deadband, the turn is skipped. Otherwise `FunctionCalls_OA.send_turn_command(angle)` emits `NAV angle=X dist_ft=0.01`, and the ESP32's existing NAV handler turns-then-replies-READY without driving forward. Silence timeouts, deadband hits, SL failures, and exhausted retries all fall through to `ARRIVED` so the caregiver alert is not gated on the refinement succeeding.

### World-frame heading cross-check

Each terminal UART reply from the ESP32 carries a cumulative `heading=<float>` token (the robot's world-frame orientation since boot). The Pi stores this in `_current_world_heading`. On the FIRST localization of a cry event, the orchestrator computes `_event_baseline_bearing = wrap(current_heading + angle_deg)` — the world-frame bearing to the source at the start of the event. In `FINAL_TURN`, the refined robot-relative angle is lifted to world-frame the same way, and the signed delta against the baseline is checked against `FINAL_TURN_MAX_WORLD_DEVIATION` (default 45°). Deltas larger than that are logged and the turn is skipped — typically close-range SL noise or a wrong DOAnet source pick. Baseline clears on `_return_to_listening`; `_current_world_heading` persists across events because it tracks the robot's physical orientation. Old ESP32 firmware without the `heading=` token leaves both attributes `None`, disabling the cross-check (behavior falls back to pre-feature FINAL_TURN).

---

## Detection Buffer Flow

Two-pass detection avoids the blind window that a stop-and-re-record approach would create:

1. **Streaming pass** — `FunctionCalls_BCD` feeds 1 s chunks through the 6-stage detection pipeline. On a confirmed cry, `_on_cry_detected` is called with a `DetectionResult` holding the 48 kHz buffer and in-clip cry regions.
2. **Buffer pass** — `_localize_and_navigate` (daemon thread) consumes that same buffer and runs DOAnet on the cry regions. No audio is discarded; the mic stream keeps running.

`RELISTEN` reuses the exact same callback — the next detected cry populates a fresh `_last_detection` and spawns a new localize-and-navigate thread. The relisten counter survives across that thread boundary so `MAX_RELISTEN` is enforced per cry event, not per thread.

---

## Subsystem Wrappers (`FunctionCalls_*`)

Each wrapper keeps the orchestrator independent of its subsystem's internals.

| Wrapper | Exposes |
|---------|---------|
| `FunctionCalls_BCD.BabyCryDetection` | `start(on_cry_callback)`, `stop()`, `reset()`, `detect_from_audio()`, `filter_for_localization()` |
| `FunctionCalls_SL.SoundLocalization` | `localize(audio, sr)` → `{direction_deg, distance_ft, sources}` |
| `FunctionCalls_OA` | `send_nav_command(direction_deg, distance_ft)`, `send_turn_command(direction_deg)` (turn-only for FINAL_TURN), `send_cancel()`, `wait_for_response(timeout)` — after a terminal reply, `last_response_heading` holds the ESP32's world-frame heading (or `None` if the reply did not carry one) |
| `FunctionCalls_App.MobileApp` | `start_server()`, `send_alert(confidence)`, `reset_notification()`, `stop()` |

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
├── test_mode.py                  # Dry-run state-machine stepper
├── test_mic_levels.py            # Per-channel RMS check for the mic array
├── bcd_recording_test.py         # BCD-only recording harness
└── AUDIO_SAMPLE_RATE_FLOW.md     # Sample-rate handling across subsystems
```

---

## Related Documentation

- [Project Overview](../README.md) — full system architecture and quick start
- [Audio Sample-Rate Flow](AUDIO_SAMPLE_RATE_FLOW.md) — where 16 kHz and 48 kHz audio live
- [Baby Cry Detection](../baby_cry_detection/README.md) — upstream detector that supplies the buffer
- [Sound Localization](../SoundLocalization/README.md) — DOAnet bearing estimation
- [Obstacle Avoidance](../ObstacleAvoidance/README.md) — ESP32 firmware that receives NAV commands
