#!/usr/bin/env python3
"""
Test script for the robot navigation system.

Six modes:

  Manual UART pokers (no BCD, no SL — just write NAV strings):
    --angle <deg>          Send a single NAV command and wait for ESP32 reply
    --interactive          Type angle/distance pairs at a prompt
    --preset               Run a fixed sequence of preset turn angles
    --audio <path.wav>     Run SL on a wav file then send the result

  Live pipeline (mic capture -> BCD -> SL -> NAV, like main.py):
    --live-turn            Turn toward each cry source but DO NOT drive
                           (distance clamped to TURN_ONLY_DIST_FT)
    --live-drive           Turn + drive with obstacle avoidance
                           (full main.py path, minus the mobile app)

Usage examples:
    # Manual modes
    python test_nav.py --angle 90 --distance 3
    python test_nav.py --interactive
    python test_nav.py --audio test_cry.wav --device-name "TI USB Audio"

    # Live modes (require a 4-mic array and a connected ESP32)
    python test_nav.py --live-turn  --device-name "TI USB Audio" -q
    python test_nav.py --live-drive --device-name "TI USB Audio" -q
"""

import argparse
import serial
import time
import sys
import os

# Module-level constant referenced by both run_interactive() and the
# lazy-loaded LiveTestOrchestrator class below.
#
# Distance BELOW the ESP32 arrival threshold (1.64ft / 0.5m). The
# 404NewObstacleAvoidance2.py firmware turns to face the sound source
# UNCONDITIONALLY ("Face the sound source first (always, even at
# threshold)" — firmware:382), then checks the arrival threshold.
# With dist_ft <= 1.64ft, the threshold check fires immediately after
# the turn → robot rotates and stops without driving.
#
# 1.0ft (~0.305m) gives a comfortable margin below the 0.5m threshold so
# encoder/calibration noise can't push remaining_m back over. Do NOT
# raise above ~1.6ft without retesting — anything >= 1.64ft will commit
# the robot to driving the full distance.
TURN_ONLY_DIST_FT = 1.0


# LiveTestOrchestrator is defined lazily by _ensure_live_test_orchestrator()
# below to avoid paying the cost of importing main.py (which transitively
# pulls in tensorflow, librosa, pyaudio, soundfile, etc.) when running
# the lightweight manual modes (--angle, --interactive, --audio, --preset).
LiveTestOrchestrator = None


def _ensure_live_test_orchestrator():
    """Define `LiveTestOrchestrator` on first call.

    Lazy-loads `main.Orchestrator` and defines a thin subclass that:

    1. Overrides `_send_and_wait_esp32` so that when `turn_only=True` the
       distance sent to the ESP32 is clamped to `TURN_ONLY_DIST_FT` (1.0ft,
       below the 0.5m arrival threshold). The firmware turns first then
       checks arrival, so the robot rotates to face the cry and stops
       without driving.
    2. Shortens `ARRIVED_DURATION` from 30s to 5s so the test loop returns
       to LISTENING quickly between cries.

    Everything else (BCD callback, SL inference, RELISTEN, BUMPED handling,
    shutdown) is inherited from `main.Orchestrator` unchanged.

    Idempotent — safe to call repeatedly. After the first call,
    `test_nav.LiveTestOrchestrator` is the subclass.
    """
    global LiveTestOrchestrator
    if LiveTestOrchestrator is not None:
        return LiveTestOrchestrator

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import Orchestrator  # noqa: E402  (lazy: heavy ML/audio deps)

    class _LiveTestOrchestrator(Orchestrator):
        ARRIVED_DURATION = 5.0  # overrides Orchestrator.ARRIVED_DURATION = 30.0

        def __init__(self, args, turn_only: bool):
            self._turn_only = turn_only
            super().__init__(args)

        def _localize_and_navigate(self):
            # Drive mode behaves exactly like main.py — no patching.
            if not self._turn_only:
                return super()._localize_and_navigate()

            # Turn-only mode: the SL/ML distance is irrelevant because
            # _send_and_wait_esp32 will clamp the value to TURN_ONLY_DIST_FT
            # anyway. The inherited validation block in main.Orchestrator
            # rejects 0/inf/NaN distances and returns to LISTENING before
            # _send_and_wait_esp32 ever gets called — which would mean a
            # broken distance pipeline silently disables --live-turn even
            # when SL gives us a perfectly valid angle.
            #
            # Workaround: monkey-patch main.predict_ml_distance to return
            # TURN_ONLY_DIST_FT for the duration of this call. The validation
            # then passes, _send_and_wait_esp32 receives (angle, TURN_ONLY_DIST_FT),
            # and our existing override clamps it to TURN_ONLY_DIST_FT (no-op).
            import main  # already loaded by this point — cheap

            real_predict = main.predict_ml_distance
            main.predict_ml_distance = lambda *args, **kwargs: TURN_ONLY_DIST_FT
            try:
                return super()._localize_and_navigate()
            finally:
                main.predict_ml_distance = real_predict

        def _send_and_wait_esp32(self, angle_deg, distance_ft):
            # NOTE: positional super-call mirrors main.Orchestrator's
            # signature exactly. If main.Orchestrator._send_and_wait_esp32
            # signature changes, this needs an update — the unit test in
            # test_live_orchestrator.py is the canary.
            if self._turn_only:
                distance_ft = TURN_ONLY_DIST_FT
            return super()._send_and_wait_esp32(angle_deg, distance_ft)

    LiveTestOrchestrator = _LiveTestOrchestrator
    return LiveTestOrchestrator


# Module-level reference so the ctypes callback isn't garbage-collected.
# A dangling C function pointer would segfault the process. Same fix
# main.py uses (main.py:46-48, main.py:560-572).
_alsa_error_handler = None


def _suppress_alsa_warnings():
    """Install a no-op ALSA error handler to silence libasound stderr noise.

    Used by the live test modes (--live-turn, --live-drive) which open the
    mic array via BabyCryDetection. On Linux/Pi this loads libasound.so.2
    and registers a no-op error callback. On non-Linux platforms this is
    a no-op (the OSError from missing libasound is swallowed).
    """
    global _alsa_error_handler
    os.environ["JACK_NO_START_SERVER"] = "1"
    import ctypes  # lazy — manual modes never call this function

    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so.2")
        _alsa_error_handler = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
        )(lambda *_: None)
        asound.snd_lib_error_set_handler(_alsa_error_handler)
    except OSError:
        pass  # not on Linux / no libasound — fine, nothing to suppress


def _configure_live_logging(debug: bool, quiet: bool) -> None:
    """Configure root + module loggers for live test modes.

    Mirrors main.py:537-553 — same format, same DEBUG/INFO gating, same
    quiet allowlist. Kept as a separate copy here so test_nav.py can be
    used without touching main.py (per the design constraint), with a
    minor delta: includes "test_nav" in the allowlist defensively in
    case a future module logger is added.
    """
    import logging  # lazy: only paid by live modes

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
        for name in (
            "__main__",
            "main",
            "test_nav",  # defensive — preempts future module-logger addition
            "FunctionCalls_BCD",
            "FunctionCalls_SL",
            "FunctionCalls_OA",
        ):
            logging.getLogger(name).setLevel(logging.INFO)


def run_live_test(args, turn_only: bool):
    """Live BCD -> SL -> NAV pipeline test.

    Mirrors main.py's startup but skips the mobile app entirely. When
    turn_only is True the LiveTestOrchestrator clamps NAV distance to
    TURN_ONLY_DIST_FT (below the firmware arrival threshold) so the
    robot turns to face the cry source and stops without driving. When
    turn_only is False, the full main.py pipeline runs (turn + drive +
    obstacle avoidance).

    Args:
        args: argparse Namespace from main(). Must have: device_index,
            device_name, no_app, single_model, quiet (and any other
            attributes main.Orchestrator.__init__ reads).
        turn_only: True for --live-turn, False for --live-drive.

    Note:
        This function mutates `args.no_app` in place (forces it to True).
        Callers passing the same Namespace into multiple dispatchers
        should be aware of the side effect.
    """
    # Live modes never use Flask/camera/email — force them off regardless
    # of any --no-app flag the user passed (or didn't pass).
    args.no_app = True

    # Resolve --device-name to a numeric device index (mirrors main.py)
    if args.device_name and args.device_index is None:
        from record_samtry import find_device_by_name  # lazy

        args.device_index = find_device_by_name(args.device_name)
        if args.device_index is None:
            print(
                f"ERROR: No audio device matching --device-name "
                f"{args.device_name!r} was found."
            )
            sys.exit(1)

    if args.device_index is None:
        print(
            "ERROR: Live modes require --device-index or --device-name "
            "to select a microphone array."
        )
        sys.exit(1)

    _suppress_alsa_warnings()

    # Lazy-create the LiveTestOrchestrator class (also triggers the
    # heavyweight main.py / FunctionCalls_* / record / record_samtry
    # imports — this is the natural pay-the-cost boundary for live modes).
    _ensure_live_test_orchestrator()

    mode_name = "TURN-ONLY" if turn_only else "FULL DRIVE"
    print(f"\nStarting live test: {mode_name}")
    print(f"  Device index: {args.device_index}")
    if turn_only:
        print(f"  Distance override: {TURN_ONLY_DIST_FT}ft (turn only, no drive)")
    else:
        print(f"  Distance:          from SL (full drive + OA)")
    print()

    runner = LiveTestOrchestrator(args, turn_only=turn_only)
    runner.run()


BAUD_RATE = 115200
SERIAL_PORT = "/dev/serial0"


def open_serial(port, baud):
    try:
        s = serial.Serial(port, baud, timeout=1)
        time.sleep(0.5)  # let ESP32 settle
        print(f"Serial opened: {port} @ {baud}")
        return s
    except Exception as e:
        print(f"ERROR: Cannot open {port}: {e}")
        sys.exit(1)


def send_nav(ser, angle_deg, distance_ft):
    """Send NAV command and print ESP32 response."""
    cmd = f"NAV angle={angle_deg:.3f} dist_ft={distance_ft:.3f}\n"
    print(f"\n{'='*50}")
    print(f"  Sending:    {cmd.strip()}")
    print(f"  Angle:      {angle_deg:.1f} deg", end="")
    if abs(angle_deg) < 5:
        print(" (FORWARD)")
    elif angle_deg > 0:
        print(f" (RIGHT {angle_deg:.0f} deg)")
    else:
        print(f" (LEFT {abs(angle_deg):.0f} deg)")
    print(f"  Distance:   {distance_ft:.1f} ft ({distance_ft * 0.3048:.2f} m)")
    print(f"{'='*50}")

    ser.write(cmd.encode())

    # Wait for terminal response (READY, RELISTEN, or BUMPED)
    print("  Waiting for ESP32 response...", end="", flush=True)
    start = time.time()
    timeout = 60  # seconds
    terminal = {"READY", "RELISTEN", "BUMPED"}
    while time.time() - start < timeout:
        if ser.in_waiting:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                elapsed = time.time() - start
                if line == "BUMPED":
                    print(f"\n  *** BUMP SENSOR TRIGGERED *** ({elapsed:.1f}s)")
                    print("  Robot halted — power cycle ESP32 to resume")
                    return line
                elif line in terminal:
                    print(f"\n  Response:   {line} ({elapsed:.1f}s)")
                    return line
                else:
                    # Non-terminal (e.g. OBSTACLE) — print and keep waiting
                    print(
                        f"\n  ESP32:      {line} ({elapsed:.1f}s)", end="", flush=True
                    )
        time.sleep(0.1)

    print(f"\n  TIMEOUT after {timeout}s — no response from ESP32")
    return None


def run_interactive(ser):
    """Interactive mode: type angle,distance pairs to test turns."""
    print("\nInteractive mode — test robot navigation")
    print("Type: <angle>            (turn only, no driving)")
    print("      <angle> <dist_ft>  (turn + drive)")
    print("      'q' to quit\n")
    print("Common test angles:")
    print("    0   = straight ahead")
    print("   45   = 45 deg right")
    print("   90   = 90 deg right")
    print("  180   = behind (turns right 180)")
    print("  270   = 90 deg left  (normalizes to -90)")
    print("  315   = 45 deg left  (normalizes to -45)")
    print("  -90   = 90 deg left")
    print()

    # TURN_ONLY_DIST_FT is defined at module scope (below the firmware's
    # 0.5m / 1.64ft arrival threshold). The firmware turns first then
    # checks arrival, so the robot rotates and stops without driving.

    while True:
        try:
            raw = input("nav> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() == "q":
            break

        parts = raw.split()
        try:
            angle = float(parts[0])
            if len(parts) > 1:
                dist = float(parts[1])
            else:
                dist = TURN_ONLY_DIST_FT
                print(f"  (turn only — sending {dist}ft to trigger immediate arrival)")
        except (ValueError, IndexError):
            print("  Usage: <angle> [distance_ft]")
            continue

        send_nav(ser, angle, dist)


def run_sl_test(ser, audio_path, device_name, single_model=None):
    """Run SL on audio file, then send result to ESP32."""
    # Add project paths
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sl_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "SoundLocalization"
    )
    sys.path.insert(0, sl_dir)

    from FunctionCalls_SL import SoundLocalization
    import soundfile as sf

    print(f"Loading audio: {audio_path}")
    audio, sample_rate = sf.read(audio_path)
    num_channels = audio.shape[1] if audio.ndim > 1 else 1
    print(f"  {audio.shape[0]/sample_rate:.1f}s, {num_channels}ch, {sample_rate}Hz")

    print("Initializing SL...")
    sl = SoundLocalization(single_model=single_model)

    print("Running localization...")
    loc = sl.localize(
        audio_data=audio,
        sample_rate=sample_rate,
        num_channels=num_channels,
    )

    angle = loc["direction_deg"]
    dist = loc["distance_ft"]
    sources = loc["sources"]

    print(f"\nSL Result:")
    print(f"  Direction:  {angle:.1f} deg")
    print(f"  Distance:   {dist:.1f} ft")
    print(f"  Sources:    {sources}")

    confirm = input(
        f"\nSend NAV angle={angle:.1f} dist={dist:.1f}ft to robot? [y/N] "
    ).strip()
    if confirm.lower() == "y":
        send_nav(ser, angle, dist)
    else:
        print("Cancelled.")


def run_preset_tests(ser):
    """Run a sequence of preset angles with short distance to test turns only."""
    print(
        "\nPreset turn tests (distance=0.6ft / 0.18m — just enough to clear arrival threshold)"
    )
    print(
        "The robot should turn to face each direction, drive briefly, then send READY.\n"
    )

    tests = [
        (0, "Straight ahead"),
        (45, "45 deg right"),
        (90, "90 deg right"),
        (180, "Behind (180 deg)"),
        (270, "90 deg left (270 -> -90)"),
        (315, "45 deg left (315 -> -45)"),
    ]

    for angle, desc in tests:
        print(f"\n--- Test: {desc} (angle={angle}) ---")
        confirm = input("  Press Enter to send (or 's' to skip, 'q' to quit): ").strip()
        if confirm.lower() == "q":
            break
        if confirm.lower() == "s":
            continue
        send_nav(ser, angle, 2.0)
        time.sleep(1)  # pause between tests


def main():
    parser = argparse.ArgumentParser(description="Test robot navigation turns via UART")
    parser.add_argument(
        "--port",
        default=SERIAL_PORT,
        help="Serial port (manual modes only — live modes use /dev/serial0)",
    )
    parser.add_argument(
        "--baud", type=int, default=BAUD_RATE, help="Baud rate (manual modes only)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--angle", type=float, help="Angle in degrees to send (manual)")
    group.add_argument(
        "--interactive", action="store_true", help="Interactive manual NAV mode"
    )
    group.add_argument("--audio", type=str, help="Audio file to run SL on (one-shot)")
    group.add_argument("--preset", action="store_true", help="Run preset turn tests")
    group.add_argument(
        "--live-turn",
        action="store_true",
        help="Live BCD -> SL -> turn only (no drive)",
    )
    group.add_argument(
        "--live-drive",
        action="store_true",
        help="Live BCD -> SL -> full drive with OA (production path)",
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=2.0,
        help="Distance in feet (default: 2) — manual modes only",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="PyAudio device index (live modes / --audio)",
    )
    parser.add_argument(
        "--device-name",
        type=str,
        default=None,
        help="Find audio device by name substring (live modes / --audio)",
    )
    parser.add_argument(
        "--single-model",
        type=str,
        default=None,
        help="Use a single SL model for all angles (e.g. New_Test_Model.h5)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Mute INFO messages (live modes only)",
    )
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Disable mobile app (forced on for --live-turn/--live-drive regardless)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging (live modes only)"
    )

    args = parser.parse_args()

    # --- Live modes: skip open_serial(), use Orchestrator's UART ---
    if args.live_turn or args.live_drive:
        _configure_live_logging(debug=args.debug, quiet=args.quiet)
        run_live_test(args, turn_only=args.live_turn)
        return

    # --- Manual modes: existing behavior, unchanged ---
    ser = open_serial(args.port, args.baud)
    try:
        if args.interactive:
            run_interactive(ser)
        elif args.preset:
            run_preset_tests(ser)
        elif args.audio:
            run_sl_test(
                ser, args.audio, args.device_name, single_model=args.single_model
            )
        else:
            send_nav(ser, args.angle, args.distance)
    finally:
        ser.close()
        print("\nSerial closed.")


if __name__ == "__main__":
    main()
