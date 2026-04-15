"""
Combines all Functions for Pi Integration

This system uses a 4-microphone array and Raspberry Pi 5 to detect baby cries
in real-time, filters the audio, and determines the baby's location using sound
localization. The robot navigates towards the baby's location using obstacle
avoidance and ESP32 UART commands. Once the robot arrives at the baby's location
the caregiver gets a notification of the confidence level of the detected cry
and a live video to view from the camera on the robot.

State machine:
    LISTENING   -> Low-power cry detection active
    LOCALIZING  -> Cry detected, running sound localization
    NAVIGATING  -> ESP32 navigating to baby with obstacle avoidance
    ARRIVED     -> Robot at baby, streaming video + email sent
    RELISTEN    -> ESP32 hit dead end, re-localize

# Normal live mode
    python Pi_Integration/main.py --device-name "TI USB Audio" -q
    python Pi_Integration/main.py --device-name "TI USB Audio" -q --no-app
    python Pi_Integration/main.py --device-index 0 -q

# For testing at different pipeline depths, use test_mode.py:
    python Pi_Integration/test_mode.py --device-name "TI USB Audio" --bcd -q
    python Pi_Integration/test_mode.py --device-name "TI USB Audio" --bcd-sl -q
    python Pi_Integration/test_mode.py --device-name "TI USB Audio" --bcd-sl-oa -q


"""

import argparse
import enum
import logging
import os
import sys
import threading
import time

# Silence JACK/ALSA C-level stderr noise BEFORE importing anything that touches audio.
# These env vars stop JACK from trying to autostart a server; FunctionCalls_BCD
# wraps the pipeline start in an fd-level stderr redirect to catch anything the
# env vars don't suppress.
os.environ.setdefault("JACK_NO_START_SERVER", "1")
os.environ.setdefault("JACK_NO_AUDIO_RESERVATION", "1")

import numpy as np

from FunctionCalls_OA import ObstacleAvoidance
from FunctionCalls_SL import SoundLocalization
from record_samtry import predict_ml_distance

# Keep module-level reference to the ALSA error handler so it is never
# garbage-collected while the process is running (a dangling C callback would segfault).
_alsa_error_handler = None

logger = logging.getLogger(__name__)


class State(enum.Enum):
    LISTENING = "LISTENING"
    LOCALIZING = "LOCALIZING"
    NAVIGATING = "NAVIGATING"
    ARRIVED = "ARRIVED"
    RELISTEN = "RELISTEN"


class Orchestrator:
    """Central state machine coordinating all subsystems."""

    ARRIVED_DURATION = 30.0    # seconds to stay in ARRIVED before returning to LISTENING
    MAX_RELISTEN = 3           # max ESP32 RELISTEN responses allowed per cry event
    RELISTEN_TIMEOUT = 60.0    # seconds to wait in RELISTEN for a fresh cry before giving up
    NAV_TIMEOUT = 120.0        # seconds before treating ESP32 as hung
    MIN_CRY_DURATION = 5.0     # minimum seconds of cry audio required for localization

    def __init__(self, args):
        self.state = State.LISTENING
        self._state_lock = threading.Lock()
        self._last_detection = None
        self._shutdown = threading.Event()
        self._device_index = args.device_index
        # Relisten tracking — survives across thread boundaries because a RELISTEN
        # response ends the current _localize_and_navigate thread and the NEXT cry
        # detection starts a new one.  The counter tells that new thread how many
        # relistens have already happened for the current cry event.
        self._relisten_count = 0
        self._relisten_timer = None  # threading.Timer, active only while in RELISTEN state
        try:
            self.oa = ObstacleAvoidance()
        except Exception as e:
            logger.error(f"Could not open UART: {e}")
            print("\nERROR: UART connection required for production mode.")
            print("Check that /dev/serial0 is available and the ESP32 is connected.")
            print("For testing without UART, use test_mode.py with --bcd or --bcd-sl.")
            sys.exit(1)

        # --- Initialize subsystems ---
        logger.info("Initializing subsystems...")

        from FunctionCalls_BCD import BabyCryDetection

        self.bcd = BabyCryDetection(
            device_index=args.device_index,
            verbose=not args.quiet,
        )

        if not args.no_app:
            from FunctionCalls_App import MobileApp
            self.app = MobileApp()
        else:
            self.app = None
            logger.info("Mobile app disabled (--no-app)")

        self.sl = SoundLocalization(single_model=getattr(args, 'single_model', None))

        logger.info("All subsystems initialized")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _set_state(self, new_state: State) -> None:
        with self._state_lock:
            old = self.state
            self.state = new_state
        logger.info(f"State: {old.value} -> {new_state.value}")

    def _return_to_listening(self) -> None:
        """Atomic full-reset: clear all per-cry-event state and go LISTENING.

        This is the "full reset" path used on errors, timeouts, false positives,
        successful arrivals, and max-relisten giveup.  The state transition,
        counter reset, and timer cancellation all happen inside the state lock
        so a concurrent cry-detection callback cannot observe a half-reset
        Orchestrator (e.g. state=RELISTEN but counter=0).
        """
        with self._state_lock:
            prev_state = self.state
            was_navigating = prev_state in (State.NAVIGATING, State.LOCALIZING, State.RELISTEN)
            # Atomically clear per-event state under the lock.
            self._cancel_relisten_timer_locked()
            self._relisten_count = 0
            self.state = State.LISTENING

        if prev_state != State.LISTENING:
            logger.info(f"State: {prev_state.value} -> {State.LISTENING.value}")
        self._do_listening_cleanup(was_navigating=was_navigating)

    def _do_listening_cleanup(self, was_navigating: bool) -> None:
        """I/O-side cleanup after a transition to LISTENING.

        The caller must have ALREADY transitioned state atomically under
        the lock.  This method does the non-atomic parts (UART, BCD reset,
        app notification) outside the lock so slow I/O doesn't block
        concurrent cry callbacks.
        """
        if was_navigating:
            try:
                self.oa.send_cancel()
            except Exception as e:
                logger.warning(f"Failed to send CANCEL to ESP32: {e}")
        self.bcd.reset()
        if self.app:
            self.app.reset_notification()

    def _cancel_relisten_timer(self) -> None:
        """Cancel the relisten silence-timeout timer if one is active.

        Safe to call without holding the state lock — Timer.cancel() is
        thread-safe and we clear our reference under assumption the caller
        is the only writer.  For strict consistency with concurrent resets,
        prefer _cancel_relisten_timer_locked() from inside a state_lock block.
        """
        if self._relisten_timer is not None:
            self._relisten_timer.cancel()
            self._relisten_timer = None

    def _cancel_relisten_timer_locked(self) -> None:
        """Cancel the relisten timer.  Caller must hold self._state_lock."""
        if self._relisten_timer is not None:
            self._relisten_timer.cancel()
            self._relisten_timer = None

    # ------------------------------------------------------------------
    #! Cry detection callback (runs on detector's thread)
    # ------------------------------------------------------------------
    def _on_cry_detected(self, detection) -> None:
        """Called by BabyCryDetection when a cry is confirmed.

        Cries are accepted in BOTH LISTENING (fresh event) and RELISTEN
        (continuing an event after ESP32 asked us to retry).  The relisten
        counter is NOT reset here — it persists across the thread boundary
        so _enter_relisten() can enforce MAX_RELISTEN across multiple
        navigate attempts for the same baby.
        """
        # Write _last_detection atomically with the state change so the
        # navigation thread always sees the matching detection object.
        with self._state_lock:
            if self.state not in (State.LISTENING, State.RELISTEN):
                if self.state == State.ARRIVED:
                    logger.warning(
                        "Cry detected while ARRIVED — robot already at baby, "
                        "ignoring until LISTENING resumes"
                    )
                else:
                    logger.info(
                        f"Cry detected but state is {self.state.value} — ignoring"
                    )
                return
            if self._shutdown.is_set():
                return
            prev_state = self.state
            self._last_detection = detection
            self.state = State.LOCALIZING

        # A fresh cry arrived in time — cancel the silence-timeout timer if
        # we were in RELISTEN waiting for one.  Done outside the lock because
        # Timer.cancel() is thread-safe and we don't want to hold the state
        # lock across any I/O.
        if prev_state == State.RELISTEN:
            self._cancel_relisten_timer()

        logger.info(
            f"Cry confirmed (confidence={detection.confidence:.2%}), "
            f"State: {prev_state.value} -> {State.LOCALIZING.value}"
        )

        # Run localization + navigation on a separate thread so the
        # detector callback returns quickly.
        threading.Thread(target=self._localize_and_navigate, daemon=True).start()

    # ------------------------------------------------------------------
    #! UART communication with ESP32
    # ------------------------------------------------------------------
    def _send_and_wait_esp32(self, angle_deg, distance_ft):
        """
        Send NAV command over UART and block until ESP32 replies.

        Returns:
            True  if ESP32 sent READY (arrived at baby)
            False if ESP32 sent RELISTEN (obstacle avoided, need re-localization)
            None  if shutdown was requested, timeout, or BUMPED (emergency halt)
        """
        self.oa.send_nav_command(angle_deg, distance_ft)

        deadline = time.monotonic() + self.NAV_TIMEOUT

        while not self._shutdown.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error(
                    f"ESP32 navigation timed out after {self.NAV_TIMEOUT:.0f}s "
                    "— returning to LISTENING"
                )
                return None

            response = self.oa.wait_for_response(timeout=min(1.0, remaining))
            if response == "READY":
                logger.info("ESP32 ready — arrived at target")
                return True
            elif response == "RELISTEN":
                logger.info("ESP32 obstacle avoided — re-localizing")
                return False
            elif response == "BUMPED":
                logger.critical(
                    "ESP32 BUMP SENSOR TRIGGERED — robot halted. "
                    "Power cycle ESP32 to resume."
                )
                return None

        # Shutdown was requested
        return None

    # ------------------------------------------------------------------
    # Audio for localization
    # ------------------------------------------------------------------
    def _get_localization_audio(self):
        """Extract cry audio for direction + raw audio for distance.

        Delegates to BabyCryDetection so this class stays focused on
        state-machine logic.  Always uses the last detection's buffer —
        RELISTEN also reads from a fresh buffer because a fresh
        detection was captured by the live BCD pipeline before the
        cry callback fired.
        """
        with self._state_lock:
            detection = self._last_detection
        return self.bcd.extract_cry_audio_from_detection(
            detection, min_duration=self.MIN_CRY_DURATION,
        )

    # ------------------------------------------------------------------
    #! Localize -> Navigate -> Arrive pipeline
    # ------------------------------------------------------------------
    def _localize_and_navigate(self):
        """Run one localize + navigate cycle for the last detected cry.

        This function handles a SINGLE navigation attempt, not a loop.
        If the ESP32 responds with RELISTEN, the thread hands off to
        `_enter_relisten()` and exits.  The next cry detection (while
        in RELISTEN state) starts a fresh `_localize_and_navigate` thread,
        preserving the relisten counter across that boundary.
        """
        # --- Get audio ---
        try:
            cry_audio, sr, raw_audio = self._get_localization_audio()
        except Exception as e:
            logger.error(f"Failed to get localization audio: {e}", exc_info=True)
            self._return_to_listening()
            return

        # No cry regions — false positive, give up this cry event entirely
        if cry_audio is None:
            self._return_to_listening()
            return

        # --- Filter cry regions for direction (phase-preserving bandpass + spectral subtraction) ---
        filtered_audio = self.bcd.filter_for_localization(cry_audio)

        # --- Localize direction (filtered audio) + distance (raw audio) ---
        try:
            loc = self.sl.localize(
                audio_data=filtered_audio,
                sample_rate=sr,
                num_channels=filtered_audio.shape[1] if filtered_audio.ndim > 1 else 1,
            )
            angle_deg = loc['direction_deg']

            # Distance from raw audio (matches record_samtry.py)
            try:
                distance_ft = predict_ml_distance(raw_audio, sr)
                logger.info(f"ML distance prediction: {distance_ft:.1f} ft")
            except Exception as ml_err:
                logger.error(f"ML distance prediction failed: {ml_err}")
                distance_ft = loc['distance_ft']

            logger.info(
                f"Localization (relisten {self._relisten_count}/{self.MAX_RELISTEN}): "
                f"{angle_deg:.1f} deg at {distance_ft:.1f} ft — {loc['sources']}"
            )
        except Exception as e:
            logger.error(f"Localization failed: {e}", exc_info=True)
            self._return_to_listening()
            return

        if (distance_ft <= 0 or not np.isfinite(distance_ft)
                or not np.isfinite(angle_deg)):
            logger.warning(
                f"Invalid localization (angle={angle_deg}, dist={distance_ft}) "
                "— returning to LISTENING"
            )
            self._return_to_listening()
            return

        # --- Navigate ---
        self._set_state(State.NAVIGATING)
        try:
            arrived = self._send_and_wait_esp32(angle_deg, distance_ft)
        except Exception as e:
            logger.error(f"UART error during navigation: {e}", exc_info=True)
            self._return_to_listening()
            return

        if arrived is None:
            if not self._shutdown.is_set():
                logger.error("Navigation timed out — returning to LISTENING")
                self._return_to_listening()
            return

        if arrived:
            self._on_arrived()
            return

        # ESP32 requested re-localization — hand off to _enter_relisten, which
        # will either transition to RELISTEN (waiting for fresh cry via the
        # live BCD pipeline) or give up if MAX_RELISTEN is exceeded.  This
        # thread exits; the next cry callback will spawn a new one.
        self._enter_relisten()

    # ------------------------------------------------------------------
    #! Relisten handling — wait for a fresh cry via the live BCD pipeline
    # ------------------------------------------------------------------
    def _enter_relisten(self) -> None:
        """Transition to RELISTEN state to wait for a fresh cry.

        Increments the relisten counter.  If MAX_RELISTEN is exceeded,
        fully resets via _return_to_listening.  Otherwise sets state
        to RELISTEN and arms a silence-timeout timer: if no cry is
        heard within RELISTEN_TIMEOUT seconds, the timer fires and
        gives up.

        Does not block — the caller's thread exits after this returns.
        The next cry detection will spawn a fresh _localize_and_navigate
        thread that reads the (now-incremented) counter.
        """
        with self._state_lock:
            self._relisten_count += 1
            count = self._relisten_count
            prev_state = self.state
            # Cancel any previous timer before deciding next state — even if
            # we're about to give up, we must not leave an armed timer around.
            self._cancel_relisten_timer_locked()

            if count <= self.MAX_RELISTEN:
                # Atomically transition to RELISTEN so a racing cry callback
                # either sees prev_state (and gets ignored) or RELISTEN
                # (and gets accepted) — never a half-updated state.
                self.state = State.RELISTEN
                should_give_up = False
            else:
                should_give_up = True

        if should_give_up:
            logger.error(
                f"Max relisten count ({self.MAX_RELISTEN}) exceeded — "
                "giving up on this cry event"
            )
            self._return_to_listening()
            return

        logger.info(f"State: {prev_state.value} -> {State.RELISTEN.value}")
        logger.warning(
            f"ESP32 requested RELISTEN {count}/{self.MAX_RELISTEN} — "
            f"waiting {self.RELISTEN_TIMEOUT:.0f}s for a fresh cry"
        )

        # Re-arm BCD so it's ready to produce a new detection.  We do not
        # call _return_to_listening here because that would cancel the
        # cry event entirely (reset counter, reset notification).
        self.bcd.reset()

        # Arm the silence-timeout timer.  If a cry arrives, _on_cry_detected
        # cancels it.  If nothing arrives, it fires _relisten_timeout_expired.
        # We assign under the lock so _cancel_relisten_timer_locked calls
        # from other threads see a consistent reference.
        timer = threading.Timer(
            self.RELISTEN_TIMEOUT, self._relisten_timeout_expired
        )
        timer.daemon = True
        with self._state_lock:
            # Defensive: if state changed during bcd.reset() (shutdown, race),
            # don't arm the timer — it would fire with a stale assumption.
            if self.state == State.RELISTEN:
                self._relisten_timer = timer
                timer.start()
            else:
                logger.info("State changed during RELISTEN setup — not arming timer")

    def _relisten_timeout_expired(self) -> None:
        """Called when RELISTEN_TIMEOUT elapses without a fresh cry.

        Resets the cry event and returns to full LISTENING.  This is
        the Option-B timeout: a period of silence means the previous
        cry event is stale and a new cry should be treated as fresh.

        Race-safe: the state check AND the transition happen inside the
        same state_lock acquisition, so a cry callback racing with this
        timer cannot see stale state.
        """
        with self._state_lock:
            if self.state != State.RELISTEN:
                # A cry was detected just as the timer fired, or the
                # orchestrator has moved past RELISTEN for some other reason.
                # Whoever transitioned us out of RELISTEN owns the cleanup.
                return
            # Atomically claim the transition: counter + timer + state all
            # cleared under the lock, so any cry callback blocked on the lock
            # will wake up seeing state=LISTENING and treat it as fresh.
            self._cancel_relisten_timer_locked()
            self._relisten_count = 0
            self.state = State.LISTENING

        logger.info(f"State: {State.RELISTEN.value} -> {State.LISTENING.value}")
        logger.warning(
            f"Relisten silence timeout ({self.RELISTEN_TIMEOUT:.0f}s) — "
            "no fresh cry heard, giving up on this cry event"
        )
        # We were navigating (conceptually) before entering RELISTEN, so
        # tell ESP32 to stand down.
        self._do_listening_cleanup(was_navigating=True)

    # ------------------------------------------------------------------
    #! Arrival handling
    # ------------------------------------------------------------------
    def _on_arrived(self) -> None:
        """Robot has reached the baby."""
        # Successful navigation — the cry event is resolved, so clear relisten
        # tracking.  _return_to_listening() below also clears it, but doing it
        # here makes the invariant explicit for readers.
        self._cancel_relisten_timer()
        self._relisten_count = 0

        self._set_state(State.ARRIVED)
        with self._state_lock:
            confidence = self._last_detection.confidence if self._last_detection else 0.0

        logger.info("Robot arrived at baby - sending alert")
        if self.app:
            try:
                self.app.send_alert(confidence)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}", exc_info=True)

        logger.info(f"Waiting {self.ARRIVED_DURATION}s before returning to LISTENING")
        self._shutdown.wait(timeout=self.ARRIVED_DURATION)

        if not self._shutdown.is_set():
            self._return_to_listening()

    # ------------------------------------------------------------------
    #! Main run loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start all subsystems and block until shutdown."""
        if self.app:
            try:
                self.app.start_server()
            except Exception as e:
                logger.error(f"Flask app failed to start: {e}", exc_info=True)
                self.app = None

        self.bcd.start(on_cry_callback=self._on_cry_detected)

        # Wait for audio buffer to fill before accepting detections
        buf = self.bcd._detector.audio_buffer
        ctx = self.bcd._detector.context_duration
        print(f"\n  Filling audio buffer ({ctx:.0f}s)...", end="", flush=True)
        while not self._shutdown.is_set() and not buf.has_duration(ctx):
            self._shutdown.wait(timeout=0.5)
        print(" ready.\n")

        self._set_state(State.LISTENING)

        print("=" * 60)
        print("  Baby Monitor Robot - ACTIVE")
        print("=" * 60)
        print(f"  State:       {self.state.value}")
        if self.app:
            print(f"  Stream URL:  {self.app.stream_url}")
        print(f"  UART port:   {self.oa.port}")
        print("=" * 60)
        print("  Press Ctrl+C to stop")
        print()

        try:
            while not self._shutdown.is_set():
                self._shutdown.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully stop all subsystems."""
        logger.info("Shutting down orchestrator...")
        self._shutdown.set()
        self._cancel_relisten_timer()
        if self.app:
            self.app.stop()
        self.bcd.stop()
        self.oa.close()
        logger.info("Orchestrator shut down")

        def _force_exit():
            time.sleep(3)
            logging.shutdown()
            os._exit(0)
        threading.Thread(target=_force_exit, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(
        description='Baby Monitor Robot - Pi Integration Orchestrator'
    )
    parser.add_argument(
        '--device-index', type=int, default=None,
        help='PyAudio device index for microphone array',
    )
    parser.add_argument(
        '--device-name', type=str, default=None,
        help="Find device by name substring (e.g., 'TI USB Audio')",
    )
    parser.add_argument(
        '--no-app', action='store_true',
        help='Disable mobile app (Flask server, mDNS, camera)',
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable DEBUG-level logging',
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Mute BCD processing steps, debug output, and root logger INFO messages',
    )
    parser.add_argument(
        '--single-model', type=str, default=None,
        help='Use a single SL model for all angles (e.g. New_Test_Model.h5)',
    )

    args = parser.parse_args()

    # --- Logging ---
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        for name in (
            '__main__',
            'FunctionCalls_BCD',
            'FunctionCalls_SL',
            'FunctionCalls_OA',
            'FunctionCalls_App',
        ):
            logging.getLogger(name).setLevel(logging.INFO)

    # --- Device lookup ---
    if args.device_name:
        from record_samtry import find_device_by_name
        args.device_index = find_device_by_name(args.device_name)

    # --- Suppress ALSA/JACK warnings ---
    os.environ["JACK_NO_START_SERVER"] = "1"
    global _alsa_error_handler
    import ctypes
    try:
        asound = ctypes.cdll.LoadLibrary('libasound.so.2')
        _alsa_error_handler = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
        )(lambda *_: None)
        asound.snd_lib_error_set_handler(_alsa_error_handler)
    except OSError:
        pass

    # --- Run ---
    orchestrator = Orchestrator(args)
    orchestrator.run()


if __name__ == '__main__':
    main()
