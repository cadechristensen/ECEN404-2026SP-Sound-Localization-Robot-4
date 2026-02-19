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
    python Pi_Integration/main.py --device-index 2 -q

# Test mode with an existing audio file
    python Pi_Integration/main.py --test-audio path/to/recording.wav -q
    
"""

import argparse
import enum
import logging
import os
import threading
import time
import numpy as np
import serial
from math import radians, cos, sin

from FunctionCalls_SL import SoundLocalization

ser = serial.Serial(
    port='/dev/serial0',
    baudrate=115200,
    timeout=1
    )

logger = logging.getLogger(__name__)


class State(enum.Enum):
    LISTENING = "LISTENING"
    LOCALIZING = "LOCALIZING"
    NAVIGATING = "NAVIGATING"
    ARRIVED = "ARRIVED"
    RELISTEN = "RELISTEN"


class Orchestrator:
    """Central state machine coordinating all subsystems."""

    # How long to stay in ARRIVED before returning to LISTENING
    ARRIVED_DURATION = 30.0  # seconds

    def __init__(self, args):
        self.state = State.LISTENING
        self._state_lock = threading.Lock()
        self._last_detection = None
        self._shutdown = threading.Event()
        self._test_audio = args.test_audio  # None in normal mode

        # --- Initialize subsystems ---
        logger.info("Initializing subsystems...")

        # BCD is needed in both live and test mode.
        # OA and App require Pi-specific packages (pyserial, flask)
        # so they are imported lazily and skipped in test mode.
        from FunctionCalls_BCD import BabyCryDetection

        self.bcd = BabyCryDetection(
            device_index=args.device_index,
            num_channels=args.channels,
            device=args.device,
            verbose=not args.quiet,
        )

        from FunctionCalls_App import MobileApp
        self.app = MobileApp()

        self.sl = SoundLocalization(
            models_dir=args.models_dir,
            task_id=args.task_id,
        )

        logger.info("All subsystems initialized")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _set_state(self, new_state: State) -> None:
        with self._state_lock:
            old = self.state
            self.state = new_state
        logger.info(f"State: {old.value} -> {new_state.value}")

    # ------------------------------------------------------------------
    #! Cry detection callback (runs on detector's thread)
    # ------------------------------------------------------------------
    def _on_cry_detected(self, detection) -> None:
        """Called by BabyCryDetection when a cry is confirmed."""
        with self._state_lock:
            if self.state != State.LISTENING:
                logger.info("Cry detected but not in LISTENING state — ignoring")
                return

        logger.info(f"Cry confirmed (confidence={detection.confidence:.2%}), transitioning to LOCALIZING")
        self._last_detection = detection
        self._set_state(State.LOCALIZING)

        # Run localization + navigation on a separate thread so the
        # detector callback returns quickly.
        threading.Thread(target=self._localize_and_navigate, daemon=True).start()

    # ------------------------------------------------------------------
    #! Send NAV command to ESP32 and wait for response
    # ------------------------------------------------------------------
    def _send_and_wait_esp32(self, x, y, distance_m):
        """
        Send NAV command over UART and block until ESP32 replies.

        Returns:
            True  if ESP32 sent READY (arrived at baby)
            False if ESP32 sent RELISTEN (obstacle avoided, need re-localization)
        """
        cmd = f"NAV x={x:.3f} y={y:.3f} d={distance_m:.3f}\n"
        print(f"Sending -> ESP32: {cmd.strip()}")

        ser.write(cmd.encode())
        time.sleep(0.3)

        # Obstacle avoidance algorithm is now running on ESP32
        # until obstacle is avoided or target is reached
        while True:
            msg = ser.readline().decode().strip()
            if not msg:
                continue

            print("RX:", msg)

            if msg == "RELISTEN":
                print("Obstacle avoided. Relistening...")
                return False
            elif msg == "READY":
                print("ESP32 ready")
                return True

    # ------------------------------------------------------------------
    #! Localize -> Navigate -> Arrive pipeline
    # ------------------------------------------------------------------
    def _localize_and_navigate(self):
        """Run localization, send nav command, wait for ESP32 response."""
        detection = self._last_detection
        if detection is None:
            logger.error("No detection data available")
            self._set_state(State.LISTENING)
            return

        # Use filtered audio if available, otherwise raw buffer
        audio = detection.filtered_audio if detection.filtered_audio is not None else detection.audio_buffer
        if audio is None:
            logger.error("No audio data in detection result")
            self._set_state(State.LISTENING)
            return

        # --- Localize ---
        try:
            loc = self.sl.localize(
                audio_data=audio,
                sample_rate=16000,  # BCD runs at 16 kHz
                num_channels=audio.shape[1] if audio.ndim > 1 else 1,
            )
            direction = loc['direction_deg']
            distance_ft = loc['distance_ft']
            rads = radians(direction)
            x = distance_ft * cos(rads)
            y = distance_ft * sin(rads)
            distance_m = distance_ft * 0.3048

            logger.info(
                f"Localization: {direction:.1f} deg at {distance_ft:.1f} ft - {loc['sources']} "
                f"NAV x={x:.3f} y={y:.3f} d={distance_m:.3f}m"
            )
        except Exception as e:
            logger.error(f"Localization failed: {e}", exc_info=True)
            self._set_state(State.LISTENING)
            if self.bcd:
                self.bcd.reset()
            return

        if distance_ft <= 0:
            logger.warning("Invalid distance - returning to LISTENING")
            self._set_state(State.LISTENING)
            if self.bcd:
                self.bcd.reset()
            return

        # --- Navigate ---
        self._set_state(State.NAVIGATING)
        arrived = self._send_and_wait_esp32(x, y, distance_m)

        if arrived:
            self._on_arrived()
        else:
            # RELISTEN — re-localize and navigate again
            logger.info("ESP32 requests re-localization")
            self._set_state(State.RELISTEN)
            self._set_state(State.LOCALIZING)
            self._localize_and_navigate()

    # ------------------------------------------------------------------
    #! Arrival handling
    # ------------------------------------------------------------------
    def _on_arrived(self) -> None:
        """Robot has reached the baby."""
        self._set_state(State.ARRIVED)
        confidence = self._last_detection.confidence if self._last_detection else 0.0

        logger.info("Robot arrived at baby - sending alert")
        if self.app:
            self.app.send_alert(confidence)

        # Stay in ARRIVED for a while, then return to listening
        logger.info(f"Waiting {self.ARRIVED_DURATION}s before returning to LISTENING")
        self._shutdown.wait(timeout=self.ARRIVED_DURATION)

        if not self._shutdown.is_set():
            self._set_state(State.LISTENING)
            if self.bcd:
                self.bcd.reset()

    # ------------------------------------------------------------------
    #! Main run loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start all subsystems and block until shutdown."""
        # Start Flask server (always-on for when needed)
        if self.app:
            self.app.start_server()

        if self._test_audio:
            self._run_test_mode()
            return

        # Start cry detection in LISTENING mode
        self.bcd.start(on_cry_callback=self._on_cry_detected)

        self._set_state(State.LISTENING)

        print()
        print("=" * 60)
        print("  Baby Monitor Robot — ACTIVE")
        print("=" * 60)
        print(f"  State:       {self.state.value}")
        print(f"  Stream URL:  {self.app.stream_url}")
        print(f"  UART port:   {ser.port}")
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

    def _run_test_mode(self) -> None:
        """Load an audio file and run it through BCD -> localization."""
        import soundfile as sf

        filepath = os.path.abspath(self._test_audio)
        logger.info(f"TEST MODE: Loading audio from {filepath}")

        audio_data, sample_rate = sf.read(filepath)  # (samples,) or (samples, channels)
        num_channels = audio_data.shape[1] if audio_data.ndim > 1 else 1
        logger.info(
            f"Loaded: {audio_data.shape}, sr={sample_rate}, channels={num_channels}"
        )

        # --- Stage 1: Baby Cry Detection ---
        self._set_state(State.LISTENING)
        # BCD expects 16 kHz, 4-channel audio; trim channels and resample if needed
        bcd_audio = audio_data
        if bcd_audio.ndim == 2 and bcd_audio.shape[1] > 4:
            logger.info(f"Trimming from {bcd_audio.shape[1]} to 4 channels for BCD")
            bcd_audio = bcd_audio[:, :4]
        if sample_rate != 16000:
            import librosa
            logger.info(f"Resampling from {sample_rate} Hz to 16000 Hz for BCD...")
            if bcd_audio.ndim == 2:
                channels = []
                for ch in range(bcd_audio.shape[1]):
                    channels.append(
                        librosa.resample(
                            bcd_audio[:, ch].astype(np.float32),
                            orig_sr=sample_rate,
                            target_sr=16000,
                        )
                    )
                bcd_audio = np.column_stack(channels)
            else:
                bcd_audio = librosa.resample(
                    bcd_audio.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=16000,
                )

        detection = self.bcd.detect_from_audio(bcd_audio)

        print()
        print("=" * 60)
        print("  TEST MODE — Baby Cry Detection Result")
        print("=" * 60)
        print(f"  Is Cry:      {detection.is_cry}")
        print(f"  Confidence:  {detection.confidence:.2%}")
        print("=" * 60)

        if not detection.is_cry:
            print("  No cry detected — skipping localization.")
            self.shutdown()
            return

        self._last_detection = detection

        # --- Stage 2: Sound Localization ---
        # Use filtered audio if available (preserves phase), at original sample rate
        # for DOAnet (will be resampled to 48 kHz inside FunctionCalls_SL)
        if detection.filtered_audio is not None:
            loc_audio = detection.filtered_audio
            loc_sr = 16000  # filtered audio is at BCD's sample rate
        else:
            loc_audio = audio_data
            loc_sr = sample_rate

        loc_channels = loc_audio.shape[1] if loc_audio.ndim > 1 else 1

        self._set_state(State.LOCALIZING)
        try:
            loc = self.sl.localize(
                audio_data=loc_audio,
                sample_rate=loc_sr,
                num_channels=loc_channels,
            )
            direction = loc['direction_deg']
            distance_ft = loc['distance_ft']

            print()
            print("=" * 60)
            print("  TEST MODE — Localization Result")
            print("=" * 60)
            print(f"  Direction:  {direction:.1f} degrees")
            print(f"  Distance:   {distance_ft:.1f} ft")
            print(f"  Sources:    {loc['sources']}")
            print("=" * 60)
        except Exception as e:
            logger.error(f"Localization failed: {e}", exc_info=True)
            self.shutdown()
            return

        if distance_ft <= 0:
            logger.warning("Invalid distance — skipping navigation")
            self.shutdown()
            return

        # --- Stage 3: Navigate via ESP32 ---
        rads = radians(direction)
        x = distance_ft * cos(rads)
        y = distance_ft * sin(rads)
        distance_m = distance_ft * 0.3048

        self._set_state(State.NAVIGATING)
        arrived = self._send_and_wait_esp32(x, y, distance_m)

        if arrived:
            self._on_arrived()
        else:
            print("  ESP32 requested RELISTEN (test mode - stopping)")
            print()

        self.shutdown()

    def shutdown(self) -> None:
        """Gracefully stop all subsystems."""
        logger.info("Shutting down orchestrator...")
        self._shutdown.set()
        if self.bcd:
            self.bcd.stop()
        if ser and ser.is_open:
            ser.close()
            logger.info("UART closed")
        if self.app:
            self.app.stop()
        logger.info("Orchestrator shut down")


def main():
    parser = argparse.ArgumentParser(
        description='Baby Monitor Robot - Pi Integration Orchestrator'
    )
    parser.add_argument(
        '--test-audio', type=str, default=None,
        help='Path to a WAV file to test with (skips live detection, runs localization + navigation)',
    )
    parser.add_argument(
        '--device-index', type=int, default=None,
        help='PyAudio device index for microphone array',
    )
    parser.add_argument(
        '--channels', type=int, default=4,
        help='Number of microphone channels (default: 4)',
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Torch device (default: cpu)',
    )
    parser.add_argument(
        '--models-dir', type=str, default='.',
        help='Directory containing DOAnet model files relative to SoundLocalization/ (default: .)',
    )
    parser.add_argument(
        '--task-id', type=str, default='6',
        help='DOAnet task ID (default: 6)',
    )

    parser.add_argument(
        '--debug', action='store_true',
        help='Enable DEBUG-level logging',
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Mute BCD processing steps, debug output, and root logger INFO messages',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    if args.quiet:
        # Suppress root logger INFO (model loading, filter init, etc.)
        logging.getLogger().setLevel(logging.WARNING)
        # Keep named loggers at INFO so orchestrator/subsystem logs still show
        for name in ('__main__', 'FunctionCalls_BCD', 'FunctionCalls_SL'):
            logging.getLogger(name).setLevel(logging.INFO)

    orchestrator = Orchestrator(args)
    orchestrator.run()


if __name__ == '__main__':
    main()
