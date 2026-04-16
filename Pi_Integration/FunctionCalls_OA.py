# Function Calls for Obstacle Avoidance (UART to ESP32)

import math
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ObstacleAvoidance:
    """
    Pi-side UART communication with the ESP32.

    Protocol (Pi -> ESP32):
        "NAV angle=<float> dist_ft=<float>\n"
        where angle is bearing in degrees, dist_ft is distance in feet.

    Protocol (ESP32 -> Pi):
        "READY heading=<float>\n"    — robot arrived at baby
        "OBSTACLE\n"                 — obstacle encountered, ESP32 handling avoidance autonomously
        "RELISTEN heading=<float>\n" — dead end hit, need re-localization
        "BUMPED heading=<float>\n"   — bump sensor triggered, ESP32 halted

    Terminal replies carry the ESP32's cumulative world_heading (degrees).
    Pi reads it via `last_response_heading` after each wait_for_response() call.
    Heading is optional — old firmware without it sets last_response_heading=None.
    """

    # Responses that end the wait loop
    TERMINAL_RESPONSES = {"READY", "RELISTEN", "BUMPED"}

    def __init__(self, port: str = "/dev/serial0", baudrate: int = 115200):
        self._port_name = port
        self._baudrate = baudrate
        self._serial = None
        self.last_response_heading: Optional[float] = (
            None  # world_heading from last terminal reply
        )
        self._open()

    @property
    def port(self) -> str:
        return self._port_name

    def _open(self) -> None:
        import serial

        self._serial = serial.Serial(
            port=self._port_name,
            baudrate=self._baudrate,
            timeout=1,
        )
        # Flush any stale data
        self._serial.reset_input_buffer()
        logger.info(f"UART opened on {self._port_name} @ {self._baudrate} baud")

    def send_nav_command(self, direction_deg: float, distance_ft: float) -> None:
        """
        Send a navigation command to the ESP32.

        Args:
            direction_deg: Direction to sound source in degrees.
            distance_ft: Distance to sound source in feet.

        Raises:
            ValueError: If direction_deg or distance_ft is non-finite or out of range.
        """
        if not math.isfinite(direction_deg) or not math.isfinite(distance_ft):
            raise ValueError(
                f"NAV values must be finite: angle={direction_deg}, dist_ft={distance_ft}"
            )
        if distance_ft <= 0:
            raise ValueError(f"distance_ft must be positive, got {distance_ft}")
        # cmd = f"NAV x={direction_deg:.3f} y={distance_ft:.3f} d={distance_ft:.3f}\n"
        cmd = f"NAV angle={direction_deg:.3f} dist_ft={distance_ft:.3f}\n"
        self._serial.write(cmd.encode("utf-8"))
        logger.info(f"Sent to ESP32: {cmd.strip()}")

    def send_turn_command(self, direction_deg: float) -> None:
        """
        Send a turn-only NAV command to the ESP32.

        Sends dist_ft=0.01 (~3 mm, well under ARRIVAL_THRESHOLD of 0.762 m),
        so the ESP32 turns to face direction_deg, then immediately replies
        READY without driving forward. Used by the FINAL_TURN state to
        refine facing angle after arrival.

        Args:
            direction_deg: Bearing to turn to, in degrees.

        Raises:
            ValueError: If direction_deg is non-finite.
        """
        if not math.isfinite(direction_deg):
            raise ValueError(f"direction_deg must be finite, got {direction_deg}")
        cmd = f"NAV angle={direction_deg:.3f} dist_ft=0.010\n"
        self._serial.write(cmd.encode("utf-8"))
        logger.info(f"Sent to ESP32 (turn-only): {cmd.strip()}")

    @staticmethod
    def _parse_terminal_line(line: str):
        """Split a response line into (verb, heading_or_None).

        A terminal line is ``<VERB>[ heading=<float>]``.  Returns the verb
        as the first whitespace-separated token and parses the optional
        ``heading=`` token.  Returns (None, None) if the first token is
        not a known terminal verb.
        """
        parts = line.split()
        if not parts:
            return None, None
        verb = parts[0]
        heading = None
        for tok in parts[1:]:
            if tok.startswith("heading="):
                try:
                    heading = float(tok.split("=", 1)[1])
                except (ValueError, IndexError):
                    heading = None
                break
        return verb, heading

    def wait_for_response(self, timeout: float = 60.0) -> Optional[str]:
        """
        Block until the ESP32 sends READY, RELISTEN, or BUMPED.

        Continuously reads lines, ignoring empty reads and informational
        messages (e.g. OBSTACLE), until a terminal response or timeout.

        Terminal replies may carry `heading=<float>` — when present, the
        heading is stored in `self.last_response_heading` for the caller
        to read.  Missing or malformed heading tokens leave the attribute
        as None.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            "READY", "RELISTEN", "BUMPED", or None on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._serial or not self._serial.is_open:
                return None
            try:
                raw = self._serial.readline()
            except (OSError, TypeError):
                return None
            if not raw:
                logger.debug(f"ESP32 no response (serial timeout {timeout:.1f}s)")
                continue

            response = raw.decode("utf-8", errors="replace").strip()
            if not response:
                continue

            logger.info(f"ESP32 RX: {response}")

            verb, heading = self._parse_terminal_line(response)
            if verb in self.TERMINAL_RESPONSES:
                self.last_response_heading = heading
                return verb

            # Non-terminal (e.g. OBSTACLE) — ESP32 is alive, keep waiting
            logger.info(f"ESP32 sent '{response}' (non-terminal), still waiting...")

        # Timeout — caller handles the retry/escalation logic
        return None

    def send_cancel(self) -> None:
        """Send CANCEL to stop the ESP32 immediately."""
        if not self._serial or not self._serial.is_open:
            return
        try:
            self._serial.write("CANCEL\n".encode("utf-8"))
            logger.info("Sent CANCEL to ESP32")
        except (OSError, TypeError):
            logger.debug("Could not send CANCEL — port already closed")

    def close(self) -> None:
        """Close the serial port."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("UART closed")
