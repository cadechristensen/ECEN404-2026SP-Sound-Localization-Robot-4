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
        "READY\n"    — robot arrived at baby
        "OBSTACLE\n" — obstacle encountered, ESP32 handling avoidance autonomously
        "RELISTEN\n" — dead end hit, need re-localization
    """

    # Responses that end the wait loop
    TERMINAL_RESPONSES = {'READY', 'RELISTEN', 'BUMPED'}

    def __init__(self, port: str = '/dev/serial0', baudrate: int = 115200):
        self._port_name = port
        self._baudrate = baudrate
        self._serial = None
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
        #cmd = f"NAV x={direction_deg:.3f} y={distance_ft:.3f} d={distance_ft:.3f}\n"
        cmd = f"NAV angle={direction_deg:.3f} dist_ft={distance_ft:.3f}\n"
        self._serial.write(cmd.encode('utf-8'))
        logger.info(f"Sent to ESP32: {cmd.strip()}")

    def wait_for_response(self, timeout: float = 60.0) -> Optional[str]:
        """
        Block until the ESP32 sends READY or RELISTEN.

        Continuously reads lines, ignoring empty reads and informational
        messages (e.g. OBSTACLE), until a terminal response or timeout.

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

            response = raw.decode('utf-8', errors='replace').strip()
            if not response:
                continue

            logger.info(f"ESP32 RX: {response}")

            if response in self.TERMINAL_RESPONSES:
                return response

            # Non-terminal (e.g. OBSTACLE) — ESP32 is alive, keep waiting
            logger.info(f"ESP32 sent '{response}' (non-terminal), still waiting...")

        # Timeout — caller handles the retry/escalation logic
        return None

    def send_cancel(self) -> None:
        """Send CANCEL to stop the ESP32 immediately."""
        if not self._serial or not self._serial.is_open:
            return
        try:
            self._serial.write("CANCEL\n".encode('utf-8'))
            logger.info("Sent CANCEL to ESP32")
        except (OSError, TypeError):
            logger.debug("Could not send CANCEL — port already closed")

    def close(self) -> None:
        """Close the serial port."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("UART closed")
