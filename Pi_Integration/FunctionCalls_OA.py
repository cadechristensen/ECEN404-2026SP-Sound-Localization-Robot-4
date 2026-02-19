# Function Calls for Obstacle Avoidance (UART to ESP32)

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Feet to meters conversion factor
FT_TO_M = 0.3048


class ObstacleAvoidance:
    """
    Pi-side UART communication with the ESP32 running ObstacleAvoidanceAlgorithm.

    Protocol (Pi -> ESP32):
        "NAV x=<float> y=<float> d=<float>\n"
        where x, y are Cartesian target coords (meters) and d is straight-line distance (meters).

    Protocol (ESP32 -> Pi):
        "READY\n"    — robot arrived at baby
        "OBSTACLE\n" — obstacle encountered, ESP32 handling avoidance autonomously
        "RELISTEN\n" — dead end hit, need re-localization
    """

    VALID_RESPONSES = {'READY', 'OBSTACLE', 'RELISTEN'}

    def __init__(self, port: str = '/dev/serial0', baudrate: int = 115200):
        self._port_name = port
        self._baudrate = baudrate
        self._serial: Optional[serial.Serial] = None
        self._open()

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

        Converts DOAnet direction (degrees, 0=front, 90=right, clockwise)
        and distance (feet) into Cartesian x, y (meters) for the ESP32.

        Coordinate convention (matching ESP32 parser):
            x = -sin(deg) * distance   (positive x = left)
            y =  cos(deg) * distance   (positive y = forward)
        """
        distance_m = distance_ft * FT_TO_M
        rad = math.radians(direction_deg)

        x = -math.sin(rad) * distance_m
        y = math.cos(rad) * distance_m

        cmd = f"NAV x={x:.3f} y={y:.3f} d={distance_m:.3f}\n"
        self._serial.write(cmd.encode('utf-8'))
        logger.info(f"Sent to ESP32: {cmd.strip()}")

    def wait_for_response(self, timeout: float = 60.0) -> Optional[str]:
        """
        Block until the ESP32 sends a response line.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            One of "READY", "OBSTACLE", "RELISTEN", or None on timeout.
        """
        self._serial.timeout = timeout
        try:
            raw = self._serial.readline()
        finally:
            self._serial.timeout = 1  # restore default

        if not raw:
            logger.warning(f"UART read timed out after {timeout}s")
            return None

        response = raw.decode('utf-8', errors='replace').strip()
        logger.info(f"ESP32 response: {response}")

        if response in self.VALID_RESPONSES:
            return response

        logger.warning(f"Unexpected UART response: {response}")
        return response

    def close(self) -> None:
        """Close the serial port."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("UART closed")
