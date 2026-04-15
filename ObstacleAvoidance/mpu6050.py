"""
MPU6050 Gyroscope driver for ESP32 MicroPython.

Provides closed-loop turn-by-angle using the Z-axis gyroscope instead of
time-based calibration. Drop-in replacement for the time-based turn_by_angle.

Wiring:
    MPU6050 VCC → ESP32 3.3V
    MPU6050 GND → ESP32 GND
    MPU6050 SDA → ESP32 GPIO 21 (configurable)
    MPU6050 SCL → ESP32 GPIO 22 (configurable)

Usage:
    from mpu6050 import MPU6050

    gyro = MPU6050(sda_pin=21, scl_pin=22)
    gyro.calibrate()  # hold robot still for 2 seconds

    # Then replace time-based turn_by_angle with:
    gyro.turn_by_angle(90, turn_right_func, turn_left_func, stop_func)
"""

from machine import Pin, I2C
from time import ticks_us, ticks_diff, sleep_ms
import struct


class MPU6050:
    """MPU6050 gyroscope driver for closed-loop turning."""

    # MPU6050 registers
    ADDR = 0x68
    PWR_MGMT_1 = 0x6B
    GYRO_CONFIG = 0x1B
    GYRO_ZOUT_H = 0x47  # Z-axis gyro high byte

    # Gyro sensitivity (degrees/s per LSB) for each range setting
    GYRO_SCALES = {
        0: 131.0,    # ±250 °/s
        1: 65.5,     # ±500 °/s
        2: 32.8,     # ±1000 °/s
        3: 16.4,     # ±2000 °/s
    }

    def __init__(self, sda_pin=21, scl_pin=22, i2c_id=0, gyro_range=1, upside_down=False):
        """
        Initialize MPU6050.

        Args:
            sda_pin: GPIO pin for I2C SDA
            scl_pin: GPIO pin for I2C SCL
            i2c_id: I2C bus number (0 or 1)
            gyro_range: 0=±250°/s, 1=±500°/s, 2=±1000°/s, 3=±2000°/s
                ±500°/s is a good default for robot turns — fast enough
                for quick spins but still precise enough for small angles.
            upside_down: Set True if the MPU6050 is mounted face-down
                (flips the Z-axis sign so turns read correctly).
        """
        self.i2c = I2C(i2c_id, sda=Pin(sda_pin), scl=Pin(scl_pin), freq=400000)
        self.scale = self.GYRO_SCALES[gyro_range]
        self.z_offset = 0.0  # calibration offset (drift compensation)
        self.z_sign = -1.0 if upside_down else 1.0

        # Wake up MPU6050 (it starts in sleep mode)
        self.i2c.writeto_mem(self.ADDR, self.PWR_MGMT_1, b'\x00')
        sleep_ms(100)

        # Set gyro range
        self.i2c.writeto_mem(self.ADDR, self.GYRO_CONFIG, bytes([gyro_range << 3]))
        sleep_ms(10)

        print(f"MPU6050 initialized (gyro range: ±{[250,500,1000,2000][gyro_range]} deg/s)")

    def _read_gyro_z_raw(self):
        """Read raw Z-axis gyroscope value (signed 16-bit)."""
        data = self.i2c.readfrom_mem(self.ADDR, self.GYRO_ZOUT_H, 2)
        raw = struct.unpack('>h', data)[0]  # big-endian signed short
        return raw

    def read_gyro_z(self):
        """Read Z-axis angular velocity in degrees/second, with offset removed."""
        raw = self._read_gyro_z_raw()
        return ((raw / self.scale) - self.z_offset) * self.z_sign

    def calibrate(self, samples=200, delay_ms=10):
        """
        Calibrate gyro drift. Robot must be stationary during calibration.

        Averages readings over ~2 seconds to find the zero-rate offset.
        This offset is subtracted from all future readings.
        """
        print("Calibrating gyro (hold robot still)...", end="")
        total = 0.0
        for _ in range(samples):
            total += self._read_gyro_z_raw() / self.scale
            sleep_ms(delay_ms)
        self.z_offset = total / samples
        print(f" done (offset: {self.z_offset:.3f} deg/s)")

    def turn_by_angle(self, target_deg, turn_right_func, turn_left_func, stop_func,
                      tolerance=2.0, timeout_ms=10000):
        """
        Turn the robot by a specific angle using gyro feedback.

        Args:
            target_deg: Degrees to turn. Positive = right, negative = left.
            turn_right_func: Callable that starts right turn motors.
            turn_left_func: Callable that starts left turn motors.
            stop_func: Callable that stops all motors.
            tolerance: Stop within this many degrees of target (default: 2°).
            timeout_ms: Safety timeout in milliseconds (default: 10s).

        Returns:
            float: Actual angle turned (for debugging/logging).
        """
        if abs(target_deg) < 1:
            return 0.0

        accumulated = 0.0
        target = abs(target_deg)

        # Start turning
        if target_deg > 0:
            turn_right_func()
        else:
            turn_left_func()

        last_time = ticks_us()
        start_time = last_time

        while abs(accumulated) < (target - tolerance):
            # Safety timeout
            if ticks_diff(ticks_us(), start_time) > timeout_ms * 1000:
                print(f"Turn timeout! Reached {accumulated:.1f} deg of {target_deg} deg")
                break

            # Read angular velocity
            gz = self.read_gyro_z()  # deg/s

            # Calculate time step
            now = ticks_us()
            dt = ticks_diff(now, last_time) / 1_000_000.0  # microseconds to seconds
            last_time = now

            # Integrate: angle += angular_velocity * time
            accumulated += gz * dt

            sleep_ms(5)  # ~200Hz sample rate

        stop_func()
        sleep_ms(50)  # let motors settle

        actual = abs(accumulated)
        print(f"Turn complete: target={target_deg:.1f} deg, actual={actual:.1f} deg")
        return accumulated

    def test(self, duration_s=5):
        """
        Print live gyro readings for testing. Useful to verify the sensor works.

        Args:
            duration_s: How long to print readings.
        """
        print(f"Gyro Z readings for {duration_s}s (rotate robot to see values change):")
        start = ticks_us()
        while ticks_diff(ticks_us(), start) < duration_s * 1_000_000:
            gz = self.read_gyro_z()
            print(f"  Z: {gz:+7.2f} deg/s")
            sleep_ms(100)
