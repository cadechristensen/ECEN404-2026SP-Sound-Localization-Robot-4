# GPIO Scanner — paste into ESP32 MicroPython REPL to detect connected pins
#
# How it works:
#   1. Set internal PULL_DOWN, read value
#   2. Set internal PULL_UP, read value
#   3. If the pin follows both pulls → FLOATING (nothing connected)
#   4. If the pin stays HIGH or LOW regardless → something is driving it (CONNECTED)
#   5. Input-only pins (34-39) have no internal pulls, so we just read state
#
# Usage:
#   screen /dev/serial0 115200   (or /dev/ttyUSB0 over USB)
#   Then paste this script

from machine import Pin
import time

# All usable ESP32 GPIOs (excluding 1/3 = REPL UART, 6-11 = flash)
ALL_GPIOS = [0, 2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19,
             21, 22, 23, 25, 26, 27, 32, 33, 34, 35, 36, 39]

# Known pin assignments from CurrentObstacleAvoidance.py
KNOWN = {
    0:  "Ultrasonic RIGHT echo (IN)",
    2:  "Onboard LED (OUT)",
    4:  "Ultrasonic RIGHT trig (OUT)",
    12: "Motor LEFT PWM",
    13: "Ultrasonic REAR echo (IN)",
    14: "Ultrasonic REAR trig (OUT)",
    16: "UART RX (from Pi)",
    17: "UART TX (to Pi)",
    19: "Ultrasonic LEFT trig (OUT)",
    21: "Ultrasonic LEFT echo (IN)",
    22: "I2C SDA (MPU6050)",
    23: "I2C SCL (MPU6050)",
    27: "Motor RIGHT PWM",
    34: "Encoder (Hall sensor, IN)",
}

# Input-only pins (no internal pull-up/pull-down)
INPUT_ONLY = {34, 35, 36, 39}

def test_pin(gpio):
    """Test if a pin has something connected by toggling internal pulls.

    Returns:
        (status, pull_down_val, pull_up_val)
        status: 'CONNECTED-HIGH', 'CONNECTED-LOW', 'FLOATING', or 'INPUT-ONLY'
    """
    if gpio in INPUT_ONLY:
        p = Pin(gpio, Pin.IN)
        val = p.value()
        return ('INPUT-ONLY', val, val)

    # Read with pull-down (should go LOW if floating)
    p = Pin(gpio, Pin.IN, Pin.PULL_DOWN)
    time.sleep_ms(5)
    val_down = p.value()

    # Read with pull-up (should go HIGH if floating)
    p = Pin(gpio, Pin.IN, Pin.PULL_UP)
    time.sleep_ms(5)
    val_up = p.value()

    if val_down == 0 and val_up == 1:
        return ('FLOATING', val_down, val_up)
    elif val_down == 1 and val_up == 1:
        return ('CONNECTED-HIGH', val_down, val_up)
    elif val_down == 0 and val_up == 0:
        return ('CONNECTED-LOW', val_down, val_up)
    else:
        # val_down=1, val_up=0 — unusual, likely weak external pull
        return ('CONNECTED-?', val_down, val_up)


print("=" * 65)
print("ESP32 GPIO Connection Scanner")
print("=" * 65)
print(f"{'GPIO':>6}  {'Status':>15}  {'PD':>3} {'PU':>3}  {'Assignment'}")
print("-" * 65)

connected = []
floating = []

for gpio in ALL_GPIOS:
    try:
        status, v_down, v_up = test_pin(gpio)
        assign = KNOWN.get(gpio, "")

        if status == 'FLOATING':
            marker = "  "
            floating.append(gpio)
        elif status == 'INPUT-ONLY':
            marker = "? "
        else:
            marker = ">>"
            connected.append(gpio)

        print(f"{marker}{gpio:>4}  {status:>15}  {v_down:>3} {v_up:>3}  {assign}")
    except Exception as e:
        print(f"  {gpio:>4}  {'ERROR':>15}  {e}")

print("=" * 65)
print(f"\n>> CONNECTED ({len(connected)} pins):")
for g in connected:
    assign = KNOWN.get(g, "unknown")
    print(f"     GPIO {g:>2} — {assign}")

print(f"\n   FLOATING ({len(floating)} pins):")
for g in floating:
    print(f"     GPIO {g:>2} — not connected")

print(f"\n?  INPUT-ONLY pins (34-39): can't test pulls, read state only")
print("=" * 65)
