from machine import Pin, PWM, time_pulse_us, UART
import time
from time import ticks_us, ticks_diff, sleep_ms
import math

print("Starting 3-Sensor Obstacle Avoidance + Sound Navigation")

# =============================================================================
# UART SETUP (ESP32 <-> Raspberry Pi)
# =============================================================================
uart = UART(1, baudrate=115200, tx=Pin(17), rx=Pin(16))
print("UART initialized")

# =============================================================================
# MOTOR SETUP (Sabertooth R/C Mode)
# =============================================================================
PWM_FREQ = 50
motor_left  = PWM(Pin(12), freq=PWM_FREQ)
motor_right = PWM(Pin(27), freq=PWM_FREQ)

def pulse_us_to_duty(us):
    return int(us / 20000 * 65535)

# --- Duty values (microseconds) ---
STOP_LEFT_US     = 1480
STOP_RIGHT_US    = 1230

FORWARD_LEFT_US  = 1200
FORWARD_RIGHT_US = 1080

# Turn RIGHT: left wheel forward, right wheel back
TURN_RIGHT_LEFT_US  = 1200   # left  motor duty while turning right
TURN_RIGHT_RIGHT_US = 1400   # right motor duty while turning right
TURN_RIGHT_DURATION = 0.62   # seconds for a 90° right turn

# Turn LEFT: left wheel back, right wheel forward
TURN_LEFT_LEFT_US  = 1600    # left  motor duty while turning left
TURN_LEFT_RIGHT_US = 1080    # right motor duty while turning left
TURN_LEFT_DURATION = 0.65    # seconds for a 90° left turn

def stop():
    motor_left.duty_u16(pulse_us_to_duty(STOP_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(STOP_RIGHT_US))

def forward():
    motor_left.duty_u16(pulse_us_to_duty(FORWARD_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(FORWARD_RIGHT_US))

def _turn_right_raw():
    """Spin motors for a right turn — caller controls timing."""
    motor_left.duty_u16(pulse_us_to_duty(TURN_RIGHT_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(TURN_RIGHT_RIGHT_US))

def _turn_left_raw():
    """Spin motors for a left turn — caller controls timing."""
    motor_left.duty_u16(pulse_us_to_duty(TURN_LEFT_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(TURN_LEFT_RIGHT_US))

# =============================================================================
# ENCODER SETUP  (distance tracking only — no turn counting)
# =============================================================================
ENCODER_PIN   = 34
COUNTS_PER_42CM = 12
DIST_PER_COUNT  = 0.42 / COUNTS_PER_42CM   # metres per encoder count ≈ 0.035 m

count         = 0
last_time_us  = 0
DEBOUNCE_US   = 50000

def encoder_isr(pin):
    global count, last_time_us
    now = ticks_us()
    if ticks_diff(now, last_time_us) > DEBOUNCE_US:
        count += 1
        last_time_us = now

enc = Pin(ENCODER_PIN, Pin.IN)
enc.irq(trigger=Pin.IRQ_FALLING, handler=encoder_isr)

# =============================================================================
# UNIT CONVERSION
# =============================================================================
FT_TO_M = 0.3048

def feet_to_meters(ft):
    return ft * FT_TO_M

# =============================================================================
# NAVIGATION PARAMETERS
# =============================================================================
ARRIVAL_THRESHOLD = 0.5      # metres — stop this close to the target

nav_active           = False
nav_dir              = None
nav_distance_m       = 0.0    # target distance in metres (converted from ft)
nav_start_count      = 0
distance_travelled   = 0.0    # metres covered since last nav command
obstacle_encountered = False
nav_angle            = 0.0    # bearing to sound source (degrees)

# =============================================================================
# ULTRASONIC SENSORS
# =============================================================================
def make_ultrasonic(trig, echo):
    return {"trig": Pin(trig, Pin.OUT), "echo": Pin(echo, Pin.IN)}

physical_sensors = {
    "rear_physical":  make_ultrasonic(14, 13),
    "left_physical":  make_ultrasonic(19, 21),
    "right_physical": make_ultrasonic(4, 0)
}

ultra = {
    "front": physical_sensors["rear_physical"],
    "left":  physical_sensors["right_physical"],
    "right": physical_sensors["left_physical"]
}

def get_distance(sensor, samples=3):
    trig = sensor["trig"]
    echo = sensor["echo"]
    readings = []
    for _ in range(samples):
        trig.value(0)
        time.sleep_us(2)
        trig.value(1)
        time.sleep_us(10)
        trig.value(0)
        duration = time_pulse_us(echo, 1, 30000)
        if duration > 0:
            d = (duration * 0.0343) / 2   # cm
            if d > 2:
                readings.append(d)
        time.sleep(0.005)
    if readings:
        readings.sort()
        return readings[len(readings) // 2]
    return 400

def read_distances():
    return {k: get_distance(v) for k, v in ultra.items()}

# =============================================================================
# OBSTACLE AVOIDANCE PARAMETERS
# =============================================================================
SAFE_FRONT = 50   # cm
SAFE_SIDE  = 75   # cm

# FSM STATES
GO_STRAIGHT  = 0
FOLLOW_LEFT  = 1
FOLLOW_RIGHT = 2
state = GO_STRAIGHT

# =============================================================================
# STARTUP
# =============================================================================
led = Pin(2, Pin.OUT)
for _ in range(4):
    led.toggle()
    time.sleep(0.3)
stop()
print("System ACTIVE")
print("-" * 60)

# =============================================================================
# UART PARSING FOR NAVIGATION
# =============================================================================
def parse_uart():
    global nav_active, nav_dir, nav_distance_m, nav_start_count
    global obstacle_encountered, nav_angle, distance_travelled

    if uart.any():
        msg = uart.readline().decode().strip()
        print("UART RX:", msg)

        if msg.startswith("NAV"):
            # Expected format: "NAV x=-1.234 y=2.345 d=3.456"
            # Distance 'd' is assumed to be in FEET — converted to metres here
            parts = msg.split()
            x   = float(parts[1].split("=")[1])
            y   = float(parts[2].split("=")[1])
            d_ft = float(parts[3].split("=")[1])

            nav_distance_m = feet_to_meters(y)
            nav_angle      = x   # signed degrees
            nav_dir        = ("FORWARD" if abs(nav_angle) < 5
                              else ("RIGHT" if nav_angle > 0 else "LEFT"))

            # Reset tracking for fresh run
            nav_start_count      = count
            distance_travelled   = 0.0
            obstacle_encountered = False
            nav_active           = True

            print(f"NAV cmd  → d={d_ft:.2f} ft ({nav_distance_m:.2f} m), "
                  f"angle={nav_angle:.1f}°, dir={nav_dir}")

# =============================================================================
# HELPER — MOVE DISTANCE (encoder-based)
# =============================================================================
def move_distance(target_m, motion_fn=forward):
    """Drive until encoder counts cover target_m metres, then stop."""
    global distance_travelled
    start = count
    target_counts = round(target_m / DIST_PER_COUNT)
    motion_fn()
    while (count - start) < target_counts:
        sleep_ms(2)
    stop()
    distance_travelled += target_m

# =============================================================================
# HELPER — TURN BY ANGLE (time-based, 90° calibrated)
# =============================================================================
def turn_by_angle(angle_deg):
    """
    Rotate by angle_deg using timed turns.
    Positive angle → turn right; negative angle → turn left.
    Duration scales linearly from the 90° calibration times.
    """
    if abs(angle_deg) < 1:
        return

    fraction = abs(angle_deg) / 90.0

    if angle_deg > 0:
        duration = TURN_RIGHT_DURATION * fraction
        _turn_right_raw()
    else:
        duration = TURN_LEFT_DURATION * fraction
        _turn_left_raw()

    time.sleep(duration)
    stop()
    sleep_ms(50)   # brief settle before next move

# =============================================================================
# MAIN LOOP
# =============================================================================
while True:
    parse_uart()
    d = read_distances()
    f, l, r = d["front"], d["left"], d["right"]
    front_blocked = f < SAFE_FRONT
    left_blocked  = l < SAFE_SIDE
    right_blocked = r < SAFE_SIDE

    print(f"S:{state} | F:{f:.0f} L:{l:.0f} R:{r:.0f} | "
          f"NAV:{nav_dir if nav_active else 'IDLE'} | "
          f"Travelled:{distance_travelled:.2f} m / "
          f"{nav_distance_m:.2f} m")

    # =========================================================================
    # NAVIGATION TOWARD SOUND
    # =========================================================================
    if nav_active and not obstacle_encountered:
        remaining_m = nav_distance_m - distance_travelled

        # ---- Arrived? ----
        if remaining_m <= ARRIVAL_THRESHOLD:
            stop()
            nav_active = False
            uart.write("READY\n")
            print("✓ Arrived within threshold, ready for communication")
            state = GO_STRAIGHT
            continue

        # ---- Face the sound source ----
        if abs(nav_angle) >= 5:
            turn_by_angle(nav_angle)
            nav_angle = 0.0   # heading is now aligned; clear so we don't re-turn

        # ---- Drive remaining distance (with live obstacle check) ----
        start = count
        target_counts = round(remaining_m / DIST_PER_COUNT)
        forward()
        while (count - start) < target_counts:
            # re-check front sensor while driving
            live_f = get_distance(ultra["front"])
            if live_f < SAFE_FRONT:
                stop()
                obstacle_encountered = True
                break
            sleep_ms(20)

        if not obstacle_encountered:
            stop()
            distance_travelled += (count - start) * DIST_PER_COUNT

    # =========================================================================
    # OBSTACLE AVOIDANCE
    # =========================================================================
    if front_blocked or obstacle_encountered:
        stop()
        uart.write("OBSTACLE\n")
        obstacle_encountered = True
        print("⚠ Obstacle ahead! Choosing escape direction...")

        # Update distance actually covered before stopping
        # (already updated in the driving loop above for mid-move stops)

        # Choose escape direction
        if left_blocked and not right_blocked:
            print("→ Turning RIGHT (left side blocked)")
            turn_by_angle(90)
            state = FOLLOW_RIGHT
        elif right_blocked and not left_blocked:
            print("← Turning LEFT (right side blocked)")
            turn_by_angle(-90)
            state = FOLLOW_LEFT
        else:
            if l >= r:
                print("← Turning LEFT (more clearance)")
                turn_by_angle(-90)
                state = FOLLOW_LEFT
            else:
                print("→ Turning RIGHT (more clearance)")
                turn_by_angle(90)
                state = FOLLOW_RIGHT

        stop()
        uart.write("RELISTEN\n")
        nav_active           = False
        obstacle_encountered = False
        # distance_travelled is preserved so the next NAV command
        # can issue a corrected remaining distance from the Pi side.
        print("Waiting for updated NAV command after re-listen...")

    time.sleep(0.05)
