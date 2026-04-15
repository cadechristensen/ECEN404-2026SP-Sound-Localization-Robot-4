from machine import Pin, PWM, time_pulse_us, UART, I2C
import time
from time import ticks_us, ticks_diff, sleep_ms
import struct

print("Starting 3-Sensor Obstacle Avoidance + Sound Navigation")

# =============================================================================
# UART SETUP (ESP32 <-> Raspberry Pi)
# =============================================================================
uart = UART(1, baudrate=115200, tx=Pin(17), rx=Pin(16), timeout=50)
print("UART initialized")

# =============================================================================
# MOTOR SETUP (Sabertooth R/C Mode)
# =============================================================================
PWM_FREQ = 50
motor_left = PWM(Pin(12), freq=PWM_FREQ)
motor_right = PWM(Pin(27), freq=PWM_FREQ)


def pulse_us_to_duty(us):
    return int(us / 20000 * 65535)


STOP_LEFT_US = 1480
STOP_RIGHT_US = 1230
FORWARD_LEFT_US = 1600
FORWARD_RIGHT_US = 1700
TURN_RIGHT_LEFT_US = 1600
TURN_RIGHT_RIGHT_US = 1050
TURN_LEFT_LEFT_US = 1050
TURN_LEFT_RIGHT_US = 1700

MAX_CORRECTION_US = 80
HEADING_KP = 15


def stop():
    motor_left.duty_u16(pulse_us_to_duty(STOP_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(STOP_RIGHT_US))


def forward_corrected(correction_us=0):
    left_us = max(
        STOP_LEFT_US + 50,
        min(FORWARD_LEFT_US + MAX_CORRECTION_US, int(FORWARD_LEFT_US + correction_us)),
    )
    right_us = max(
        STOP_RIGHT_US + 50,
        min(
            FORWARD_RIGHT_US + MAX_CORRECTION_US, int(FORWARD_RIGHT_US - correction_us)
        ),
    )
    motor_left.duty_u16(pulse_us_to_duty(left_us))
    motor_right.duty_u16(pulse_us_to_duty(right_us))


def _turn_right_raw():
    motor_left.duty_u16(pulse_us_to_duty(TURN_RIGHT_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(TURN_RIGHT_RIGHT_US))


def _turn_left_raw():
    motor_left.duty_u16(pulse_us_to_duty(TURN_LEFT_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(TURN_LEFT_RIGHT_US))


# =============================================================================
# BUMP SENSORS
# =============================================================================
bump_left = Pin(25, Pin.IN, Pin.PULL_UP)
bump_right = Pin(32, Pin.IN, Pin.PULL_UP)


def check_bumps():
    return not bump_left.value() or not bump_right.value()


def bump_halt():
    stop()
    uart.write("BUMPED\n")
    print("!!! BUMP — EMERGENCY HALT — power cycle to restart !!!")
    led = Pin(2, Pin.OUT)
    while True:
        led.toggle()
        sleep_ms(200)


# =============================================================================
# MPU6050 GYROSCOPE
# =============================================================================
MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
GYRO_CONFIG = 0x1B
GYRO_ZOUT_H = 0x47
GYRO_SCALE = 65.5

i2c = I2C(0, sda=Pin(22), scl=Pin(23), freq=400000)
i2c.writeto_mem(MPU_ADDR, PWR_MGMT_1, b"\x00")
sleep_ms(100)
i2c.writeto_mem(MPU_ADDR, GYRO_CONFIG, bytes([1 << 3]))
sleep_ms(10)
print("MPU6050 initialized")

z_offset = 0.0
heading = 0.0
last_gyro_t = ticks_us()


def read_gyro_z_raw():
    data = i2c.readfrom_mem(MPU_ADDR, GYRO_ZOUT_H, 2)
    return struct.unpack(">h", data)[0]


def calibrate_gyro(samples=200, delay_ms=10):
    global z_offset
    print("Calibrating gyro — hold robot still...", end="")
    total = 0.0
    for _ in range(samples):
        total += read_gyro_z_raw() / GYRO_SCALE
        sleep_ms(delay_ms)
    z_offset = total / samples
    print(f" done (offset: {z_offset:.3f} deg/s)")


def read_gyro_z():
    return -((read_gyro_z_raw() / GYRO_SCALE) - z_offset)


def reset_heading():
    global heading, last_gyro_t
    heading = 0.0
    last_gyro_t = ticks_us()


def update_heading_and_drive():
    global heading, last_gyro_t
    gz = read_gyro_z()
    now = ticks_us()
    dt = ticks_diff(now, last_gyro_t) / 1_000_000.0
    last_gyro_t = now
    heading += gz * dt
    correction = max(
        -MAX_CORRECTION_US, min(MAX_CORRECTION_US, int(HEADING_KP * heading))
    )
    forward_corrected(correction)


def drive_ms(duration_ms):
    """
    Drive forward with heading hold for exactly duration_ms.
    Motor gets a fresh PWM pulse every 10ms — Sabertooth never starves.
    """
    start = ticks_us() // 1000
    while (ticks_us() // 1000 - start) < duration_ms:
        if check_bumps():
            bump_halt()
        update_heading_and_drive()
        sleep_ms(10)


# =============================================================================
# GYRO TURN
# =============================================================================
def turn_by_angle(target_deg, tolerance=2.0, timeout_ms=15000):
    """Positive = right, negative = left."""
    if abs(target_deg) < 1:
        return 0.0
    accumulated = 0.0
    if target_deg > 0:
        _turn_right_raw()
    else:
        _turn_left_raw()
    last_time = ticks_us()
    start_time = last_time
    while abs(accumulated) < (abs(target_deg) - tolerance):
        if check_bumps():
            bump_halt()
        if ticks_diff(ticks_us(), start_time) > timeout_ms * 1000:
            print(f"Turn timeout at {accumulated:.1f} deg")
            break
        gz = read_gyro_z()
        now = ticks_us()
        dt = ticks_diff(now, last_time) / 1_000_000.0
        last_time = now
        accumulated += gz * dt
        sleep_ms(5)
    stop()
    sleep_ms(100)
    print(f"Turn done: target={target_deg:.1f}  actual={accumulated:.1f} deg")
    return accumulated


# =============================================================================
# ENCODER SETUP
# =============================================================================
ENCODER_PIN = 34
DIST_PER_COUNT = 0.119063

count = 0
last_time_us = 0
DEBOUNCE_US = 50000


def encoder_isr(pin):
    global count, last_time_us
    now = ticks_us()
    if ticks_diff(now, last_time_us) > DEBOUNCE_US:
        count += 1
        last_time_us = now


enc = Pin(ENCODER_PIN, Pin.IN)
enc.irq(trigger=Pin.IRQ_FALLING, handler=encoder_isr)


# =============================================================================
# ULTRASONIC SENSORS
# =============================================================================
def make_ultrasonic(trig, echo):
    return {"trig": Pin(trig, Pin.OUT), "echo": Pin(echo, Pin.IN)}


physical_sensors = {
    "rear_physical": make_ultrasonic(33, 13),
    "right_physical": make_ultrasonic(19, 21),
    "left_physical": make_ultrasonic(4, 0),
}

ultra = {
    "front": physical_sensors["rear_physical"],
    "left": physical_sensors["right_physical"],
    "right": physical_sensors["left_physical"],
}


def get_distance(sensor):
    trig, echo = sensor["trig"], sensor["echo"]
    trig.value(0)
    time.sleep_us(2)
    trig.value(1)
    time.sleep_us(10)
    trig.value(0)
    duration = time_pulse_us(echo, 1, 30000)
    if duration > 0:
        d = (duration * 0.0343) / 2
        if d > 2:
            return d
    return 400


def read_front():
    return get_distance(ultra["front"])


def read_sides():
    l = get_distance(ultra["left"])
    r = get_distance(ultra["right"])
    return l, r


def confirm_obstacle_front():
    """
    3 readings with motor pulses between each via drive_ms.
    Needs 2 of 3 below SAFE_FRONT to confirm real obstacle.
    Sabertooth never sees a signal gap.
    """
    confirm_count = 0
    for _ in range(3):
        drive_ms(15)
        check = get_distance(ultra["front"])
        print(f"  Confirming... {check:.0f}cm")
        if check < SAFE_FRONT:
            confirm_count += 1
    return confirm_count >= 2


# =============================================================================
# SENSOR FLAG
# =============================================================================
SENSORS_ENABLED = True

# =============================================================================
# OBSTACLE / CLEARANCE THRESHOLDS
# =============================================================================
SAFE_FRONT = 50
SAFE_SIDE = 60
CLEAR_SIDE = 75
# Stop distance when the front ultrasonic is looking at the confirmed sound
# source (not a generic obstacle). Keeps the robot body from pressing up
# against the person holding the baby. See NAVIGATION PARAMETERS for the
# target-vs-obstacle disambiguation constants.
TARGET_CLEARANCE = 30  # cm

# =============================================================================
# UNIT CONVERSION
# =============================================================================
FT_TO_M = 0.3048


def feet_to_meters(ft):
    return ft * FT_TO_M


# =============================================================================
# NAVIGATION PARAMETERS
# =============================================================================
ARRIVAL_THRESHOLD = 0.762
HEADING_OFFSET = 0
DRIVE_TIMEOUT_MS = 30000
# --- Target-vs-obstacle disambiguation ----------------------------------------
# The front ultrasonic sensor sits physically forward of the mic array (which
# is the navigation reference — SL predicts nav_distance_m from the mic array
# to the sound source). So the sensor enters "obstacle range" of the target
# well before the encoder reports arrival. We convert the ultrasonic reading
# into a position-from-origin estimate and compare it to nav_distance_m to
# decide whether the thing in front is the target or a real obstacle between
# the robot and the target.
SENSOR_OFFSET_M = 0.18  # 7 inches — front ultrasonic → mic array center
TARGET_MATCH_TOLERANCE_M = 0.30  # slack for SL distance noise + target body depth

nav_active = False
nav_dir = None
nav_distance_m = 0.0
nav_start_count = 0
distance_travelled = 0.0
nav_angle = 0.0

# =============================================================================
# OBSTACLE AVOIDANCE STATES
# =============================================================================
GO_STRAIGHT = "GO_STRAIGHT"
AVOID_SIDE = "AVOID_SIDE"
WAITING = "WAITING"

CLEAR_TIMEOUT_MS = 30000  # max time to drive past obstacle after turning back (ms)

avoid_state = GO_STRAIGHT
avoid_dir = None
waiting_start = 0
WAITING_TIMEOUT_MS = 60000  # 60s — if Pi doesn't send new NAV, return to idle

# =============================================================================
# STARTUP
# =============================================================================
led = Pin(2, Pin.OUT)
for _ in range(4):
    led.toggle()
    time.sleep(0.3)

stop()
calibrate_gyro()
print("System ACTIVE")
print("-" * 60)

# =============================================================================
# UART PARSING
# =============================================================================
cancelled = False  # set by CANCEL command, checked in driving loop


def parse_uart():
    global nav_active, nav_dir, nav_distance_m, nav_start_count
    global nav_angle, distance_travelled, avoid_state, avoid_dir, cancelled

    if uart.any():
        raw = uart.readline()
        if raw is None:
            return
        try:
            msg = raw.decode().strip()
        except (UnicodeError, ValueError):
            print("UART: bad bytes, skipping")
            return
        print("UART RX:", msg)

        if msg == "CANCEL":
            stop()
            nav_active = False
            avoid_state = GO_STRAIGHT
            avoid_dir = None
            cancelled = True
            print("CANCEL received — stopped, returning to idle")
            return

        if msg.startswith("NAV"):
            try:
                parts = msg.split()
                x = float(parts[1].split("=")[1])
                d_ft = float(parts[2].split("=")[1])

                nav_distance_m = feet_to_meters(d_ft)
                nav_angle = x + HEADING_OFFSET
                if nav_angle > 180:
                    nav_angle -= 360
                elif nav_angle < -180:
                    nav_angle += 360

                nav_dir = (
                    "FORWARD"
                    if abs(nav_angle) < 5
                    else ("RIGHT" if nav_angle > 0 else "LEFT")
                )

                nav_start_count = count
                distance_travelled = 0.0
                nav_active = True
                avoid_state = GO_STRAIGHT
                avoid_dir = None

                print(
                    f"NAV → d={d_ft:.2f} ft ({nav_distance_m:.2f} m), "
                    f"angle={nav_angle:.1f}°, dir={nav_dir}"
                )
            except (ValueError, IndexError) as e:
                print(f"NAV parse error: {e}")


# =============================================================================
# MAIN LOOP
# =============================================================================
while True:

    if check_bumps():
        bump_halt()

    parse_uart()

    # =========================================================================
    # WAITING — stopped after relisten, do nothing until NAV arrives
    # =========================================================================
    if avoid_state == WAITING:
        stop()
        if ticks_diff(ticks_us(), waiting_start) > WAITING_TIMEOUT_MS * 1000:
            print("WAITING timeout — Pi never sent new NAV, returning to idle")
            avoid_state = GO_STRAIGHT
        sleep_ms(50)
        continue

    # =========================================================================
    # NAVIGATION — only runs in GO_STRAIGHT with an active NAV command
    # =========================================================================
    if nav_active and avoid_state == GO_STRAIGHT:
        # Face the sound source first (always, even at threshold)
        if abs(nav_angle) >= 5:
            turn_by_angle(nav_angle)
            nav_angle = 0.0
            reset_heading()

        remaining_m = nav_distance_m - distance_travelled

        # Arrived?
        if remaining_m <= ARRIVAL_THRESHOLD:
            stop()
            nav_active = False
            uart.write("READY\n")
            print("Arrived — READY for communication")
            sleep_ms(50)
            continue

        # Drive remaining distance with heading hold
        start = count
        target_counts = round(remaining_m / DIST_PER_COUNT)
        drive_start_ms = ticks_us() // 1000
        reset_heading()
        forward_corrected(0)

        timed_out = False
        obstacle_hit = False
        cancelled = False
        target_reached = False

        while (count - start) < target_counts:
            if check_bumps():
                bump_halt()

            # Check for CANCEL from Pi during driving
            parse_uart()
            if cancelled:
                print("CANCEL during drive — stopping")
                break

            if (ticks_us() // 1000 - drive_start_ms) > DRIVE_TIMEOUT_MS:
                print("Drive timeout — possible stall")
                timed_out = True
                break

            gz = read_gyro_z()
            now = ticks_us()
            dt = ticks_diff(now, last_gyro_t) / 1_000_000.0
            last_gyro_t = now
            heading += gz * dt
            correction = max(
                -MAX_CORRECTION_US, min(MAX_CORRECTION_US, int(HEADING_KP * heading))
            )
            forward_corrected(correction)

            if SENSORS_ENABLED:
                live_f = read_front()
                if live_f < SAFE_FRONT:
                    # Confirm before making any decision
                    if confirm_obstacle_front():
                        # Disambiguate: is this the sound source, or a real
                        # obstacle between us and the source?
                        #
                        # The ultrasonic sensor sits SENSOR_OFFSET_M in front
                        # of the mic array, so an object it sees at live_f cm
                        # is at this position in nav coordinates (meters from
                        # the mic's starting point for this NAV command):
                        travelled_now = (
                            distance_travelled + (count - start) * DIST_PER_COUNT
                        )
                        object_pos = travelled_now + SENSOR_OFFSET_M + (live_f / 100.0)

                        if abs(object_pos - nav_distance_m) <= TARGET_MATCH_TOLERANCE_M:
                            # It's the target. Close the gap to clearance.
                            if live_f <= TARGET_CLEARANCE:
                                print(
                                    f"Target reached at {live_f:.0f}cm "
                                    f"(clearance={TARGET_CLEARANCE}cm, "
                                    f"object_pos={object_pos:.2f}m "
                                    f"≈ target={nav_distance_m:.2f}m) — ARRIVED"
                                )
                                target_reached = True
                                break
                            # Target in view but still further than clearance
                            # — keep approaching, next iteration will re-check.
                            print(
                                f"Target ahead at {live_f:.0f}cm "
                                f"(object_pos={object_pos:.2f}m "
                                f"≈ target={nav_distance_m:.2f}m) "
                                f"— closing to {TARGET_CLEARANCE}cm clearance"
                            )
                        else:
                            print(
                                f"Real obstacle at {live_f:.0f}cm "
                                f"(object_pos={object_pos:.2f}m "
                                f"vs target={nav_distance_m:.2f}m) "
                                f"— entering avoidance"
                            )
                            obstacle_hit = True
                            break
                    # False reading — keep driving, confirm kept motors alive

            sleep_ms(10)

        stop()
        distance_travelled += (count - start) * DIST_PER_COUNT

        # CANCEL already handled — nav_active set to False by parse_uart
        if cancelled:
            cancelled = False
            continue

        # Front ultrasonic saw the target, stopped at TARGET_CLEARANCE short
        if target_reached:
            nav_active = False
            uart.write("READY\n")
            print("Target acquired via front ultrasonic — READY for communication")
            sleep_ms(50)
            continue

        if timed_out and (count - start) == 0:
            print("Encoder failure — assuming arrived after timeout")
            nav_active = False
            uart.write("READY\n")
            sleep_ms(50)
            continue

        if obstacle_hit:
            # Fresh side read now that we have stopped
            l, r = read_sides()
            if l >= r:
                avoid_dir = "LEFT"
                print("Turning LEFT to go around")
                turn_by_angle(-90)
            else:
                avoid_dir = "RIGHT"
                print("Turning RIGHT to go around")
                turn_by_angle(90)
            reset_heading()
            avoid_state = AVOID_SIDE
            uart.write("OBSTACLE\n")
            print(
                f"Following obstacle — watching "
                f"{'RIGHT' if avoid_dir == 'LEFT' else 'LEFT'} sensor..."
            )
            continue

    # =========================================================================
    # OBSTACLE AVOIDANCE STATE MACHINE
    # =========================================================================

    # ---- GO_STRAIGHT: obstacle detected between drive iterations ----
    if avoid_state == GO_STRAIGHT and nav_active:
        f = read_front()
        if f < SAFE_FRONT:
            if confirm_obstacle_front():
                stop()
                uart.write("OBSTACLE\n")
                print(f"Obstacle at {f:.0f}cm — choosing side...")
                l, r = read_sides()
                if l >= r:
                    avoid_dir = "LEFT"
                    print("Turning LEFT to go around")
                    turn_by_angle(-90)
                else:
                    avoid_dir = "RIGHT"
                    print("Turning RIGHT to go around")
                    turn_by_angle(90)
                reset_heading()
                avoid_state = AVOID_SIDE
                print(
                    f"Following obstacle — watching "
                    f"{'RIGHT' if avoid_dir == 'LEFT' else 'LEFT'} sensor..."
                )

    # ---- AVOID_SIDE: driving alongside obstacle ----
    elif avoid_state == AVOID_SIDE:
        f = read_front()
        l, r = read_sides()
        wall_dist = r if avoid_dir == "LEFT" else l
        print(
            f"[AVOID_SIDE] F:{f:4.0f} L:{l:4.0f} R:{r:4.0f} cm | heading:{heading:+.2f}°"
        )

        if f < SAFE_FRONT:
            if confirm_obstacle_front():
                stop()
                print("Dead end — turning 180° and relistening")
                if avoid_dir == "LEFT":
                    turn_by_angle(-180)
                else:
                    turn_by_angle(180)
                reset_heading()
                stop()
                nav_active = False
                avoid_state = WAITING
                waiting_start = ticks_us()
                avoid_dir = None
                uart.write("RELISTEN\n")
                print("Sent RELISTEN after dead end")

        elif wall_dist > CLEAR_SIDE:
            # Obstacle end detected — go around the corner
            stop()
            print(
                f"Obstacle cleared (side={wall_dist:.0f}cm) — turning back and driving past"
            )

            # Turn back toward original direction
            if avoid_dir == "LEFT":
                turn_by_angle(90)
            else:
                turn_by_angle(-90)
            reset_heading()

            # Drive forward checking both side sensors:
            # wait to see the obstacle, then wait for it to clear
            print("Driving forward — watching both sensors for obstacle...")
            saw_obstacle = False
            clear_start_ms = ticks_us() // 1000

            while (ticks_us() // 1000 - clear_start_ms) < CLEAR_TIMEOUT_MS:
                if check_bumps():
                    bump_halt()
                parse_uart()
                if cancelled:
                    break

                fc = read_front()
                if fc < SAFE_FRONT and confirm_obstacle_front():
                    print("Front obstacle while clearing — relistening")
                    break

                l, r = read_sides()

                if l < SAFE_SIDE or r < SAFE_SIDE:
                    saw_obstacle = True
                    print(f"[CLEARING] L:{l:4.0f} R:{r:4.0f} cm (obstacle alongside)")
                elif saw_obstacle:
                    print(f"[CLEARING] L:{l:4.0f} R:{r:4.0f} cm — passed obstacle!")
                    break

                update_heading_and_drive()
                sleep_ms(15)

            stop()

            # Done — relisten from new position
            nav_active = False
            avoid_state = WAITING
            waiting_start = ticks_us()
            avoid_dir = None
            uart.write("RELISTEN\n")
            print("Obstacle cleared — sent RELISTEN")

        else:
            update_heading_and_drive()
            drive_ms(15)
