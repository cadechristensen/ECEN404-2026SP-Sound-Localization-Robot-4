from machine import Pin, PWM, time_pulse_us, UART, I2C
import time
from time import ticks_us, ticks_diff, sleep_ms
import struct
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
STOP_LEFT_US     = 1460
STOP_RIGHT_US    = 1230
 
FORWARD_LEFT_US  = 1600
FORWARD_RIGHT_US = 1700
 
# Turn RIGHT: left wheel forward, right wheel backward
TURN_RIGHT_LEFT_US  = 1600
TURN_RIGHT_RIGHT_US = 1050
 
# Turn LEFT: left wheel backward, right wheel forward
TURN_LEFT_LEFT_US  = 1050
TURN_LEFT_RIGHT_US = 1700
 
def stop():
    motor_left.duty_u16(pulse_us_to_duty(STOP_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(STOP_RIGHT_US))
 
def forward():
    motor_left.duty_u16(pulse_us_to_duty(FORWARD_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(FORWARD_RIGHT_US))
 
def _turn_right_raw():
    motor_left.duty_u16(pulse_us_to_duty(TURN_RIGHT_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(TURN_RIGHT_RIGHT_US))
 
def _turn_left_raw():
    motor_left.duty_u16(pulse_us_to_duty(TURN_LEFT_LEFT_US))
    motor_right.duty_u16(pulse_us_to_duty(TURN_LEFT_RIGHT_US))
 
# =============================================================================
# MPU6050 GYROSCOPE SETUP
# =============================================================================
MPU_ADDR    = 0x68
PWR_MGMT_1  = 0x6B
GYRO_CONFIG = 0x1B
GYRO_ZOUT_H = 0x47
GYRO_SCALE  = 65.5   # ±500 deg/s range
 
i2c = I2C(0, sda=Pin(22), scl=Pin(23), freq=400000)
 
# Wake up MPU6050
i2c.writeto_mem(MPU_ADDR, PWR_MGMT_1, b'\x00')
sleep_ms(100)
 
# Set gyro range to ±500 deg/s
i2c.writeto_mem(MPU_ADDR, GYRO_CONFIG, bytes([1 << 3]))
sleep_ms(10)
print("MPU6050 initialized")
 
z_offset = 0.0
 
def read_gyro_z_raw():
    data = i2c.readfrom_mem(MPU_ADDR, GYRO_ZOUT_H, 2)
    return struct.unpack('>h', data)[0]
 
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
    raw = read_gyro_z_raw()
    return (raw / GYRO_SCALE) - z_offset
 
# =============================================================================
# ENCODER SETUP (distance tracking only)
# =============================================================================
ENCODER_PIN     = 34
COUNTS_PER_42CM = 12
DIST_PER_COUNT  = 0.42 / COUNTS_PER_42CM   # ~0.035 m per count
 
count        = 0
last_time_us = 0
DEBOUNCE_US  = 50000
 
def encoder_isr(pin):
    global count, last_time_us
    now = ticks_us()
    if ticks_diff(now, last_time_us) > DEBOUNCE_US:
        count += 1
        last_time_us = now
 
enc = Pin(ENCODER_PIN, Pin.IN)
enc.irq(trigger=Pin.IRQ_FALLING, handler=encoder_isr)
 
# =============================================================================
# SENSOR FLAG — set False if sensors not plugged in
# =============================================================================
SENSORS_ENABLED = False
 
# =============================================================================
# UNIT CONVERSION
# =============================================================================
FT_TO_M = 0.3048
 
def feet_to_meters(ft):
    return ft * FT_TO_M
 
# =============================================================================
# NAVIGATION PARAMETERS
# =============================================================================
ARRIVAL_THRESHOLD = 0.5      # metres
HEADING_OFFSET    = 0        # degrees — adjust if mic array is offset from robot forward
DRIVE_TIMEOUT_MS  = 30000    # max drive time before stall assumed (30s)
 
nav_active           = False
nav_dir              = None
nav_distance_m       = 0.0
nav_start_count      = 0
distance_travelled   = 0.0
obstacle_encountered = False
nav_angle            = 0.0
 
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
    if not SENSORS_ENABLED:
        return {"front": 400, "left": 400, "right": 400}
    return {k: get_distance(v) for k, v in ultra.items()}
 
# =============================================================================
# OBSTACLE AVOIDANCE PARAMETERS
# =============================================================================
SAFE_FRONT = 50   # cm
SAFE_SIDE  = 75   # cm
 
# =============================================================================
# STARTUP
# =============================================================================
led = Pin(2, Pin.OUT)
for _ in range(4):
    led.toggle()
    time.sleep(0.3)
stop()
calibrate_gyro()   # calibrate gyro on startup — hold robot still
print("System ACTIVE")
print("-" * 60)
 
# =============================================================================
# UART PARSING FOR NAVIGATION
# =============================================================================
def parse_uart():
    global nav_active, nav_dir, nav_distance_m, nav_start_count
    global obstacle_encountered, nav_angle, distance_travelled
 
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
 
        if msg.startswith("NAV"):
            try:
                # Expected format: "NAV x=-1.234 y=2.345 d=3.456"
                # x = angle in degrees, d = distance in feet
                parts = msg.split()
                x    = float(parts[1].split("=")[1])
                d_ft = float(parts[3].split("=")[1])
 
                nav_distance_m = feet_to_meters(d_ft)
                nav_angle      = x + HEADING_OFFSET
                if nav_angle > 180:
                    nav_angle -= 360
                elif nav_angle < -180:
                    nav_angle += 360
                nav_dir        = ("FORWARD" if abs(nav_angle) < 5
                                  else ("RIGHT" if nav_angle > 0 else "LEFT"))
 
                nav_start_count      = count
                distance_travelled   = 0.0
                obstacle_encountered = False
                nav_active           = True
 
                print(f"NAV cmd  → d={d_ft:.2f} ft ({nav_distance_m:.2f} m), "
                      f"angle={nav_angle:.1f}°, dir={nav_dir}")
            except (ValueError, IndexError) as e:
                print(f"NAV parse error: {e}")
 
# =============================================================================
# HELPER — TURN BY ANGLE (gyro closed-loop)
# =============================================================================
def turn_by_angle(target_deg, tolerance=2.0, timeout_ms=10000):
    """
    Closed-loop gyro turn.
    Positive = right, negative = left.
    Resets encoder count after turn so stray counts don't affect distance.
    """
    global count
 
    if abs(target_deg) < 1:
        return 0.0
 
    accumulated = 0.0
    target      = abs(target_deg)
 
    if target_deg > 0:
        _turn_right_raw()
    else:
        _turn_left_raw()
 
    last_time  = ticks_us()
    start_time = last_time
 
    while abs(accumulated) < (target - tolerance):
        if ticks_diff(ticks_us(), start_time) > timeout_ms * 1000:
            print(f"Turn timeout! Reached {accumulated:.1f} of {target_deg} deg")
            break
        gz  = read_gyro_z()
        now = ticks_us()
        dt  = ticks_diff(now, last_time) / 1_000_000.0
        last_time = now
        accumulated += gz * dt
        sleep_ms(5)
 
    stop()
    sleep_ms(50)
    count = 0   # discard encoder counts from turn
    print(f"Turn done: target={target_deg:.1f}, actual={abs(accumulated):.1f} deg")
    return accumulated
 
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
 
    print(f"F:{f:.0f} L:{l:.0f} R:{r:.0f} | "
          f"NAV:{nav_dir if nav_active else 'IDLE'} | "
          f"Travelled:{distance_travelled:.2f} m / {nav_distance_m:.2f} m")
 
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
            print("Arrived within threshold, ready for communication")
            continue
 
        # ---- Face the sound source ----
        if abs(nav_angle) >= 5:
            turn_by_angle(nav_angle)
            nav_angle = 0.0   # aligned, don't re-turn
 
        # ---- Drive remaining distance ----
        start = count
        target_counts = round(remaining_m / DIST_PER_COUNT)
        drive_start_ms = ticks_us() // 1000
        forward()
        while (count - start) < target_counts:
            if (ticks_us() // 1000 - drive_start_ms) > DRIVE_TIMEOUT_MS:
                print("Drive timeout — possible stall")
                break
            if SENSORS_ENABLED:
                live_f = get_distance(ultra["front"], samples=1)
                if live_f < SAFE_FRONT:
                    obstacle_encountered = True
                    break
            sleep_ms(20)
 
        stop()
        distance_travelled += (count - start) * DIST_PER_COUNT
 
    # =========================================================================
    # OBSTACLE AVOIDANCE
    # =========================================================================
    if front_blocked or obstacle_encountered:
        stop()
        uart.write("OBSTACLE\n")
        obstacle_encountered = True
        print("Obstacle ahead! Choosing escape direction...")
 
        if left_blocked and not right_blocked:
            print("Turning RIGHT (left side blocked)")
            turn_by_angle(90)
        elif right_blocked and not left_blocked:
            print("Turning LEFT (right side blocked)")
            turn_by_angle(-90)
        else:
            if l >= r:
                print("Turning LEFT (more clearance)")
                turn_by_angle(-90)
            else:
                print("Turning RIGHT (more clearance)")
                turn_by_angle(90)
 
        stop()
        uart.write("RELISTEN\n")
        nav_active           = False
        obstacle_encountered = False
        print("Waiting for updated NAV command after re-listen...")
 
    time.sleep(0.05)
