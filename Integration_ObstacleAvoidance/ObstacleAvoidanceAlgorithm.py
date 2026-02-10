from machine import Pin, PWM, time_pulse_us, UART
import time

print("Starting 3-Sensor Obstacle Avoidance + Sound Navigation")

# =============================================================================
# UART SETUP (ESP32 <-> Raspberry Pi)
# =============================================================================
uart = UART(1, baudrate=115200, tx=Pin(25), rx=Pin(26))
uart.write("READY\n")

# =============================================================================
# MOTOR SETUP (Sabertooth R/C Mode)
# =============================================================================
PWM_FREQ = 50
motor_left = PWM(Pin(12), freq=PWM_FREQ)
motor_right = PWM(Pin(27), freq=PWM_FREQ)

def pulse_us_to_duty(us):
    return int(us / 20000 * 65535)

STOP    = pulse_us_to_duty(1500)
FORWARD = pulse_us_to_duty(1400)
REVERSE = pulse_us_to_duty(1600)

def stop():
    motor_left.duty_u16(STOP)
    motor_right.duty_u16(STOP)

def forward():
    motor_left.duty_u16(FORWARD)
    motor_right.duty_u16(FORWARD)

def reverse():
    motor_left.duty_u16(REVERSE)
    motor_right.duty_u16(REVERSE)

def turn_left():
    motor_left.duty_u16(REVERSE)
    motor_right.duty_u16(FORWARD)

def turn_right():
    motor_left.duty_u16(FORWARD)
    motor_right.duty_u16(REVERSE)

# =============================================================================
# NAVIGATION PARAMETERS (FROM SOUND LOCALIZATION)
# =============================================================================
ROBOT_SPEED = 0.30   # meters per second (CALIBRATE THIS)

nav_active = False
nav_dir = None
nav_distance = 0.0
nav_end_time = 0.0

def parse_uart():
    global nav_active, nav_dir, nav_distance, nav_end_time

    if uart.any():
        msg = uart.readline().decode().strip()
        print("UART:", msg)

        if msg.startswith("NAV"):
            parts = msg.split()
            x = float(parts[1].split("=")[1])
            y = float(parts[2].split("=")[1])
            nav_distance = float(parts[3].split("=")[1])

            if abs(y) > abs(x):
                nav_dir = "FORWARD"
            else:
                nav_dir = "LEFT" if x < 0 else "RIGHT"

            move_time = nav_distance / ROBOT_SPEED
            nav_end_time = time.time() + move_time
            nav_active = True

# =============================================================================
# ULTRASONIC SENSORS (REMAPPED)
# =============================================================================
def make_ultrasonic(trig, echo):
    return {"trig": Pin(trig, Pin.OUT), "echo": Pin(echo, Pin.IN)}

physical_sensors = {
    "rear_physical":  make_ultrasonic(14, 13),
    "left_physical":  make_ultrasonic(19, 21),
    "right_physical": make_ultrasonic(4, 16)
}

ultra = {
    "front": physical_sensors["rear_physical"],
    "left":  physical_sensors["right_physical"],
    "right": physical_sensors["left_physical"]
}

# =============================================================================
# DISTANCE MEASUREMENT
# =============================================================================
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
            d = (duration * 0.0343) / 2
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
# PARAMETERS
# =============================================================================
SAFE_FRONT = 50
SAFE_SIDE  = 75
TURN_TIME  = 0.85
FOLLOW_STEP = 0.25

# =============================================================================
# FSM STATES
# =============================================================================
GO_STRAIGHT  = 0
FOLLOW_LEFT  = 1
FOLLOW_RIGHT = 2
REVERSE_OUT  = 3

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
# MAIN LOOP
# =============================================================================
while True:
    parse_uart()

    d = read_distances()
    f, l, r = d["front"], d["left"], d["right"]

    front_blocked = f < SAFE_FRONT
    left_blocked  = l < SAFE_SIDE
    right_blocked = r < SAFE_SIDE

    print(f"State:{state} | F:{f:.1f} L:{l:.1f} R:{r:.1f} | NAV:{nav_dir}")

    # -------------------------------------------------------------------------
    if state == GO_STRAIGHT:
        if front_blocked:
            stop()
            uart.write("OBSTACLE\n")
            nav_active = False

            if left_blocked and not right_blocked:
                turn_right()
                time.sleep(TURN_TIME)
                state = FOLLOW_RIGHT

            elif right_blocked and not left_blocked:
                turn_left()
                time.sleep(TURN_TIME)
                state = FOLLOW_LEFT

            else:
                if l > r:
                    turn_left()
                    state = FOLLOW_LEFT
                else:
                    turn_right()
                    state = FOLLOW_RIGHT
                time.sleep(TURN_TIME)

            stop()

        else:
            if nav_active:
                if time.time() > nav_end_time:
                    stop()
                    nav_active = False
                    uart.write("RELISTEN\n")
                else:
                    if nav_dir == "FORWARD":
                        forward()
                    elif nav_dir == "LEFT":
                        turn_left()
                    elif nav_dir == "RIGHT":
                        turn_right()
            else:
                stop()

            time.sleep(FOLLOW_STEP)

    # -------------------------------------------------------------------------
    elif state == FOLLOW_LEFT:
        if front_blocked:
            stop()
            turn_right()
            time.sleep(2 * TURN_TIME)
            stop()
            uart.write("RELISTEN\n")
            state = GO_STRAIGHT
        else:
            forward()
            time.sleep(FOLLOW_STEP)

    # -------------------------------------------------------------------------
    elif state == FOLLOW_RIGHT:
        if front_blocked:
            stop()
            turn_left()
            time.sleep(2 * TURN_TIME)
            stop()
            uart.write("RELISTEN\n")
            state = GO_STRAIGHT
        else:
            forward()
            time.sleep(FOLLOW_STEP)

    time.sleep(0.05)

