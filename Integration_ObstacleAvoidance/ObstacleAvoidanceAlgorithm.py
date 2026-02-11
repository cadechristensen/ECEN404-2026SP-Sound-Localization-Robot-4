from machine import Pin, PWM, time_pulse_us, UART
import time

print("Starting 3-Sensor Obstacle Avoidance + Sound Navigation")

# =============================================================================
# UART SETUP (ESP32 <-> Raspberry Pi)
# =============================================================================
uart = UART(1, baudrate=115200, tx=Pin(25), rx=Pin(26))
print("UART initialized")

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
ARRIVAL_THRESHOLD = 0.5  # Stop when within 0.5m of target

nav_active = False
nav_dir = None
nav_distance = 0.0
nav_start_time = 0.0
nav_duration = 0.0
obstacle_encountered = False

def parse_uart():
    global nav_active, nav_dir, nav_distance, nav_start_time, nav_duration, obstacle_encountered

    if uart.any():
        msg = uart.readline().decode().strip()
        print("UART RX:", msg)

        if msg.startswith("NAV"):
            # Parse: "NAV x=-1.234 y=2.345 d=3.456"
            parts = msg.split()
            x = float(parts[1].split("=")[1])
            y = float(parts[2].split("=")[1])
            nav_distance = float(parts[3].split("=")[1])

            print(f"Navigation command: x={x:.2f}, y={y:.2f}, dist={nav_distance:.2f}m")

            # Determine primary direction based on coordinates
            if abs(y) > abs(x):
                nav_dir = "FORWARD"
            else:
                nav_dir = "LEFT" if x < 0 else "RIGHT"

            # Calculate navigation duration
            nav_duration = nav_distance / ROBOT_SPEED
            nav_start_time = time.time()
            nav_active = True
            obstacle_encountered = False
            
            print(f"Direction: {nav_dir}, Duration: {nav_duration:.1f}s")

# =============================================================================
# ULTRASONIC SENSORS (REMAPPED TO MATCH PHYSICAL LAYOUT)
# =============================================================================
def make_ultrasonic(trig, echo):
    return {"trig": Pin(trig, Pin.OUT), "echo": Pin(echo, Pin.IN)}

# Physical sensor wiring
physical_sensors = {
    "rear_physical":  make_ultrasonic(14, 13),
    "left_physical":  make_ultrasonic(19, 21),
    "right_physical": make_ultrasonic(4, 16)
}

# Logical mapping (what we call "front" is physically rear)
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
            d = (duration * 0.0343) / 2  # cm
            if d > 2:
                readings.append(d)

        time.sleep(0.005)

    if readings:
        readings.sort()
        return readings[len(readings) // 2]  # median

    return 400  # max range if no valid reading

def read_distances():
    return {k: get_distance(v) for k, v in ultra.items()}

# =============================================================================
# OBSTACLE AVOIDANCE PARAMETERS
# =============================================================================
SAFE_FRONT = 50   # cm - stop if obstacle closer than this
SAFE_SIDE  = 75   # cm - side clearance needed
TURN_TIME  = 0.85 # seconds for 90° turn
FOLLOW_STEP = 0.25 # seconds between movements

# =============================================================================
# FSM STATES
# =============================================================================
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
# MAIN LOOP
# =============================================================================
while True:
    parse_uart()

    d = read_distances()
    f, l, r = d["front"], d["left"], d["right"]

    front_blocked = f < SAFE_FRONT
    left_blocked  = l < SAFE_SIDE
    right_blocked = r < SAFE_SIDE

    print(f"S:{state} | F:{f:.0f} L:{l:.0f} R:{r:.0f} | NAV:{nav_dir if nav_active else 'IDLE'}")

    # =========================================================================
    # CHECK IF NAVIGATION COMPLETE
    # =========================================================================
    if nav_active and not obstacle_encountered:
        elapsed = time.time() - nav_start_time
        
        if elapsed >= nav_duration:
            # Navigation time complete - arrived at baby
            stop()
            nav_active = False
            
            # Signal Raspberry Pi we're ready for video communication
            uart.write("READY\n")
            print("✓ At baby - READY for communication")
            state = GO_STRAIGHT
            continue

    # =========================================================================
    # STATE: GO_STRAIGHT (NORMAL NAVIGATION)
    # =========================================================================
    if state == GO_STRAIGHT:
        if front_blocked:
            stop()
            uart.write("OBSTACLE\n")
            obstacle_encountered = True
            
            print("⚠ Obstacle ahead! Choosing escape direction...")

            # Choose escape direction based on side clearance
            if left_blocked and not right_blocked:
                print("→ Turning RIGHT to avoid")
                turn_right()
                time.sleep(TURN_TIME)
                state = FOLLOW_RIGHT

            elif right_blocked and not left_blocked:
                print("← Turning LEFT to avoid")
                turn_left()
                time.sleep(TURN_TIME)
                state = FOLLOW_LEFT

            else:
                # Both sides blocked or both clear - choose based on more space
                if l > r:
                    print("← Turning LEFT (more space)")
                    turn_left()
                    state = FOLLOW_LEFT
                else:
                    print("→ Turning RIGHT (more space)")
                    turn_right()
                    state = FOLLOW_RIGHT
                time.sleep(TURN_TIME)

            stop()

        else:
            # No obstacle in front
            if nav_active:
                # Execute navigation command
                if nav_dir == "FORWARD":
                    forward()
                elif nav_dir == "LEFT":
                    turn_left()
                elif nav_dir == "RIGHT":
                    turn_right()
            else:
                # No active navigation
                stop()

            time.sleep(FOLLOW_STEP)

    # =========================================================================
    # STATE: FOLLOW_LEFT (WALL FOLLOWING)
    # =========================================================================
    elif state == FOLLOW_LEFT:
        if front_blocked:
            # Hit dead end while wall following
            stop()
            print("⚠ Dead end - reversing direction")
            turn_right()
            time.sleep(2 * TURN_TIME)  # 180° turn
            stop()
            
            # Request new localization
            uart.write("RELISTEN\n")
            nav_active = False
            obstacle_encountered = False
            state = GO_STRAIGHT
        else:
            # Keep moving forward along left wall
            forward()
            time.sleep(FOLLOW_STEP)

    # =========================================================================
    # STATE: FOLLOW_RIGHT (WALL FOLLOWING)
    # =========================================================================
    elif state == FOLLOW_RIGHT:
        if front_blocked:
            # Hit dead end while wall following
            stop()
            print("⚠ Dead end - reversing direction")
            turn_left()
            time.sleep(2 * TURN_TIME)  # 180° turn
            stop()
            
            # Request new localization
            uart.write("RELISTEN\n")
            nav_active = False
            obstacle_encountered = False
            state = GO_STRAIGHT
        else:
            # Keep moving forward along right wall
            forward()
            time.sleep(FOLLOW_STEP)

    time.sleep(0.05)
