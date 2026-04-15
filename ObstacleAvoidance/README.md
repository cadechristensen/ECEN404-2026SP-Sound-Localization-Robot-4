# Obstacle Avoidance

Firmware that runs on the ESP32 DevKit. Receives NAV commands from the Raspberry Pi over UART, drives the robot toward a bearing + distance, and handles obstacle avoidance along the way using three ultrasonic sensors, an MPU6050 gyro for heading hold, a wheel encoder for odometry, and bump sensors for hard contact.

> **Language:** MicroPython, not Arduino. The ESP32 is flashed with `.py` modules via Thonny or ampy — there are no `.ino` files in this project.

---

## Entry Points

| File | Purpose |
|------|---------|
| `404NewObstacleAvoidance2.py` | Main firmware — UART parser + navigation state machine + motor loop |
| `mpu6050.py` | MPU6050 IMU driver (I2C bus 0, address `0x68`) |
| `gpio_scanner.py` | One-shot utility for checking which GPIO pins report active |
| `Info.txt` | Local setup notes |

Flash the main file to the ESP32; it starts automatically on boot and blocks until a NAV command arrives on `/dev/serial0`.

---

## UART Protocol

Baud 115200, line-terminated ASCII, half-duplex.

| Direction | Message | Meaning |
|-----------|---------|---------|
| Pi → ESP32 | `NAV angle=<deg> dist_ft=<ft>\n` | Drive to bearing (0° forward, positive = right) + distance |
| Pi → ESP32 | `CANCEL\n` | Abort the current NAV and return to idle |
| ESP32 → Pi | `READY\n` | Arrived at target |
| ESP32 → Pi | `OBSTACLE\n` | Obstacle detected — entering wall-following avoidance |
| ESP32 → Pi | `RELISTEN\n` | Dead end — orchestrator should re-localize |
| ESP32 → Pi | `BUMPED\n` | Bump sensor triggered — firmware halts |

---

## Navigation Flow

1. Receive `NAV angle=X dist_ft=Y`.
2. Turn in place by `X` degrees using the gyro-stabilized turn loop (bumper-aware, times out at 15 s).
3. Drive forward with heading hold: a P-loop on gyro-integrated heading error feeds a PWM correction into the differential drive.
4. While driving:
   - Track distance via the KY-024 wheel encoder.
   - Poll the front ultrasonic; if the reading is below `SAFE_FRONT`, confirm with three readings before acting on it.
   - **Target-vs-obstacle disambiguation:** estimate object position in NAV coordinates as `distance_travelled + SENSOR_OFFSET_M + (front_cm / 100)` and compare against `nav_distance_m`:
     - Within `TARGET_MATCH_TOLERANCE_M` → it's the sound source → close to `TARGET_CLEARANCE` → emit `READY`.
     - Otherwise → real obstacle → emit `OBSTACLE` → enter wall-following.
5. **Wall-following:** turn perpendicular to the wall, drive alongside tracking the side ultrasonic, corner around the obstacle when the side opens up (`CLEAR_SIDE`), and resume heading toward the target. If the obstacle never ends within bounds, emit `RELISTEN`.

---

## Hardware Pinout

| Pin | Component |
|-----|-----------|
| 12 | Motor left PWM (Sabertooth CH1, 50 Hz) |
| 27 | Motor right PWM (Sabertooth CH2, 50 Hz) |
| 22, 23 | I2C0 SDA / SCL — MPU6050 @ `0x68` |
| 25 | Bump sensor left (active-low, pull-up) |
| 32 | Bump sensor right (active-low, pull-up) |
| 33, 13 | Front ultrasonic TRIG / ECHO (HC-SR04) |
| 19, 21 | Right ultrasonic TRIG / ECHO |
| 4, 0   | Left ultrasonic TRIG / ECHO |
| 34 | Wheel encoder (KY-024, falling-edge IRQ) |

---

## Tuning Parameters

Defined near the top of `404NewObstacleAvoidance2.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `SAFE_FRONT` | 50 cm | Front obstacle-confirmation threshold |
| `SAFE_SIDE` / `CLEAR_SIDE` | 60 / 75 cm | Wall-following side-distance gates |
| `TARGET_CLEARANCE` | 30 cm | Stop distance when the front is the sound source |
| `ARRIVAL_THRESHOLD` | 0.762 m | Encoder distance tolerance for "arrived" |
| `SENSOR_OFFSET_M` | 0.18 m | Front ultrasonic offset from mic-array center |
| `TARGET_MATCH_TOLERANCE_M` | 0.30 m | Slack when comparing ultrasonic vs SL distance |
| `MAX_CORRECTION_US` | 80 µs | Maximum PWM correction from heading-hold loop |
| `HEADING_KP` | 15 | Proportional gain on gyro heading error |
| `DIST_PER_COUNT` | 0.119 m | Wheel encoder count → meters |

---

## Motor Calibration

Sabertooth 2×12 in R/C mode expects 1000–2000 µs pulses at 50 Hz. Per-motor stop values were hand-calibrated to find the deadband:

| Motor | Stop (µs) | Forward (µs) | Reverse (µs) |
|-------|-----------|--------------|--------------|
| Left  | 1480 | 1600 | 1050 |
| Right | 1230 | 1700 | 1050 |

---

## File Layout

```
ObstacleAvoidance/
├── 404NewObstacleAvoidance2.py   # Main firmware
├── mpu6050.py                    # MPU6050 IMU driver
├── gpio_scanner.py               # GPIO pin scan utility
└── Info.txt                      # Setup notes
```

---

## Related Documentation

- [Project Overview](../README.md) — full system architecture
- [Pi Integration](../Pi_Integration/README.md) — the orchestrator that issues NAV commands
