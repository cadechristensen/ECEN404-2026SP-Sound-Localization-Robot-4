# Sound-Localization-Team-4

**ECEN 404 Capstone — Team 64 · "Bring The Hertz"**

An autonomous baby monitoring robot that detects infant cries in real time, localizes the sound source using a 4-microphone array, navigates toward the baby while avoiding obstacles, and notifies the caregiver with a confidence score and a live video feed. All inference runs on-device on a Raspberry Pi 5.

---

## Key Features

- **Real-time cry detection** — CNN-Transformer hybrid, 97.93% accuracy, 100–200 ms inference on Pi CPU
- **Sound localization** — DOAnet CRNN with five angle-specific models for direction-of-arrival estimation
- **Obstacle avoidance** — ESP32 (MicroPython) with Sabertooth motor driver and gyro-stabilized heading hold
- **Mobile caregiver alerts** — Flask server pushes notifications plus a live video stream
- **Edge deployment** — Low-power listening mode, phase-preserving audio pipeline, calibrated 0.92 detection threshold

---

## System Architecture

```
LOW-POWER LISTENING   (1 s chunks, fast gate)
        ↓
CRY DETECTED          (confidence ≥ 0.92)
        ↓
TEMPORAL SMOOTHING    (3+ consecutive frames)
        ↓
PHASE-PRESERVING FILTER  (48 kHz, 4-channel, cry isolated)
        ↓
SOUND LOCALIZATION    (DOAnet → direction + distance)
        ↓
NAVIGATE              (Pi → ESP32 over UART, ultrasonic avoidance)
        ↓
CAREGIVER ALERT       (mobile notification + live video)
        ↓
RETURN TO LISTENING
```

---

## Repository Structure

```
Sound-Localization-Team-4/
├── baby_cry_detection/     # CNN-Transformer cry detection — training, evaluation, deployment
├── SoundLocalization/      # DOAnet CRNN direction-of-arrival estimation (angle-specific models)
├── ObstacleAvoidance/      # ESP32 MicroPython motor control, ultrasonic + gyro navigation
├── Pi_Integration/         # System orchestrator — state machine, UART link, subsystem wrappers
├── AppMobile/              # Flask alert server + mobile app (email, notifications, video stream)
├── hardware_setup/         # Pi-side PCM6260 mic-array bring-up (firmware load, gain, register decode)
└── README.md               # This file
```

---

## Hardware

| Component | Purpose |
|-----------|---------|
| Raspberry Pi 5 (8 GB) | Main compute — cry detection, localization, orchestrator |
| TI PCM6260-Q1 codec + 4× AOM-5024L-HD-R | USB 4-channel microphone array (I2C address 0x48) |
| ESP32 DevKit | Motor control, sensor polling, navigation state machine |
| Sabertooth 2×12 motor driver | Dual-channel DC motor control (R/C mode) |
| MPU6050 | 6-DOF IMU — heading hold during forward drive |
| HC-SR04 ultrasonic sensors (3×) | Forward + left/right obstacle detection |
| KY-024 encoder | Distance odometry |
| IG-42GM gear motors (2×) | Drive wheels |

---

## Quick Start

### Development — training the cry detection model

```bash
cd baby_cry_detection
pip install -r requirements.txt

# Preprocess the dataset (first run only)
python scripts/preprocess_dataset.py --output data/processed/v1

# Train
python training/main.py train

# Evaluate with temperature-scaling calibration
python training/main.py evaluate --model results/train_*/model_best.pth
```

### Deployment — running the full system on the Pi

```bash
# One-time mic-array bring-up (TI PCM6260 codec: I2C verify, firmware load, gain set)
cd hardware_setup
sudo ./setup.sh 0x50

# Install Pi-specific dependencies
cd ../baby_cry_detection/deployment
pip3 install -r requirements-pi.txt

# Launch the orchestrator (starts BCD, SL, UART link, and Flask app)
python3 Pi_Integration/main.py --device-name "TI USB Audio" -q
```

> See [`hardware_setup/README.md`](hardware_setup/README.md) for details on the mic-array setup step.

### ESP32 firmware

Flash `ObstacleAvoidance/404NewObstacleAvoidance2.py` to the ESP32 via Thonny or ampy. The ESP32 communicates with the Pi over UART at 115200 baud on `/dev/serial0`.

---

## Subsystem Documentation

### Subsystem READMEs

- [Baby Cry Detection](baby_cry_detection/README.md) — CNN-Transformer model, training, evaluation, filtering, deployment
- [Sound Localization](SoundLocalization/README.md) — DOAnet CRNN with angle-specific models for direction-of-arrival
- [Obstacle Avoidance](ObstacleAvoidance/README.md) — ESP32 MicroPython firmware — motor control, ultrasonic avoidance, gyro heading hold
- [Pi Integration](Pi_Integration/README.md) — orchestrator state machine and subsystem wrappers
- [Mobile App & Alert Server](AppMobile/README.md) — Flask server + Android companion app
- [Hardware Setup](hardware_setup/README.md) — Pi-side mic-array bring-up (PCM6260 firmware, gain, diagnostics)

### Additional Guides

- [Deployment Guide](baby_cry_detection/deployment/docs/DEPLOYMENT_GUIDE.md) — Pi-specific setup, thresholds, and troubleshooting
- [Audio Sample-Rate Flow](Pi_Integration/AUDIO_SAMPLE_RATE_FLOW.md) — how 16 kHz BCD and 48 kHz SL pipelines coexist
- [Dataset Summary](baby_cry_detection/docs/DATASET_SUMMARY.md) — sources, class balance, preprocessing

---

## UART Protocol (Pi ↔ ESP32)

| Direction | Message | Meaning |
|-----------|---------|---------|
| Pi → ESP32 | `NAV angle=<deg> dist_ft=<ft>\n` | Drive to bearing + distance |
| ESP32 → Pi | `READY\n` | Arrived at target |
| ESP32 → Pi | `OBSTACLE\n` | Obstacle encountered, wall-following |
| ESP32 → Pi | `RELISTEN\n` | Dead end — re-localize from current position |

`angle`: 0° forward, positive = right, negative = left. `dist_ft`: source distance in feet.

---

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.93 % |
| Precision | 97.93 % |
| Recall | 97.93 % |
| ROC-AUC | 99.50 % |
| Inference latency (Pi 5 CPU) | 100–200 ms |

Model: CNN-Transformer hybrid (d_model = 384, 8 heads, 4 layers) trained on ~8,700 audio samples across 59 negative categories. Ensemble of top-3 checkpoints with post-hoc temperature scaling.

---

## Team

**ECEN 404 — Team 64 · "Bring The Hertz"**
Texas A&M University · Electrical & Computer Engineering · Spring 2026.

---

## Acknowledgments

- CryCeleb2023, ICSD, and Donate-a-Cry dataset contributors
- LibriSpeech (adult-speech hard negatives) and ESC-50 (environmental sounds)
- PyTorch, torchaudio, librosa, and the broader open-source audio-ML ecosystem

---

## License

For educational and research purposes (ECEN 404 Capstone).
