# Sound-Localization-Team-4

A complete AI-powered baby monitoring system combining real-time cry detection with sound localization for autonomous robot navigation.

---

## Overview

This system uses a 4-microphone array and Raspberry Pi 5 to detect baby cries in real-time, filters the audio, and determines the baby's location using sound localization. The robot navigates towards the baby's location using obstacle avoidance and ESP32 UART commands. Once the robot arrives at the baby's location the caregiver would get a notification of the confidence level of the detected cry and a live video to view from the camera on the robot. Designed for deployment on Raspberry Pi 5 for robotic baby monitoring applications.

### Key Features

- **Real-time Baby Cry Detection** - CNN-Transformer hybrid model
- **Audio Filtering** - Isolates baby cry frequencies (100-3000 Hz)
- **Low-Power Listening Mode** - Optimized for battery operation
- **Two-Stage Detection** - Fast initial detection + TTA confirmation
- **Raspberry Pi Optimized Cry Detection** - Designed for edge deployment

---

## System Architecture
```

Architecture:
    LOW-POWER LISTENING MODE (1-sec chunks, quick detection)
        |
        v
    CRY DETECTED? (>50% confidence, 3 consecutive positives)
        |
        v
    CAPTURE CONTEXT (3-5 seconds of audio)
        |
        v
    CONFIRM WITH TTA (>85% confidence)
        |
        v
    AUDIO FILTERING (noise removal, frequency isolation)
        |
        v
    AUDIO FOR LOCALIZATION (48kHz, 4-channel, cry isolated)
        |
        v
    SOUND LOCALIZATION (DOAnet: direction + distance)
        |
        v
    NAVIGATE ROBOT (ESP32 UART commands)
        |
        v
    NOTIFY CAREGIVER (notification with confidence, view live video from robot)
        |
        v
    RETURN TO LOW-POWER MODE
```



