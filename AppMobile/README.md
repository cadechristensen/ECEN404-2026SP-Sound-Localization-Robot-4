# Mobile App & Alert Server

Delivers caregiver notifications when the robot reaches a crying baby. The Pi side runs a Flask server that streams live video from the camera and exposes status endpoints. The Android companion app polls the server over the local network, raises push notifications, and displays the live feed.

---

## Architecture

```
           Pi (Flask + mDNS)                     Android app
    ┌─────────────────────────────┐      ┌───────────────────────┐
    │  /video    (MJPEG stream)   │◄─────│  Video viewer         │
    │  /status   (JSON poll)      │◄─────│  Notification poller  │
    │  /trigger  (POST alert)     │      │                       │
    │  /reset    (POST ack)       │      └───────────────────────┘
    │  SMTP email alert           │
    │  mDNS: _babyrobot._tcp      │
    └─────────────────────────────┘
             ▲
             │ send_alert(confidence)
             │
        Orchestrator
```

The Android app discovers the Pi via mDNS, so no manual IP configuration is needed on a fresh network.

---

## Entry Points

| File | Purpose |
|------|---------|
| `AppMobileFire.py` | Flask server — video stream, status endpoints, mDNS registration, SMTP email alert |
| `MobileAppAndroidStudio/` | Android Studio project (Kotlin) for the companion app |

The Flask server is started automatically by the Pi orchestrator via `FunctionCalls_App.MobileApp.start_server()`.

---

## Flask Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Landing page |
| `GET` | `/video` | MJPEG live video stream from the camera |
| `GET` | `/status` | Returns `{triggered: bool, confidence: float}` — polled by the Android app |
| `POST` | `/trigger` | Called by the orchestrator on a confirmed cry — sets `triggered=true` |
| `POST` | `/reset` | Called by the app once the caregiver acknowledges — clears the flag |

---

## Android Companion App

`MobileAppAndroidStudio/Capstone403/` is a standard Android Studio project targeting the endpoints above.

Responsibilities:

- Resolve the Pi via mDNS on the local Wi-Fi.
- Poll `/status` at a fixed interval; on `triggered=true` raise a system notification that includes the confidence score.
- Open the MJPEG stream from `/video` inside an in-app viewer.
- `POST /reset` after the caregiver taps the acknowledge action.

Build with Android Studio Giraffe or later. Min / target SDK versions live in `app/build.gradle`.

---

## Email Notifications

`send_stream_email()` in `AppMobileFire.py` sends an SMTP message to a configured address when the robot arrives at the baby. The email includes the confidence score and a link back to the server's video stream, so caregivers who don't have the app installed still get an alert.

---

## Runtime Configuration

Network and capture settings are defined near the top of `AppMobileFire.py`:

| Constant | Purpose |
|----------|---------|
| `FLASK_PORT` | Port the Flask server listens on |
| `MDNS_NAME` | Service name advertised on the local network |
| `CAMERA_INDEX` | OpenCV capture index for the camera |

---

## File Layout

```
AppMobile/
├── AppMobileFire.py               # Flask server + mDNS + SMTP email
└── MobileAppAndroidStudio/        # Android Studio project (Kotlin companion app)
    └── Capstone403/               # App module (Gradle build, activities, resources)
```

---

## Related Documentation

- [Project Overview](../README.md) — full system architecture
- [Pi Integration](../Pi_Integration/README.md) — orchestrator that calls `start_server()` and `send_alert()` on a confirmed cry
