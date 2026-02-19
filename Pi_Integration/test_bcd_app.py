"""
BCD + App integration test.

Tests the path:
    audio file -> BabyCryDetection -> MobileApp.send_alert -> /status endpoint

Run from Pi_Integration/:
    python test_bcd_app.py
    python test_bcd_app.py path/to/audio.wav -q
"""

import os
import sys
import json
import time
import logging
import argparse
import urllib.request

# Make AppMobile importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'AppMobile'))

import numpy as np

from FunctionCalls_BCD import BabyCryDetection
from FunctionCalls_App import MobileApp
import AppMobileFire as AM


def make_synthetic_cry(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate a synthetic cry-like signal (amplitude-modulated tone) as a
    4-channel array. Used when no audio file is provided.
    """
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    # Baby cry fundamental ~350 Hz with 5 Hz AM envelope
    signal = 0.5 * np.sin(2 * np.pi * 350 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t))
    signal = signal.astype(np.float32)
    # Stack to 4 channels with small per-channel noise
    channels = [signal + 0.01 * np.random.randn(len(signal)).astype(np.float32)
                for _ in range(4)]
    return np.column_stack(channels)


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(path)
    if sr != target_sr:
        import librosa
        print(f"  Resampling {sr} Hz -> {target_sr} Hz...")
        if audio.ndim == 2:
            channels = [librosa.resample(audio[:, ch].astype(np.float32),
                                         orig_sr=sr, target_sr=target_sr)
                        for ch in range(audio.shape[1])]
            audio = np.column_stack(channels)
        else:
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def main():
    parser = argparse.ArgumentParser(description="BCD + App integration test")
    parser.add_argument("audio", nargs="?", default=None,
                        help="Path to a WAV file (omit to use synthetic signal)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress BCD step-by-step output")
    args = parser.parse_args()

    SR = 16000

    # --- 1. Load or generate audio ---
    print("=" * 60)
    if args.audio:
        print(f"  Loading audio: {args.audio}")
        audio_data, sr = load_audio(args.audio, target_sr=SR)
        print(f"  Shape: {audio_data.shape}, sr={sr}")
        if audio_data.ndim == 2 and audio_data.shape[1] > 4:
            print(f"  Trimming from {audio_data.shape[1]} to 4 channels (matching main.py)")
            audio_data = audio_data[:, :4]
            print(f"  Shape after trim: {audio_data.shape}")
    else:
        print("  No audio file provided — using synthetic cry signal")
        audio_data = make_synthetic_cry(duration_s=3.0, sample_rate=SR)
        print(f"  Shape: {audio_data.shape}, sr={SR}")
    print("=" * 60)

    # --- 2. Start Flask server ---
    print("\n[App] Starting Flask server...")
    app = MobileApp()
    app.start_server()
    time.sleep(1.0)  # let Flask bind
    print(f"[App] Stream URL: {app.stream_url}")

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # --- 3. Run BCD ---
    print("\n[BCD] Running baby cry detection...")
    bcd = BabyCryDetection(verbose=not args.quiet)
    detection = bcd.detect_from_audio(audio_data)

    print()
    print("=" * 60)
    print("  Baby Cry Detection Result")
    print("=" * 60)
    print(f"  Is cry:     {detection.is_cry}")
    print(f"  Confidence: {detection.confidence:.2%}")
    print("=" * 60)

    # --- 4. Fire alert if cry detected ---
    if detection.is_cry:
        print("\n[App] Cry detected — sending alert...")
        app.send_alert(detection.confidence)
        print("[App] Alert sent")
    else:
        print("\n[App] No cry detected — skipping alert")
        print("      (Use a real baby cry WAV to test the full alert path)")

    # --- 5. Check notification_status in memory ---
    print()
    print("=" * 60)
    print("  notification_status (in-process)")
    print("=" * 60)
    for k, v in AM.notification_status.items():
        print(f"  {k}: {v}")

    # --- 6. Poll /status endpoint ---
    print()
    print("=" * 60)
    print("  GET /status (HTTP)")
    print("=" * 60)
    try:
        raw = urllib.request.urlopen("https://megacephalic-tarsha-dextrorotatory.ngrok-free.dev", timeout=5).read()
        resp = json.loads(raw)
        for k, v in resp.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print("=" * 60)

    print("\n[Server still running — open https://megacephalic-tarsha-dextrorotatory.ngrok-free.dev in a browser]")
    input("Press Enter to stop...\n")
    app.stop()
    print("Done.")


if __name__ == "__main__":
    main()
