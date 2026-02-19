"""
Generate sound-localization training data by running the BCD pipeline on
recorded or pre-existing audio and saving the filtered multichannel output.

Usage:
    # From an existing WAV file
    python Pi_Integration/generate_sl_training_data.py --input Pi_Integration/1ft_set4.wav --label 1ft_test -q

    # Live recording from mic array
    python Pi_Integration/generate_sl_training_data.py --device-index 2 --duration 10 --label 3ft_90deg -q
"""

import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Path setup (same pattern as FunctionCalls_BCD.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from FunctionCalls_BCD import BabyCryDetection
from FunctionCalls_SL import SoundLocalization

logger = logging.getLogger(__name__)

BCD_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def record_audio(device_index, duration, channels, sample_rate):
    """Record audio from a PyAudio device and return (np.ndarray, sample_rate)."""
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024,
        )
        logger.info(
            f"Recording {duration}s from device {device_index} "
            f"({channels}ch, {sample_rate}Hz)..."
        )
        frames = []
        for _ in range(int(sample_rate / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.float32))
        stream.stop_stream()
        stream.close()
    finally:
        pa.terminate()

    audio = np.concatenate(frames)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    logger.info(f"Recorded {audio.shape[0] / sample_rate:.2f}s")
    return audio, sample_rate


# ---------------------------------------------------------------------------
# Resampling helper (from main.py pattern)
# ---------------------------------------------------------------------------
def resample_audio(audio, orig_sr, target_sr):
    """Resample audio array to target_sr. Returns resampled ndarray."""
    if orig_sr == target_sr:
        return audio
    import librosa

    logger.info(f"Resampling from {orig_sr} Hz to {target_sr} Hz...")
    if audio.ndim == 2:
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(
                librosa.resample(
                    audio[:, ch].astype(np.float32),
                    orig_sr=orig_sr,
                    target_sr=target_sr,
                )
            )
        return np.column_stack(channels)
    return librosa.resample(
        audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
    )


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "filename",
    "is_cry",
    "confidence",
    "direction_deg",
    "distance_ft",
    "sources",
    "loudness",
    "duration_s",
    "num_channels",
    "sample_rate",
    "label",
    "timestamp",
]


def append_metadata(csv_path, row: dict):
    """Append a row to the metadata CSV, creating the file+header if needed."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args):
    ts = time.strftime("%Y%m%d_%H%M%S")
    label_tag = f"_{args.label}" if args.label else ""
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load or record audio ---
    if args.input:
        filepath = os.path.abspath(args.input)
        logger.info(f"Loading audio from {filepath}")
        audio, sample_rate = sf.read(filepath)
    else:
        audio, sample_rate = record_audio(
            args.device_index, args.duration, args.channels, args.sample_rate
        )

    num_channels = audio.shape[1] if audio.ndim > 1 else 1

    # Trim to requested channel count
    if audio.ndim == 2 and num_channels > args.channels:
        logger.info(f"Trimming from {num_channels} to {args.channels} channels")
        audio = audio[:, : args.channels]
        num_channels = args.channels

    # Save raw audio
    raw_name = f"{ts}{label_tag}_raw.wav"
    raw_path = os.path.join(args.output_dir, raw_name)
    sf.write(raw_path, audio, sample_rate)
    logger.info(f"Saved raw audio: {raw_path}")

    # --- Resample to 16 kHz for BCD ---
    bcd_audio = resample_audio(audio, sample_rate, BCD_SAMPLE_RATE)

    # --- BCD pipeline ---
    logger.info("Initializing BCD...")
    bcd = BabyCryDetection(
        num_channels=num_channels,
        device=args.device,
        verbose=not args.quiet,
    )
    detection = bcd.detect_from_audio(bcd_audio)

    print()
    print("=" * 50)
    print(f"  Is Cry:     {detection.is_cry}")
    print(f"  Confidence: {detection.confidence:.2%}")
    print("=" * 50)

    # --- Prepare metadata row ---
    row = {
        "filename": raw_name,
        "is_cry": detection.is_cry,
        "confidence": f"{detection.confidence:.4f}",
        "direction_deg": "",
        "distance_ft": "",
        "sources": "",
        "loudness": "",
        "duration_s": f"{audio.shape[0] / sample_rate:.2f}",
        "num_channels": num_channels,
        "sample_rate": BCD_SAMPLE_RATE,
        "label": args.label or "",
        "timestamp": ts,
    }

    # --- If cry detected: save filtered audio and run SL ---
    if detection.is_cry and detection.filtered_audio is not None:
        filtered = detection.filtered_audio
        filt_name = f"{ts}{label_tag}_filtered.wav"
        filt_path = os.path.join(args.output_dir, filt_name)
        sf.write(filt_path, filtered, BCD_SAMPLE_RATE)
        logger.info(f"Saved filtered audio: {filt_path}")
        row["filename"] = filt_name

        # --- Sound localization ---
        logger.info("Initializing SL...")
        sl = SoundLocalization(models_dir=args.models_dir, task_id=args.task_id)
        loc_channels = filtered.shape[1] if filtered.ndim > 1 else 1

        try:
            loc = sl.localize(
                audio_data=filtered,
                sample_rate=BCD_SAMPLE_RATE,
                num_channels=loc_channels,
            )
            row["direction_deg"] = f"{loc['direction_deg']:.1f}"
            row["distance_ft"] = f"{loc['distance_ft']:.1f}"
            row["sources"] = loc["sources"]
            # Extract loudness from first source (e.g. "Loudness: 0.85")
            import re
            loud_match = re.search(r"Loudness:\s*([\d.]+)", loc["sources"])
            row["loudness"] = loud_match.group(1) if loud_match else ""

            print()
            print("=" * 50)
            print(f"  Direction:  {loc['direction_deg']:.1f} degrees")
            print(f"  Distance:   {loc['distance_ft']:.1f} ft")
            print(f"  Sources:    {loc['sources']}")
            print("=" * 50)
        except Exception as e:
            logger.error(f"Localization failed: {e}", exc_info=True)
    elif not detection.is_cry:
        print("  No cry detected — skipping localization.")

    # --- Append to metadata CSV ---
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    append_metadata(csv_path, row)
    logger.info(f"Metadata appended to {csv_path}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate sound-localization training data via the BCD pipeline"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i", type=str, help="Path to an existing WAV file"
    )
    input_group.add_argument(
        "--device-index", type=int, help="PyAudio device index for live recording"
    )

    parser.add_argument(
        "--duration", type=float, default=10,
        help="Recording duration in seconds (only with --device-index, default: 10)",
    )
    parser.add_argument(
        "--channels", type=int, default=4,
        help="Number of mic channels (default: 4)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=48000,
        help="Recording sample rate (default: 48000, resampled to 16kHz for BCD)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default=os.path.join(_SCRIPT_DIR, "sl_training_data"),
        help="Output directory (default: Pi_Integration/sl_training_data)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (default: cpu)"
    )
    parser.add_argument(
        "--models-dir", type=str, default=".",
        help="DOAnet models dir (default: .)",
    )
    parser.add_argument(
        "--task-id", type=str, default="6", help="DOAnet task ID (default: 6)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Mute BCD steps and root logger",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Optional label tag for filenames (e.g., 3ft_90deg)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        for name in ("__main__", "FunctionCalls_BCD", "FunctionCalls_SL"):
            logging.getLogger(name).setLevel(logging.INFO)

    run(args)


if __name__ == "__main__":
    main()
