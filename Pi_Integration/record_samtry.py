import argparse
import csv
import ctypes
import ctypes.util
import datetime
import logging
import math
import os
import re
import sys

# from CodeTest.Integration_ML.Sound_Characterization import multichannel_detector

# Suppress noisy ALSA/JACK warnings from PyAudio device enumeration
os.environ["JACK_NO_START_SERVER"] = "1"
try:
    _ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
    )

    def _py_error_handler(filename, line, function, err, fmt):
        pass

    _c_error_handler = _ERROR_HANDLER_FUNC(_py_error_handler)
    _asound = ctypes.cdll.LoadLibrary(
        ctypes.util.find_library("asound") or "libasound.so.2"
    )
    _asound.snd_lib_error_set_handler(_c_error_handler)
except Exception:
    pass

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
# Stderr suppression for PyAudio init (silences JACK warnings)
# ---------------------------------------------------------------------------
def _quiet_pyaudio_init():
    """Create a PyAudio instance with stderr suppressed to hide JACK warnings."""
    import pyaudio

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        pa = pyaudio.PyAudio()
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)
    return pa


# ---------------------------------------------------------------------------
# Device lookup
# ---------------------------------------------------------------------------
def find_device_by_name(name):
    """Find a PyAudio device index by substring match on its name."""
    pa = _quiet_pyaudio_init()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if name.lower() in info["name"].lower():
                logger.info(f"Matched device '{info['name']}' at index {i}")
                return i
    finally:
        pa.terminate()
    raise ValueError(f"No audio device matching '{name}' found")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def record_audio(device_index, duration, channels, sample_rate):
    """Record audio from a PyAudio device and return (np.ndarray, sample_rate)."""
    import pyaudio

    pa = _quiet_pyaudio_init()
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
        for _ in range(math.ceil(sample_rate / 1024 * duration)):
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
# ---------------------------------------------------------------------------
# ML Distance Prediction Extractor
def predict_ml_distance(audio_data, sample_rate):
    import joblib
    import pandas as pd
    import librosa
    import numpy as np
    import os

    TRAINING_SR = 48000

    # Ensure this matches where your model actually lives!
    models_folder = os.path.join(_PROJECT_ROOT, "SoundLocalization", "models")
    model_path = os.path.join(
        models_folder, "distance_model_rawApril6_angle_try1.joblib"
    )
    features_path = os.path.join(
        models_folder, "feature_names_rawApril6_angle_try1.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model file! Checked exactly here: {model_path}"
        )
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Missing features file! Checked exactly here: {features_path}"
        )

    dist_model = joblib.load(model_path)
    dist_features = joblib.load(features_path)

    # 1. Average all channels together first
    y_mono = np.mean(audio_data, axis=1) if audio_data.ndim > 1 else audio_data
    y_mono = y_mono.astype(np.float32)

    # 2. Resample to 48kHz if needed
    if sample_rate != TRAINING_SR:
        y_mono = librosa.resample(y_mono, orig_sr=sample_rate, target_sr=TRAINING_SR)
        sample_rate = TRAINING_SR

    # 3. Audio Trimming (SYNCED TO top_db=60 TO MATCH TRAINING)
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

    y_trimmed, _ = librosa.effects.trim(y_mono, top_db=60)

    if len(y_trimmed) < FRAME_LENGTH:
        y_trimmed = y_mono

    features = {}

    # --- 4. EXACT FEATURE EXTRACTION (Only calculating what the model kept) ---

    # RMS (Volume)
    rms = librosa.feature.rms(
        y=y_trimmed, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    features["rms_std"] = np.std(rms_db)

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y=y_trimmed, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    # MFCCs
    mfccs = librosa.feature.mfcc(
        y=y_trimmed,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Python arrays are 0-indexed, but feature names are 1-indexed (mfcc_mean_1 = index 0)
    features["mfcc_mean_1"] = mfccs_mean[0]
    features["mfcc_std_1"] = mfccs_std[0]
    features["mfcc_std_3"] = mfccs_std[2]
    features["mfcc_mean_4"] = mfccs_mean[3]
    features["mfcc_std_13"] = mfccs_std[12]

    # 5. Build DataFrame and enforce exact column order
    features_df = pd.DataFrame([features])
    features_df = features_df[dist_features]

    # 6. Predict
    prediction = dist_model.predict(features_df)[0]

    # IMPORTANT: If your training script console output said "Winning Setting -> USE_LOG_DISTANCE: False",
    # you MUST remove the `np.exp()` wrapper below and just use: real_distance_ft = prediction
    real_distance_ft = prediction

    return real_distance_ft


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    # --- BCD pipeline (initial screening only, TTA disabled) ---
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
        "sample_rate": sample_rate,
        "label": args.label or "",
        "timestamp": ts,
    }

    # --- If cry detected: extract cry regions from 48kHz original ---
    if detection.is_cry:
        if not detection.cry_regions:
            print("  No cry regions — skipping localization.")
        else:
            # Extract cry regions from 48kHz original (single extraction, reused for save + SL)
            total_samples = audio.shape[0]
            chunks_48k = []
            for start, end in detection.cry_regions:
                s_idx = int(start * sample_rate)
                e_idx = min(int(end * sample_rate), total_samples)
                chunks_48k.append(audio[s_idx:e_idx])

            if not chunks_48k:
                print("  Empty cry region chunks — skipping localization.")
            else:
                cry_audio_48k = np.concatenate(chunks_48k, axis=0)

                # Save unfiltered cry regions
                filt_name = f"{ts}{label_tag}_filtered.wav"
                filt_path = os.path.join(args.output_dir, filt_name)
                sf.write(filt_path, cry_audio_48k, sample_rate)
                logger.info(
                    f"Saved filtered audio (48kHz, phase-preserved): {filt_path}"
                )
                row["filename"] = filt_name
                row["sample_rate"] = sample_rate

                # Phase-preserving filtering for SL (bandpass + spectral subtraction)
                loc_audio = bcd.filter_for_localization(cry_audio_48k)

                logger.info("Initializing SL...")
                sl = SoundLocalization(
                    models_dir=args.models_dir,
                    task_id=args.task_id,
                    single_model=args.single_model,
                )

                try:
                    loc = sl.localize(
                        audio_data=loc_audio,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                    )

                    # --- ML DISTANCE PREDICTION ---
                    # Pass `audio` (full raw recording) so distance features
                    # are computed on the same signal as training.
                    try:
                        predicted_distance = predict_ml_distance(audio, sample_rate)
                    except Exception as ml_err:
                        logger.error(f"ML Distance prediction failed: {ml_err}")
                        predicted_distance = loc.get("distance_ft", 0)

                    # Source selection now handled in function_calls.py
                    # (picks source closest to loudest mic by RMS)
                    row["direction_deg"] = f"{loc.get('direction_deg', 0):.1f}"
                    row["distance_ft"] = f"{predicted_distance:.1f}"
                    row["sources"] = loc.get("sources", "")

                    # Extract loudness for Source 0 (the best source)
                    raw_sources = str(loc.get("sources", ""))
                    loud_match = re.search(r"Loudness:\s*([\d.]+)", raw_sources)
                    row["loudness"] = loud_match.group(1) if loud_match else ""

                    print()
                    print("=" * 50)
                    print(f"  Direction:   {loc.get('direction_deg', 0):.1f} degrees")
                    print(f"  Distance:    {predicted_distance:.1f} ft (ML Predicted)")
                    print(f"  Sources:     {raw_sources}")
                    print("=" * 50)
                except Exception as e:
                    logger.error(f"Localization failed: {e}", exc_info=True)
    else:
        print("  No cry detected — skipping localization.")

    # Store ground truth from label if available (for training data)
    if args.label:
        label_match = re.match(r"(\d+)ft_(\d+)deg", args.label)
        if label_match:
            row["distance_ft"] = label_match.group(1)
            row["direction_deg"] = label_match.group(2)

    # --- Append to metadata CSV ---
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    append_metadata(csv_path, row)
    logger.info(f"Metadata appended to {csv_path}")
    print()


# ---------------------------------------------------------------------------
# Batch recording mode
# ---------------------------------------------------------------------------
def run_batch(args):
    """Systematic batch recording for SL training data collection.

    Guides the user through recording at each angle and distance combination.
    Automatically labels each recording with the correct angle_deg and distance_ft.
    """
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    distances = [1, 2, 3, 5, 7, 10]

    os.makedirs(args.output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print("  SL Training Data Collection — Batch Mode")
    print("=" * 60)
    print(f"  Angles:    {angles}")
    print(f"  Distances: {distances} ft")
    print(f"  Total:     {len(angles) * len(distances)} recordings")
    print(f"  Duration:  {args.duration}s each")
    print(f"  Output:    {args.output_dir}")
    print("=" * 60)
    print()
    print("Place the sound source at the specified angle and distance,")
    print("then press Enter to record. Type 's' to skip, 'q' to quit.")
    print()

    for dist in distances:
        for angle in angles:
            label = f"{dist}ft_{angle}deg"
            prompt = f"  [{label}] Place source at {angle} deg, {dist} ft. "
            try:
                response = input(
                    prompt + "Press Enter to record (s=skip, q=quit): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBatch recording stopped.")
                return

            if response.lower() == "q":
                print("Batch recording stopped.")
                return
            if response.lower() == "s":
                print(f"  Skipped {label}")
                continue

            # Override label for this recording
            args.label = label
            args.input = None  # Force live recording

            try:
                run(args)
                print(f"  Recorded {label}")
            except Exception as e:
                print(f"  ERROR recording {label}: {e}")

    print()
    print("Batch recording complete!")
    print(f"Check {args.output_dir}/metadata.csv for all labels.")


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
    input_group.add_argument(
        "--device-name",
        type=str,
        help="Find device by name substring (e.g., 'TI USB Audio')",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10,
        help="Recording duration in seconds (only with --device-index, default: 10)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=4,
        help="Number of mic channels (default: 4)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Recording sample rate (default: 48000, resampled to 16kHz for BCD)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "sl_training_data"),
        help="Output directory (default: Pi_Integration/sl_training_data)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (default: cpu)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "SoundLocalization", "models"),
        help="DOAnet models dir (default: SoundLocalization/models)",
    )
    parser.add_argument(
        "--task-id", type=str, default="6", help="DOAnet task ID (default: 6)"
    )
    parser.add_argument(
        "--single-model",
        type=str,
        default=None,
        help="Use a single SL model for all angles (e.g. New_Test_Model.h5)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Mute BCD steps and root logger",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label tag for filenames (e.g., 3ft_90deg)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Systematic batch recording at all angle/distance combinations",
    )

    args = parser.parse_args()

    if args.device_name:
        args.device_index = find_device_by_name(args.device_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        for name in ("__main__", "FunctionCalls_BCD", "FunctionCalls_SL"):
            logging.getLogger(name).setLevel(logging.INFO)

    if args.batch:
        run_batch(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
