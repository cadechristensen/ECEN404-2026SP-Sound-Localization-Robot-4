"""
Multichannel recorder for TI PCM6260-Q1 mic array.

Records 4-channel audio, applies phase-preserving noise filtering via
AudioFilteringPipeline, and writes the result as 32-bit float WAV files.

Output is intended as training data for the sound localization model.
This script does not interact with the localization or robot-navigation
pipelines in any way.
"""

import os
import sys
import argparse
import time
import numpy as np
import pyaudio
import soundfile as sf
from pathlib import Path
from datetime import datetime

# Insert the project root (two directories up) so that src.* imports resolve
# regardless of whether the script is invoked directly or via the systemd service.
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from config_pi import ConfigPi
from audio_filtering import AudioFilteringPipeline


def list_devices():
    """Print all available PyAudio input devices and return."""
    p = pyaudio.PyAudio()
    print("\nAvailable input devices:")
    print("-" * 60)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
            print(f"        Input channels : {info['maxInputChannels']}")
            print(f"        Default rate   : {info['defaultSampleRate']}")
    p.terminate()


def record_audio(config, device_index, duration):
    """
    Record multichannel audio using a blocking read loop.

    Args:
        config  : ConfigPi instance (provides PI_SAMPLE_RATE, PI_CHANNELS,
                  PI_BUFFER_SIZE).
        device_index : PyAudio device index to record from.
        duration     : Recording length in seconds.

    Returns:
        numpy.ndarray of shape (N, 4) with dtype float32.
        N == sample_rate * duration (exact, because frames are requested
        in fixed-size chunks until the target count is reached).
    """
    sample_rate = config.PI_SAMPLE_RATE
    channels = config.PI_CHANNELS
    buffer_size = config.PI_BUFFER_SIZE
    total_frames = sample_rate * duration

    p = pyaudio.PyAudio()

    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sample_rate,
            input=True,
            output=False,
            input_device_index=device_index,
            frames_per_buffer=buffer_size,
            start=False,
        )
    except OSError as exc:
        print(f"\nERROR: Failed to open audio stream on device {device_index}.")
        print(f"  {exc}")
        print("  Possible cause: ALSA driver on this device does not support")
        print("  float32 input.  Try a different device or rebuild PyAudio")
        print("  against a backend that supports paFloat32.")
        p.terminate()
        sys.exit(1)

    stream.start_stream()
    print("Recording ...", flush=True)

    frames_collected = 0
    chunks = []

    while frames_collected < total_frames:
        remaining = total_frames - frames_collected
        read_count = min(buffer_size, remaining)
        data = stream.read(read_count, exception_on_overflow=False)
        # paFloat32 + N channels -> 4 bytes * channels per frame
        chunk = np.frombuffer(data, dtype=np.float32).reshape(-1, channels)
        chunks.append(chunk)
        frames_collected += chunk.shape[0]

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.concatenate(chunks, axis=0)
    # Trim to exactly total_frames in case the last chunk overshot
    return audio[:total_frames].astype(np.float32)


def save_wav(audio, output_path, sample_rate):
    """
    Write a multichannel float32 WAV file.

    Args:
        audio       : numpy.ndarray, shape (N, channels), dtype float32.
        output_path : destination file path (str or Path).
        sample_rate : integer sample rate for the WAV header.
    """
    sf.write(str(output_path), audio, sample_rate, subtype='FLOAT')


def main():
    parser = argparse.ArgumentParser(
        description="Record 4-channel audio from PCM6260-Q1 and save "
                    "phase-preserving filtered WAV files."
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print available input devices and exit."
    )
    parser.add_argument(
        "--device-index", type=int, default=None,
        help="PyAudio device index to record from (required for recording)."
    )
    parser.add_argument(
        "--duration", type=int, default=10,
        help="Recording duration in seconds (default: 10)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./recordings",
        help="Directory for output WAV files (default: ./recordings). "
             "Created if it does not exist."
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Also save the unfiltered 4-channel WAV alongside the filtered one."
    )

    args = parser.parse_args()

    # --list-devices: print and exit immediately
    if args.list_devices:
        list_devices()
        return

    # --device-index is mandatory for recording; fail explicitly rather than
    # silently falling back to whatever PyAudio considers default.
    if args.device_index is None:
        print("ERROR: --device-index is required for recording.")
        print("       Run with --list-devices to see available devices.")
        sys.exit(1)

    # Initialise config and filtering pipeline
    config = ConfigPi()
    pipeline = AudioFilteringPipeline(config)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = config.PI_SAMPLE_RATE
    channels = config.PI_CHANNELS

    # Timestamp used for both filenames in this recording session
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filtered_path = output_dir / f"multichannel_filtered_{timestamp}.wav"
    raw_path = output_dir / f"multichannel_raw_{timestamp}.wav"

    # --- Config summary banner ---
    print("=" * 60, flush=True)
    print("  MULTICHANNEL RECORDER", flush=True)
    print("=" * 60, flush=True)
    print(f"  Device index     : {args.device_index}", flush=True)
    print(f"  Duration         : {args.duration} s", flush=True)
    print(f"  Sample rate      : {sample_rate} Hz", flush=True)
    print(f"  Channels         : {channels}", flush=True)
    print(f"  Output directory : {output_dir.resolve()}", flush=True)
    print(f"  Save raw         : {args.save_raw}", flush=True)
    print("-" * 60, flush=True)
    print(f"  Filter params:", flush=True)
    print(f"    High-pass cutoff       : {config.HIGHPASS_CUTOFF} Hz", flush=True)
    print(f"    Band-pass             : {config.BANDPASS_LOW}-{config.BANDPASS_HIGH} Hz", flush=True)
    print(f"    Spectral sub strength : {config.NOISE_REDUCE_STRENGTH}", flush=True)
    print("=" * 60, flush=True)

    # --- 3-second countdown ---
    for t in (3, 2, 1):
        print(f"  Starting in {t} ...", flush=True)
        time.sleep(1)

    # --- Record ---
    raw = record_audio(config, args.device_index, args.duration)

    # --- Filter (VAD disabled: full clip is filtered unconditionally) ---
    print("Filtering ...", flush=True)
    result = pipeline.preprocess_audio(raw, apply_vad=False, apply_filtering=True)
    filtered = result['filtered']

    # Ensure filtered is a plain float32 numpy array (pipeline may return a tensor)
    if hasattr(filtered, 'numpy'):
        filtered = filtered.numpy()
    filtered = np.asarray(filtered, dtype=np.float32)

    # --- Save ---
    if args.save_raw:
        save_wav(raw, raw_path, sample_rate)
        print(f"  Raw WAV saved      : {raw_path}", flush=True)

    save_wav(filtered, filtered_path, sample_rate)
    print(f"  Filtered WAV saved : {filtered_path}", flush=True)

    # --- Final summary ---
    raw_samples = raw.shape[0]
    filtered_samples = filtered.shape[0]
    print("-" * 60, flush=True)
    print("  SUMMARY", flush=True)
    print(f"    Raw samples      : {raw_samples}  "
          f"({raw_samples / sample_rate:.3f} s)", flush=True)
    print(f"    Filtered samples : {filtered_samples}  "
          f"({filtered_samples / sample_rate:.3f} s)", flush=True)
    if raw_samples != filtered_samples:
        print(f"    Note: filtered output is {raw_samples - filtered_samples} "
              f"sample(s) shorter due to spectral-subtraction istft rounding.",
              flush=True)
    if args.save_raw:
        print(f"    Raw file size    : {raw_path.stat().st_size} bytes", flush=True)
    print(f"    Filtered file size: {filtered_path.stat().st_size} bytes", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
