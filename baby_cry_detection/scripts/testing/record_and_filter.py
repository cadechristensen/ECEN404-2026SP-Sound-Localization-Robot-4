"""
Record and Filter Audio Script

Combines audio recording and baby cry filtering into a single workflow.
Records audio from microphone and immediately processes it through the BabyCryAudioFilter.

Usage:
    python scripts/testing/record_and_filter.py --list-devices
    python scripts/testing/record_and_filter.py --model results/model.pth
    python scripts/testing/record_and_filter.py --model results/model.pth --duration 10 --threshold 0.7 --plot
    
    python scripts/testing/record_and_filter.py --device 1 --duration 10 --count 2 --plot --acoustic --both
    python scripts/testing/record_and_filter.py --device 15 --duration 10 --count 2 --plot --acoustic --both --quiet
"""

# Add project root to path
import sys
import os
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import pyaudio
import wave
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from src.config import Config
from src.audio_filter import BabyCryAudioFilter
from scripts.testing.test_my_audio import visualize_filtering_pipeline


def calculate_audio_db_levels(audio_data: np.ndarray, sample_rate: int = 48000,
                               reference_spl_db: float = 94.0) -> dict:
    """
    Calculate dB levels for audio data.

    This function calculates:
    - dBFS (dB relative to Full Scale)
    - Estimated dB SPL (calibrated to reference)
    - Peak dB levels

    Args:
        audio_data: Audio samples (int16 format or normalized float)
        sample_rate: Sample rate in Hz
        reference_spl_db: Reference SPL calibration point (default: 94 dB SPL = 0 dBFS)
                         Common microphone calibration uses 94 dB SPL @ 1 kHz = 1 Pascal

    Returns:
        Dictionary with dB measurements
    """
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float = audio_data.astype(np.float32)

    if len(audio_float) == 0:
        return {
            'rms_dbfs': -np.inf,
            'peak_dbfs': -np.inf,
            'estimated_spl_db': -np.inf,
            'peak_spl_db': -np.inf
        }

    rms = np.sqrt(np.mean(audio_float ** 2))
    peak = np.max(np.abs(audio_float))

    if rms > 0:
        rms_dbfs = 20 * np.log10(rms)
    else:
        rms_dbfs = -np.inf

    if peak > 0:
        peak_dbfs = 20 * np.log10(peak)
    else:
        peak_dbfs = -np.inf

    estimated_spl_db = reference_spl_db + rms_dbfs
    peak_spl_db = reference_spl_db + peak_dbfs

    return {
        'rms_dbfs': rms_dbfs,
        'peak_dbfs': peak_dbfs,
        'estimated_spl_db': estimated_spl_db,
        'peak_spl_db': peak_spl_db,
        'rms_linear': rms,
        'peak_linear': peak
    }


def analyze_audio_file_db(filename: str) -> Tuple[dict, dict]:
    """
    Analyze dB levels of an audio file (all channels and per-channel).

    Args:
        filename: Path to WAV file

    Returns:
        Tuple of (overall_stats, per_channel_stats)
    """
    try:
        with wave.open(filename, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            num_frames = wf.getnframes()
            audio_bytes = wf.readframes(num_frames)

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        if num_channels > 1:
            audio_data = audio_data.reshape(-1, num_channels)
            overall_audio = audio_data.flatten()
        else:
            overall_audio = audio_data

        overall_stats = calculate_audio_db_levels(overall_audio, sample_rate)
        overall_stats['num_channels'] = num_channels
        overall_stats['duration_seconds'] = num_frames / sample_rate
        overall_stats['sample_rate'] = sample_rate

        per_channel_stats = {}
        if num_channels > 1:
            for ch in range(num_channels):
                channel_data = audio_data[:, ch]
                per_channel_stats[f'channel_{ch+1}'] = calculate_audio_db_levels(
                    channel_data, sample_rate
                )

        return overall_stats, per_channel_stats

    except Exception as e:
        print(f"Error analyzing audio file: {e}")
        return {}, {}


class AudioRecorder:
    """Audio recorder with device selection and error handling."""

    def __init__(self, sample_rate: int = 48000, channels: int = 4,
                 chunk_size: int = 1024, audio_format: int = pyaudio.paInt16,
                 record_channels: int = 8):
        """
        Initialize AudioRecorder with specified parameters.

        Args:
            sample_rate: Sampling frequency in Hz (default: 48000)
            channels: Number of audio channels to save (default: 4)
            chunk_size: Number of frames per buffer
            audio_format: PyAudio format (default: 16-bit PCM)
            record_channels: Number of channels to record from device (default: 8)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.record_channels = record_channels
        self.chunk_size = chunk_size
        self.audio_format = audio_format
        self.audio = None

    def list_devices(self) -> None:
        """List all available audio input devices."""
        audio = pyaudio.PyAudio()
        print("\n" + "=" * 60)
        print("Available Audio Input Devices:")
        print("=" * 60)

        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"Device {i}: {info['name']}")
                print(f"  - Channels: {info['maxInputChannels']}")
                print(f"  - Sample Rate: {int(info['defaultSampleRate'])} Hz")
                print()

        audio.terminate()

    def record_audio(self, filename: str, duration: int = 10,
                    device_index: Optional[int] = None, quiet: bool = False) -> bool:
        """
        Record audio from microphone and save to WAV file.

        Args:
            filename: Output WAV filename
            duration: Recording duration in seconds
            device_index: Specific device index (None for default)
            quiet: Enable quiet mode for minimal output

        Returns:
            True if recording successful, False otherwise
        """
        try:
            self.audio = pyaudio.PyAudio()

            # Open audio stream
            stream_kwargs = {
                'format': self.audio_format,
                'channels': self.record_channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size
            }

            if device_index is not None:
                stream_kwargs['input_device_index'] = device_index
                if not quiet:
                    device_info = self.audio.get_device_info_by_index(device_index)
                    print(f"\nUsing device: {device_info['name']}")

            stream = self.audio.open(**stream_kwargs)

            if quiet:
                print(f"Recording {duration}s...")
            else:
                print(f"\nRecording {self.record_channels} channels for {duration} seconds...")
                print(f"(Saving first {self.channels} channels with audio)")
                print("-" * 50)

            frames = []
            total_chunks = int(self.sample_rate / self.chunk_size * duration)

            # Record audio with progress indicator
            for i in range(total_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)

                # Progress indicator every second
                if (i + 1) % int(self.sample_rate / self.chunk_size) == 0:
                    elapsed = (i + 1) // int(self.sample_rate / self.chunk_size)
                    if quiet:
                        print(f"{elapsed}s", end=' ', flush=True)
                    else:
                        print(f"  {elapsed}/{duration} seconds", end='\r')

            if quiet:
                print()  # New line after progress
            else:
                print(f"\nRecording finished.")

            # Clean up stream
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

            # Process audio: extract first 4 channels from 8-channel recording
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.reshape(-1, self.record_channels)
            audio_data = audio_data[:, :self.channels]
            audio_data = audio_data.flatten()

            # Save to WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())

            # Display file info
            if not quiet:
                file_size = os.path.getsize(filename)
                print(f"Audio saved to: {filename}")
                print(f"  - Size: {file_size / 1024:.2f} KB")
                print(f"  - Duration: {duration}s")
                print(f"  - Sample Rate: {self.sample_rate} Hz")
                print(f"  - Channels Saved: {self.channels} (recorded {self.record_channels}, kept first {self.channels})")

            if not quiet:
                print("\nAnalyzing audio levels...")

            overall_stats, per_channel_stats = analyze_audio_file_db(filename)

            if overall_stats:
                if quiet:
                    # Quiet mode: only show estimated SPL
                    print(f"Estimated SPL: {overall_stats['estimated_spl_db']:.1f} dB")
                else:
                    # Verbose mode: show all details
                    print(f"\nAUDIO LEVEL ANALYSIS:")
                    print(f"  Overall (all channels):")
                    print(f"    - RMS Level: {overall_stats['rms_dbfs']:.1f} dBFS")
                    print(f"    - Peak Level: {overall_stats['peak_dbfs']:.1f} dBFS")
                    print(f"    - Estimated SPL (calibrated): {overall_stats['estimated_spl_db']:.1f} dB SPL")
                    print(f"    - Peak SPL: {overall_stats['peak_spl_db']:.1f} dB SPL")

                    if per_channel_stats:
                        print(f"\n  Per-channel RMS levels:")
                        for ch_name, ch_stats in per_channel_stats.items():
                            ch_num = ch_name.split('_')[1]
                            print(f"    - Channel {ch_num}: {ch_stats['rms_dbfs']:.1f} dBFS ({ch_stats['estimated_spl_db']:.1f} dB SPL)")

                    print(f"\n  Note: SPL estimates assume 94 dB SPL calibration reference.")
                    print(f"        Actual SPL depends on microphone sensitivity and gain settings.")

            return True

        except Exception as e:
            print(f"\nError during recording: {e}")
            if self.audio:
                self.audio.terminate()
            return False

    #! Name of File
    def get_next_filename(self, base_name: str = "rec_demo", extension: str = ".wav",
                         output_dir: Optional[Path] = None) -> str:
        """
        Get next available filename with incremented number.

        Args:
            base_name: Base filename prefix
            extension: File extension
            output_dir: Directory to save files (default: current directory)

        Returns:
            Next available filename (e.g., "demo/rec_demo1.wav")
        """
        if output_dir is None:
            output_dir = Path(".")

        counter = 1
        while True:
            filename = output_dir / f"{base_name}{counter}{extension}"
            if not filename.exists():
                return str(filename)
            counter += 1


def filter_audio_file(input_path: str, audio_filter: BabyCryAudioFilter,
                     cry_threshold: float = 0.5,
                     use_acoustic_features: bool = True,
                     generate_plots: bool = False,
                     output_mode: str = "full_length",
                     output_dir: Optional[Path] = None,
                     quiet: bool = False) -> Optional[dict]:
    """
    Filter recorded audio file for baby cry detection.

    Args:
        input_path: Path to recorded audio file
        audio_filter: Initialized BabyCryAudioFilter instance
        cry_threshold: Detection threshold (0.0-1.0)
        use_acoustic_features: Whether to use acoustic validation filter
        generate_plots: Generate visualization plots of filtering stages
        output_mode: Output format - "full_length", "cry_only", or "both"
        output_dir: Directory to save output files (default: same as input)
        quiet: Enable quiet mode for minimal output

    Returns:
        Dictionary with processing results, or None if error occurred
    """
    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        return None

    # Generate output filename
    input_file = Path(input_path)
    if output_dir is None:
        output_dir = input_file.parent
    output_path = output_dir / f"{input_file.stem}_filtered{input_file.suffix}"

    if quiet:
        print("Processing...")
    else:
        print("\n" + "=" * 80)
        print("FILTERING AUDIO FILE")
        print("=" * 80)
        print(f"Input: {input_path}")
        print(f"Model: {'Using ML model' if audio_filter.model is not None else 'Acoustic features only'}")
        print(f"Binary Classification: ML-ONLY (with acoustic validation filter)")
        print(f"Acoustic validation: {'ENABLED' if use_acoustic_features else 'DISABLED'}")
        print(f"Threshold: {cry_threshold}")
        print(f"Output mode: {output_mode}")
        print(f"Output will be saved to: {output_path}")
        print()

    try:
        results = audio_filter.process_audio_file(
            input_path=str(input_path),
            output_path=str(output_path),
            cry_threshold=cry_threshold,
            use_acoustic_features=use_acoustic_features,
            output_mode=output_mode
        )

        # Display results
        if quiet:
            # Quiet mode: minimal output with segments and confidence
            print(f"Detected {results['num_cry_segments']} segments")
            if results.get('cry_segments_with_prob'):
                for i, (start, end, prob) in enumerate(results['cry_segments_with_prob'], 1):
                    duration = end - start
                    print(f"  {i}. {start:.2f}s - {end:.2f}s ({duration:.2f}s) - {prob*100:.1f}%")
            print(f"Average confidence: {results['avg_confidence']:.1f}%")
            for mode, filepath in results['output_files'].items():
                print(f"Saved: {filepath}")
        else:
            # Verbose mode: detailed output
            print("\n" + "=" * 80)
            print("FILTERING RESULTS")
            print("=" * 80)
            print(f"Total duration: {results['total_duration']:.2f} seconds")
            print(f"Cry duration: {results['cry_duration']:.2f} seconds ({results['duration_percentage']:.1f}% of file)")
            if results['output_mode'] in ['cry_only', 'both']:
                print(f"Cry-only duration: {results['cry_only_duration']:.2f} seconds")
            print(f"Number of cry segments: {results['num_cry_segments']}")
            print()
            print("MODEL CONFIDENCE METRICS (Softmax Probabilities):")
            print(f"  Average confidence: {results['avg_confidence']:.1f}%")
            print(f"  Confidence range: {results['min_confidence']:.1f}% - {results['max_confidence']:.1f}%")

            if results['cry_segments']:
                print("\nCry segments detected:")
                for i, (start, end) in enumerate(results['cry_segments'][:10], 1):
                    duration = end - start
                    print(f"  {i}. {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")

                if len(results['cry_segments']) > 10:
                    print(f"  ... and {len(results['cry_segments']) - 10} more segments")

            print("\nOutput files:")
            for mode, filepath in results['output_files'].items():
                print(f"  {mode}: {filepath}")

        # Generate visualization plots if requested
        if generate_plots:
            try:
                if not quiet:
                    print("\nGenerating visualization plots...")
                # Create visualization subdirectory
                viz_dir = output_dir / "filtering_visualizations"
                viz_dir.mkdir(parents=True, exist_ok=True)

                plot_path = visualize_filtering_pipeline(
                    input_path,
                    audio_filter,
                    cry_threshold,
                    use_acoustic_features,
                    results,
                    output_dir=viz_dir
                )
                if not quiet:
                    print(f"Visualization plots saved to: {plot_path}")
                    print("=" * 80)
            except Exception as e:
                if not quiet:
                    print(f"Warning: Could not generate plots: {e}")
                    import traceback
                    traceback.print_exc()

        return results

    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point for record and filter script."""
    parser = argparse.ArgumentParser(
        description='Record audio and immediately filter for baby cry detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default model:
  python scripts/testing/record_and_filter.py
  python scripts/testing/record_and_filter.py --duration 10 --threshold 0.7 --plot
  python scripts/testing/record_and_filter.py --count 3 --acoustic

  # Use a different model:
  python scripts/testing/record_and_filter.py --model other_model.pth

  # Other options:
  python scripts/testing/record_and_filter.py --device 2 --cry-only
  python scripts/testing/record_and_filter.py --list-devices
        """
    )

    # Create demo output directory
    output_dir = Path("demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recording options
    parser.add_argument('--list-devices', '-l', action='store_true',
                       help='List all available audio input devices and exit')
    parser.add_argument('--device', '-i', type=int, default=None,
                       help='Audio input device index (default: system default)')
    parser.add_argument('--duration', '-d', type=int, default=10,
                       help='Recording duration in seconds (default: 10)')
    parser.add_argument('--count', '-c', type=int, default=1,
                       help='Number of recordings to make (default: 1)')
    parser.add_argument('--no-prompt', action='store_true',
                       help='Disable prompt before each recording (default: prompt enabled)')

    # Recording format options
    parser.add_argument('--sample-rate', '-r', type=int, default=48000,
                       help='Sample rate in Hz (default: 48000)')
    parser.add_argument('--channels', type=int, default=4,
                       help='Number of audio channels to save (default: 4)')
    parser.add_argument('--record-channels', type=int, default=8,
                       help='Number of channels to record from device (default: 8)')

    #! Model configuration options
    parser.add_argument('--model', '-m', type=str,
                       default="results/train_2025-11-24_19-10-26/evaluations/eval_2025-11-24_19-38-25/calibrated_model.pth",
                       help='Path to trained model')

    # Filtering options
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Detection threshold 0.0-1.0 (default: 0.5, lower=more sensitive)')
    parser.add_argument('--acoustic', action='store_true',
                       help='Enable acoustic feature validation (binary accept/reject filter)')
    parser.add_argument('--plot', '-p', action='store_true',
                       help='Generate visualization plots of filtering stages')

    # Output options
    parser.add_argument('--cry-only', action='store_true',
                       help='Output only concatenated cry segments (removes silence)')
    parser.add_argument('--both', action='store_true',
                       help='Output both full-length and cry-only versions')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode for demos - minimal output (progress, dB SPL, confidence, files)')

    args = parser.parse_args()

    # Initialize recorder
    recorder = AudioRecorder(
        sample_rate=args.sample_rate,
        channels=args.channels,
        record_channels=args.record_channels
    )

    # List devices if requested
    if args.list_devices:
        recorder.list_devices()
        return

    # Validate model path
    model_path = args.model
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please specify a valid model path using --model argument")
        return

    if not args.quiet:
        print(f"\nUsing model: {model_path}")

    # Initialize audio filter
    if not args.quiet:
        print("\n" + "=" * 80)
        print("INITIALIZING AUDIO FILTER")
        print("=" * 80)

    try:
        config = Config()
        audio_filter = BabyCryAudioFilter(config=config, model_path=model_path, verbose=not args.quiet)
        if not args.quiet:
            print("Audio filter initialized successfully")
    except Exception as e:
        print(f"Error initializing audio filter: {e}")
        import traceback
        traceback.print_exc()
        return

    # Determine output mode
    output_mode = "full_length"
    if args.cry_only:
        output_mode = "cry_only"
    elif args.both:
        output_mode = "both"

    if not args.quiet:
        print("\n" + "=" * 80)
        print("RECORD AND FILTER SESSION")
        print("=" * 80)
        print(f"Recordings to make: {args.count}")
        print(f"Duration per recording: {args.duration}s")
        print(f"Detection threshold: {args.threshold}")
        print(f"Acoustic validation: {'ENABLED' if args.acoustic else 'DISABLED'}")
        print(f"Output mode: {output_mode}")
        print(f"Visualization plots: {'ENABLED' if args.plot else 'DISABLED'}")
        print("=" * 80)

    # Record and filter loop
    success_count = 0
    filtered_count = 0

    for i in range(1, args.count + 1):
        if not args.quiet:
            print(f"\n{'=' * 80}")
            print(f"RECORDING {i}/{args.count}")
            print("=" * 80)

        # Prompt user unless disabled
        if not args.no_prompt:
            response = input("Press Enter to start recording (or 'q' to quit): ").strip().lower()
            if response == 'q':
                print("Session cancelled by user.")
                break

        # Get next available filename in demo directory
        filename = recorder.get_next_filename(output_dir=output_dir)

        # Record audio
        if recorder.record_audio(filename, duration=args.duration, device_index=args.device, quiet=args.quiet):
            success_count += 1

            # Filter the recorded audio
            results = filter_audio_file(
                input_path=filename,
                audio_filter=audio_filter,
                cry_threshold=args.threshold,
                use_acoustic_features=args.acoustic,
                generate_plots=args.plot,
                output_mode=output_mode,
                output_dir=output_dir,
                quiet=args.quiet
            )

            if results:
                filtered_count += 1
            else:
                if not args.quiet:
                    print("WARNING: Filtering failed for this recording")
        else:
            if not args.quiet:
                print("WARNING: Recording failed, skipping filtering")

    # Final summary
    if not args.quiet:
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)
        print(f"Recordings successful: {success_count}/{args.count}")
        print(f"Files filtered: {filtered_count}/{success_count}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
