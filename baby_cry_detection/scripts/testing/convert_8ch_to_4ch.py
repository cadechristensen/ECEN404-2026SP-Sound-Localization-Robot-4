"""
Audio Channel Converter for TI Board Recordings

Converts 8-channel WAV files to 4-channel by extracting only the first 4 channels.
Useful for processing existing TI board recordings that contain 4 empty channels.

Usage:
    python convert_8ch_to_4ch.py input.wav
    python convert_8ch_to_4ch.py input.wav --output custom_output.wav
    python convert_8ch_to_4ch.py *.wav  # Process multiple files
"""

import wave
import os
import sys
import argparse
import numpy as np
from pathlib import Path


def process_audio_file(input_file: str, output_file: str = None,
                       input_channels: int = 8, output_channels: int = 4) -> bool:
    """
    Convert multi-channel WAV file to fewer channels by extracting first N channels.

    Args:
        input_file: Path to input WAV file
        output_file: Path to output WAV file (optional)
        input_channels: Expected number of input channels (default: 8)
        output_channels: Number of channels to keep (default: 4)

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Generate output filename if not provided
        if output_file is None:
            path = Path(input_file)
            output_file = str(path.parent / f"{path.stem}_4ch{path.suffix}")

        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return False

        # Open and read input WAV file
        with wave.open(input_file, 'rb') as wf_in:
            # Get audio parameters
            n_channels = wf_in.getnchannels()
            sampwidth = wf_in.getsampwidth()
            framerate = wf_in.getframerate()
            n_frames = wf_in.getnframes()

            print(f"\nProcessing: {input_file}")
            print(f"  Input channels: {n_channels}")
            print(f"  Sample width: {sampwidth} bytes")
            print(f"  Sample rate: {framerate} Hz")
            print(f"  Total frames: {n_frames}")

            # Validate channel count
            if n_channels < output_channels:
                print(f"Error: Input file has only {n_channels} channels, cannot extract {output_channels}")
                return False

            if n_channels == output_channels:
                print(f"Warning: Input already has {output_channels} channels, no conversion needed")
                return False

            # Read all audio data
            audio_bytes = wf_in.readframes(n_frames)

        # Convert to numpy array based on sample width
        if sampwidth == 2:  # 16-bit audio
            dtype = np.int16
        elif sampwidth == 4:  # 32-bit audio
            dtype = np.int32
        elif sampwidth == 1:  # 8-bit audio
            dtype = np.uint8
        else:
            print(f"Error: Unsupported sample width: {sampwidth}")
            return False

        # Process audio data
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)

        # Reshape to (samples, channels)
        audio_data = audio_data.reshape(-1, n_channels)

        # Extract only the first N channels
        audio_data = audio_data[:, :output_channels]

        # Flatten back to 1D array
        audio_data = audio_data.flatten()

        # Write output WAV file
        with wave.open(output_file, 'wb') as wf_out:
            wf_out.setnchannels(output_channels)
            wf_out.setsampwidth(sampwidth)
            wf_out.setframerate(framerate)
            wf_out.writeframes(audio_data.tobytes())

        # Display results
        output_size = os.path.getsize(output_file)
        print(f"Success: Converted to {output_channels} channels")
        print(f"  Output file: {output_file}")
        print(f"  Output size: {output_size / 1024:.2f} KB")

        return True

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False


def main():
    """Main entry point for audio channel converter."""
    parser = argparse.ArgumentParser(
        description='Convert 8-channel WAV files to 4-channel (TI board recordings)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_8ch_to_4ch.py recording.wav           # Creates recording_4ch.wav
  python convert_8ch_to_4ch.py audio.wav -o output.wav # Custom output name
  python convert_8ch_to_4ch.py *.wav                   # Process all WAV files
  python convert_8ch_to_4ch.py file.wav --keep 2       # Keep only first 2 channels
        """
    )

    parser.add_argument('input_files', nargs='+',
                       help='Input WAV file(s) to process')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output filename (only for single file processing)')
    parser.add_argument('--input-channels', type=int, default=8,
                       help='Expected number of input channels (default: 8)')
    parser.add_argument('--keep', '-k', type=int, default=4,
                       help='Number of channels to keep (default: 4)')

    args = parser.parse_args()

    # Check if output is specified with multiple files
    if len(args.input_files) > 1 and args.output:
        print("Error: --output can only be used with a single input file")
        sys.exit(1)

    print("="*60)
    print("Audio Channel Converter - TI Board Processing")
    print("="*60)

    # Process files
    success_count = 0
    total_files = len(args.input_files)

    for input_file in args.input_files:
        # Use custom output only if provided and single file
        output_file = args.output if len(args.input_files) == 1 else None

        if process_audio_file(input_file, output_file,
                             args.input_channels, args.keep):
            success_count += 1
        print()  # Empty line between files

    print("="*60)
    print(f"Completed: {success_count}/{total_files} files processed successfully")
    print("="*60)


if __name__ == "__main__":
    main()
