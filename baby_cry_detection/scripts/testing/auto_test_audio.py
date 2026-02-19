"""
Audio Recording Test Script for TI Board Integration

Records audio from microphone (TI board or default device) and saves as WAV files.
Automatically increments filename numbers to avoid overwriting existing recordings.

Usage:
    python auto_test_audio.py --list-devices
    python auto_test_audio.py [--device DEVICE_INDEX] [--count NUM_RECORDINGS] [--duration SECONDS]

"""

#python scripts/testing/auto_test_audio.py --list-devices
#python scripts/testing/auto_test_audio.py --device 1 --count 2 --duration 10   
#python scripts/testing/auto_test_audio.py --device # --count # --duration 10

import pyaudio
import wave
import os
import argparse
import numpy as np
from typing import Optional

#python -c "import torchaudio; waveform, sr = torchaudio.load('examples/audio_test1.wav'); print(f'Sample rate: {sr} Hz'); print(f'Channels:{waveform.shape[0]}'); print(f'Duration: {waveform.shape[1]/sr:.2f} seconds'); print(f'Samples: {waveform.shape[1]}')")


class AudioRecorder:
    """Professional audio recorder with device selection and error handling."""

    def __init__(self, sample_rate: int = 48000, channels: int = 4,
                 chunk_size: int = 1024, audio_format: int = pyaudio.paInt16,
                 record_channels: int = 8):
        """
        Initialize AudioRecorder with specified parameters.

        Args:
            sample_rate: Sampling frequency in Hz (default: 48000)
            channels: Number of audio channels to save (default: 4 for TI board)
            chunk_size: Number of frames per buffer
            audio_format: PyAudio format (default: 16-bit PCM)
            record_channels: Number of channels to record from device (default: 8)
        """
        self.sample_rate = sample_rate
        self.channels = channels  # Channels to save
        self.record_channels = record_channels  # Channels to record
        self.chunk_size = chunk_size
        self.audio_format = audio_format
        self.audio = None

    def list_devices(self) -> None:
        """List all available audio input devices."""
        audio = pyaudio.PyAudio()
        print("\n" + "="*60)
        print("Available Audio Input Devices:")
        print("="*60)

        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"Device {i}: {info['name']}")
                print(f"  - Channels: {info['maxInputChannels']}")
                print(f"  - Sample Rate: {int(info['defaultSampleRate'])} Hz")
                print()

        audio.terminate()

    def record_audio(self, filename: str, duration: int = 10,
                    device_index: Optional[int] = None) -> bool:
        """
        Record audio from microphone and save to WAV file.

        Args:
            filename: Output WAV filename
            duration: Recording duration in seconds
            device_index: Specific device index (None for default)

        Returns:
            True if recording successful, False otherwise
        """
        try:
            self.audio = pyaudio.PyAudio()

            # Open audio stream
            stream_kwargs = {
                'format': self.audio_format,
                'channels': self.record_channels,  # Record all 8 channels
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size
            }

            if device_index is not None:
                stream_kwargs['input_device_index'] = device_index
                device_info = self.audio.get_device_info_by_index(device_index)
                print(f"\nUsing device: {device_info['name']}")

            stream = self.audio.open(**stream_kwargs)

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
                    print(f"  {elapsed}/{duration} seconds", end='\r')

            print(f"\nRecording finished.")

            # Clean up stream
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

            # Process audio: extract first 4 channels from 8-channel recording
            # Convert bytes to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

            # Reshape to (samples, channels)
            audio_data = audio_data.reshape(-1, self.record_channels)

            # Extract only the first 'channels' (default 4)
            audio_data = audio_data[:, :self.channels]

            # Flatten back to 1D array
            audio_data = audio_data.flatten()

            # Save to WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())

            # Display file info
            file_size = os.path.getsize(filename)
            print(f"Audio saved to: {filename}")
            print(f"  - Size: {file_size / 1024:.2f} KB")
            print(f"  - Duration: {duration}s")
            print(f"  - Sample Rate: {self.sample_rate} Hz")
            print(f"  - Channels Saved: {self.channels} (recorded {self.record_channels}, kept first {self.channels})")

            return True

        except Exception as e:
            print(f"\nError during recording: {e}")
            if self.audio:
                self.audio.terminate()
            return False

    #! Name of File
    def get_next_filename(self, base_name: str = "rec_demo", extension: str = ".wav") -> str:
        """
        Get next available filename with incremented number.

        Args:
            base_name: Base filename prefix
            extension: File extension

        Returns:
            Next available filename (e.g., "audiotest1.wav")
        """
        counter = 1
        while True:
            filename = f"{base_name}{counter}{extension}"
            if not os.path.exists(filename):
                return filename
            counter += 1


def main():
    """Main entry point for audio recording script."""
    parser = argparse.ArgumentParser(
        description='Record audio from microphone (TI board or default device)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_test_audio.py                          # Record 3 files, 10s each
  python auto_test_audio.py --duration 5 --count 5   # Record 5 files, 5s each
  python auto_test_audio.py --list-devices           # List available devices
  python auto_test_audio.py --device 2 --count 1     # Use specific device
        """
    )

    parser.add_argument('--duration', '-d', type=int, default=10,
                       help='Recording duration in seconds (default: 10)')
    parser.add_argument('--count', '-c', type=int, default=3,
                       help='Number of recordings to make (default: 3)')
    parser.add_argument('--device', '-i', type=int, default=None,
                       help='Audio input device index (default: system default)')
    parser.add_argument('--list-devices', '-l', action='store_true',
                       help='List all available audio input devices and exit')
    parser.add_argument('--sample-rate', '-r', type=int, default=48000,
                       help='Sample rate in Hz (default: 48000)')
    parser.add_argument('--channels', type=int, default=4,
                       help='Number of audio channels to save (default: 4 for TI board)')
    parser.add_argument('--record-channels', type=int, default=8,
                       help='Number of channels to record from device (default: 8 for TI board)')
    parser.add_argument('--no-prompt', action='store_true',
                       help='Disable prompt before each recording (default: prompt enabled)')

    args = parser.parse_args()

    # Initialize recorder
    recorder = AudioRecorder(sample_rate=args.sample_rate,
                           channels=args.channels,
                           record_channels=args.record_channels)

    # List devices if requested
    if args.list_devices:
        recorder.list_devices()
        return

    print("\n" + "="*60)
    print("Audio Recording Test - TI Board Integration")
    print("="*60)

    # Record multiple files
    success_count = 0
    for i in range(1, args.count + 1):
        filename = recorder.get_next_filename()
        print(f"\n[Recording {i}/{args.count}]")

        # Prompt user unless disabled
        if not args.no_prompt:
            response = input("Press Enter to start recording (or 'q' to quit): ").strip().lower()
            if response == 'q':
                print("Recording session cancelled by user.")
                break

        if recorder.record_audio(filename, duration=args.duration,
                                device_index=args.device):
            success_count += 1
        else:
            print("WARNING: Skipping to next recording...")

    print("\n" + "="*60)
    print(f"Completed: {success_count}/{args.count} recordings successful")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
        