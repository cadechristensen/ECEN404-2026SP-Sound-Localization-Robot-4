"""
Example demonstrating dual-output audio filtering feature.

This script shows how to use the new output modes:
- full_length: Preserves original audio length, mutes non-cry segments
- cry_only: Concatenates only cry segments, removes silence
- both: Generates both output versions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import torch
import torchaudio
from src.config import Config
from src.audio_filter import BabyCryAudioFilter


def example_full_length_output():
    """Example 1: Full-length output (default behavior)."""
    print("=" * 80)
    print("EXAMPLE 1: Full-Length Output (Preserves Original Length)")
    print("=" * 80)

    config = Config()
    model_path = "results/train_2025-11-17_19-15-25/model_inference.pth"
    audio_filter = BabyCryAudioFilter(config=config, model_path=model_path)

    input_path = "examples/audio_test1.wav"
    output_path = "examples/audio_test1_full_example.wav"

    results = audio_filter.process_audio_file(
        input_path=input_path,
        output_path=output_path,
        cry_threshold=0.5,
        output_mode="full_length"
    )

    print(f"\nOutput saved to: {results['output_files']['full_length']}")
    print(f"File duration: {results['total_duration']:.2f}s (unchanged)")
    print(f"Cry content: {results['cry_duration']:.2f}s ({results['duration_percentage']:.1f}%)")
    print()


def example_cry_only_output():
    """Example 2: Cry-only output (concatenated segments)."""
    print("=" * 80)
    print("EXAMPLE 2: Cry-Only Output (Concatenated Cry Segments)")
    print("=" * 80)

    config = Config()
    model_path = "results/train_2025-11-17_19-15-25/model_inference.pth"
    audio_filter = BabyCryAudioFilter(config=config, model_path=model_path)

    input_path = "examples/audio_test1.wa"
    output_path = "examples/audio_test1_cut_example.wav"

    results = audio_filter.process_audio_file(
        input_path=input_path,
        output_path=output_path,
        cry_threshold=0.5,
        output_mode="cry_only"
    )

    print(f"\nOutput saved to: {results['output_files']['cry_only']}")
    print(f"Original duration: {results['total_duration']:.2f}s")
    print(f"Cry-only duration: {results['cry_only_duration']:.2f}s")
    print(f"Compression ratio: {results['cry_only_duration'] / results['total_duration'] * 100:.1f}%")
    print()


def example_both_outputs():
    """Example 3: Both output modes simultaneously."""
    print("=" * 80)
    print("EXAMPLE 3: Both Output Modes")
    print("=" * 80)

    config = Config()
    model_path = "results/model_best.pth"
    audio_filter = BabyCryAudioFilter(config=config, model_path=model_path)

    input_path = "path/to/your/audio.wav"
    output_path = "output/audio_filtered.wav"

    results = audio_filter.process_audio_file(
        input_path=input_path,
        output_path=output_path,
        cry_threshold=0.5,
        output_mode="both"
    )

    print("\nOutput files:")
    for mode, filepath in results['output_files'].items():
        print(f"  {mode}: {filepath}")

    print(f"\nOriginal duration: {results['total_duration']:.2f}s")
    print(f"Full-length output: {results['total_duration']:.2f}s (with {results['cry_duration']:.2f}s cry content)")
    print(f"Cry-only output: {results['cry_only_duration']:.2f}s")
    print()


def example_multichannel_audio():
    """Example 4: Multi-channel audio processing (preserves all channels)."""
    print("=" * 80)
    print("EXAMPLE 4: Multi-Channel Audio (4-Channel Mic Array)")
    print("=" * 80)

    config = Config()
    model_path = "results/model_best.pth"
    audio_filter = BabyCryAudioFilter(config=config, model_path=model_path)

    input_path = "path/to/4channel_audio.wav"
    output_path = "output/4channel_filtered.wav"

    results = audio_filter.process_audio_file(
        input_path=input_path,
        output_path=output_path,
        cry_threshold=0.5,
        output_mode="both"
    )

    print("\nOutput files (all preserve 4-channel structure):")
    for mode, filepath in results['output_files'].items():
        print(f"  {mode}: {filepath}")

    print("\nNote: Phase relationships between channels are preserved")
    print("This is essential for sound localization and beamforming.")
    print()


def example_edge_cases():
    """Example 5: Handling edge cases."""
    print("=" * 80)
    print("EXAMPLE 5: Edge Case Handling")
    print("=" * 80)

    config = Config()
    model_path = "results/model_best.pth"
    audio_filter = BabyCryAudioFilter(config=config, model_path=model_path)

    print("\nCase 1: Audio with no cry detected")
    print("-" * 40)

    results = audio_filter.process_audio_file(
        input_path="path/to/silent_audio.wav",
        output_path="output/no_cry.wav",
        cry_threshold=0.5,
        output_mode="cry_only"
    )

    if results['num_cry_segments'] == 0:
        print("No cry segments detected")
        print(f"Cry-only output duration: {results['cry_only_duration']:.2f}s (empty)")

    print("\nCase 2: Very short cry segments")
    print("-" * 40)

    results = audio_filter.process_audio_file(
        input_path="path/to/brief_cries.wav",
        output_path="output/brief_cries.wav",
        cry_threshold=0.7,
        output_mode="both"
    )

    print(f"Detected {results['num_cry_segments']} brief cry segments")
    print(f"Average segment duration: {results['cry_duration'] / max(1, results['num_cry_segments']):.2f}s")

    print("\nCase 3: Continuous crying (high cry percentage)")
    print("-" * 40)

    results = audio_filter.process_audio_file(
        input_path="path/to/continuous_cry.wav",
        output_path="output/continuous_cry.wav",
        cry_threshold=0.5,
        output_mode="both"
    )

    print(f"Cry percentage: {results['duration_percentage']:.1f}%")
    if results['duration_percentage'] > 80:
        print("High cry content - full_length and cry_only outputs will be very similar")

    print()


def example_direct_api_usage():
    """Example 6: Using the extract_cry_segments_only method directly."""
    print("=" * 80)
    print("EXAMPLE 6: Direct API Usage (Advanced)")
    print("=" * 80)

    config = Config()
    model_path = "results/model_best.pth"
    audio_filter = BabyCryAudioFilter(config=config, model_path=model_path)

    audio, sr = torchaudio.load("path/to/your/audio.wav")
    audio = audio[0] if audio.shape[0] == 1 else audio.transpose(0, 1)

    if sr != audio_filter.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, audio_filter.sample_rate)
        audio = resampler(audio)

    print("Step 1: Detect cry segments...")
    if audio.dim() > 1:
        isolated_audio, cry_segments, all_segments = audio_filter.isolate_baby_cry_multichannel(
            audio, cry_threshold=0.5
        )
    else:
        isolated_audio, cry_segments, all_segments = audio_filter.isolate_baby_cry(
            audio, cry_threshold=0.5
        )

    print(f"Found {len(cry_segments)} cry segments")

    print("\nStep 2: Extract cry-only audio...")
    cry_only_audio = audio_filter.extract_cry_segments_only(audio, cry_segments)

    print(f"Original audio: {len(audio) / audio_filter.sample_rate:.2f}s")
    print(f"Cry-only audio: {len(cry_only_audio) / audio_filter.sample_rate:.2f}s")

    print("\nStep 3: Save outputs...")
    if cry_only_audio.dim() > 1:
        torchaudio.save("output/cry_only_custom.wav",
                       cry_only_audio.transpose(0, 1),
                       audio_filter.sample_rate)
    else:
        torchaudio.save("output/cry_only_custom.wav",
                       cry_only_audio.unsqueeze(0),
                       audio_filter.sample_rate)

    print("Custom processing complete!")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("DUAL-OUTPUT AUDIO FILTERING EXAMPLES")
    print("=" * 80)
    print()

    print("This script demonstrates the new dual-output feature for audio filtering.")
    print("You can now generate two types of outputs:")
    print("  1. Full-length: Original length preserved, non-cry segments muted")
    print("  2. Cry-only: Concatenated cry segments, silence removed")
    print()
    print("Choose an example to run:")
    print("  1. Full-length output (default)")
    print("  2. Cry-only output")
    print("  3. Both outputs")
    print("  4. Multi-channel audio")
    print("  5. Edge cases")
    print("  6. Direct API usage")
    print("  7. Run all examples")
    print()

    choice = input("Enter choice (1-7): ").strip()

    examples = {
        "1": example_full_length_output,
        "2": example_cry_only_output,
        "3": example_both_outputs,
        "4": example_multichannel_audio,
        "5": example_edge_cases,
        "6": example_direct_api_usage,
    }

    if choice == "7":
        for func in examples.values():
            func()
    elif choice in examples:
        examples[choice]()
    else:
        print("Invalid choice. Please run again.")


if __name__ == "__main__":
    print("\nNOTE: This is an example script. Update the file paths before running.")
    print("Example paths like 'path/to/your/audio.wav' need to be replaced with actual files.")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        sys.exit(0)

    main()
