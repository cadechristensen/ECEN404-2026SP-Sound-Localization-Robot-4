"""
Test your own audio files with acoustic feature-based filtering.

Usage:
    python scripts/testing/test_my_audio.py my_audio.wav
    python scripts/testing/test_my_audio.py my_audio.wav --threshold 0.92 --plot --acoustic
    python scripts/testing/test_my_audio.py my_audio.wav --model other_model.pth
"""

# Add project root to path
import sys
import os
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import sys
import argparse
from pathlib import Path
import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from src.config import Config
from src.audio_filtering import BabyCryAudioFilter
from src.data_preprocessing import AudioPreprocessor


def _merge_overlapping_segments(segments):
    """Merge overlapping time segments to avoid double-counting duration."""
    if not segments:
        return []
    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]
    for current_start, current_end in sorted_segments[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged

def visualize_filtering_pipeline(input_path: str, audio_filter: BabyCryAudioFilter,
                                 cry_threshold: float = 0.5,
                                 use_acoustic_features: bool = True,
                                 processing_results: dict = None,
                                 show_ml_predictions: bool = False,
                                 output_dir: Path = None):
    """
    Generate visualization plots showing all filtering stages.

    Args:
        input_path: Path to audio file
        audio_filter: Initialized BabyCryAudioFilter
        cry_threshold: Detection threshold
        use_acoustic_features: Whether acoustic features were used
        processing_results: Results from process_audio_file (optional, for consistency)
        show_ml_predictions: Whether to show the ML predictions plot (default: False)
        output_dir: Directory to save visualization (default: input_file.parent / 'filtering_visualizations')
    """
    print("\nGenerating visualization plots...")

    # Load and prepare audio
    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] > 1:
        audio = audio[0]  # Use first channel for visualization
    else:
        audio = audio[0]

    # Resample if needed
    if sr != audio_filter.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, audio_filter.sample_rate)
        audio = resampler(audio)
        sr = audio_filter.sample_rate

    # Step through the filtering pipeline
    print("  Step 1: Spectral filtering...")
    filtered_audio = audio_filter.spectral_filter(audio)

    print("  Step 2: Voice activity detection...")
    vad_mask = audio_filter.voice_activity_detection(filtered_audio)

    print("  Step 3: Spectral subtraction...")
    denoised_audio = audio_filter.spectral_subtraction(filtered_audio)

    #! Get acoustic features if enabled
    acoustic_features = True
    if use_acoustic_features:
        print("  Step 4: Computing acoustic features...")
        acoustic_features = audio_filter.compute_acoustic_features(filtered_audio)

    # Get ML predictions
    print("  Step 5: Getting ML predictions...")
    ml_segments = audio_filter.classify_audio_segments(
        filtered_audio,
        use_acoustic_validation=use_acoustic_features
    )

    # Create visualization with dynamic numbering
    # Grid is 3x3 = 9 plots (removed VAD, Acoustic Features, and Rejection Filters from visualization)
    plt.figure(figsize=(18, 12))

    # Track plot number dynamically
    plot_num = 0
    preprocessor = AudioPreprocessor(audio_filter.config)
    time_axis = np.arange(len(audio)) / sr

    # Plot: Original waveform
    plot_num += 1
    ax = plt.subplot(3, 3, plot_num)
    ax.plot(time_axis, audio.numpy(), linewidth=0.5, color='#4472C4')
    ax.set_title(f'{plot_num}. Original Waveform', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.set_xlim([0, len(audio) / sr])
    ax.grid(True, alpha=0.3)

    # Plot: Spectral filtered waveform
    plot_num += 1
    ax = plt.subplot(3, 3, plot_num)
    time_axis_filt = np.arange(len(filtered_audio)) / sr
    ax.plot(time_axis_filt, filtered_audio.numpy(), linewidth=0.5, color='#70AD47')
    ax.set_title(f'{plot_num}. Spectral Filtered (100-3000 Hz)', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.set_xlim([0, len(filtered_audio) / sr])
    ax.grid(True, alpha=0.3)

    # Plot: Denoised waveform
    plot_num += 1
    ax = plt.subplot(3, 3, plot_num)
    time_axis_den = np.arange(len(denoised_audio)) / sr
    ax.plot(time_axis_den, denoised_audio.numpy(), linewidth=0.5, color='#9966CC')
    ax.set_title(f'{plot_num}. Spectral Subtraction Applied', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.set_xlim([0, len(denoised_audio) / sr])
    ax.grid(True, alpha=0.3)

    # Compute raw dB spectrograms (no z-score normalization -- that is for model input only).
    # All three panels share vmin/vmax so amplitude reductions are visible.
    original_spec = preprocessor.amplitude_to_db(preprocessor.mel_transform(audio.unsqueeze(0))).squeeze(0)
    filtered_spec = preprocessor.amplitude_to_db(preprocessor.mel_transform(filtered_audio.unsqueeze(0))).squeeze(0)
    denoised_spec = preprocessor.amplitude_to_db(preprocessor.mel_transform(denoised_audio.unsqueeze(0))).squeeze(0)
    spec_vmin = float(original_spec.min())
    spec_vmax = float(original_spec.max())

    # Plot: Original spectrogram
    plot_num += 1
    ax = plt.subplot(3, 3, plot_num)
    im = ax.imshow(original_spec.numpy(), aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, len(audio) / sr, 0, original_spec.shape[0]],
                   vmin=spec_vmin, vmax=spec_vmax)
    ax.set_title(f'{plot_num}. Original Mel Spectrogram', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Mel Frequency Bin', fontsize=9)
    plt.colorbar(im, ax=ax, label='dB')

    # Plot: Filtered spectrogram
    plot_num += 1
    ax = plt.subplot(3, 3, plot_num)
    im = ax.imshow(filtered_spec.numpy(), aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, len(filtered_audio) / sr, 0, filtered_spec.shape[0]],
                   vmin=spec_vmin, vmax=spec_vmax)
    ax.set_title(f'{plot_num}. Filtered Mel Spectrogram', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Mel Frequency Bin', fontsize=9)
    plt.colorbar(im, ax=ax, label='dB')

    # Plot: Denoised spectrogram
    plot_num += 1
    ax = plt.subplot(3, 3, plot_num)
    im = ax.imshow(denoised_spec.numpy(), aspect='auto', origin='lower', cmap='viridis',
                   extent=[0, len(denoised_audio) / sr, 0, denoised_spec.shape[0]],
                   vmin=spec_vmin, vmax=spec_vmax)
    ax.set_title(f'{plot_num}. Denoised Mel Spectrogram', fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Mel Frequency Bin', fontsize=9)
    plt.colorbar(im, ax=ax, label='dB')

    # # Plot: Voice activity detection
    # plot_num += 1
    # ax = plt.subplot(3, 3, plot_num)
    # # Create time axis matching VAD mask length
    # frame_length = 1024
    # hop_length_vad = frame_length // 2
    # vad_time = np.arange(len(vad_mask)) * hop_length_vad / sr
    #
    # # Ensure arrays have matching lengths
    # if len(vad_time) != len(vad_mask):
    #     min_len = min(len(vad_time), len(vad_mask))
    #     vad_time = vad_time[:min_len]
    #     vad_mask_plot = vad_mask[:min_len]
    # else:
    #     vad_mask_plot = vad_mask
    #
    # # Plot waveform in background (normalized to 0-1 range)
    # audio_normalized = (audio.numpy() - audio.numpy().min()) / (audio.numpy().max() - audio.numpy().min() + 1e-8)
    # ax.plot(time_axis, audio_normalized, linewidth=0.5, color='gray', alpha=0.4, label='Waveform')
    #
    # # Overlay VAD mask
    # ax.fill_between(vad_time, 0, 1, where=vad_mask_plot.numpy(), alpha=0.3, color='green', label='Voice Activity')
    # ax.set_title(f'{plot_num}. Voice Activity Detection', fontweight='bold', fontsize=10)
    # ax.set_xlabel('Time (s)', fontsize=9)
    # ax.set_ylabel('Activity', fontsize=9)
    # ax.set_xlim([0, len(audio) / sr])
    # ax.set_ylim([0, 1])
    # ax.legend(fontsize=8)
    # ax.grid(alpha=0.3)

    # Plot: ML predictions (optional - only if enabled)
    if show_ml_predictions:
        plot_num += 1
        ax = plt.subplot(3, 3, plot_num)
        for start, end, prob, _meta in ml_segments:
            color = 'red' if prob > cry_threshold else 'orange'
            alpha = min(0.8, prob)
            ax.axvspan(start, end, alpha=alpha, color=color)
        ax.set_title(f'{plot_num}. ML Predictions (threshold={cry_threshold})', fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Cry Probability', fontsize=9)
        ax.set_xlim([0, len(audio) / sr])
        ax.set_ylim([0, 1])
        ax.axhline(y=cry_threshold, color='black', linestyle='--', label='Threshold', linewidth=1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # # Plot: Acoustic features (if enabled)
    # if acoustic_features:
    #     plot_num += 1
    #     ax = plt.subplot(3, 3, plot_num)
    #
    #     # Show harmonic scores
    #     harmonic = acoustic_features['harmonic_scores']
    #     energy = acoustic_features['energy_scores']
    #     hop_length = 512
    #
    #     # Use the shorter length for alignment
    #     min_len_features = min(len(harmonic), len(energy))
    #     time_axis_features = np.arange(min_len_features) * hop_length / sr
    #
    #     ax.plot(time_axis_features, harmonic[:min_len_features].numpy(), label='Harmonic', alpha=0.7)
    #     ax.plot(time_axis_features, energy[:min_len_features].numpy(),
    #             label='Energy', alpha=0.7)
    #     ax.set_title(f'{plot_num}. Acoustic Features', fontweight='bold', fontsize=10)
    #     ax.set_xlabel('Time (s)', fontsize=9)
    #     ax.set_ylabel('Score', fontsize=9)
    #     ax.legend(fontsize=8)
    #     ax.grid(alpha=0.3)
    #
    # # Plot: Rejection filters (if enabled)
    # if acoustic_features:
    #     plot_num += 1
    #     ax = plt.subplot(3, 3, plot_num)
    #
    #     # Get all rejection filters
    #     adult_rej = acoustic_features['adult_rejection']
    #     music_rej = acoustic_features['music_rejection']
    #     env_rej = acoustic_features['env_rejection']
    #
    #     # Use the shortest length for alignment
    #     min_len_rejection = min(len(adult_rej), len(music_rej), len(env_rej))
    #     hop_length = 512
    #     time_axis_rejection = np.arange(min_len_rejection) * hop_length / sr
    #
    #     ax.plot(time_axis_rejection, adult_rej[:min_len_rejection].numpy(),
    #              label='Adult Speech Filter', alpha=0.7)
    #     ax.plot(time_axis_rejection, music_rej[:min_len_rejection].numpy(),
    #              label='Music Filter', alpha=0.7)
    #     ax.plot(time_axis_rejection, env_rej[:min_len_rejection].numpy(),
    #              label='Environmental Filter', alpha=0.7)
    #     ax.set_title(f'{plot_num}. Rejection Filters (1=keep, 0=reject)', fontweight='bold', fontsize=10)
    #     ax.set_xlabel('Time (s)', fontsize=9)
    #     ax.set_ylabel('Score', fontsize=9)
    #     ax.legend(fontsize=7)
    #     ax.grid(alpha=0.3)

    # Plot: Final cry segments
    plot_num += 1
    ax = plt.subplot(3, 3, plot_num)

    # Use actual results if provided, otherwise compute from ml_segments
    if processing_results and 'cry_segments' in processing_results:
        cry_segments = processing_results['cry_segments']
    else:
        cry_segments = [(start, end) for start, end, prob, _meta in ml_segments if prob >= cry_threshold]

    # Show waveform with cry regions highlighted
    ax.plot(time_axis, audio.numpy(), linewidth=0.5, color='gray', alpha=0.5, label='Original')
    for start, end in cry_segments:
        ax.axvspan(start, end, alpha=0.3, color='red')

    ax.set_title(f'{plot_num}. Detected Cry Segments ({len(cry_segments)} found)',
                  fontweight='bold', fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.set_xlim([0, len(audio) / sr])
    ax.grid(alpha=0.3)

    # Plot: Summary statistics
    plot_num += 1
    ax = plt.subplot(3, 3, plot_num)
    ax.axis('off')

    # Use processing_results if provided, otherwise compute from visualization run
    if processing_results:
        total_duration = processing_results['total_duration']
        cry_duration = processing_results['cry_duration']
        duration_percentage = processing_results['duration_percentage']
        avg_confidence = processing_results['avg_confidence']
        min_confidence = processing_results['min_confidence']
        max_confidence = processing_results['max_confidence']
        num_segments = processing_results['num_cry_segments']
    else:
        total_duration = len(audio) / sr
        # Merge overlapping segments before calculating duration
        merged_segments = _merge_overlapping_segments(cry_segments)
        cry_duration = sum(end - start for start, end in merged_segments)
        duration_percentage = (cry_duration / total_duration) * 100 if total_duration > 0 else 0
        num_segments = len(cry_segments)
        avg_confidence = 0.0
        min_confidence = 0.0
        max_confidence = 0.0

    summary_text = f"""
    FILTERING SUMMARY
    ═════════════════════════════

    Total Duration: {total_duration:.2f}s
    Cry Duration: {cry_duration:.2f}s
    Duration %: {duration_percentage:.1f}% of file

    Cry Segments: {num_segments}
    Threshold: {cry_threshold}

    Model Confidence:
    Average: {avg_confidence:.1f}%
    Range: {min_confidence:.1f}%-{max_confidence:.1f}%

    Filters Applied:
    ✓ Spectral filtering (100-3000 Hz)
    ✓ Voice activity detection
    ✓ Spectral subtraction
    ✓ ML model classification
    {'✓ Acoustic feature validation' if use_acoustic_features else '✗ Acoustic features (disabled)'}
    {'✓ Adult speech rejection' if use_acoustic_features else '✗ Adult speech rejection'}
    {'✓ Music rejection' if use_acoustic_features else '✗ Music rejection'}
    {'✓ Environmental rejection' if use_acoustic_features else '✗ Environmental rejection'}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center')
    ax.set_title(f'{plot_num}. Summary', fontweight='bold', fontsize=10)

    plt.tight_layout()

    # Save figure
    input_file = Path(input_path)
    if output_dir is None:
        output_dir = input_file.parent / 'filtering_visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{input_file.stem}_filtering_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    print("=" * 80)

    plt.close()

    return output_path


def test_audio_file(input_path: str, model_path: str = None,
                   cry_threshold: float = 0.5,
                   use_acoustic_features: bool = False,
                   generate_plots: bool = False,
                   output_mode: str = "full_length"):
    """
    Test a single audio file for baby cry detection.

    Args:
        input_path: Path to your audio file
        model_path: Path to trained model (optional - can work without ML model)
        cry_threshold: Detection threshold (0.0-1.0, lower = more sensitive)
        use_acoustic_features: Whether to use acoustic validation filter (binary accept/reject) (default: False)
        generate_plots: Generate visualization plots of filtering stages
        output_mode: Output format - "full_length", "cry_only", or "both" (default: "full_length")
    """
    # Check if input file exists
    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        return

    # Initialize
    config = Config()

    audio_filter = BabyCryAudioFilter(config=config, model_path=model_path)

    # Generate output filename
    input_file = Path(input_path)
    output_path = input_file.parent / f"{input_file.stem}_filtered{input_file.suffix}"

    print("=" * 80)
    print(f"Testing Audio File: {input_path}")
    print("=" * 80)
    print(f"Model: {'Using ML model' if model_path else 'Acoustic features only'}")
    print(f"Binary Classification: ML-ONLY (with acoustic validation filter)")
    print(f"Acoustic validation: {'ENABLED' if use_acoustic_features else 'DISABLED'}")
    print(f"Threshold: {cry_threshold}")
    print(f"Output mode: {output_mode}")
    print(f"Output will be saved to: {output_path}")
    print()

    # Process the audio
    try:
        results = audio_filter.process_audio_file(
            input_path=str(input_path),
            output_path=str(output_path),
            cry_threshold=cry_threshold,
            use_acoustic_features=use_acoustic_features,
            output_mode=output_mode
        )

        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
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
                plot_path = visualize_filtering_pipeline(
                    input_path,
                    audio_filter,
                    cry_threshold,
                    use_acoustic_features,
                    results  # Pass the actual results
                )
                print(f"Visualization plots saved to: {plot_path}")
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")
                import traceback
                traceback.print_exc()

        return results

    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_files(audio_dir: str, model_path: str = None):
    """
    Test all audio files in a directory.

    Args:
        audio_dir: Directory containing audio files
        model_path: Path to trained model (optional)
    """
    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        print(f"Error: Directory not found: {audio_dir}")
        return

    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    print()

    # Process each file
    all_results = []
    for audio_file in audio_files:
        results = test_audio_file(str(audio_file), model_path)
        if results:
            all_results.append(results)
        print("\n" + "-" * 80 + "\n")

    # Summary
    if all_results:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        total_files = len(all_results)
        files_with_cries = sum(1 for r in all_results if r['num_cry_segments'] > 0)
        total_segments = sum(r['num_cry_segments'] for r in all_results)

        print(f"Files processed: {total_files}")
        print(f"Files with detected cries: {files_with_cries}")
        print(f"Total cry segments: {total_segments}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test audio files for baby cry detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default model:
  python scripts/testing/test_my_audio.py my_baby_cry.wav
  python scripts/testing/test_my_audio.py my_baby_cry.wav --threshold 0.92 --plot

  # Use a different model:
  python scripts/testing/test_my_audio.py my_baby_cry.wav --model other_model.pth

  # Additional options:
  python scripts/testing/test_my_audio.py my_baby_cry.wav --acoustic --plot
  python scripts/testing/test_my_audio.py my_baby_cry.wav --cry-only
  python scripts/testing/test_my_audio.py my_baby_cry.wav --both --plot
        """
    )

    # Positional argument
    parser.add_argument('audio_file', nargs='?', type=str,
                       help='Path to audio file to test')

    # Model configuration options
    parser.add_argument('--model', '-m', type=str,
                       default="results/train_2025-11-24_19-10-26/evaluations/eval_2025-11-24_19-38-25/calibrated_model.pth",
                       help='Path to trained model')

    # Testing options
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

    args = parser.parse_args()

    # Check if audio file was provided
    if not args.audio_file:
        parser.print_help()
        sys.exit(1)

    # Validate model path
    model_path = args.model
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please specify a valid model path using --model argument")
        sys.exit(1)

    print(f"\nUsing model: {model_path}")

    # Determine output mode
    output_mode = "full_length"
    if args.cry_only:
        output_mode = "cry_only"
    elif args.both:
        output_mode = "both"

    # Test the audio file
    test_audio_file(
        args.audio_file,
        model_path,
        args.threshold,
        use_acoustic_features=args.acoustic,
        generate_plots=args.plot,
        output_mode=output_mode
    )
