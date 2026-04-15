"""
Dual-channel voting cry detection for 4-channel mic array audio.

Selects the two best channels by cry-band SNR, runs the full filtering pipeline
on each, then combines results via SNR-weighted voting or logical OR.
All 4 channels are preserved with phase relationships intact for sound localization.
The filtered output (cry regions only, all channels) is ready for the Sound Localization model.
"""

import sys
import os
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, List

from src.config import Config
from src.audio_filtering import BabyCryAudioFilter
from deployment.multichannel_detector import EnhancedSNRComputation
from scripts.testing.test_my_audio import visualize_filtering_pipeline, _merge_overlapping_segments


def load_and_prepare_multichannel_audio(
    input_path: str,
    target_sample_rate: int = 16000,
    num_channels: int = 4
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to 4-channel format if needed.

    Args:
        input_path: Path to audio file
        target_sample_rate: Target sample rate
        num_channels: Number of channels to output (default: 4)

    Returns:
        Tuple of (multichannel_audio, sample_rate)
        Audio shape: (num_samples, num_channels)
    """
    audio, sr = torchaudio.load(input_path)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
        sr = target_sample_rate

    # Convert to numpy
    audio_np = audio.numpy()

    # Handle different input formats
    if audio_np.shape[0] == 1:
        # Mono - duplicate to all channels
        audio_np = np.tile(audio_np.T, (1, num_channels))
        print(f"Converted mono audio to {num_channels} channels by duplication")
    elif audio_np.shape[0] == 2:
        # Stereo - duplicate each channel twice to get 4 channels
        left = audio_np[0:1, :].T
        right = audio_np[1:2, :].T
        audio_np = np.hstack([left, right, left, right])
        print(f"Converted stereo audio to {num_channels} channels by duplication")
    elif audio_np.shape[0] >= num_channels:
        # Multi-channel - use first N channels and transpose
        audio_np = audio_np[:num_channels, :].T
        #print(f"Using first {num_channels} channels from {audio_np.shape[1]}-channel audio")
    else:
        # Less than required channels - pad by duplication
        audio_np = audio_np.T
        while audio_np.shape[1] < num_channels:
            audio_np = np.hstack([audio_np, audio_np[:, :num_channels - audio_np.shape[1]]])
        print(f"Padded {audio_np.shape[1]} channels to {num_channels} channels")

    # Ensure correct shape: (num_samples, num_channels)
    if audio_np.shape[1] != num_channels:
        raise ValueError(f"Expected {num_channels} channels, got {audio_np.shape[1]}")

    return audio_np, sr


def extract_cry_segments_multichannel(audio: np.ndarray, cry_segments: List[Tuple[float, float]], sample_rate: int) -> np.ndarray:
    """
    Extract and concatenate cry segments from all channels, preserving phase.

    Segments are extracted simultaneously across all channels so inter-channel
    phase relationships remain intact within each segment. Caller must pass
    merged (non-overlapping) segments to avoid duplicating audio.

    Args:
        audio: Original multi-channel audio (num_samples, num_channels)
        cry_segments: Merged list of (start_time, end_time) in seconds
        sample_rate: Audio sample rate

    Returns:
        Concatenated cry-only audio (total_cry_samples, num_channels)
    """
    chunks = []
    for start, end in cry_segments:
        start_idx = int(start * sample_rate)
        end_idx = min(int(end * sample_rate), audio.shape[0])
        chunks.append(audio[start_idx:end_idx])
    return np.concatenate(chunks, axis=0)


def test_multichannel_audio(
    input_path: str,
    model_path: str,
    cry_threshold: float = 0.5,
    voting_strategy: str = "weighted",
    generate_plots: bool = False,
    num_channels: int = 4
):
    """
    Dual-channel voting cry detection on 4-channel mic array audio.

    Pipeline per selected channel (via isolate_baby_cry):
        1. Spectral filtering (bandpass)
        2. Voice activity detection
        3. ML segment classification (on raw audio, same as training)
        4. Spectral subtraction
        5. Cry segment extraction

    Voting combines the two channels' confidences to make the final
    cry/no-cry decision. Output is the original 4-channel audio masked
    to cry regions only (phase preserved for sound localization).

    Args:
        input_path: Path to audio file
        model_path: Path to trained model
        cry_threshold: Detection threshold (0.0-1.0)
        voting_strategy: "weighted" (SNR-weighted) or "logical_or"
        generate_plots: Generate filtering pipeline visualization
        num_channels: Number of channels (default: 4)
    """
    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        return

    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        return

    print("=" * 60)
    print("Dual-Channel Voting Cry Detection")
    print("=" * 60)
    print(f"Input:     {input_path}")
    print(f"Threshold: {cry_threshold}  |  Voting: {voting_strategy}  |  Plots: {'On' if generate_plots else 'Off'}")

    # Initialize
    config = Config()
    audio_filter = BabyCryAudioFilter(config=config, model_path=model_path, verbose=False)
    snr_computer = EnhancedSNRComputation(sample_rate=config.SAMPLE_RATE)

    # Load audio
    audio, sr = load_and_prepare_multichannel_audio(input_path, config.SAMPLE_RATE, num_channels)
    peak = float(np.max(np.abs(audio)))
    #peak_db = (20 * np.log10(peak) + 94.0) if peak > 0 else -np.inf
    print(f"Duration: {audio.shape[0] / sr:.2f}s  |  Channels: {num_channels}")
    print()

    # --- Channel Selection by cry-band SNR ---
    snr_scores = snr_computer.compute_snr_all_channels(audio)
    sorted_indices = np.argsort(snr_scores)[::-1]
    selected_channels = sorted_indices[:2].tolist()

    snr_labels = []
    for ch in range(num_channels):
        tag = " [P]" if ch == selected_channels[0] else " [S]" if ch == selected_channels[1] else ""
        snr_labels.append(f"Ch{ch}: {snr_scores[ch]:5.1f} dB{tag}")
    print("SNR:  " + "  ".join(snr_labels))
    print()

    # --- Full filtering pipeline on each selected channel ---
    channel_results = {}
    channel_confidences = []

    for ch in selected_channels:
        channel_audio = torch.from_numpy(audio[:, ch]).float()

        _, cry_segments_ch, all_segments_ch = audio_filter.isolate_baby_cry(
            channel_audio,
            cry_threshold=cry_threshold,
            use_acoustic_features=False
        )

        confidence = max((prob for _, _, prob in all_segments_ch), default=0.0)

        channel_results[ch] = {
            'cry_segments': cry_segments_ch,
            'all_segments': all_segments_ch,
            'confidence': confidence
        }
        channel_confidences.append(confidence)
        print(f"  Ch{ch}: {len(all_segments_ch)} segments ({len(cry_segments_ch)} cry)  |  max conf: {confidence:.4f}")

    # --- Dual-Channel Voting ---
    if voting_strategy == "weighted":
        weights = np.array([snr_scores[ch] for ch in selected_channels])
        weights = np.exp(weights / 10.0)
        weights = weights / weights.sum()
        final_confidence = float(np.sum(np.array(channel_confidences) * weights))
    elif voting_strategy == "logical_or":
        final_confidence = max(channel_confidences)

    is_cry = final_confidence >= cry_threshold
    agreement = 1.0 - abs(channel_confidences[0] - channel_confidences[1])
    agreement_label = "high" if agreement > 0.8 else "moderate" if agreement > 0.6 else "low"
    print(f"  Voting: {final_confidence:.4f}  |  Agreement: {agreement:.2%} ({agreement_label})")

    # --- Detection Result ---
    print()
    print("=" * 60)
    print(f"  RESULT: {'CRY DETECTED' if is_cry else 'NO CRY DETECTED'}")
    print(f"  Confidence: {final_confidence:.2%}  |  Threshold: {cry_threshold:.2%}")
    print("=" * 60)
    print()

    # --- Extract cry segments from all 4 channels (phase preserved for Sound Localization) ---
    primary_ch = selected_channels[0]
    cry_segments = _merge_overlapping_segments(channel_results[primary_ch]['cry_segments'])

    input_file = Path(input_path)
    output_path = None

    if is_cry and cry_segments:
        filtered_audio = extract_cry_segments_multichannel(audio, cry_segments, sr)
        for i, (start, end) in enumerate(cry_segments, 1):
            print(f"  {i}. {start:.2f}s - {end:.2f}s")

        output_path = input_file.parent / f"{input_file.stem}_filtered.wav"
        output_tensor = torch.from_numpy(filtered_audio.T).float()  # (channels, samples)
        torchaudio.save(str(output_path), output_tensor, sr)
        print(f"Saved: {output_path}  ({filtered_audio.shape[0] / sr:.2f}s, {num_channels}-ch)")
    else:
        print("No cry detected.")
    print()

    # --- Visualization (filtering pipeline plots, same as record_and_filter) ---
    if generate_plots:
        print("Generating filtering pipeline plots...")
        try:
            viz_dir = input_file.parent / "filtering_visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

            # Save primary channel as mono wav -- visualize_filtering_pipeline loads from file
            primary_audio = torch.from_numpy(audio[:, primary_ch]).float().unsqueeze(0)
            primary_wav = str(viz_dir / f"{input_file.stem}_ch{primary_ch}.wav")
            torchaudio.save(primary_wav, primary_audio, sr)

            # Build processing_results for consistent viz summary panel
            all_segs = channel_results[primary_ch]['all_segments']
            total_dur = audio.shape[0] / sr
            cry_dur = sum(end - start for start, end in cry_segments)
            cry_probs = [p for _, _, p in all_segs if p >= cry_threshold]
            processing_results = {
                'cry_segments': cry_segments,
                'total_duration': total_dur,
                'cry_duration': cry_dur,
                'duration_percentage': (cry_dur / total_dur * 100) if total_dur > 0 else 0.0,
                'avg_confidence': (sum(cry_probs) / len(cry_probs) * 100) if cry_probs else 0.0,
                'min_confidence': (min(cry_probs) * 100) if cry_probs else 0.0,
                'max_confidence': (max(cry_probs) * 100) if cry_probs else 0.0,
                'num_cry_segments': len(cry_segments)
            }

            plot_path = visualize_filtering_pipeline(
                primary_wav,
                audio_filter,
                cry_threshold,
                use_acoustic_features=True,
                processing_results=processing_results,
                output_dir=viz_dir
            )
            print(f"Visualization saved to: {plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
            import traceback
            traceback.print_exc()

    return {
        'is_cry': is_cry,
        'confidence': final_confidence,
        'selected_channels': selected_channels,
        'channel_confidences': channel_confidences,
        'snr_scores': snr_scores,
        'agreement': agreement,
        'voting_strategy': voting_strategy,
        'cry_segments': cry_segments,
        'output_path': str(output_path) if output_path else None
    }


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Dual-channel voting cry detection for 4-channel mic array audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_multichannel_audio.py audio.wav
  python test_multichannel_audio.py audio.wav --model path/to/model.pth
  python test_multichannel_audio.py audio.wav --threshold 0.92 --voting logical_or
  python test_multichannel_audio.py audio.wav --plot
        """
    )

    parser.add_argument('input_audio', type=str,
                       help='Path to input audio file')
    parser.add_argument('--model', '-m', type=str,
                        default="baby_cry_detection/results/train_2025-11-24_19-10-26/evaluations/eval_2025-11-24_19-38-25/calibrated_model.pth",
                        help='Path to trained model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (0.0-1.0, default: 0.5)')
    parser.add_argument('--voting', type=str, default='weighted',
                       choices=['weighted', 'logical_or'],
                       help='Voting strategy (default: weighted)')
    parser.add_argument('--plot', '-p', action='store_true',
                       help='Generate filtering pipeline visualization')
    parser.add_argument('--channels', type=int, default=4,
                       help='Number of channels (default: 4)')

    args = parser.parse_args()

    test_multichannel_audio(
        input_path=args.input_audio,
        model_path=args.model,
        cry_threshold=args.threshold,
        voting_strategy=args.voting,
        generate_plots=args.plot,
        num_channels=args.channels
    )


if __name__ == "__main__":
    main()
