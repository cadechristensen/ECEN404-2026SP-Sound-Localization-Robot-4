#!/usr/bin/env python3
"""
Test script for baby cry detection model and audio filtering.

This script tests two things:
1. Model Inference: Load audio, run through model, display classification and confidence
2. Audio Filtering: If baby cry detected, apply filtering and save filtered .wav file

Usage:
    python3 test_model_and_filter.py --model /path/to/model.pth --audio /path/to/audio.wav [--output output.wav]

    # Example:
    python3 test_model_and_filter.py --model models/model_quantized.pth --audio test_audio.wav --output filtered_output.wav
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

# Insert the project root (two directories up) so that src.* imports resolve
# when this script is run directly from deployment/raspberry_pi/.
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import Config
from src.model import create_model
from src.audio_filter import BabyCryAudioFilter


class BabyCryModelTester:
    """Test harness for baby cry detection model and filtering."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize tester.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on ('cpu' or 'cuda')
        """
        self.config = Config()
        self.device = torch.device(device if device else
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))

        logging.info(f"Using device: {self.device}")

        # Load model
        self.model = create_model(self.config).to(self.device)
        self._load_checkpoint(model_path)
        self.model.eval()
        logging.info(f"Model loaded from {model_path}")

        # Initialize audio filter
        self.audio_filter = BabyCryAudioFilter(self.config, model_path)
        logging.info("Audio filter initialized")

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            f_min=self.config.F_MIN,
            f_max=self.config.F_MAX
        ).to(self.device)

    def _load_checkpoint(self, model_path: str):
        """
        Load model weights from checkpoint.

        Handles three checkpoint formats:
        - Pi-quantized: {'model': <quantized model>, 'pi_optimized': True}
        - Calibrated:   {'model_state_dict': ..., 'temperature': T, ...}
        - Standard:     {'model_state_dict': ...} or raw state dict
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if not isinstance(checkpoint, dict):
            # Raw model object saved directly
            self.model = checkpoint.to(self.device)

        elif checkpoint.get('pi_optimized'):
            # Quantized model: cannot use state_dict, replace model entirely
            self.model = checkpoint['model'].to(self.device)
            logging.info("Loaded Pi-quantized model")

        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Calibrated model: TemperatureScaledModel saved a second temperature
            # on top of BabyCryClassifier's own self.temperature.
            # Fold both into model.temperature so forward() applies the correct scaling.
            if 'temperature' in checkpoint:
                self.model.temperature.data *= checkpoint['temperature']
                logging.info(f"Calibrated temperature {checkpoint['temperature']:.4f} applied")

        else:
            self.model.load_state_dict(checkpoint)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.config.SAMPLE_RATE:
            logging.info(f"Resampling from {sample_rate} to {self.config.SAMPLE_RATE}")
            resampler = torchaudio.transforms.Resample(sample_rate, self.config.SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = self.config.SAMPLE_RATE

        return waveform.squeeze(0), sample_rate

    def preprocess_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio for model input.

        Args:
            waveform: Audio waveform

        Returns:
            Preprocessed audio tensor
        """
        # Ensure correct duration
        target_length = int(self.config.DURATION * self.config.SAMPLE_RATE)
        if len(waveform) < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - len(waveform)))
        elif len(waveform) > target_length:
            waveform = waveform[:target_length]

        return waveform

    def audio_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to mel spectrogram."""
        waveform = waveform.to(self.device).unsqueeze(0)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale and normalize
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(0)  # (1, 1, n_mels, time)

        return mel_spec

    def predict(self, audio: torch.Tensor) -> Tuple[bool, float]:
        """
        Predict baby cry classification.

        Args:
            audio: Audio waveform

        Returns:
            Tuple of (is_cry, confidence_score)
        """
        # Preprocess
        waveform = self.preprocess_audio(audio)
        spectrogram = self.audio_to_spectrogram(waveform)

        # Predict
        with torch.no_grad():
            logits = self.model(spectrogram)
            probs = torch.softmax(logits, dim=1)

            # Class 0: Not baby cry, Class 1: Baby cry
            not_cry_prob = probs[0, 0].item()
            cry_prob = probs[0, 1].item()

        is_cry = cry_prob > not_cry_prob
        confidence = max(cry_prob, not_cry_prob)

        return is_cry, confidence, cry_prob, not_cry_prob

    def filter_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply audio filtering to isolate baby cry.

        Args:
            audio: Audio waveform

        Returns:
            Filtered audio waveform
        """
        # Convert to tensor if needed
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio).float()

        # Apply filtering
        filtered, cry_segments = self.audio_filter.isolate_baby_cry(
            audio,
            cry_threshold=0.6
        )

        return filtered

    def save_audio(self, waveform: torch.Tensor, output_path: str, sample_rate: int = 16000):
        """
        Save audio waveform to file.

        Args:
            waveform: Audio waveform
            output_path: Output file path
            sample_rate: Sample rate
        """
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform).float()

        # Add channel dimension if needed
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        torchaudio.save(output_path, waveform, sample_rate)
        logging.info(f"Saved audio to {output_path}")

    def test(self, audio_path: str, output_path: Optional[str] = None, threshold: float = 0.5):
        """
        Run complete test: inference + filtering.

        Args:
            audio_path: Path to input audio file
            output_path: Path to save filtered audio (optional)
            threshold: Classification threshold for baby cry (0-1)
        """
        print("\n" + "="*70)
        print("BABY CRY DETECTION - MODEL TEST")
        print("="*70)

        # Load audio
        print(f"\n[1] Loading audio: {audio_path}")
        try:
            waveform, sample_rate = self.load_audio(audio_path)
            print(f"    ✓ Loaded successfully")
            print(f"      Duration: {len(waveform) / sample_rate:.2f} seconds")
            print(f"      Sample rate: {sample_rate} Hz")
            print(f"      Samples: {len(waveform)}")
        except Exception as e:
            logging.error(f"Failed to load audio: {e}")
            return

        # Run inference
        print(f"\n[2] Running model inference...")
        try:
            is_cry, confidence, cry_prob, not_cry_prob = self.predict(waveform)
            print(f"    ✓ Inference complete")
            print(f"\n    Classification Results:")
            print(f"    ├─ Baby Cry Probability: {cry_prob:.1%}")
            print(f"    ├─ Not Baby Cry Probability: {not_cry_prob:.1%}")
            print(f"    ├─ Confidence: {confidence:.1%}")
            print(f"    └─ Decision: {'BABY CRY DETECTED ✓' if is_cry else 'NOT BABY CRY'}")
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            return

        # Apply filtering if baby cry detected
        print(f"\n[3] Audio Filtering...")
        if is_cry and output_path:
            try:
                print(f"    Baby cry detected! Applying audio filtering...")
                filtered_audio = self.filter_audio(waveform)
                print(f"    ✓ Filtering complete")
                print(f"      Filtered audio duration: {len(filtered_audio) / sample_rate:.2f} seconds")

                # Save filtered audio
                self.save_audio(filtered_audio, output_path, sample_rate)
                print(f"\n    Filtered audio saved to: {output_path}")

            except Exception as e:
                logging.error(f"Filtering failed: {e}")
        elif not is_cry and output_path:
            print(f"    No baby cry detected - skipping filtering")
        elif is_cry and not output_path:
            print(f"    Baby cry detected but no output path specified")
            print(f"    Use --output to save filtered audio")
        else:
            print(f"    No filtering performed")

        # Summary
        print(f"\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Input file: {audio_path}")
        if output_path:
            print(f"Output file: {output_path}")
        print(f"\nResult: {'BABY CRY' if is_cry else 'NOT BABY CRY'}")
        print(f"Confidence: {confidence:.1%}")
        print("="*70 + "\n")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Test baby cry detection model and audio filtering'
    )
    parser.add_argument('--model', type=str, default='../calibrated_model.pth',
                       help='Path to trained model checkpoint (default: ../calibrated_model.pth)')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to test audio file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save filtered audio (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (0-1)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create tester and run test
    tester = BabyCryModelTester(args.model, device=args.device)
    tester.test(args.audio, output_path=args.output, threshold=args.threshold)


if __name__ == "__main__":
    main()
