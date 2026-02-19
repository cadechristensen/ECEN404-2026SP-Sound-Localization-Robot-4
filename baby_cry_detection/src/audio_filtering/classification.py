"""
ML model integration and audio segment classification.

Binary classification for baby cry detection using trained neural network models.
"""

import os
import logging
import torch
from typing import List, Tuple, Optional, Dict


class AudioClassifier:
    """
    ML-based audio segment classifier for baby cry detection.

    Uses trained neural network models with optional confidence calibration
    and acoustic feature validation.
    """

    def __init__(self, config, preprocessor, acoustic_extractor=None, calibrator=None):
        """
        Initialize audio classifier.

        Args:
            config: Configuration object with model parameters
            preprocessor: Audio preprocessor for feature extraction
            acoustic_extractor: Optional acoustic feature extractor
            calibrator: Optional confidence calibrator
        """
        self.config = config
        self.preprocessor = preprocessor
        self.acoustic_extractor = acoustic_extractor
        self.calibrator = calibrator
        self.model = None
        self.sample_rate = config.SAMPLE_RATE

    def load_model(self, model_path: str):
        """
        Load trained baby cry detection model.

        Args:
            model_path: Path to trained model checkpoint
        """
        from ..model import create_model

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if 'model' in checkpoint and not isinstance(checkpoint['model'], dict):
                # Quantized model format: checkpoint['model'] is the actual model object
                self.model = checkpoint['model']
            else:
                # Standard format: load state_dict into a fresh model
                self.model = create_model(self.config)

                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

            self.model.eval()
            logging.info(f"Classifier model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            self.model = None

    def classify_segments(self, audio: torch.Tensor,
                         segment_duration: float = 3.0,
                         overlap: float = 0.5,
                         use_acoustic_validation: bool = True) -> List[Tuple[float, float, float, Dict]]:
        """
        Classify audio segments using the trained model with acoustic validation.

        Args:
            audio: Input audio tensor (mono)
            segment_duration: Duration of each segment in seconds
            overlap: Overlap ratio between segments
            use_acoustic_validation: Whether to apply acoustic feature validation

        Returns:
            List of (start_time, end_time, cry_probability, metadata) tuples
            metadata contains: {'raw_prob', 'calibrated_prob', 'validated', 'rejection_reason'}
        """
        if self.model is None:
            print("Warning: No model loaded, skipping classification")
            return []

        segment_samples = int(segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))

        segments = []

        for start_idx in range(0, len(audio) - segment_samples + 1, hop_samples):
            end_idx = start_idx + segment_samples
            segment = audio[start_idx:end_idx]

            # Preprocess segment
            try:
                mel_spec = self.preprocessor.extract_log_mel_spectrogram(segment)
                # Add batch dimension and channel dimension for model input (B, C, H, W)
                mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)

                # Get prediction
                with torch.no_grad():
                    outputs = self.model(mel_spec)

                    # Apply calibration if available
                    if self.calibrator is not None:
                        probabilities = self.calibrator.calibrate_probabilities(outputs)
                        calibrated_prob = probabilities[0, 1].item() if self.config.NUM_CLASSES == 2 else probabilities[0, 0].item()
                    else:
                        probabilities = torch.softmax(outputs, dim=1)
                        calibrated_prob = probabilities[0, 1].item() if self.config.NUM_CLASSES == 2 else probabilities[0, 0].item()

                    # Store raw probability (before calibration)
                    raw_probabilities = torch.softmax(outputs, dim=1)
                    raw_prob = raw_probabilities[0, 1].item() if self.config.NUM_CLASSES == 2 else raw_probabilities[0, 0].item()

                start_time = start_idx / self.sample_rate
                end_time = end_idx / self.sample_rate

                # Extract acoustic features for validation
                metadata = {
                    'raw_prob': raw_prob,
                    'calibrated_prob': calibrated_prob,
                    'validated': True,
                    'rejection_reason': None
                }

                final_prob = calibrated_prob

                if use_acoustic_validation and self.acoustic_extractor is not None:
                    # Extract acoustic features for this segment
                    from ..acoustic_features import validate_cry_binary

                    acoustic_features = self.acoustic_extractor.extract_all_features(segment)

                    # Apply BINARY acoustic validation (accept/reject only)
                    is_valid, reason = validate_cry_binary(
                        acoustic_features,
                        calibrated_prob,
                        threshold=0.5
                    )

                    metadata['validated'] = is_valid
                    metadata['rejection_reason'] = reason if not is_valid else None
                    metadata['acoustic_features'] = {
                        'pitch_mean': acoustic_features['pitch_mean'],
                        'hnr_mean': acoustic_features['hnr_mean'],
                        'duration': acoustic_features['duration']
                    }

                    # Keep original ML probability, but set to 0 if rejected by validation
                    final_prob = calibrated_prob if is_valid else 0.0

                segments.append((start_time, end_time, final_prob, metadata))

            except Exception as e:
                print(f"Error processing segment {start_idx}-{end_idx}: {e}")
                continue

        return segments

    def load_calibrator(self, calibrator_path: str, num_classes: int = 2):
        """
        Load confidence calibrator.

        Args:
            calibrator_path: Path to calibrator checkpoint
            num_classes: Number of output classes
        """
        from ..calibration import ConfidenceCalibrator

        if os.path.exists(calibrator_path):
            self.calibrator = ConfidenceCalibrator(method="temperature")
            self.calibrator.load(calibrator_path, num_classes=num_classes)
            print(f"Loaded calibrator from {calibrator_path}")
        else:
            print(f"Warning: Calibrator not found at {calibrator_path}")
