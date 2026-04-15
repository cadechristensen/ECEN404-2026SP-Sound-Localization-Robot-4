"""
Inference module for baby cry detection.
Provides functionality to predict on single audio files.
"""

import torch
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Union

try:
    from .config import Config
    from .model import create_model
    from .data_preprocessing import AudioPreprocessor
except ImportError:
    import sys
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from config import Config  # type: ignore
    from model import create_model  # type: ignore
    from data_preprocessing import AudioPreprocessor  # type: ignore


class BabyCryPredictor:
    """
    Predictor class for baby cry detection on individual audio files.
    """

    def __init__(self, config: Config):
        """
        Initialize predictor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model = None
        self.preprocessor = AudioPreprocessor(config)

    def load_model(self, model_path: Path):
        """
        Load trained model.

        Args:
            model_path: Path to model checkpoint
        """
        logging.info(f"Loading model from {model_path}")

        # Attempt the safe (weights_only=True) load first.  Quantized model
        # checkpoints store a full nn.Module object (pickle) which cannot be
        # loaded with weights_only=True; fall back to the full unpickle only
        # in that case so state-dict checkpoints always use the safer path.
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            logging.warning(
                f"Could not load {model_path} with weights_only=True "
                "(quantized model or legacy format). "
                "Falling back to full unpickle — ensure the checkpoint is from a trusted source."
            )
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model' in checkpoint and not isinstance(checkpoint['model'], dict):
            # Quantized model format: checkpoint['model'] is the actual model object
            # Note: Quantized models cannot be moved to different devices, keep on CPU
            self.model = checkpoint['model']
            if self.device.type != 'cpu':
                logging.warning("Quantized model detected. Forcing device to CPU (quantized models cannot run on other devices)")
                self.device = torch.device('cpu')
        else:
            # Standard format: load state_dict into a fresh model
            self.model = create_model(self.config)

            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)

        self.model.eval()

        logging.info("Model loaded successfully")

    def predict(self, audio_path: Path) -> Dict:
        """
        Predict on a single audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logging.info(f"Processing audio file: {audio_path}")

        try:
            # Preprocess audio
            spectrogram = self.preprocessor.process_audio_file(audio_path)

            # Add batch and channel dimensions
            spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time_steps)
            spectrogram = spectrogram.to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(spectrogram)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item() * 100

            # Get class label
            class_label = self.config.CLASS_LABELS[predicted_class]

            # Prepare result
            result = {
                'file_path': str(audio_path),
                'predicted_class': predicted_class,
                'predicted_label': class_label,
                'confidence': confidence,
                'probabilities': {
                    'non_cry': probabilities[0, 0].item() * 100,
                    'cry': probabilities[0, 1].item() * 100
                }
            }

            logging.info(f"Prediction: {class_label} ({confidence:.1f}% confidence)")

            return result

        except Exception as e:
            logging.error(f"Prediction failed for {audio_path}: {e}")
            raise

    def predict_with_threshold(self, audio_path: Path, threshold: Optional[float] = None) -> Dict:
        """
        Predict with confidence threshold.

        Args:
            audio_path: Path to audio file
            threshold: Confidence threshold (uses config default if None)

        Returns:
            Dictionary containing prediction results with threshold decision
        """
        if threshold is None:
            threshold = self.config.CONFIDENCE_THRESHOLD

        result = self.predict(audio_path)

        # Apply threshold
        if result['predicted_label'] == 'cry' and result['confidence'] < threshold * 100:
            result['threshold_decision'] = 'uncertain'
            result['threshold_message'] = f"Predicted cry but confidence ({result['confidence']:.1f}%) below threshold ({threshold*100:.1f}%)"
        else:
            result['threshold_decision'] = result['predicted_label']
            result['threshold_message'] = f"Confident prediction"

        result['threshold_used'] = threshold

        return result

    def predict_batch(self, audio_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Predict on multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of prediction results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(Path(audio_path))
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to process {audio_path}: {e}")
                results.append({
                    'file_path': str(audio_path),
                    'error': str(e)
                })

        return results
