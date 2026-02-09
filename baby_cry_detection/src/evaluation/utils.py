"""
Utility functions for model evaluation.

This module contains helper functions including TTA, ensemble models,
prediction generation, and error logging.
Designed for binary classification (non-cry vs cry detection).
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import logging


def predict_with_tta(
    model: nn.Module,
    spectrograms: torch.Tensor,
    device: torch.device,
    n_augments: int = 5
) -> torch.Tensor:
    """
    Test-time augmentation for more robust predictions.

    Args:
        model: Trained model
        spectrograms: Input spectrograms (batch_size, 1, n_mels, time_steps)
        device: torch device
        n_augments: Number of augmented versions to average

    Returns:
        Averaged predictions (batch_size, num_classes)
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        outputs = model(spectrograms)
        predictions.append(outputs)

        for _ in range(n_augments - 1):
            shift_amount = torch.randint(-5, 6, (1,)).item()
            aug_spec = torch.roll(spectrograms, shifts=shift_amount, dims=-1)

            noise = torch.randn_like(aug_spec) * 0.01
            aug_spec = aug_spec + noise

            outputs = model(aug_spec)
            predictions.append(outputs)

    avg_outputs = torch.mean(torch.stack(predictions), dim=0)
    return avg_outputs


class EnsembleModel:
    """
    Ensemble multiple model checkpoints for better predictions.
    Averages predictions from the last N best checkpoints to improve robustness.
    """

    def __init__(self, model_paths: List[Path], config, device: torch.device):
        """
        Initialize ensemble.

        Args:
            model_paths: List of paths to model checkpoints
            config: Config object
            device: torch device
        """
        from ..model import create_model

        self.models = []
        self.device = device
        self.config = config

        logging.info(f"Loading {len(model_paths)} models for ensemble...")
        for i, path in enumerate(model_paths):
            if not path.exists():
                logging.warning(f"Model path not found: {path}, skipping...")
                continue

            checkpoint = torch.load(path, map_location='cpu', weights_only=False)

            model = create_model(config).to(device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            self.models.append(model)
            logging.info(f"  Loaded model {i+1}/{len(model_paths)}: {path.name}")

        if not self.models:
            raise ValueError("No valid models loaded for ensemble!")

        logging.info(f"Ensemble ready with {len(self.models)} models")

    def eval(self):
        """Set all models to evaluation mode."""
        for model in self.models:
            model.eval()
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models and average predictions.
        """
        with torch.no_grad():
            predictions = [model(x) for model in self.models]
            return torch.mean(torch.stack(predictions), dim=0)

    @staticmethod
    def from_results_dir(results_dir: Path, config, device: torch.device) -> 'EnsembleModel':
        """
        Load ensemble from a results directory using saved metadata.

        Args:
            results_dir: Path to results directory
            config: Config object
            device: torch device

        Returns:
            Initialized EnsembleModel
        """
        ensemble_metadata_path = results_dir / "ensemble_checkpoints.json"

        if not ensemble_metadata_path.exists():
            logging.warning("ensemble_checkpoints.json not found, searching for checkpoints...")
            checkpoint_paths = sorted(results_dir.glob("model_epoch_*.pth"))
            if not checkpoint_paths:
                raise FileNotFoundError(f"No model checkpoints found in {results_dir}")
            n_models = getattr(config, 'ENSEMBLE_N_MODELS', 3)
            checkpoint_paths = checkpoint_paths[-n_models:]
        else:
            with open(ensemble_metadata_path, 'r') as f:
                metadata = json.load(f)
            checkpoint_paths = [results_dir / path for path in metadata['checkpoint_paths']]
            logging.info(f"Loading ensemble from metadata: {len(checkpoint_paths)} checkpoints")

        return EnsembleModel(checkpoint_paths, config, device)


def generate_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
    tta_n_augments: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions for a dataset.

    Args:
        model: Model to evaluate
        data_loader: DataLoader for the dataset
        device: torch device
        use_tta: Whether to use test-time augmentation
        tta_n_augments: Number of augmentations for TTA

    Returns:
        Tuple of (predictions, probabilities, true_labels)
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        desc = "Generating predictions (TTA)" if use_tta else "Generating predictions"
        for batch_data in tqdm(data_loader, desc=desc):
            if len(batch_data) == 3:
                spectrograms, labels, _ = batch_data
            else:
                spectrograms, labels = batch_data

            spectrograms = spectrograms.to(device, non_blocking=True)
            labels = labels.cpu().numpy()

            if use_tta:
                outputs = predict_with_tta(model, spectrograms, device, tta_n_augments)
            else:
                outputs = model(spectrograms)

            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels)

    return (
        np.array(all_predictions),
        np.array(all_probabilities),
        np.array(all_labels)
    )


def generate_predictions_and_log_errors(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_labels: List[str],
    dataset_name: str,
    results_dir: Path,
    use_tta: bool = False,
    tta_n_augments: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions and log misclassified files.

    Args:
        model: Model to evaluate
        data_loader: DataLoader for the dataset
        device: torch device
        class_labels: List of class label names
        dataset_name: Name of dataset (train/val/test)
        results_dir: Directory to save error log
        use_tta: Whether to use test-time augmentation
        tta_n_augments: Number of augmentations for TTA

    Returns:
        Tuple of (predictions, probabilities, true_labels)
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    misclassified_files = []

    uses_sampler = data_loader.sampler is not None

    with torch.no_grad():
        desc = "Generating predictions (TTA)" if use_tta else "Generating predictions"
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=desc)):
            if len(batch_data) == 3:
                spectrograms, labels, indices = batch_data
                indices = indices.cpu().numpy()
                has_indices = True
            else:
                spectrograms, labels = batch_data
                indices = None
                has_indices = False

            spectrograms = spectrograms.to(device, non_blocking=True)
            labels = labels.cpu().numpy()

            if use_tta:
                outputs = predict_with_tta(model, spectrograms, device, tta_n_augments)
            else:
                outputs = model(spectrograms)

            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            for i, (pred, true_label, prob) in enumerate(zip(predictions, labels, probabilities)):
                filename = "unknown"
                if hasattr(data_loader.dataset, 'audio_files') and has_indices:
                    try:
                        dataset_idx = indices[i]
                        if 0 <= dataset_idx < len(data_loader.dataset.audio_files):
                            file_path, _ = data_loader.dataset.audio_files[dataset_idx]
                            filename = str(file_path)
                    except Exception as e:
                        logging.debug(f"Could not retrieve filename using index {indices[i]}: {e}")
                elif hasattr(data_loader.dataset, 'audio_files') and not uses_sampler:
                    try:
                        sample_idx = batch_idx * data_loader.batch_size + i
                        if sample_idx < len(data_loader.dataset.audio_files):
                            file_path, _ = data_loader.dataset.audio_files[sample_idx]
                            filename = str(file_path)
                    except Exception as e:
                        logging.debug(f"Could not retrieve filename for batch {batch_idx}, item {i}: {e}")

                if pred != true_label:
                    true_label_name = class_labels[true_label]
                    pred_label_name = class_labels[pred]
                    confidence = prob[pred] * 100

                    misclassified_files.append({
                        'filename': filename,
                        'true_label': true_label_name,
                        'predicted_label': pred_label_name,
                        'confidence': float(confidence),
                        'probabilities': [float(p) for p in prob]
                    })

                    logging.error(f"MISCLASSIFIED - File: {filename}, True: {true_label_name}, Predicted: {pred_label_name} ({confidence:.1f}% confidence)")

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels)

    if misclassified_files:
        error_log_path = results_dir / f"misclassified_{dataset_name}.json"
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(misclassified_files, f, indent=2, ensure_ascii=False)

        logging.info(f"Saved {len(misclassified_files)} misclassified files to {error_log_path}")

        summary_path = results_dir / f"misclassified_{dataset_name}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"MISCLASSIFIED FILES SUMMARY - {dataset_name.upper()} SET\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total misclassifications: {len(misclassified_files)}\n\n")

            for item in misclassified_files:
                f.write(f"File: {item['filename']}\n")
                f.write(f"  True Label: {item['true_label']}\n")
                f.write(f"  Predicted: {item['predicted_label']} ({item['confidence']:.1f}% confidence)\n")
                f.write(f"  Probabilities: non_cry={item['probabilities'][0]:.3f}, cry={item['probabilities'][1]:.3f}\n")
                f.write("-" * 40 + "\n")

        logging.info(f"Summary saved to {summary_path}")
    else:
        logging.info(f"No misclassifications found in {dataset_name} set!")

    return (
        np.array(all_predictions),
        np.array(all_probabilities),
        np.array(all_labels)
    )


def setup_module_aliases():
    """Setup module aliases to handle different import paths when unpickling models."""
    import sys
    try:
        from .. import model, config
    except ImportError:
        import model
        import config

    if 'src' not in sys.modules:
        if __package__:
            sys.modules['src'] = sys.modules[__package__.split('.')[0]]
        else:
            import types
            src_module = types.ModuleType('src')
            sys.modules['src'] = src_module

    if 'src.model' not in sys.modules:
        sys.modules['src.model'] = model
    if 'src.config' not in sys.modules:
        sys.modules['src.config'] = config

    if 'model' not in sys.modules:
        sys.modules['model'] = model
    if 'config' not in sys.modules:
        sys.modules['config'] = config
