"""
Custom Dataset class for baby cry detection.
Handles data loading, preprocessing, and augmentation for PyTorch training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

try:
    from .config import Config
    from .data_preprocessing import AudioPreprocessor, AudioAugmentation, collect_audio_files, collect_noise_files, get_class_weights
except ImportError:
    from config import Config  # type: ignore
    from data_preprocessing import AudioPreprocessor, AudioAugmentation, collect_audio_files, collect_noise_files, get_class_weights  # type: ignore


def collate_with_indices(batch):
    """
    Custom collate function that preserves original dataset indices.
    Handles 2-tuple (spec, label), 3-tuple (spec, label, idx),
    and 4-tuple (spec, label, idx, sample_weight).

    Args:
        batch: List of tuples from dataset

    Returns:
        Tuple of tensors matching the input tuple length.
    """
    n = len(batch[0])
    if n == 4:
        # Phase 3: has indices and per-sample loss weights
        specs, labels, indices, weights = zip(*batch)
        specs = torch.stack(specs, dim=0)
        labels = torch.stack(labels, dim=0)
        indices = torch.tensor(indices, dtype=torch.long)
        weights = torch.tensor(weights, dtype=torch.float32)
        return specs, labels, indices, weights
    elif n == 3:
        # Has indices
        specs, labels, indices = zip(*batch)
        specs = torch.stack(specs, dim=0)
        labels = torch.stack(labels, dim=0)
        indices = torch.tensor(indices, dtype=torch.long)
        return specs, labels, indices
    else:
        # No indices (backward compatibility)
        specs, labels = zip(*batch)
        specs = torch.stack(specs, dim=0)
        labels = torch.stack(labels, dim=0)
        return specs, labels


def collate_filter_corrupt(batch):
    """
    Collate function that filters out corrupt samples (label == -1) before batching.
    Used for validation and test DataLoaders to prevent zero-tensor poisoning of metrics.

    Returns None if all samples in batch are corrupt (caller must skip).
    """
    clean_batch = [item for item in batch if item[1].item() != -1]
    if not clean_batch:
        return None
    return collate_with_indices(clean_batch)


class BabyCryDataset(Dataset):
    """
    Custom Dataset class for baby cry detection.
    Handles loading, preprocessing, and augmentation of audio data.
    """

    def __init__(
        self,
        audio_files: List[Tuple[Path, str]],
        config: Optional[Config] = None,
        is_training: bool = True,
        augment: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            audio_files: List of (file_path, label) tuples
            config: Configuration object. Defaults to Config().
            is_training: Whether this is for training (affects augmentation)
            augment: Whether to apply data augmentation
        """
        if config is None:
            config = Config()
        self.audio_files = audio_files
        self.config = config
        self.is_training = is_training
        self.augment = augment and is_training

        # Initialize preprocessor and augmentation
        # Disable advanced filtering during training for clean model learning
        # Filtering is only for inference/deployment
        self.preprocessor = AudioPreprocessor(config, use_advanced_filtering=False)

        # Get noise files for background augmentation if augmentation is enabled
        noise_files = None
        if self.augment:
            noise_files = collect_noise_files(config.DATA_DIR, config.SUPPORTED_FORMATS)

        self.augmentation = AudioAugmentation(config, noise_files) if self.augment else None

        # Binary classification label encoding
        self.label_to_idx = {'non_cry': 0, 'cry': 1}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Log dataset statistics
        self._log_dataset_info()

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (spectrogram, label, original_idx)
        """
        file_path, label_str = self.audio_files[idx]

        try:
            # Load and preprocess audio
            if self.augment and np.random.random() > 0.5:
                # Apply augmentation at the waveform level
                waveform, _ = self.preprocessor.load_audio(file_path)
                waveform = self.preprocessor.pad_or_truncate(waveform)
                waveform = self.augmentation.random_augment(waveform)
                spectrogram = self.preprocessor.extract_log_mel_spectrogram(waveform)
            else:
                # Standard preprocessing; pass training flag so random vs center
                # crop matches whether we are in a training or eval split.
                spectrogram = self.preprocessor.process_audio_file(
                    file_path, training=self.augment
                )

            # QUICK WIN 1: Apply SpecAugment to mel-spectrogram during training
            if self.augment and hasattr(self.augmentation, 'spec_augment'):
                spectrogram = self.augmentation.spec_augment(spectrogram)

            # Convert label to index
            label = torch.tensor(self.label_to_idx[label_str], dtype=torch.long)

            # Add channel dimension for CNN input
            spectrogram = spectrogram.unsqueeze(0)  # Shape: (1, n_mels, time_steps)

            return spectrogram, label, idx

        except Exception as e:
            # During training, a zero spectrogram with a real label is training-data
            # poisoning — the model learns "silence == cry/non-cry" from every
            # corrupt file in every epoch.  Re-raise so the DataLoader surfaces the
            # failure loudly.  During inference (is_training=False) we fall back to
            # a zero tensor so a single bad file does not abort an evaluation run (C4).
            logging.error(f"Failed to load {file_path}: {e}")
            if self.is_training:
                raise
            # Return sentinel label -1 for corrupt eval files.
            # The collate_filter_corrupt function will drop these samples before
            # they reach the model, preventing zero-tensor poisoning of eval metrics.
            dummy_spec = torch.zeros((1, self.config.N_MELS,
                                    int(self.config.DURATION * self.config.SAMPLE_RATE // self.config.HOP_LENGTH) + 1))
            return dummy_spec, torch.tensor(-1, dtype=torch.long), idx

    def _log_dataset_info(self):
        """Log information about the dataset."""
        label_counts = {}
        for _, label in self.audio_files:
            label_counts[label] = label_counts.get(label, 0) + 1

        logging.info(f"Dataset initialized with {len(self.audio_files)} samples")
        for label, count in label_counts.items():
            logging.info(f"  {label}: {count} samples")


class DatasetManager:
    """
    Manager class for handling dataset creation and data loading.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the dataset manager.

        Args:
            config: Configuration object. Defaults to Config().
        """
        if config is None:
            config = Config()
        self.config = config
        self.audio_files = None
        self.class_weights = None

    def prepare_datasets(self) -> Tuple[BabyCryDataset, BabyCryDataset, BabyCryDataset]:
        """
        Prepare train, validation, and test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Collect all audio files
        logging.info("Collecting audio files...")
        self.audio_files = collect_audio_files(
            self.config.DATA_DIR,
            self.config.SUPPORTED_FORMATS,
        )

        if not self.audio_files:
            raise ValueError(f"No audio files found in {self.config.DATA_DIR}")

        logging.info(f"Found {len(self.audio_files)} total audio files")

        # Calculate class weights for balanced training with emphasis on cry detection.
        cry_weight_multiplier = self.config.CRY_WEIGHT_MULTIPLIER

        # Binary class labels
        class_labels_list = ['non_cry', 'cry']

        self.class_weights = get_class_weights(
            self.audio_files,
            cry_weight_multiplier=cry_weight_multiplier,
            class_labels=class_labels_list,
        )

        # Log class weights
        weight_info = ", ".join([f"{label}={self.class_weights[i]:.3f}" for i, label in enumerate(class_labels_list)])
        logging.info(f"Class weights (cry_multiplier={cry_weight_multiplier}): {weight_info}")

        # Split data into train, validation, and test sets
        train_files, temp_files = train_test_split(
            self.audio_files,
            test_size=(self.config.VAL_RATIO + self.config.TEST_RATIO),
            random_state=42,
            stratify=[label for _, label in self.audio_files]
        )

        val_files, test_files = train_test_split(
            temp_files,
            test_size=self.config.TEST_RATIO / (self.config.VAL_RATIO + self.config.TEST_RATIO),
            random_state=42,
            stratify=[label for _, label in temp_files]
        )

        # Create datasets
        train_dataset = BabyCryDataset(
            train_files,
            self.config,
            is_training=True,
            augment=True
        )

        val_dataset = BabyCryDataset(
            val_files,
            self.config,
            is_training=False,
            augment=False
        )

        test_dataset = BabyCryDataset(
            test_files,
            self.config,
            is_training=False,
            augment=False
        )

        logging.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def create_data_loaders(
        self,
        train_dataset: BabyCryDataset,
        val_dataset: BabyCryDataset,
        test_dataset: BabyCryDataset,
        use_weighted_sampling: bool = True,
        num_workers: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for training, validation, and testing.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            use_weighted_sampling: Whether to use weighted sampling for balanced training
            num_workers: Number of data loader workers (defaults to config.NUM_WORKERS)

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if num_workers is None:
            num_workers = self.config.NUM_WORKERS

        # pin_memory only benefits CUDA DMA transfers; on the Raspberry Pi (CPU-only)
        # it wastes locked memory without any throughput gain and risks OOM (C3).
        pin_memory = torch.cuda.is_available()

        # Dataset is nearly balanced (0.96:1 ratio). WeightedRandomSampler
        # combined with FocalLoss class weights and CRY_WEIGHT_MULTIPLIER
        # results in excessive class weighting. Use shuffle=True instead.
        _persistent = num_workers > 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=_persistent,
            drop_last=True,
            collate_fn=collate_with_indices
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=_persistent,
            collate_fn=collate_filter_corrupt
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=_persistent,
            collate_fn=collate_filter_corrupt
        )

        return train_loader, val_loader, test_loader

    def get_sample_input_shape(self) -> torch.Size:
        """
        Get the shape of a sample input for model initialization.

        Returns:
            Input shape as torch.Size
        """
        if not self.audio_files:
            self.audio_files = collect_audio_files(
                self.config.DATA_DIR,
                self.config.SUPPORTED_FORMATS,
            )

        # Create a temporary dataset to get input shape.
        # __getitem__ returns a 3-tuple (spectrogram, label, idx); unpack accordingly (C1).
        temp_dataset = BabyCryDataset([self.audio_files[0]], self.config, is_training=False, augment=False)
        sample_input, _, _ = temp_dataset[0]

        return sample_input.shape


def test_dataset():
    """
    Test function to verify dataset functionality.
    """
    # Initialize configuration
    config = Config()

    # Create dataset manager
    manager = DatasetManager(config)

    try:
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = manager.prepare_datasets()

        # Create data loaders
        train_loader, val_loader, test_loader = manager.create_data_loaders(
            train_dataset, val_dataset, test_dataset
        )

        # Test loading a batch
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle 4-tuple, 3-tuple, and 2-tuple batch formats
            if len(batch_data) >= 3:
                spectrograms, labels = batch_data[0], batch_data[1]
            else:
                spectrograms, labels = batch_data

            print(f"Batch {batch_idx}:")
            print(f"  Spectrograms shape: {spectrograms.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels}")
            break

        print("Dataset test completed successfully!")

    except Exception as e:
        print(f"Dataset test failed: {e}")


if __name__ == "__main__":
    test_dataset()