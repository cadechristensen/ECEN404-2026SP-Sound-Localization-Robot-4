"""
Custom Dataset class for baby cry detection.
Handles data loading, preprocessing, and augmentation for PyTorch training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    Handles both (spec, label) and (spec, label, idx) tuples.

    Args:
        batch: List of tuples from dataset

    Returns:
        Tuple of (spectrograms, labels) or (spectrograms, labels, indices) as tensors
    """
    if len(batch[0]) == 3:
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


class BabyCryDataset(Dataset):
    """
    Custom Dataset class for baby cry detection.
    Handles loading, preprocessing, and augmentation of audio data.
    """

    def __init__(
        self,
        audio_files: List[Tuple[Path, str]],
        config: Config = Config(),
        is_training: bool = True,
        augment: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            audio_files: List of (file_path, label) tuples
            config: Configuration object
            is_training: Whether this is for training (affects augmentation)
            augment: Whether to apply data augmentation
        """
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

        # Create label encoding based on config mode
        if config.MULTI_CLASS_MODE:
            self.label_to_idx = {label: idx for idx, label in config.MULTI_CLASS_LABELS.items()}
        else:
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
                # Standard preprocessing
                spectrogram = self.preprocessor.process_audio_file(file_path)

            # QUICK WIN 1: Apply SpecAugment to mel-spectrogram during training
            if self.augment and hasattr(self.augmentation, 'spec_augment'):
                spectrogram = self.augmentation.spec_augment(spectrogram)

            # Convert label to index
            label = torch.tensor(self.label_to_idx[label_str], dtype=torch.long)

            # Add channel dimension for CNN input
            spectrogram = spectrogram.unsqueeze(0)  # Shape: (1, n_mels, time_steps)

            return spectrogram, label, idx

        except Exception as e:
            # If loading fails, return a zero spectrogram with the correct shape
            logging.warning(f"Failed to load {file_path}: {e}")
            dummy_spec = torch.zeros((1, self.config.N_MELS,
                                    int(self.config.DURATION * self.config.SAMPLE_RATE // self.config.HOP_LENGTH) + 1))
            label = torch.tensor(self.label_to_idx[label_str], dtype=torch.long)
            return dummy_spec, label, idx

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

    def __init__(self, config: Config = Config()):
        """
        Initialize the dataset manager.

        Args:
            config: Configuration object
        """
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
            multi_class=self.config.MULTI_CLASS_MODE
        )

        if not self.audio_files:
            raise ValueError(f"No audio files found in {self.config.DATA_DIR}")

        logging.info(f"Found {len(self.audio_files)} total audio files")

        # Calculate class weights for balanced training with emphasis on cry detection
        cry_weight_multiplier = getattr(self.config, 'CRY_WEIGHT_MULTIPLIER', 2.0)
        baby_noncry_weight_cap = getattr(self.config, 'BABY_NONCRY_WEIGHT_CAP', None)

        # Get class labels in the correct order based on mode
        if self.config.MULTI_CLASS_MODE:
            class_labels_list = [self.config.MULTI_CLASS_LABELS[i] for i in sorted(self.config.MULTI_CLASS_LABELS.keys())]
        else:
            class_labels_list = ['non_cry', 'cry']

        self.class_weights = get_class_weights(
            self.audio_files,
            cry_weight_multiplier=cry_weight_multiplier,
            class_labels=class_labels_list,
            baby_noncry_weight_cap=baby_noncry_weight_cap
        )

        # Log class weights
        weight_info = ", ".join([f"{label}={self.class_weights[i]:.3f}" for i, label in enumerate(class_labels_list)])
        cap_info = f", baby_noncry_cap={baby_noncry_weight_cap}" if baby_noncry_weight_cap else ""
        logging.info(f"Class weights (cry_multiplier={cry_weight_multiplier}{cap_info}): {weight_info}")

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
        num_workers: int = None
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
        # Create weighted sampler for balanced training
        train_sampler = None
        if use_weighted_sampling and self.class_weights is not None:
            # Calculate sample weights
            sample_weights = []
            for _, label_str in train_dataset.audio_files:
                label_idx = train_dataset.label_to_idx[label_str]
                sample_weights.append(self.class_weights[label_idx].item())

            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

        # Create data loaders with custom collate function to preserve indices
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            sampler=train_sampler,
            shuffle=(train_sampler is None),  # Don't shuffle if using sampler
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_with_indices
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_with_indices
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_with_indices
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
                multi_class=self.config.MULTI_CLASS_MODE
            )

        # Create a temporary dataset to get input shape
        temp_dataset = BabyCryDataset([self.audio_files[0]], self.config, is_training=False, augment=False)
        sample_input, _ = temp_dataset[0]

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
            # Handle both old format (specs, labels) and new format (specs, labels, indices)
            if len(batch_data) == 3:
                spectrograms, labels, _ = batch_data
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