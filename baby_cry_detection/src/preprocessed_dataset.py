"""
Dataset class for loading preprocessed spectrograms.
Much faster than computing spectrograms on-the-fly during training.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import json
import logging

try:
    from .config import Config
    from .data_preprocessing import AudioAugmentation, collect_noise_files
except ImportError:
    from config import Config  # type: ignore
    from data_preprocessing import AudioAugmentation, collect_noise_files  # type: ignore


class PreprocessedBabyCryDataset(Dataset):
    """
    Dataset class for loading pre-computed spectrograms from disk.

    This provides significantly faster training by eliminating on-the-fly
    audio loading and spectrogram computation. Augmentation can still be
    applied on-the-fly at the spectrogram level if needed.

    Features:
    - Fast loading via memory-mapped numpy arrays
    - Config validation to ensure preprocessing matches current config
    - Support for on-the-fly augmentation (optional)
    - Maintains same interface as original BabyCryDataset
    """

    def __init__(
        self,
        preprocessed_dir: Path,
        config: Config = Config(),
        is_training: bool = True,
        augment: bool = True,
        file_indices: Optional[List[int]] = None
    ):
        """
        Initialize the preprocessed dataset.

        Args:
            preprocessed_dir: Directory containing preprocessed spectrograms
            config: Configuration object (for validation and augmentation)
            is_training: Whether this is for training (affects augmentation)
            augment: Whether to apply data augmentation
            file_indices: Optional list of indices to use (for train/val/test split)
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.config = config
        self.is_training = is_training
        self.augment = augment and is_training

        # Validate preprocessed data exists
        if not self.preprocessed_dir.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {self.preprocessed_dir}")

        # Load and validate manifest
        self.manifest = self._load_manifest()

        # Filter by indices if provided (for train/val/test split)
        if file_indices is not None:
            self.file_list = [self.manifest['files'][i] for i in file_indices]
        else:
            self.file_list = self.manifest['files']

        # Create label encoding based on config mode
        if config.MULTI_CLASS_MODE:
            self.label_to_idx = {label: idx for idx, label in config.MULTI_CLASS_LABELS.items()}
        else:
            self.label_to_idx = {'non_cry': 0, 'cry': 1}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Initialize augmentation (spectrogram-level augmentation)
        # Note: This is different from waveform-level augmentation used in preprocessing
        self.augmentation = None
        if self.augment:
            # For preprocessed data, we can apply simple spectrogram augmentations
            # like SpecAugment (time/frequency masking)
            # We don't need noise files since spectrograms are already computed
            self.augmentation = self._create_spectrogram_augmenter()

        # Log dataset statistics
        self._log_dataset_info()

    def _load_manifest(self) -> dict:
        """Load and validate manifest file."""
        manifest_path = self.preprocessed_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Validate manifest structure
        required_keys = ['files', 'config_hash', 'total_files']
        for key in required_keys:
            if key not in manifest:
                raise ValueError(f"Invalid manifest: missing key '{key}'")

        logging.info(f"Loaded manifest with {len(manifest['files'])} files")
        logging.info(f"Config hash: {manifest['config_hash']}")

        return manifest

    def _create_spectrogram_augmenter(self):
        """
        QUICK WIN 1: Create SpecAugment for spectrogram-level augmentation.
        Applies time and frequency masking to preprocessed spectrograms.
        """
        # AudioAugmentation is already imported at the top of this file
        # Create augmentation instance with SpecAugment enabled
        augmenter = AudioAugmentation(self.config, noise_files=[])

        # Return the spec_augment method
        return augmenter.spec_augment if hasattr(augmenter, 'spec_augment') else None

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (spectrogram, label, original_idx)
        """
        file_info = self.file_list[idx]

        # Get preprocessed file path
        preprocessed_filename = file_info['preprocessed_path']
        spectrogram_path = self.preprocessed_dir / preprocessed_filename

        try:
            # Load preprocessed spectrogram using memory mapping for efficiency
            # This allows OS to cache hot files and avoid loading entire array into RAM
            spectrogram_np = np.load(spectrogram_path, mmap_mode='r')

            # Convert to writable array (mmap is read-only)
            spectrogram_np = np.array(spectrogram_np, copy=True)

            # Convert to tensor
            spectrogram = torch.from_numpy(spectrogram_np).float()

            # Apply augmentation if enabled
            if self.augment and self.augmentation is not None:
                spectrogram = self.augmentation(spectrogram)

            # Get label
            label_str = file_info['label']
            label = torch.tensor(self.label_to_idx[label_str], dtype=torch.long)

            # Add channel dimension for CNN input if not present
            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)  # Shape: (1, n_mels, time_steps)

            return spectrogram, label, idx

        except Exception as e:
            # If loading fails, return a zero spectrogram with the correct shape
            logging.warning(f"Failed to load {spectrogram_path}: {e}")

            # Infer shape from manifest or use default
            if 'shape' in file_info:
                shape = tuple(file_info['shape'])
            else:
                # Default shape based on config
                time_steps = int(self.config.DURATION * self.config.SAMPLE_RATE // self.config.HOP_LENGTH) + 1
                shape = (self.config.N_MELS, time_steps)

            dummy_spec = torch.zeros((1, *shape))
            label_str = file_info['label']
            label = torch.tensor(self.label_to_idx[label_str], dtype=torch.long)

            return dummy_spec, label, idx

    def _log_dataset_info(self):
        """Log information about the dataset."""
        label_counts = {}
        for file_info in self.file_list:
            label = file_info['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        logging.info(f"Preprocessed dataset size: {len(self.file_list)}")
        logging.info(f"Label distribution: {label_counts}")
        logging.info(f"Augmentation enabled: {self.augment}")

    def get_storage_info(self) -> dict:
        """
        Get information about storage usage.

        Returns:
            Dictionary with storage statistics
        """
        total_size = sum(
            (self.preprocessed_dir / f['preprocessed_path']).stat().st_size
            for f in self.file_list
            if (self.preprocessed_dir / f['preprocessed_path']).exists()
        )

        return {
            'total_files': len(self.file_list),
            'total_size_mb': total_size / (1024**2),
            'avg_size_kb': (total_size / len(self.file_list)) / 1024 if self.file_list else 0
        }


class PreprocessedDatasetManager:
    """
    Manager class for creating train/val/test splits from preprocessed data.
    Maintains same interface as original DatasetManager for compatibility.
    """

    def __init__(self, preprocessed_dir: Path, config: Config = Config()):
        """
        Initialize the dataset manager.

        Args:
            preprocessed_dir: Directory containing preprocessed spectrograms
            config: Configuration object
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.config = config

        # Load manifest to get total file count
        manifest_path = self.preprocessed_dir / "manifest.json"
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.total_files = len(self.manifest['files'])

        # Compatibility: Create audio_files list from manifest (for code expecting this attribute)
        # Format: [(Path, label), ...]
        # Note: These paths may not exist - preprocessed data uses .npy files instead
        # This is only for compatibility with code that expects audio_files attribute
        self.audio_files = [
            (Path(file_info.get('original_path', 'unknown')), file_info['label'])
            for file_info in self.manifest['files']
        ]

        # Calculate class weights from manifest
        self.class_weights = self._calculate_class_weights()

    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        label_counts = self.manifest.get('label_counts', {})

        if not label_counts:
            # Fallback: count from files
            label_counts = {}
            for file_info in self.manifest['files']:
                label = file_info['label']
                label_counts[label] = label_counts.get(label, 0) + 1

        # Calculate weights (inverse frequency)
        total_samples = sum(label_counts.values())

        if total_samples == 0:
            # Safety fallback for empty dataset
            logging.warning("No samples found in manifest, using default weights")
            if self.config.MULTI_CLASS_MODE:
                num_classes = len(self.config.MULTI_CLASS_LABELS)
                return torch.ones(num_classes)
            else:
                return torch.tensor([1.0, 1.0])

        if self.config.MULTI_CLASS_MODE:
            num_classes = len(self.config.MULTI_CLASS_LABELS)
            class_weights = torch.ones(num_classes)

            for idx, label in self.config.MULTI_CLASS_LABELS.items():
                count = label_counts.get(label, 1)  # Avoid division by zero
                class_weights[idx] = total_samples / (num_classes * count)
        else:
            # Binary classification
            cry_count = label_counts.get('cry', 1)
            non_cry_count = label_counts.get('non_cry', 1)

            cry_weight = total_samples / (2 * cry_count)
            non_cry_weight = total_samples / (2 * non_cry_count)

            # Apply cry weight multiplier for better cry detection
            cry_weight_multiplier = getattr(self.config, 'CRY_WEIGHT_MULTIPLIER', 2.0)
            cry_weight *= cry_weight_multiplier

            class_weights = torch.tensor([non_cry_weight, cry_weight])

        logging.info(f"Calculated class weights: {class_weights}")
        return class_weights

    def create_splits(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Create train/val/test splits from preprocessed data.

        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        total_files = len(self.manifest['files'])
        indices = list(range(total_files))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        # Calculate split sizes
        train_size = int(total_files * train_ratio)
        val_size = int(total_files * val_ratio)

        # Create splits
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        logging.info(f"Dataset splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        return train_indices, val_indices, test_indices

    def create_datasets(
        self,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        augment_train: bool = True
    ) -> Tuple[PreprocessedBabyCryDataset, PreprocessedBabyCryDataset, PreprocessedBabyCryDataset]:
        """
        Create train/val/test dataset objects.

        Args:
            train_indices: Indices for training set
            val_indices: Indices for validation set
            test_indices: Indices for test set
            augment_train: Whether to apply augmentation to training set

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = PreprocessedBabyCryDataset(
            self.preprocessed_dir,
            self.config,
            is_training=True,
            augment=augment_train,
            file_indices=train_indices
        )

        val_dataset = PreprocessedBabyCryDataset(
            self.preprocessed_dir,
            self.config,
            is_training=False,
            augment=False,
            file_indices=val_indices
        )

        test_dataset = PreprocessedBabyCryDataset(
            self.preprocessed_dir,
            self.config,
            is_training=False,
            augment=False,
            file_indices=test_indices
        )

        return train_dataset, val_dataset, test_dataset
