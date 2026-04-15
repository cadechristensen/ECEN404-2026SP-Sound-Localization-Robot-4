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


def _compute_config_hash(config: Config) -> str:
    """Compute the preprocessing config hash for the given config.

    Mirrors ``preprocess_dataset.compute_config_hash`` exactly so the two
    values are always comparable.  Only preprocessing-relevant parameters are
    hashed; augmentation settings are intentionally excluded.

    Args:
        config: Configuration object.

    Returns:
        16-character hex hash string.
    """
    import hashlib

    config_dict = {
        "SAMPLE_RATE": config.SAMPLE_RATE,
        "DURATION": config.DURATION,
        "N_MELS": config.N_MELS,
        "N_FFT": config.N_FFT,
        "HOP_LENGTH": config.HOP_LENGTH,
        "F_MIN": config.F_MIN,
        "F_MAX": config.F_MAX,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def get_subcategory(original_path: str) -> str:
    """Extract the subcategory from a manifest original_path.

    Examples:
        'hard_negatives\\baby_noncry\\file.wav' -> 'baby_noncry'
        'hard_negatives/adult_scream/file.wav'  -> 'adult_scream'
        'cry_baby\\cry\\file.wav'               -> 'cry'
        'cry_baby/cry_ICSD/file.wav'            -> 'cry_ICSD'

    Returns:
        Subcategory string (directory name one level below the top-level folder).
    """
    parts = Path(original_path).parts
    if len(parts) > 1 and parts[0] in ('hard_negatives', 'cry_baby'):
        return parts[1]
    return parts[0]


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
        config: Optional[Config] = None,
        is_training: bool = True,
        augment: bool = True,
        file_indices: Optional[List[int]] = None
    ):
        """
        Initialize the preprocessed dataset.

        Args:
            preprocessed_dir: Directory containing preprocessed spectrograms
            config: Configuration object (for validation and augmentation). Defaults to Config().
            is_training: Whether this is for training (affects augmentation)
            augment: Whether to apply data augmentation
            file_indices: Optional list of indices to use (for train/val/test split)
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        if config is None:
            config = Config()
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

        # Binary classification label encoding
        self.label_to_idx = {'non_cry': 0, 'cry': 1}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Pre-compute per-sample loss weights from category (Phase 3).
        # Used to penalize hard-negative categories (baby_noncry, adult_scream) harder.
        category_loss_weights = getattr(config, 'CATEGORY_LOSS_WEIGHTS', {})
        self.sample_loss_weights: List[float] = []
        for file_info in self.file_list:
            original_path = file_info.get('original_path', '')
            cat = get_subcategory(original_path) if original_path else ''
            self.sample_loss_weights.append(category_loss_weights.get(cat, 1.0))

        # Initialize augmentation (spectrogram-level augmentation).
        # Note: This is different from waveform-level augmentation used in preprocessing.
        # We store the full AudioAugmentation object (not just a bound method) so the
        # owner is reachable, inspectable, and safely picklable by DataLoader workers (W6).
        self._augmentation_obj: Optional[AudioAugmentation] = None
        self.augmentation = None
        if self.augment:
            # For preprocessed data, we apply SpecAugment (time/frequency masking) only;
            # no noise files are needed — waveform mixing was done at preprocessing time.
            self._augmentation_obj = AudioAugmentation(self.config, noise_files=[])
            self.augmentation = self._augmentation_obj.spec_augment

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

        # Validate that the preprocessed data was generated with the current config.
        stored_hash = manifest["config_hash"]
        current_hash = _compute_config_hash(self.config)
        if stored_hash != current_hash:
            logging.warning(
                "Preprocessed data config hash mismatch. "
                f"Data was processed with hash {stored_hash} but current config hash is {current_hash}. "
                "Consider repreprocessing with: "
                "python scripts/preprocess_dataset.py --output data/processed/vNEXT"
            )

        return manifest

    @property
    def audio_files(self):
        """Compatibility property for evaluation error logging.

        Returns a list of (Path, label_idx) tuples matching the interface
        expected by evaluation/utils.py's generate_predictions_and_log_errors.
        """
        return [
            (Path(f.get('original_path', 'unknown')), self.label_to_idx.get(f['label'], 0))
            for f in self.file_list
        ]

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (spectrogram, label, original_idx, sample_loss_weight)
        """
        file_info = self.file_list[idx]

        # Get preprocessed file path
        preprocessed_filename = file_info['preprocessed_path']
        spectrogram_path = self.preprocessed_dir / preprocessed_filename

        # Per-sample loss weight for this category (Phase 3)
        sample_weight = self.sample_loss_weights[idx]

        try:
            # Load preprocessed spectrogram.  mmap_mode='r' gives the OS page-cache a
            # chance to satisfy repeated accesses without re-reading from disk, but the
            # np.array(..., copy=True) call below immediately copies all bytes into a
            # new heap array so we can pass it to torch without touching the mmap again.
            # The copy is required because mmap arrays are read-only and torch requires
            # a writable buffer (W9 — the previous comment incorrectly claimed "avoid
            # loading entire array into RAM").
            spectrogram_np = np.load(spectrogram_path, mmap_mode='r')
            spectrogram_np = np.array(spectrogram_np, copy=True)

            # Convert to tensor
            spectrogram = torch.from_numpy(spectrogram_np).float()

            # Add channel dimension before augmentation so spec_augment always receives
            # a 2-D (n_mels, time_steps) tensor regardless of whether the .npy was
            # saved with or without a channel dim (W8).
            if spectrogram.dim() == 2:
                spectrogram = spectrogram.unsqueeze(0)  # (1, n_mels, time_steps)
            # spec_augment expects (n_mels, time_steps) — squeeze channel for the call
            # then restore it afterwards.
            if self.augment and self._augmentation_obj is not None:
                spec_2d = spectrogram.squeeze(0)
                # Apply random duration simulation FIRST to break zero-padding bias.
                # This must precede SpecAugment because it re-normalizes the spectrogram.
                spec_2d = self._augmentation_obj.random_duration_simulation(spec_2d)
                # Then apply SpecAugment (time/frequency masking)
                spec_2d = self._augmentation_obj.spec_augment(spec_2d)
                spectrogram = spec_2d.unsqueeze(0)

            # Get label
            label_str = file_info['label']
            label = torch.tensor(self.label_to_idx[label_str], dtype=torch.long)

            return spectrogram, label, idx, sample_weight  # shape: (1, n_mels, time_steps)

        except Exception as e:
            # During training a zero spectrogram with a real label poisons the model
            # (it learns "silence == cry/non_cry").  Re-raise so the DataLoader surfaces
            # the failure.  During inference we fall back to a zero tensor so a single
            # bad cached file does not abort an evaluation run (C4).
            logging.error(f"Failed to load {spectrogram_path}: {e}")
            if self.is_training:
                raise

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

            return dummy_spec, label, idx, sample_weight

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
            Dictionary with storage statistics, including a count of missing files (W7).
        """
        present_sizes = []
        missing_count = 0
        for f in self.file_list:
            p = self.preprocessed_dir / f['preprocessed_path']
            if p.exists():
                present_sizes.append(p.stat().st_size)
            else:
                missing_count += 1

        if missing_count:
            logging.warning(
                f"get_storage_info: {missing_count}/{len(self.file_list)} preprocessed files missing"
            )

        total_size = sum(present_sizes)
        present_count = len(present_sizes)
        return {
            'total_files': len(self.file_list),
            'present_files': present_count,
            'missing_files': missing_count,
            'total_size_mb': total_size / (1024**2),
            # avg over present files only — missing files are excluded from the sum (W7)
            'avg_size_kb': (total_size / present_count) / 1024 if present_count else 0,
        }


class PreprocessedDatasetManager:
    """
    Manager class for creating train/val/test splits from preprocessed data.
    Maintains same interface as original DatasetManager for compatibility.
    """

    def __init__(self, preprocessed_dir: Path, config: Optional[Config] = None):
        """
        Initialize the dataset manager.

        Args:
            preprocessed_dir: Directory containing preprocessed spectrograms
            config: Configuration object. Defaults to Config().
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        if config is None:
            config = Config()
        self.config = config

        # Load manifest to get total file count — mirror the validation that
        # PreprocessedBabyCryDataset._load_manifest performs (C2).
        manifest_path = self.preprocessed_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        try:
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed manifest JSON at {manifest_path}: {exc}") from exc
        required_keys = ['files', 'config_hash', 'total_files']
        for key in required_keys:
            if key not in self.manifest:
                raise ValueError(f"Invalid manifest: missing key '{key}'")

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
            return torch.tensor([1.0, 1.0])

        # Binary classification
        cry_count = label_counts.get('cry', 1)
        non_cry_count = label_counts.get('non_cry', 1)

        cry_weight = total_samples / (2 * cry_count)
        non_cry_weight = total_samples / (2 * non_cry_count)

        # Apply cry weight multiplier from config.
        cry_weight *= self.config.CRY_WEIGHT_MULTIPLIER

        class_weights = torch.tensor([non_cry_weight, cry_weight])

        logging.info(f"Calculated class weights: {class_weights}")
        return class_weights

    def compute_sample_weights(self, file_indices: List[int]) -> List[float]:
        """Compute per-sample sampling weights for WeightedRandomSampler.

        Uses CATEGORY_SAMPLING_WEIGHTS from config to oversample hard-negative
        categories (e.g., baby_noncry 4x, adult_scream 3x).

        Args:
            file_indices: Indices into manifest['files'] for the training split.

        Returns:
            List of float weights, one per sample in file_indices.
        """
        category_weights = getattr(self.config, 'CATEGORY_SAMPLING_WEIGHTS', {})
        weights = []
        category_counts: Dict[str, int] = {}
        for idx in file_indices:
            original_path = self.manifest['files'][idx].get('original_path', '')
            cat = get_subcategory(original_path) if original_path else ''
            w = category_weights.get(cat, 1.0)
            weights.append(w)
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Log category distribution for visibility
        upsampled = {cat: cnt for cat, cnt in category_counts.items()
                     if category_weights.get(cat, 1.0) > 1.0}
        if upsampled:
            logging.info(f"WeightedRandomSampler category oversampling: "
                         f"{', '.join(f'{cat} ({cnt} samples, {category_weights[cat]}x)' for cat, cnt in upsampled.items())}")

        return weights

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

        # Extract labels for stratified splitting
        labels = [self.manifest['files'][i]['label'] for i in indices]

        try:
            from sklearn.model_selection import train_test_split

            # Stratified split: train vs (val+test)
            train_indices, temp_indices, train_labels, temp_labels = train_test_split(
                indices, labels,
                test_size=(val_ratio + test_ratio),
                random_state=random_seed,
                stratify=labels
            )

            # Stratified split: val vs test
            relative_test_ratio = test_ratio / (val_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=relative_test_ratio,
                random_state=random_seed,
                stratify=temp_labels
            )

            logging.info("Using stratified splitting for consistent class ratios across splits")

        except ImportError:
            logging.warning("sklearn not available — falling back to random (non-stratified) split")
            if shuffle:
                rng = np.random.default_rng(random_seed)
                rng.shuffle(indices)

            train_size = int(total_files * train_ratio)
            val_size = int(total_files * val_ratio)
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
