"""
Data preprocessing module for baby cry detection.
Handles audio loading, feature extraction using log-mel spectrograms,
and data augmentation techniques.
"""

import random
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import warnings
import logging

try:
    from .config import Config
except ImportError:
    from config import Config  # type: ignore

# AudioFilteringPipeline is not currently available
AudioFilteringPipeline = None

class AudioPreprocessor:
    """
    Audio preprocessing class for baby cry detection.
    Handles loading, resampling, and feature extraction from audio files.
    """

    def __init__(self, config: Optional[Config] = None, use_advanced_filtering: bool = True):
        """
        Initialize the audio preprocessor.

        Args:
            config: Configuration object containing processing parameters. Defaults to Config().
            use_advanced_filtering: Whether to use advanced filtering techniques (VAD, noise reduction)
        """
        if config is None:
            config = Config()
        self.config = config
        self.sample_rate = config.SAMPLE_RATE
        self.duration = config.DURATION
        self.n_mels = config.N_MELS
        self.n_fft = config.N_FFT
        self.hop_length = config.HOP_LENGTH
        self.f_min = config.F_MIN
        self.f_max = config.F_MAX
        self.use_advanced_filtering = use_advanced_filtering

        # Initialize mel-spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        )

        # Initialize amplitude to decibel transform
        self.amplitude_to_db = T.AmplitudeToDB()

        # Cache T.Resample objects keyed by original sample rate to avoid
        # re-constructing the resampler on every load_audio call (W2).
        self._resamplers: Dict[int, T.Resample] = {}

        # Initialize advanced filtering pipeline
        self.filtering_pipeline = None
        if use_advanced_filtering and AudioFilteringPipeline is not None:
            try:
                self.filtering_pipeline = AudioFilteringPipeline(config)
            except Exception as e:
                logging.warning(f"Failed to initialize filtering pipeline: {e}")
                self.filtering_pipeline = None

    def load_audio(self, file_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample to target sample rate.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_tensor, original_sample_rate)
        """
        try:
            # Use torchaudio for efficient loading
            waveform, original_sr = torchaudio.load(str(file_path))

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary — reuse cached T.Resample to avoid
            # re-building the filter bank on every call (W2).
            if original_sr != self.sample_rate:
                if original_sr not in self._resamplers:
                    self._resamplers[original_sr] = T.Resample(original_sr, self.sample_rate)
                waveform = self._resamplers[original_sr](waveform)

            return waveform.squeeze(0), original_sr

        except Exception as e:
            # Fallback to librosa for formats not supported by torchaudio
            try:
                waveform, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
                return torch.tensor(waveform, dtype=torch.float32), sr
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio file {file_path}: {e2}")

    def pad_or_truncate(self, waveform: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Pad or truncate audio to fixed duration.

        Args:
            waveform: Input audio waveform
            training: If True, use random crop for data augmentation; if False
                      use deterministic center crop so TTA passes are comparable (W3).

        Returns:
            Fixed-length audio waveform
        """
        target_length = int(self.sample_rate * self.duration)

        if len(waveform) > target_length:
            if training:
                # Random crop during training for augmentation diversity
                start_idx = np.random.randint(0, len(waveform) - target_length + 1)
            else:
                # Deterministic center crop during inference / TTA
                start_idx = (len(waveform) - target_length) // 2
            waveform = waveform[start_idx:start_idx + target_length]
        elif len(waveform) < target_length:
            # Pad with zeros
            padding = target_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform

    def extract_log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-mel spectrogram from audio waveform.

        Args:
            waveform: Input audio waveform

        Returns:
            Log-mel spectrogram tensor
        """
        # Ensure input is 2D (batch_size=1, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale (dB)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # Calculate expected time steps for consistency
        expected_time_steps = int(self.duration * self.sample_rate // self.hop_length) + 1

        # Clamp non-finite values that arise from silent/zero-energy audio before
        # any indexing.  AmplitudeToDB of 0 → -inf; log_mel_spec.min() on such a
        # tensor returns -inf and torch.nn.functional.pad propagates it everywhere,
        # producing an all-NaN spectrogram (C4).
        log_mel_spec = torch.nan_to_num(log_mel_spec, nan=0.0, posinf=80.0, neginf=-80.0)

        # Pad or truncate the spectrogram to ensure consistent size
        current_time_steps = log_mel_spec.shape[-1]
        if current_time_steps < expected_time_steps:
            # Pad with a fixed silence constant rather than the per-sample minimum.
            # Using log_mel_spec.min() would leak audio content into the padding region,
            # giving the model a spurious per-sample feature to learn.  -80.0 dB is safe
            # because nan_to_num above already clamps neginf to -80.0, so the padding
            # value is indistinguishable from the quietest legitimate energy bin.
            padding = expected_time_steps - current_time_steps
            pad_value = -80.0
            log_mel_spec = torch.nn.functional.pad(
                log_mel_spec, (0, padding), mode='constant', value=pad_value
            )
        elif current_time_steps > expected_time_steps:
            # Truncate
            log_mel_spec = log_mel_spec[:, :, :expected_time_steps]

        # Per-sample instance normalisation to zero-mean unit-variance.
        # NOTE (W4): this makes each sample's statistics dependent only on itself,
        # so TTA augmented views will have slightly different means/stds.  To make
        # TTA logits fully comparable, compute normalization statistics on the
        # un-augmented waveform before augmenting and pass them in — or switch to
        # global dataset statistics stored in config.
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)

        return log_mel_spec.squeeze(0)  # Remove batch dimension

    def process_audio_file(
        self,
        file_path: Union[str, Path],
        apply_filtering: bool = None,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Complete preprocessing pipeline for a single audio file.

        Args:
            file_path: Path to audio file
            apply_filtering: Whether to apply advanced filtering (overrides default)
            training: If True, pad_or_truncate uses a random crop (augmentation
                diversity during training).  If False (default), uses a
                deterministic center crop so inference and TTA are reproducible.

        Returns:
            Processed log-mel spectrogram
        """
        # Load audio
        waveform, _ = self.load_audio(file_path)

        # Pad or truncate to fixed length
        waveform = self.pad_or_truncate(waveform, training=training)

        # Apply advanced filtering if enabled
        if apply_filtering is None:
            apply_filtering = self.use_advanced_filtering

        if apply_filtering and self.filtering_pipeline is not None:
            try:
                # Apply noise filtering (VAD optional for training, keep full duration)
                filtered_result = self.filtering_pipeline.preprocess_audio(
                    waveform,
                    apply_vad=False,  # Don't segment during training
                    apply_filtering=True,
                    extract_deep_features=False
                )
                waveform = filtered_result['filtered']
            except Exception as e:
                logging.debug(f"Filtering failed for {file_path}, using original audio: {e}")

        # Extract log-mel spectrogram
        log_mel_spec = self.extract_log_mel_spectrogram(waveform)

        return log_mel_spec


class AudioAugmentation:
    """
    Audio augmentation class for data augmentation during training.
    Implements various augmentation techniques to improve model robustness.
    """

    def __init__(self, config: Optional[Config] = None, noise_files: Optional[List[Path]] = None):
        """
        Initialize audio augmentation.

        Args:
            config: Configuration object containing augmentation parameters. Defaults to Config().
            noise_files: List of noise file paths for background mixing
        """
        if config is None:
            config = Config()
        self.config = config
        self.noise_factor = config.NOISE_FACTOR
        self.time_stretch_range = config.TIME_STRETCH_RATE
        self.pitch_shift_range = config.PITCH_SHIFT_STEPS
        self.sample_rate = config.SAMPLE_RATE
        self.noise_files = noise_files or []

        # QUICK WIN 1: SpecAugment parameters
        self.use_spec_augment = getattr(config, 'USE_SPEC_AUGMENT', False)
        self.time_mask_param = getattr(config, 'SPEC_AUG_TIME_MASK_PARAM', 20)
        self.freq_mask_param = getattr(config, 'SPEC_AUG_FREQ_MASK_PARAM', 15)
        self.num_time_masks = getattr(config, 'SPEC_AUG_NUM_TIME_MASKS', 2)
        self.num_freq_masks = getattr(config, 'SPEC_AUG_NUM_FREQ_MASKS', 2)

        # Load noise files for background mixing.
        # Inline loading with torchaudio avoids creating a throwaway AudioPreprocessor
        # and its associated MelSpectrogram/AmplitudeToDB transforms (S5).
        self.noise_waveforms = []
        if self.noise_files:
            _noise_resamplers: Dict[int, T.Resample] = {}
            for noise_file in self.noise_files:
                try:
                    waveform, original_sr = torchaudio.load(str(noise_file))
                    # Mono downmix
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    # Resample if needed, reusing cached resamplers
                    if original_sr != config.SAMPLE_RATE:
                        if original_sr not in _noise_resamplers:
                            _noise_resamplers[original_sr] = T.Resample(original_sr, config.SAMPLE_RATE)
                        waveform = _noise_resamplers[original_sr](waveform)
                    waveform = waveform.squeeze(0)
                    # Ensure noise is long enough for mixing
                    if len(waveform) > config.SAMPLE_RATE * config.DURATION:
                        self.noise_waveforms.append(waveform)
                except Exception:
                    continue  # Skip problematic noise files

    def add_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to the waveform.

        Args:
            waveform: Input audio waveform

        Returns:
            Augmented waveform with added noise
        """
        noise = torch.randn_like(waveform) * self.noise_factor
        return waveform + noise

    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply time stretching to the waveform.

        Args:
            waveform: Input audio waveform

        Returns:
            Time-stretched waveform
        """
        stretch_factor = np.random.uniform(
            self.time_stretch_range[0],
            self.time_stretch_range[1]
        )

        # Convert to numpy for librosa processing — detach() + cpu() required
        # before .numpy() to handle gradient-tracked and/or CUDA tensors (C1).
        waveform_np = waveform.detach().cpu().numpy()
        stretched = librosa.effects.time_stretch(waveform_np, rate=stretch_factor)

        return torch.tensor(stretched, dtype=torch.float32)

    def pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply pitch shifting to the waveform.

        Args:
            waveform: Input audio waveform

        Returns:
            Pitch-shifted waveform
        """
        n_steps = np.random.randint(
            self.pitch_shift_range[0],
            self.pitch_shift_range[1] + 1
        )

        # Convert to numpy for librosa processing — detach() + cpu() required
        # before .numpy() to handle gradient-tracked and/or CUDA tensors (C1).
        waveform_np = waveform.detach().cpu().numpy()
        shifted = librosa.effects.pitch_shift(
            waveform_np,
            sr=self.sample_rate,
            n_steps=n_steps
        )

        return torch.tensor(shifted, dtype=torch.float32)

    def add_background_noise(self, waveform: torch.Tensor, noise_level: float = None) -> torch.Tensor:
        """
        Add realistic multi-source background noise to simulate household conditions.

        Args:
            waveform: Input audio waveform (cry)
            noise_level: Mixing level for background noise (0.0 to 1.0). If None, uses variable SNR.

        Returns:
            Waveform with background noise mixed in
        """
        if not self.noise_waveforms:
            return waveform

        # Variable SNR: randomly vary the noise level to simulate different household conditions
        # Lower noise_level = louder cry relative to background (high SNR, quiet household)
        # Higher noise_level = quieter cry relative to background (low SNR, noisy household)
        if noise_level is None:
            noise_level = np.random.uniform(0.05, 0.3)  # Variable household noise levels

        # Multi-source noise mixing: randomly choose 1-3 noise sources
        num_sources = np.random.randint(1, 4)  # 1, 2, or 3 noise sources

        target_length = len(waveform)
        mixed_noise = torch.zeros_like(waveform)

        # Mix multiple noise sources together
        for _ in range(num_sources):
            # random.choice handles a list of Tensors correctly; np.random.choice
            # tries to convert the list to an array and raises ValueError on
            # Python 3.10+ / NumPy 1.24+ when elements are non-scalar Tensors (C2).
            noise_waveform = random.choice(self.noise_waveforms)

            # Extract a random segment from noise that matches waveform length
            if len(noise_waveform) <= target_length:
                # If noise is shorter, repeat it
                repeats = (target_length // len(noise_waveform)) + 1
                noise_segment = noise_waveform.repeat(repeats)[:target_length]
            else:
                # Extract random segment of required length
                start_idx = np.random.randint(0, len(noise_waveform) - target_length)
                noise_segment = noise_waveform[start_idx:start_idx + target_length]

            # Normalize each noise segment
            noise_segment = noise_segment / (torch.max(torch.abs(noise_segment)) + 1e-6)

            # Add to mixed noise with random weighting for each source
            source_weight = np.random.uniform(0.3, 1.0)
            mixed_noise += noise_segment * source_weight

        # Normalize combined noise
        mixed_noise = mixed_noise / (torch.max(torch.abs(mixed_noise)) + 1e-6)

        # Mix noise with original waveform
        mixed_waveform = waveform + (mixed_noise * noise_level)

        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(mixed_waveform))
        if max_val > 1.0:
            mixed_waveform = mixed_waveform / max_val

        return mixed_waveform

    def add_room_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add simple room reverb/echo to simulate household acoustics.

        Args:
            waveform: Input audio waveform

        Returns:
            Waveform with reverb applied
        """
        # Simple reverb using delayed copies with decay
        # Simulates sound reflections in a room
        delay_samples = int(0.05 * self.sample_rate)  # 50ms delay (small room)
        decay = 0.3  # Reverb decay factor

        # Create delayed and attenuated copies.
        # Guard against waveforms shorter than the delay — the slice
        # reverb[:len(waveform) - delay_samples] would have a negative stop
        # index when len(waveform) <= delay_samples (W5).
        reverb = torch.zeros_like(waveform)
        if len(waveform) > delay_samples:
            reverb[:len(waveform) - delay_samples] = waveform[delay_samples:] * decay

        # Add second reflection
        delay_samples_2 = int(0.08 * self.sample_rate)  # 80ms delay
        if len(waveform) > delay_samples_2:
            reverb[:len(waveform) - delay_samples_2] += waveform[delay_samples_2:] * (decay * 0.5)

        # Mix original with reverb
        reverb_waveform = waveform + reverb

        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(reverb_waveform))
        if max_val > 1.0:
            reverb_waveform = reverb_waveform / max_val

        return reverb_waveform

    def spec_augment(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        QUICK WIN 1: Apply SpecAugment (time and frequency masking) to mel-spectrogram.
        Based on Park et al. 2019 "SpecAugment: A Simple Data Augmentation Method for ASR".

        Args:
            mel_spectrogram: Input mel-spectrogram of shape (n_mels, time_steps)

        Returns:
            Augmented mel-spectrogram with masked regions
        """
        if not self.use_spec_augment:
            return mel_spectrogram

        mel_spec = mel_spectrogram.clone()
        n_mels, time_steps = mel_spec.shape

        # Apply frequency masking.
        # Clamp mask width to n_mels so that (n_mels - f) is always >= 1,
        # preventing a ValueError when freq_mask_param >= n_mels (C3).
        for _ in range(self.num_freq_masks):
            f = min(int(np.random.uniform(0, self.freq_mask_param)), n_mels - 1)
            f0 = int(np.random.uniform(0, max(1, n_mels - f)))
            mel_spec[f0:f0 + f, :] = 0

        # Apply time masking.
        # Same guard: clamp to time_steps - 1 to keep the upper bound positive (C3).
        for _ in range(self.num_time_masks):
            t = min(int(np.random.uniform(0, self.time_mask_param)), time_steps - 1)
            t0 = int(np.random.uniform(0, max(1, time_steps - t)))
            mel_spec[:, t0:t0 + t] = 0

        return mel_spec

    def random_duration_simulation(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Simulate variable-length audio at the spectrogram level.

        Randomly masks the rightmost portion of the spectrogram (simulating
        zero-padding from short audio) and re-normalizes.  This breaks the
        spurious correlation between zero-padding and the cry class: 77% of
        cry training samples are <3s and get zero-padded vs only 5% of non-cry.

        Applied to BOTH classes with equal probability so the model learns to
        detect cry features, not the padding pattern.

        Args:
            spectrogram: Input spectrogram of shape (n_mels, time_steps),
                         already z-score normalized from preprocessing.

        Returns:
            Augmented spectrogram with simulated duration variation.
        """
        use_aug = getattr(self.config, 'USE_DURATION_AUGMENT', True)
        if not use_aug:
            return spectrogram

        prob = getattr(self.config, 'DURATION_SIM_PROBABILITY', 0.5)
        if random.random() > prob:
            return spectrogram

        n_mels, time_steps = spectrogram.shape
        min_keep = getattr(self.config, 'DURATION_SIM_MIN_KEEP', 0.30)
        max_keep = getattr(self.config, 'DURATION_SIM_MAX_KEEP', 0.85)

        keep_fraction = random.uniform(min_keep, max_keep)
        keep_steps = max(1, int(time_steps * keep_fraction))

        # Mask the right portion with a fixed constant rather than the per-sample
        # signal minimum.  Using the signal minimum leaks audio content into the
        # masked region.  The spectrogram is already z-score normalized at this
        # point, so 0.0 (the distribution mean) is the natural silence surrogate
        # and is consistent across all samples.
        masked = spectrogram.clone()
        masked[:, keep_steps:] = 0.0

        # Re-normalize to zero-mean unit-variance (matching the training pipeline)
        std = masked.std()
        if std > 1e-8:
            masked = (masked - masked.mean()) / std

        return masked

    def random_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentation to the waveform.
        Enhanced for realistic household conditions.

        Args:
            waveform: Input audio waveform

        Returns:
            Randomly augmented waveform
        """
        # Randomly choose augmentations to apply
        augmentations = []

        if np.random.random() > 0.5:
            augmentations.append(self.add_gaussian_noise)

        if np.random.random() > 0.7:
            augmentations.append(self.time_stretch)

        if np.random.random() > 0.7:
            augmentations.append(self.pitch_shift)

        # Add room reverb to simulate household acoustics (40% probability)
        if np.random.random() > 0.6:
            augmentations.append(self.add_room_reverb)

        # Add background noise mixing for cry samples (INCREASED to 90% for household realism)
        if np.random.random() > 0.1:
            augmentations.append(self.add_background_noise)

        # Apply selected augmentations
        augmented_waveform = waveform.clone()
        for aug_func in augmentations:
            try:
                augmented_waveform = aug_func(augmented_waveform)
            except Exception as e:
                # Skip augmentation if it fails
                continue

        return augmented_waveform


def collect_audio_files(data_dir: Path, supported_formats: List[str]) -> List[Tuple[Path, str]]:
    """
    Collect all audio files from the data directory.
    Supports recursive searching for cry_baby and hard_negatives directories.
    Binary classification only: cry vs non_cry.

    Args:
        data_dir: Path to data directory
        supported_formats: List of supported audio formats

    Returns:
        List of tuples (file_path, label)
    """
    audio_files = []

    # Define label mapping based on directory structure
    # NOTE: 'noise' directory is excluded - noise files are used for augmentation, not as training labels

    # Binary mode: cry vs non_cry
    label_mapping = {
        'cry': 'cry',                    # Baby cry sounds (Donate-a-Cry, Hugging Face, Kaggle)
        'cry_ICSD': 'cry',               # ICSD baby cry real strong labeled samples
        'cry_crycaleb': 'cry',           # CryCeleb2023 baby cry samples
        'baby_noncry': 'non_cry',        # Non-cry baby sounds (babbling, laughing, cooing, silence)
        'adult_speech': 'non_cry',       # Adult speech/conversation (LibriSpeech)
        'environmental': 'non_cry',      # Environmental sounds (ESC-50)
        # 'noise' intentionally excluded - these are for background augmentation only
    }

    nested_dir_rules = {
        'cry_baby': 'cry',  # All files in cry_baby/* are cry samples
    }

    # ONLY collect from cry_baby and hard_negatives folders
    # All other folders (cry, adult_speech, baby_noncry, environmental, noise) are IGNORED
    for subdir in data_dir.iterdir():
        if not subdir.is_dir():
            continue

        dir_name = subdir.name

        # ONLY process cry_baby and hard_negatives directories
        if dir_name == 'cry_baby':
            # All files under cry_baby/* are cry samples
            label = nested_dir_rules['cry_baby']
            for file_path in subdir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                    audio_files.append((file_path, label))

        elif dir_name == 'hard_negatives':
            # All hard_negatives are non_cry in binary classification
            for file_path in subdir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                    audio_files.append((file_path, 'non_cry'))

        # All other directories (cry, adult_speech, baby_noncry, environmental, noise) are IGNORED

    # Deduplicate by filename stem: if the same base filename appears under multiple
    # subdirectories (e.g. 547190_Voice_AdultMale_DeathScream_09.mp3 in both adult_shout
    # and adult_scream), keep only the first occurrence.  Duplicate files with conflicting
    # labels create irreconcilable label noise that degrades model calibration.
    seen_stems: dict = {}
    deduplicated: List[Tuple[Path, str]] = []
    for file_path, label in audio_files:
        stem = file_path.name  # Use full filename (stem + ext) as key
        if stem in seen_stems:
            existing_path, existing_label = seen_stems[stem]
            if existing_label != label:
                logging.warning(
                    f"Duplicate filename with conflicting labels — keeping first occurrence:\n"
                    f"  KEPT:    {existing_path} (label='{existing_label}')\n"
                    f"  DROPPED: {file_path} (label='{label}')"
                )
            else:
                logging.debug(f"Duplicate filename (same label) dropped: {file_path}")
        else:
            seen_stems[stem] = (file_path, label)
            deduplicated.append((file_path, label))

    if len(deduplicated) < len(audio_files):
        logging.info(
            f"Deduplication: removed {len(audio_files) - len(deduplicated)} duplicate "
            f"files from {len(audio_files)} total. Using {len(deduplicated)} files."
        )

    return deduplicated


def collect_noise_files(data_dir: Path, supported_formats: List[str]) -> List[Path]:
    """
    Collect noise files for data augmentation.

    Args:
        data_dir: Path to data directory
        supported_formats: List of supported audio formats

    Returns:
        List of noise file paths
    """
    noise_files = []
    noise_dir = data_dir / 'noise'

    if noise_dir.exists() and noise_dir.is_dir():
        for file_path in noise_dir.iterdir():
            if file_path.suffix.lower() in supported_formats:
                noise_files.append(file_path)

    return noise_files


def get_class_weights(audio_files: List[Tuple[Path, str]],
                     cry_weight_multiplier: float = Config.CRY_WEIGHT_MULTIPLIER,
                     class_labels: List[str] = None) -> torch.Tensor:
    """
    Calculate class weights for balanced training with emphasis on cry detection.

    Args:
        audio_files: List of (file_path, label) tuples
        cry_weight_multiplier: Multiplier for cry class weight (default Config.CRY_WEIGHT_MULTIPLIER).
            Previously hardcoded to 2.0 which contradicted config.py (S4).
        class_labels: List of class labels in order. If None, defaults to ['non_cry', 'cry']

    Returns:
        Class weights tensor
    """
    # Default to binary classification if not specified
    if class_labels is None:
        class_labels = ['non_cry', 'cry']

    # Count samples per class
    class_counts = {}
    for _, label in audio_files:
        class_counts[label] = class_counts.get(label, 0) + 1

    # Calculate inverse frequency weights
    total_samples = sum(class_counts.values())
    num_classes = len(class_labels)
    weights = []

    for class_label in class_labels:
        if class_label in class_counts:
            weight = total_samples / (num_classes * class_counts[class_label])
            # Apply multiplier to cry class to penalize missed cries more heavily
            if class_label == 'cry':
                weight *= cry_weight_multiplier
        else:
            weight = 1.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)