"""
Data preprocessing module for baby cry detection.
Handles audio loading, feature extraction using log-mel spectrograms,
and data augmentation techniques.
"""

import os
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Tuple, List, Optional, Union
import warnings
import logging
warnings.filterwarnings("ignore")

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

    def __init__(self, config: Config = Config(), use_advanced_filtering: bool = True):
        """
        Initialize the audio preprocessor.

        Args:
            config: Configuration object containing processing parameters
            use_advanced_filtering: Whether to use advanced filtering techniques (VAD, noise reduction)
        """
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

            # Resample if necessary
            if original_sr != self.sample_rate:
                resampler = T.Resample(original_sr, self.sample_rate)
                waveform = resampler(waveform)

            return waveform.squeeze(0), original_sr

        except Exception as e:
            # Fallback to librosa for formats not supported by torchaudio
            try:
                waveform, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
                return torch.tensor(waveform, dtype=torch.float32), sr
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio file {file_path}: {e2}")

    def pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Pad or truncate audio to fixed duration.

        Args:
            waveform: Input audio waveform

        Returns:
            Fixed-length audio waveform
        """
        target_length = int(self.sample_rate * self.duration)

        if len(waveform) > target_length:
            # Randomly crop from the audio
            start_idx = np.random.randint(0, len(waveform) - target_length + 1)
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

        # Pad or truncate the spectrogram to ensure consistent size
        current_time_steps = log_mel_spec.shape[-1]
        if current_time_steps < expected_time_steps:
            # Pad with minimum value
            padding = expected_time_steps - current_time_steps
            log_mel_spec = torch.nn.functional.pad(
                log_mel_spec, (0, padding), mode='constant', value=log_mel_spec.min()
            )
        elif current_time_steps > expected_time_steps:
            # Truncate
            log_mel_spec = log_mel_spec[:, :, :expected_time_steps]

        # Normalize to [-1, 1] range
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)

        return log_mel_spec.squeeze(0)  # Remove batch dimension

    def process_audio_file(self, file_path: Union[str, Path], apply_filtering: bool = None) -> torch.Tensor:
        """
        Complete preprocessing pipeline for a single audio file.

        Args:
            file_path: Path to audio file
            apply_filtering: Whether to apply advanced filtering (overrides default)

        Returns:
            Processed log-mel spectrogram
        """
        # Load audio
        waveform, _ = self.load_audio(file_path)

        # Pad or truncate to fixed length
        waveform = self.pad_or_truncate(waveform)

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

    def __init__(self, config: Config = Config(), noise_files: Optional[List[Path]] = None):
        """
        Initialize audio augmentation.

        Args:
            config: Configuration object containing augmentation parameters
            noise_files: List of noise file paths for background mixing
        """
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

        # Load noise files for background mixing
        self.noise_waveforms = []
        if self.noise_files:
            preprocessor = AudioPreprocessor(config)
            for noise_file in self.noise_files:
                try:
                    waveform, _ = preprocessor.load_audio(noise_file)
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

        # Convert to numpy for librosa processing
        waveform_np = waveform.numpy()
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

        # Convert to numpy for librosa processing
        waveform_np = waveform.numpy()
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
            noise_waveform = np.random.choice(self.noise_waveforms)

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

        # Create delayed and attenuated copies
        reverb = torch.zeros_like(waveform)
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

        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            f = int(np.random.uniform(0, self.freq_mask_param))
            f0 = int(np.random.uniform(0, n_mels - f))
            mel_spec[f0:f0 + f, :] = 0

        # Apply time masking
        for _ in range(self.num_time_masks):
            t = int(np.random.uniform(0, self.time_mask_param))
            t0 = int(np.random.uniform(0, time_steps - t))
            mel_spec[:, t0:t0 + t] = 0

        return mel_spec

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


def collect_audio_files(data_dir: Path, supported_formats: List[str], multi_class: bool = False) -> List[Tuple[Path, str]]:
    """
    Collect all audio files from the data directory.
    Supports recursive searching for cry_baby and hard_negatives directories.

    Args:
        data_dir: Path to data directory
        supported_formats: List of supported audio formats
        multi_class: If True, use multi-class labels; if False, use binary labels

    Returns:
        List of tuples (file_path, label)
    """
    audio_files = []

    # Define label mapping based on directory structure
    # NOTE: 'noise' directory is excluded - noise files are used for augmentation, not as training labels

    if multi_class:
        # Multi-class mode: cry, adult_speech, baby_noncry, environmental
        label_mapping = {
            'cry': 'cry',                    # Baby cry sounds (Donate-a-Cry, Hugging Face, Kaggle)
            'cry_ICSD': 'cry',               # ICSD baby cry real strong labeled samples
            'cry_crycaleb': 'cry',           # CryCeleb2023 baby cry samples
            'baby_noncry': 'baby_noncry',    # Non-cry baby sounds (babbling, laughing, cooing)
            'adult_speech': 'adult_speech',  # Adult speech/conversation (LibriSpeech)
            'environmental': 'environmental', # Environmental sounds (ESC-50)
            # 'noise' intentionally excluded - these are for background augmentation only
        }

        # Special handling for nested directories
        # cry_baby subdirectories all map to 'cry'
        # hard_negatives subdirectories map based on their subdirectory names
        nested_dir_rules = {
            'cry_baby': 'cry',  # All files in cry_baby/* are cry samples
        }

        # Map hard_negatives subdirectories to appropriate classes
        hard_negatives_mapping = {
            'adult_speech': 'adult_speech',
            'adult_scream': 'adult_speech',    # Hard negative: sounds like cry but is adult
            'adult_shout': 'adult_speech',     # Hard negative: sounds like cry but is adult
            'baby_noncry': 'baby_noncry',
            'child_tantrum': 'baby_noncry',    # Hard negative: child distress but not baby cry
            'music_vocal': 'environmental',     # Hard negative: vocal music can resemble cries
            'silence': 'environmental',         # Important negative: periods of silence
            # All other environmental sounds
            'airplane': 'environmental',
            'breathing': 'environmental',
            'brushing_teeth': 'environmental',
            'can_opening': 'environmental',
            'car_horn': 'environmental',
            'cat': 'environmental',
            'chainsaw': 'environmental',
            'chirping_birds': 'environmental',
            'church_bells': 'environmental',
            'clapping': 'environmental',
            'clock_alarm': 'environmental',
            'clock_tick': 'environmental',
            'coughing': 'environmental',
            'cow': 'environmental',
            'crackling_fire': 'environmental',
            'crickets': 'environmental',
            'crow': 'environmental',
            'dog': 'environmental',
            'door_wood_creaks': 'environmental',
            'door_wood_knock': 'environmental',
            'drinking_sipping': 'environmental',
            'engine': 'environmental',
            'fireworks': 'environmental',
            'footsteps': 'environmental',
            'frog': 'environmental',
            'glass_breaking': 'environmental',
            'hand_saw': 'environmental',
            'helicopter': 'environmental',
            'hen': 'environmental',
            'insects': 'environmental',
            'keyboard_typing': 'environmental',
            'laughing': 'environmental',
            'mouse_click': 'environmental',
            'pig': 'environmental',
            'pouring_water': 'environmental',
            'rain': 'environmental',
            'rooster': 'environmental',
            'sea_waves': 'environmental',
            'sheep': 'environmental',
            'siren': 'environmental',
            'sneezing': 'environmental',
            'snoring': 'environmental',
            'thunderstorm': 'environmental',
            'toilet_flush': 'environmental',
            'train': 'environmental',
            'vacuum_cleaner': 'environmental',
            'washing_machine': 'environmental',
            'water_drops': 'environmental',
            'wind': 'environmental',
        }
    else:
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

        # In binary mode, all hard_negatives map to non_cry
        hard_negatives_mapping = {
            # All subdirectories map to non_cry in binary mode
        }

    # Helper function to recursively collect files from nested directories
    def collect_from_nested_dir(parent_dir: Path, subdirs_mapping: dict):
        """Recursively collect files from nested directories with specific label mappings."""
        collected = []
        for subdir in parent_dir.iterdir():
            if not subdir.is_dir():
                continue

            subdir_name = subdir.name
            if subdir_name in subdirs_mapping:
                label = subdirs_mapping[subdir_name]
                # Recursively find all audio files in this subdirectory
                for file_path in subdir.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                        collected.append((file_path, label))
        return collected

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
            # Map based on subdirectory names
            if multi_class:
                audio_files.extend(collect_from_nested_dir(subdir, hard_negatives_mapping))
            else:
                # In binary mode, all hard_negatives are non_cry
                for file_path in subdir.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                        audio_files.append((file_path, 'non_cry'))

        # All other directories (cry, adult_speech, baby_noncry, environmental, noise) are IGNORED

    return audio_files


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


def get_class_weights(audio_files: List[Tuple[Path, str]], cry_weight_multiplier: float = 2.0,
                     class_labels: List[str] = None, baby_noncry_weight_cap: float = None) -> torch.Tensor:
    """
    Calculate class weights for balanced training with emphasis on cry detection.

    Args:
        audio_files: List of (file_path, label) tuples
        cry_weight_multiplier: Multiplier for cry class weight (default 2.0 to penalize missed cries more)
        class_labels: List of class labels in order. If None, defaults to ['non_cry', 'cry']
        baby_noncry_weight_cap: Maximum weight cap for baby_noncry class to reduce false positives

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
            # Apply weight cap to baby_noncry to reduce false positives
            elif class_label == 'baby_noncry' and baby_noncry_weight_cap is not None:
                weight = min(weight, baby_noncry_weight_cap)
        else:
            weight = 1.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)