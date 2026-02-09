"""
Configuration module for baby cry detection model.
Contains all hyperparameters and settings for training, evaluation, and deployment.
"""

import os
from pathlib import Path

class Config:
    # Data paths
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")

    # Preprocessed data configuration
    USE_PREPROCESSED = True  # Set to True to use preprocessed spectrograms
    PREPROCESSED_DIR = Path("data/processed/v1_new")  # Directory containing preprocessed data

    # Audio processing parameters
    SAMPLE_RATE = 16000  # Standard sample rate for audio processing
    DURATION = 3.0  # Duration in seconds for each audio segment
    N_MELS = 128  # Number of mel-frequency bins
    N_FFT = 2048  # FFT window size
    HOP_LENGTH = 512  # Hop length for STFT
    F_MIN = 0  # Minimum frequency
    F_MAX = 8000  # Maximum frequency (covers baby cry frequency range)

    # Data augmentation parameters (OPTIMIZED to reduce 1.4% train-val gap and improve generalization)
    NOISE_FACTOR = 0.015  # Gaussian noise factor (increased from 0.01 for better noise robustness)
    TIME_STRETCH_RATE = [0.8, 1.2]  # Time stretch range (narrowed from [0.75, 1.25] - extreme stretching may distort cry acoustics)
    PITCH_SHIFT_STEPS = [-2, 2]  # Pitch shift range in semitones (narrowed from [-3, 3] to preserve 300-600Hz cry frequency characteristics)

    # SpecAugment parameters (QUICK WIN 1) - Applied on mel-spectrograms
    USE_SPEC_AUGMENT = False  # DISABLED - Reverting to baseline (SpecAugment reduced accuracy)
    SPEC_AUG_TIME_MASK_PARAM = 20  # Max time mask length (frames) - ~0.25s at 16kHz
    SPEC_AUG_FREQ_MASK_PARAM = 15  # Max frequency mask width (mel bins)
    SPEC_AUG_NUM_TIME_MASKS = 2  # Number of time masks to apply
    SPEC_AUG_NUM_FREQ_MASKS = 2  # Number of frequency masks to apply

    # Model architecture parameters
    INPUT_CHANNELS = 1  # Single channel mel-spectrogram
    CNN_CHANNELS = [32, 64, 128, 256]  # CNN channel progression
    CNN_KERNEL_SIZE = 3  # CNN kernel size
    CNN_DROPOUT = 0.15  # CNN dropout rate (reduced from 0.2 for 95% target - less regularization)

    # Transformer parameters
    D_MODEL = 256  # Transformer embedding dimension
    N_HEADS = 8  # Number of attention heads
    N_LAYERS = 4  # Number of transformer layers
    TRANSFORMER_DROPOUT = 0.08  # Transformer dropout rate (reduced from 0.1 for 95% target)

    #! Training parameters (UPDATED for 95% accuracy target)
    BATCH_SIZE = 128  # Batch size for training (increased from 96 for more stable gradients)
    LEARNING_RATE = 1e-4  # Initial learning rate (optimized for faster convergence)
    WEIGHT_DECAY = 1e-5  # L2 regularization
    NUM_EPOCHS = 80  # Maximum number of epochs (increased from 60 for 95% target)
    PATIENCE = 20  # Early stopping patience (increased from 15 for 95% target)
    WARMUP_EPOCHS = 5  # Learning rate warmup epochs (increased from 3)
    USE_AMP = True  # Use Automatic Mixed Precision for 2× faster training (GPU only)

    # Data split ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.20
    TEST_RATIO = 0.20

    #! Classification mode - BINARY ONLY
    # NOTE: Multi-class mode is permanently disabled per project requirements
    MULTI_CLASS_MODE = False  # ALWAYS False - Binary classification only

    # Class labels - Binary mode (ONLY mode supported)
    BINARY_CLASS_LABELS = {
        0: 'non_cry',
        1: 'cry'
    }

    # Legacy multi-class labels (kept for backwards compatibility with old code)
    # DO NOT USE - Only here to prevent errors in old result files
    MULTI_CLASS_LABELS = {
        0: 'cry',
        1: 'adult_speech',
        2: 'baby_noncry',
        3: 'environmental'
    }

    # Active class labels (binary only)
    @property
    def CLASS_LABELS(self):
        return self.BINARY_CLASS_LABELS

    @property
    def NUM_CLASSES(self):
        return len(self.BINARY_CLASS_LABELS)

    # Supported audio formats
    SUPPORTED_FORMATS = ['.wav', '.ogg', '.mp3', '.flac', '.m4a', '.3gp', '.webm', '.mp4']

    # Device configuration
    DEVICE = 'cuda' if os.getenv('CUDA_AVAILABLE', 'false').lower() == 'true' else 'cpu'
    NUM_WORKERS = 4  # Number of data loader workers

    # Results configuration
    @staticmethod
    def get_results_dir(mode: str = "train"):
        """
        Generate timestamped results directory with mode prefix.
        Uses local system time.

        Args:
            mode: One of 'train', 'eval', 'analyze', 'test'

        Returns:
            Path to results directory
        """
        # Use strftime with local time - respects system timezone settings
        import time
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        results_dir = Config.RESULTS_DIR / f"{mode}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (results_dir / "plots").mkdir(exist_ok=True)
        (results_dir / "logs").mkdir(exist_ok=True)

        return results_dir

    # Model paths
    @staticmethod
    def get_model_path(results_dir):
        """Get model save path."""
        return results_dir / "model_best.pth"

    #! Loss function parameters (UPDATED for better calibration and reduced false negatives)
    # CRY_WEIGHT_MULTIPLIER increased from 1.15 to 1.35 to further penalize missed cries
    # and reduce false negative rate in safety-critical baby cry detection
    CRY_WEIGHT_MULTIPLIER = 1.35  # Penalty multiplier for missed cries
    FOCAL_LOSS_GAMMA = 2.0  # Focusing parameter for hard examples (baseline optimal value)
    FOCAL_LOSS_ALPHA = 0.3  # Weighting factor for class balance (baseline optimal value)

    # Label smoothing parameters (ENABLED for improved calibration)
    # Label smoothing helps prevent overconfident predictions and improves model calibration,
    # which is important for threshold-based decision making in production
    USE_LABEL_SMOOTHING = True  # ENABLED - Improves calibration and reduces overconfidence
    LABEL_SMOOTHING_EPSILON = 0.1  # Smooth labels to [0.1, 0.9] (increased from 0.05 for stronger regularization)

    # Ensemble parameters (QUICK WIN 2)
    USE_ENSEMBLE = True  # Enable ensemble averaging during evaluation
    ENSEMBLE_N_MODELS = 3  # Number of best models to ensemble (last 3 best checkpoints)

    # Class weight caps (to reduce false positives for rare classes)
    #BABY_NONCRY_WEIGHT_CAP = 5.0  # Cap baby_noncry weight at 5.0 (down from automatic 7.772) to reduce false positives

    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    # Raspberry Pi optimization settings
    OPTIMIZE_FOR_MOBILE = True  # Mobile optimization

    # Inference settings
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for positive prediction
    SLIDING_WINDOW_SIZE = 3.0  # Sliding window size in seconds for real-time inference
    OVERLAP_RATIO = 0.5  # Overlap ratio for sliding window

    # Raspberry Pi optimization settings
    PI_BUFFER_SIZE = 1024  # Audio buffer size for Pi streaming
    PI_SAMPLE_RATE = 16000  # Optimized sample rate for Pi
    PI_CHANNELS = 1  # Mono audio for efficiency

    # Acoustic Feature-Based Filtering Configuration
    # NOTE: These are DISABLED during training for clean model learning
    # They are applied ONLY during inference/noise filtering pipeline
    USE_ACOUSTIC_FEATURES = False  # Disabled for training (used in inference pipeline)

    # Baby cry acoustic characteristics
    CRY_F0_MIN = 300  # Baby cry fundamental frequency minimum (Hz)
    CRY_F0_MAX = 600  # Baby cry fundamental frequency maximum (Hz)
    CRY_HARMONIC_TOLERANCE = 50  # Frequency tolerance for harmonic detection (Hz)

    # Temporal pattern parameters
    CRY_BURST_MIN = 0.3  # Minimum cry burst duration (seconds)
    CRY_BURST_MAX = 2.0  # Maximum cry burst duration (seconds)
    CRY_PAUSE_MIN = 0.1  # Minimum inhalation pause (seconds)
    CRY_PAUSE_MAX = 0.8  # Maximum inhalation pause (seconds)

    # Pitch contour parameters
    PITCH_VARIATION_MIN = 20  # Minimum pitch variation for cry (Hz)
    PITCH_VARIATION_MAX = 200  # Maximum pitch variation for cry (Hz)

    # Frequency modulation parameters
    FM_VARIATION_MIN = 5  # Minimum FM variation (Hz)
    FM_VARIATION_MAX = 20  # Maximum FM variation (Hz)

    # Energy distribution parameters
    ENERGY_CONCENTRATION_THRESHOLD = 0.3  # Minimum ratio of energy in cry band

    # Rejection filter parameters
    ADULT_F0_MIN = 80  # Adult speech fundamental frequency minimum (Hz)
    ADULT_F0_MAX = 250  # Adult speech fundamental frequency maximum (Hz)
    MUSIC_PITCH_STABILITY_THRESHOLD = 0.05  # Coefficient of variation threshold for music
    ENV_SPECTRAL_FLATNESS_THRESHOLD = 0.5  # Spectral flatness threshold for environmental sounds

    # Acoustic feature weighting (must sum to 1.0 for cry indicators)
    WEIGHT_HARMONICS = 0.25  # Weight for harmonic structure
    WEIGHT_PITCH_CONTOUR = 0.15  # Weight for pitch contours
    WEIGHT_FREQUENCY_MODULATION = 0.10  # Weight for FM detection
    WEIGHT_ENERGY_DISTRIBUTION = 0.20  # Weight for energy concentration
    # Remaining 0.30 is implicit in the base score

    # Combined prediction weighting
    WEIGHT_ML_MODEL = 0.6  # Weight for ML model predictions
    WEIGHT_ACOUSTIC_FEATURES = 0.4  # Weight for acoustic features (must sum to 1.0 with ML)

    # Advanced Filtering Configuration (2024-2025 Best Practices)
    # NOTE: This is DISABLED during training to keep the model clean
    # It is enabled ONLY in the inference/noise filtering pipeline
    USE_ADVANCED_FILTERING = False  # Disabled for training (used in inference pipeline)

    # Voice Activity Detection (VAD) parameters
    VAD_FRAME_LENGTH = 400  # 25ms at 16kHz
    VAD_HOP_LENGTH = 160    # 10ms at 16kHz
    VAD_ENERGY_THRESHOLD = 0.35  # Threshold for normalized confidence (0-1 range)
    VAD_FREQ_MIN = 200      # Baby cry starts around 200 Hz
    VAD_FREQ_MAX = 1000     # Baby cry harmonics up to ~1000 Hz

    # Noise Filtering parameters
    HIGHPASS_CUTOFF = 100   # Remove rumble below 100 Hz
    BANDPASS_LOW = 200      # Baby cry frequency range start
    BANDPASS_HIGH = 2000    # Baby cry harmonics range end
    NOISE_REDUCE_STRENGTH = 0.5  # Spectral subtraction strength (0-1)

    # Deep spectrum features (for evaluation/inference)
    USE_DEEP_SPECTRUM = False  # Enable deep spectrum features (slower but more robust)
    EXTRACT_MFCC_DELTAS = False  # Extract MFCC with delta/delta-delta
    EXTRACT_SPECTRAL_CONTRAST = False  # Extract spectral contrast
    EXTRACT_CHROMA = False  # Extract chroma features
