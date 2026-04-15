"""
Configuration module for baby cry detection model.
Contains all hyperparameters and settings for training, evaluation, and deployment.
"""

import os
from pathlib import Path
import torch

class Config:
    # Data paths
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")

    # Preprocessed data configuration
    USE_PREPROCESSED = True  # Set to True to use preprocessed spectrograms
    PREPROCESSED_DIR = Path("data/processed/v12")  # Directory containing preprocessed data

    # Audio processing parameters
    SAMPLE_RATE = 16000  # Standard sample rate for audio processing
    DURATION = 3.0  # Duration in seconds for each audio segment
    MIN_AUDIO_DURATION = 0.8  # Minimum raw audio duration (seconds) to keep during preprocessing;
    # files shorter than this produce <14 useful spectrogram frames out of ~90 (>84% zero-padding)
    N_MELS = 128  # Number of mel-frequency bins
    N_FFT = 2048  # FFT window size
    HOP_LENGTH = 512  # Hop length for STFT
    F_MIN = 0  # Minimum frequency
    F_MAX = 8000  # Maximum frequency (covers baby cry frequency range)

    # Data augmentation parameters (OPTIMIZED to reduce 1.4% train-val gap and improve generalization)
    NOISE_FACTOR = 0.025  # Gaussian noise factor (increased from 0.015 for distant/low-SNR cry robustness)
    TIME_STRETCH_RATE = [0.8, 1.2]  # Time stretch range (narrowed from [0.75, 1.25] - extreme stretching may distort cry acoustics)
    PITCH_SHIFT_STEPS = [-2, 2]  # Pitch shift range in semitones (narrowed from [-3, 3] to preserve 300-600Hz cry frequency characteristics)

    # SpecAugment parameters (QUICK WIN 1) - Applied on mel-spectrograms
    USE_SPEC_AUGMENT = True  # Re-enabled with milder params for regularization (reduces overfitting gap)
    SPEC_AUG_TIME_MASK_PARAM = 15  # Max time mask length (frames) - milder than 20
    SPEC_AUG_FREQ_MASK_PARAM = 10  # Max frequency mask width (mel bins) — restored from 8 for stronger frequency regularization
    # 2 time masks forces the model to rely less on absolute spectral shape and more on
    # temporal persistence — the key discriminator between sustained cry and short-burst
    # baby_noncry vocalizations (giggling, babbling).  Run 2 used 1 mask and produced
    # 10 high-confidence baby_noncry FPs; increasing to 2 addresses this systematically.
    SPEC_AUG_NUM_TIME_MASKS = 2  # Number of time masks (increased from 1 — see note above)
    SPEC_AUG_NUM_FREQ_MASKS = 2  # Number of frequency masks (increased from 1 — forces model to rely on broader spectral patterns rather than specific bands, reducing animal-sound FPs)

    # Random duration simulation augmentation
    # 77% of cry samples are <3s (zero-padded) vs 5% of non-cry → model learned
    # zero-padding as spurious cry correlate. This augmentation randomly truncates
    # the spectrogram and re-normalizes to break that class correlation.
    USE_DURATION_AUGMENT = True  # Simulate variable audio lengths during training
    DURATION_SIM_PROBABILITY = 0.5  # Probability of applying per sample
    DURATION_SIM_MIN_KEEP = 0.30  # Keep at least 30% of time steps (~0.9s of 3s)
    DURATION_SIM_MAX_KEEP = 0.85  # Keep at most 85% of time steps (~2.55s of 3s)

    # Model architecture parameters
    INPUT_CHANNELS = 1  # Single channel mel-spectrogram
    CNN_CHANNELS = [32, 64, 128, 256]  # CNN channel progression
    CNN_KERNEL_SIZE = 3  # CNN kernel size
    CNN_DROPOUT = 0.2  # CNN dropout rate (restored from 0.15 — reducing regularization inflates train acc but hurts generalization)

    # Transformer parameters
    D_MODEL = 384  # Transformer embedding dimension (Run 16 sweet spot: 18 FPs, 97.93% accuracy)
    N_HEADS = 8  # Number of attention heads (512/8 = 64 dim per head)
    N_LAYERS = 4  # Number of transformer layers
    TRANSFORMER_DROPOUT = 0.1  # Transformer dropout rate (restored from 0.08 — proper regularization for generalization)

    #! Training parameters (UPDATED for 95% accuracy target)
    BATCH_SIZE = 128  # Batch size for training (increased from 96 for more stable gradients)
    LEARNING_RATE = 1e-4  # Initial learning rate (optimized for faster convergence)
    WEIGHT_DECAY = 1e-5  # L2 regularization
    NUM_EPOCHS = 100  # Maximum number of epochs
    PATIENCE = 20  # Early stopping patience (increased from 12 — larger d_model=384 needs more time to converge)
    WARMUP_EPOCHS = 5  # Learning rate warmup epochs (increased from 3)
    USE_AMP = True  # Use Automatic Mixed Precision for 2× faster training (GPU only)
    MIXUP_ALPHA = 0.2  # Beta distribution parameter for mixup augmentation (0.2 = mild mixing; 0 disables)

    # Data split ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.20
    TEST_RATIO = 0.20

    #! Classification — Binary only (cry vs non_cry)
    NUM_CLASSES = 2
    CLASS_LABELS = {
        0: 'non_cry',
        1: 'cry'
    }

    # Supported audio formats
    SUPPORTED_FORMATS = ['.wav', '.ogg', '.mp3', '.flac', '.m4a', '.3gp', '.webm', '.mp4']

    # Known mislabeled / ambiguous files requiring manual review before next training run:
    # - relabeled_from_cry_weak_Infantcry_999.wav: labeled non_cry but model predicts cry at
    #   99.7% confidence. Was originally a cry sample. Likely correct model prediction on a
    #   mislabeled file. Review and restore to cry label if audio confirms cry content.
    # - Infantcry_379 (segments 0992_3331 and 5646_7008): two FN segments from same recording.
    #   Investigate whether ICSD sub-group has different recording conditions from rest of dataset.
    # - 547190_Voice_AdultMale_DeathScream_09.mp3: duplicate file in adult_shout + adult_scream.
    #   Deduplication code in collect_audio_files will keep first occurrence; remove the second
    #   copy from the data directory manually to avoid label confusion.

    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    # Per-category sampling weights for WeightedRandomSampler (Phase 3)
    # Higher weight = sampled more often per epoch. Targets hard-negative categories
    # that cause the most false positives at the 0.92 deployment threshold.
    CATEGORY_SAMPLING_WEIGHTS = {
        'baby_noncry': 6.0,   # 360 samples, dominant FP category (14 FPs, up from 5)
        'laughing': 3.0,      # 4 new high-confidence FPs, shares acoustic features with crying
        'adult_scream': 3.0,  # 100 samples, high-confidence FPs
        'adult_shout': 2.0,   # 42 samples, moderate FP contributor
        'rooster': 2.0,       # 3 FPs in latest eval (up to 95% confidence)
        'dog': 2.0,           # 1 FP at 99.2% confidence — F0 overlaps with cry (300-600Hz)
        'cow': 2.0,           # 1 FP at 99.6% confidence — animal vocalization confuses model
        'coughing': 1.5,      # Phase 3 FPs: 2 at 99%+ confidence
        'glass_breaking': 1.5,  # Phase 3 FP: 1 at 99% confidence
        'mouse_click': 1.5,   # Phase 3 FP: 1 at 98% confidence
        'brushing_teeth': 1.5,  # Phase 3 FP: 1 at 99% confidence
        'sheep': 1.5,         # 1 FP at 98.6% confidence — animal sounds confuse model
        'pig': 1.5,           # 1 FP at 98.1% confidence — animal sounds confuse model
        'crow': 2.0,          # New: 1 FP at 99.25% — bird vocalization confuses model
        'cat': 1.5,           # New: 1 FP at 97.5% — meow overlaps with cry acoustics
        'child_tantrum': 2.0, # New: 2 FPs both >92% — similar vocal quality to cry
    }
    # All other categories default to 1.0

    # Per-category loss multipliers (Phase 3)
    # Stacks on top of class-level weights — makes model pay more attention
    # to misclassifying these specific hard-negative categories.
    CATEGORY_LOSS_WEIGHTS = {
        'baby_noncry': 2.5,     # Dominant FP category — 3.0 and 3.5 both hurt recall, reverting to 2.5
        'laughing': 2.0,        # Laughter shares acoustic features with crying (3 FPs >90%)
        'adult_scream': 2.0,    # Screams trigger high-confidence FPs (2 FPs)
        'child_tantrum': 2.0,   # 2 FPs both >92% confidence
        'crow': 2.0,            # 1 FP at 99.25% confidence
        'cat': 1.5,             # 1 FP at 97.5% confidence
    }
    # All other categories default to 1.0

    #! Loss function parameters (UPDATED for better calibration and reduced false negatives)
    CRY_WEIGHT_MULTIPLIER = 1.15  # Run 18: d_model=384 L=6, reverting to 1.15 (1.20 spiked FPs)
    # Class weighting is handled entirely by CRY_WEIGHT_MULTIPLIER + inverse-frequency weights.
    USE_FOCAL_LOSS = False  # CrossEntropyLoss is correct for balanced data (0.96:1 ratio); FocalLoss gamma=2 contradicts label smoothing

    # Label smoothing parameters (ENABLED for improved calibration)
    USE_LABEL_SMOOTHING = True  # ENABLED - Improves calibration and reduces overconfidence
    LABEL_SMOOTHING_EPSILON = 0.1  # Smooth labels to [0.1, 0.9]

    # Confidence penalty — disabled (asymmetric smoothing + penalty both increased FPs in Runs 20-21)
    CONFIDENCE_PENALTY_BETA = 0.0

    # Ensemble parameters (QUICK WIN 2)
    USE_ENSEMBLE = True  # Enable ensemble averaging during evaluation
    ENSEMBLE_N_MODELS = 3  # Number of best models to ensemble (last 3 best checkpoints)

    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    # Raspberry Pi optimization settings
    OPTIMIZE_FOR_MOBILE = True  # Mobile optimization

    # Inference settings
    # NOTE: Calibration analysis shows bins [0.7-0.92] have poor accuracy despite high confidence.
    # Bimodal distribution means 94.3% of predictions are <0.1 or >0.9, so 0.92 threshold loses almost nothing.
    CONFIDENCE_THRESHOLD = 0.92
    SLIDING_WINDOW_SIZE = 3.0  # Sliding window size in seconds for real-time inference
    OVERLAP_RATIO = 0.75  # Overlap ratio for sliding window (increased from 0.5 for denser coverage of distant cries)

    # Cry-band energy gate: reject chunks dominated by low-frequency noise
    # that would otherwise trigger false positives via the zero-padding bias.
    # Baby cries have 40-70% energy in 300-3000Hz; room noise has <15%.
    CRY_BAND_ENERGY_THRESHOLD = 0.10  # Min fraction of energy in 300-3000Hz to run inference
    CRY_BAND_LOW = 300   # Lower bound of cry energy band (Hz)
    CRY_BAND_HIGH = 3000  # Upper bound of cry energy band (Hz)

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
    VAD_ENERGY_THRESHOLD = 0.20  # Threshold for normalized confidence (lowered from 0.35 for distant cry sensitivity)
    VAD_FREQ_MIN = 200      # Baby cry starts around 200 Hz
    VAD_FREQ_MAX = 1000     # Baby cry harmonics up to ~1000 Hz

    # Noise Filtering parameters
    HIGHPASS_CUTOFF = 100   # Remove rumble below 100 Hz
    BANDPASS_LOW = 100      # Full cry frequency range (100-3000 Hz)
    BANDPASS_HIGH = 3000    # Includes upper harmonics
    NOISE_REDUCE_STRENGTH = 0.3  # Spectral subtraction strength (lowered from 0.5 to preserve quiet cry harmonics)

    # Deep spectrum features (for evaluation/inference)
    USE_DEEP_SPECTRUM = False  # Enable deep spectrum features (slower but more robust)
    EXTRACT_MFCC_DELTAS = False  # Extract MFCC with delta/delta-delta
    EXTRACT_SPECTRAL_CONTRAST = False  # Extract spectral contrast
    EXTRACT_CHROMA = False  # Extract chroma features
