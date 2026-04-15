"""
Baby Cry Detector — core inference engine.

Contains the BabyCryDetector class with model loading, preprocessing,
spectrogram conversion, energy/tonality gates, TTA, and multi-channel
confirmation logic.  Separated from the audio pipeline so that offline
tests and Pi_Integration can use the detector without PyAudio.
"""

import sys
import time
import logging
import queue
import numpy as np
import torch
import torchaudio
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

import os

# Insert the project root (two directories up) so that src.* imports resolve
# regardless of whether the script is invoked directly or via the systemd service.
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# src.* modules (live in <project_root>/src/)
from src.model import create_model
from src.audio_filtering import BabyCryAudioFilter
from src.acoustic_features import AcousticFeatureExtractor, validate_cry_binary

# Local deployment modules (live alongside this file)
from config_pi import ConfigPi
from multichannel_detector import create_multichannel_detector
from audio_buffer import CircularAudioBuffer
from temporal_smoother import TemporalSmoothedDetector


@dataclass
class DetectionResult:
    """Container for detection results."""
    is_cry: bool
    confidence: float
    timestamp: float
    audio_buffer: np.ndarray
    filtered_audio: Optional[np.ndarray] = None
    cry_regions: Optional[list] = None  # List of (start_sec, end_sec) tuples


class BabyCryDetector:
    """
    Baby cry detection with low-power mode and sound localization integration.
    Designed for Raspberry Pi 5 with TI PCM6260-Q1 microphone array.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[ConfigPi] = None,
        use_tta: bool = False,
        detection_threshold: Optional[float] = None,
        confirmation_threshold: Optional[float] = None,
        device: Optional[str] = None,
        audio_device_index: Optional[int] = None,
        num_channels: int = 4,
        enable_multichannel: bool = True,
        multichannel_voting: Optional[str] = None,
        enable_temporal_smoothing: bool = True,
        temporal_window_size: Optional[int] = None,
        temporal_min_consecutive: Optional[int] = None,
        temporal_confidence_threshold: Optional[float] = None,
        verbose: bool = True
    ):
        """
        Initialize detector.

        Args:
            model_path: Path to trained model checkpoint
            config: Configuration object
            use_tta: Use test-time augmentation (slower but more accurate)
            detection_threshold: Initial detection threshold
            confirmation_threshold: Confirmation threshold for wake-up
            device: Device to run on ('cpu' or 'cuda')
            audio_device_index: PyAudio device index for microphone array
            num_channels: Number of microphone channels (4 for PCM6260-Q1)
            enable_multichannel: Enable multi-channel detection (default: True)
            multichannel_voting: Voting strategy ("weighted" or "logical_or").
                If None, falls back to config.PI_MULTICHANNEL_VOTING (default: "weighted").
                Pass explicitly to override the config value at runtime.
            enable_temporal_smoothing: Enable temporal smoothing to reduce false positives (default: True)
            temporal_window_size: Number of predictions to keep in sliding window (default: 5)
            temporal_min_consecutive: Minimum consecutive high-confidence predictions for alert (default: 3)
            temporal_confidence_threshold: Threshold for high-confidence classification (default: 0.5)
            verbose: Enable verbose BCD step output (default: True)
        """
        self.config = config or ConfigPi()
        # Use model path from config — change MODEL_PATH in config_pi.py to switch models
        if model_path is None:
            model_path = self.config.MODEL_PATH
        # Resolve voting strategy: explicit arg > config.PI_MULTICHANNEL_VOTING > hard default
        multichannel_voting = multichannel_voting or getattr(
            self.config, 'PI_MULTICHANNEL_VOTING', 'weighted'
        )
        self.use_tta = use_tta
        # Calibration temperature applied to logits before softmax at inference time.
        # Loaded from checkpoint if available (see _load_checkpoint).
        self.calibration_temperature = 1.0
        # Resolve thresholds: explicit arg > config value
        self.detection_threshold = detection_threshold if detection_threshold is not None else self.config.DETECTION_THRESHOLD
        self.confirmation_threshold = confirmation_threshold if confirmation_threshold is not None else self.config.CONFIRMATION_THRESHOLD
        self.num_channels = num_channels
        self.audio_device_index = audio_device_index

        # Device setup
        self.device = torch.device(device if device else
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))

        logging.info(f"Using device: {self.device}")

        # Load baby cry detection model
        self.model = create_model(self.config).to(self.device)
        self._load_checkpoint(model_path)
        self.model.eval()
        logging.info(f"Baby cry model loaded from {model_path}")

        # Initialize audio filter
        self.audio_filter = BabyCryAudioFilter(self.config, model_path, verbose=verbose)
        logging.info("Audio filter initialized")

        # Acoustic feature validator for final confirmation gate
        self.acoustic_validator = AcousticFeatureExtractor(sample_rate=self.config.SAMPLE_RATE)
        logging.info("Acoustic feature validator initialized")

        # Audio processing setup
        self.chunk_duration = self.config.PI_AUDIO_CHUNK_DURATION
        # Capture at 48 kHz for DOAnet phase accuracy; resample to 16 kHz for BCD.
        self.capture_sample_rate = getattr(self.config, 'CAPTURE_SAMPLE_RATE', self.config.SAMPLE_RATE)
        self.chunk_size = int(self.chunk_duration * self.capture_sample_rate)
        self.context_duration = self.config.CONTEXT_DURATION

        # Circular buffer stores 48 kHz audio (preserves phase for localization)
        self.audio_buffer = CircularAudioBuffer(
            max_duration=self.context_duration,
            sample_rate=self.capture_sample_rate,
            num_channels=self.num_channels
        )

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            f_min=self.config.F_MIN,
            f_max=self.config.F_MAX
        ).to(self.device)

        # AmplitudeToDB transform — matches training pipeline (data_preprocessing.py line 63)
        # Training uses AmplitudeToDB() with default args (stype='power', top_db=None).
        # Previous deployment code passed top_db=80 which clamps dB values differently,
        # causing a train/deploy spectrogram mismatch.
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(self.device)

        # Threading and IPC
        self.audio_queue = queue.Queue(maxsize=100)
        self.detection_queue = queue.Queue(maxsize=10)  # For localization consumer
        self.is_running = False
        self.processing_failed = False
        self.low_power_mode = True

        # State tracking
        self.last_detection_time = 0
        self.detection_cooldown = self.config.DETECTION_COOLDOWN

        # Watchdog: restart processing thread on failure (safety-critical system)
        self._restart_attempts = 0
        self._max_restart_attempts = 3  # Prevent infinite restart loops

        # Callbacks
        self.on_cry_detected: Optional[Callable] = None

        # Multi-channel detection (adaptive channel selection + dual-channel voting)
        self.multichannel_detector = None
        if enable_multichannel and self.num_channels > 1:
            self.multichannel_detector = create_multichannel_detector(
                detector=self,
                num_channels=self.num_channels,
                voting_strategy=multichannel_voting,
                sample_rate=self.config.SAMPLE_RATE
            )
            logging.info(
                f"Multi-channel detection enabled ({self.num_channels} channels, "
                f"{multichannel_voting} voting)"
            )
        else:
            if not enable_multichannel:
                logging.info("Multi-channel detection disabled by configuration")
            else:
                logging.info("Single-channel mode (multi-channel detection not available)")

        # Temporal smoothing to reduce false positives from transient sounds
        # Resolve temporal params: explicit arg > config value
        temporal_window_size = temporal_window_size if temporal_window_size is not None else self.config.TEMPORAL_WINDOW_SIZE
        temporal_min_consecutive = temporal_min_consecutive if temporal_min_consecutive is not None else self.config.TEMPORAL_MIN_CONSECUTIVE
        temporal_confidence_threshold = temporal_confidence_threshold if temporal_confidence_threshold is not None else self.config.TEMPORAL_CONFIDENCE_THRESHOLD
        self.temporal_smoother: Optional[TemporalSmoothedDetector] = None
        self.enable_temporal_smoothing = enable_temporal_smoothing
        if enable_temporal_smoothing:
            self.temporal_smoother = TemporalSmoothedDetector(
                window_size=temporal_window_size,
                min_consecutive=temporal_min_consecutive,
                confidence_threshold=temporal_confidence_threshold
            )
            logging.info(
                f"Temporal smoothing enabled: window={temporal_window_size}, "
                f"min_consecutive={temporal_min_consecutive}, "
                f"threshold={temporal_confidence_threshold}"
            )
        else:
            logging.info("Temporal smoothing disabled")

    def _load_checkpoint(self, model_path: str):
        """
        Load model weights from checkpoint.

        Handles three checkpoint formats:
        - Pi-quantized: {'model': <quantized model>, 'pi_optimized': True}
        - Calibrated:   {'model_state_dict': ..., 'temperature': T, ...}
        - Standard:     {'model_state_dict': ...} or raw state dict
        """
        # Prefer the safe weights-only load; fall back to full unpickle only for
        # quantized / legacy checkpoints that embed nn.Module objects as pickle.
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            logging.warning(
                "Could not load %s with weights_only=True "
                "(quantized / legacy checkpoint). Falling back to full unpickle — "
                "ensure the file is from a trusted source.",
                model_path,
            )
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Checkpoint at '{model_path}' is a raw model object (type: "
                f"{type(checkpoint).__name__}). Raw model objects are not supported "
                f"because replacing self.model leaves stale references in "
                f"multichannel_detector and other components. "
                f"Save with torch.save({{'model_state_dict': model.state_dict(), ...}}) instead."
            )

        elif checkpoint.get('pi_optimized'):
            # Quantized model: cannot use state_dict, replace model entirely
            self.model = checkpoint['model'].to(self.device)
            logging.info("Loaded Pi-quantized model")

        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

            # Detect ensemble checkpoint (keys start with "models.N.")
            ensemble_prefixes = {
                k.split('.')[0] + '.' + k.split('.')[1]
                for k in state_dict if k.startswith('models.')
            }
            if ensemble_prefixes:
                n_models = len(ensemble_prefixes)
                logging.info(f"Detected ensemble checkpoint with {n_models} models")
                from src.model import create_model
                import torch.nn as nn

                models = nn.ModuleList([
                    create_model(self.config).to(self.device) for _ in range(n_models)
                ])
                # Load each sub-model's state dict by stripping "models.N." prefix
                for i in range(n_models):
                    prefix = f'models.{i}.'
                    sub_sd = {
                        k[len(prefix):]: v
                        for k, v in state_dict.items()
                        if k.startswith(prefix)
                    }
                    models[i].load_state_dict(sub_sd)

                # Replace self.model with a lightweight ensemble that averages logits
                class _Ensemble(nn.Module):
                    def __init__(self, sub_models):
                        super().__init__()
                        self.models = sub_models

                    def forward(self, x):
                        return torch.mean(
                            torch.stack([m(x) for m in self.models]), dim=0
                        )

                self.model = _Ensemble(models).to(self.device)
                self.model.eval()
            else:
                self.model.load_state_dict(state_dict)

            # Calibrated model: store the post-hoc calibration temperature separately.
            # BabyCryClassifier.forward() now returns raw logits (no temperature scaling),
            # so we apply calibration_temperature explicitly before softmax in detect_cry()
            # and _sliding_window_detect(). This avoids the previous double-scaling bug
            # where both model.temperature and TemperatureScaledModel.temperature were applied.
            if 'temperature' in checkpoint:
                self.calibration_temperature = float(checkpoint['temperature'])
                logging.info(f"Calibration temperature {self.calibration_temperature:.4f} loaded")

        else:
            self.model.load_state_dict(checkpoint)

    def _resample_to_model_rate(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from capture rate to model rate (16 kHz) if needed."""
        if self.capture_sample_rate == self.config.SAMPLE_RATE:
            return audio
        import librosa
        if audio.ndim == 1:
            return librosa.resample(
                audio.astype(np.float32),
                orig_sr=self.capture_sample_rate,
                target_sr=self.config.SAMPLE_RATE,
            )
        # Multi-channel: resample each channel independently
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(
                librosa.resample(
                    audio[:, ch].astype(np.float32),
                    orig_sr=self.capture_sample_rate,
                    target_sr=self.config.SAMPLE_RATE,
                )
            )
        return np.column_stack(channels)

    def preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess multi-channel audio for model input.

        Takes multi-channel audio and extracts/processes the primary channel
        while preserving the full multi-channel data for localization.

        Args:
            audio: Multi-channel audio numpy array with shape (num_samples, num_channels)

        Returns:
            Preprocessed audio tensor from primary channel (channel 0)
        """
        # Convert to tensor
        if audio.ndim == 1:
            # Mono audio (fallback)
            waveform = torch.from_numpy(audio).float()
        else:
            # Multi-channel audio - use primary channel for detection
            # Keep all channels for phase-preserving localization
            waveform = torch.from_numpy(audio[:, 0]).float()  # Use channel 0 for detection

        # Clamp to [-1, 1] so clipped recordings (common with the PCM6260 at close
        # range) do not produce out-of-range values that distort the mel spectrogram.
        waveform = waveform.clamp(-1.0, 1.0)

        # Ensure correct duration.
        # When the input is longer than DURATION (e.g. a 5-second context buffer
        # passed by confirm_and_filter), take the LAST DURATION seconds — the most
        # recent audio — rather than the first.  Taking the first 3 s caused TTA
        # confirmation to analyse stale/pre-cry audio and systematically fail.
        target_length = int(self.config.DURATION * self.config.SAMPLE_RATE)
        if len(waveform) < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - len(waveform)))
        elif len(waveform) > target_length:
            waveform = waveform[-target_length:]

        return waveform

    def audio_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to mel spectrogram."""
        waveform = waveform.to(self.device).unsqueeze(0)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to dB scale — matches training pipeline (data_preprocessing.py).
        # Training uses AmplitudeToDB (10*log10), NOT torch.log (natural log).
        mel_spec = self.amplitude_to_db(mel_spec)
        # posinf=80.0 matches training pipeline behavior; previous posinf=0.0
        # collapsed +inf dB bins to silence, distorting the spectrogram.
        mel_spec = torch.nan_to_num(mel_spec, nan=0.0, posinf=80.0, neginf=-80.0)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(0)  # (1, 1, n_mels, time)

        return mel_spec

    def _has_cry_band_energy(self, audio: np.ndarray) -> bool:
        """
        Check if the audio chunk has sufficient energy in the cry frequency band.

        Rejects chunks dominated by low-frequency room noise / rumble that would
        otherwise trigger false positives via the zero-padding bias artifact.
        Baby cries concentrate 40-70% of energy in 300-3000 Hz; ambient room
        noise typically has <15%.

        Args:
            audio: 1-D mono audio array (any length).

        Returns:
            True if cry-band energy ratio is above threshold.
        """
        threshold = getattr(self.config, 'CRY_BAND_ENERGY_THRESHOLD', 0.10)
        band_low = getattr(self.config, 'CRY_BAND_LOW', 300)
        band_high = getattr(self.config, 'CRY_BAND_HIGH', 3000)

        fft_vals = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / self.config.SAMPLE_RATE)
        power = np.abs(fft_vals) ** 2

        total_power = power.sum()
        if total_power < 1e-10:
            return False  # near-silence

        cry_mask = (freqs >= band_low) & (freqs <= band_high)
        cry_ratio = power[cry_mask].sum() / total_power

        return cry_ratio >= threshold

    def _has_tonal_content(self, audio: np.ndarray) -> bool:
        """
        Check if audio has tonal content (low spectral flatness) in the cry band.

        Spectral flatness = geometric_mean(power) / arithmetic_mean(power).
        Values near 0 indicate tonal content (baby cries); values near 1 indicate
        broadband noise (crickets, water, brushing teeth). Adds <1ms latency.

        Args:
            audio: 1-D mono audio array.

        Returns:
            True if the audio is tonal (flatness below threshold).
        """
        threshold = getattr(self.config, 'SPECTRAL_FLATNESS_THRESHOLD', 0.25)
        band_low = getattr(self.config, 'CRY_BAND_LOW', 300)
        band_high = getattr(self.config, 'CRY_BAND_HIGH', 3000)

        fft_vals = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / self.config.SAMPLE_RATE)
        power = np.abs(fft_vals) ** 2

        # Isolate cry band
        cry_mask = (freqs >= band_low) & (freqs <= band_high)
        band_power = power[cry_mask]

        if len(band_power) == 0 or band_power.sum() < 1e-10:
            return False  # near-silence or no data in band

        # Spectral flatness: geometric mean / arithmetic mean
        # Use log-domain for numerical stability (avoids overflow in product)
        log_power = np.log(band_power + 1e-20)
        geometric_mean = np.exp(log_power.mean())
        arithmetic_mean = band_power.mean()

        flatness = geometric_mean / (arithmetic_mean + 1e-20)

        if flatness >= threshold:
            logging.debug(
                f"Spectral flatness gate: REJECTED (flatness={flatness:.3f} >= {threshold})"
            )
            return False

        return True

    def predict_with_tta(self, spectrogram: torch.Tensor, n_augments: int = 3) -> torch.Tensor:
        """Predict with test-time augmentation."""
        predictions = []

        with torch.no_grad():
            # Original prediction
            predictions.append(self.model(spectrogram))

            # Augmented predictions
            for _ in range(n_augments - 1):
                # Time shift
                shift = torch.randint(-5, 6, (1,)).item()
                aug_spec = torch.roll(spectrogram, shifts=shift, dims=-1)

                # Light noise — 0.01 matches evaluation TTA (src/evaluation/utils.py)
                # to ensure consistent augmentation strength between eval and deploy
                noise = torch.randn_like(aug_spec) * 0.01
                aug_spec = aug_spec + noise

                predictions.append(self.model(aug_spec))

        return torch.mean(torch.stack(predictions), dim=0)

    def detect_cry(self, audio: np.ndarray, use_tta: bool = False) -> Tuple[bool, float, bool]:
        """
        Detect baby cry in audio chunk.

        Args:
            audio: Audio numpy array (mono 1-D or multi-channel 2-D)
            use_tta: Whether to use TTA

        Returns:
            Tuple of (is_cry, confidence, gate_rejected)
            gate_rejected is True when rejected by pre-inference gates.
        """
        # Cry-band energy gate: reject chunks dominated by low-frequency noise
        # that would trigger false positives via the zero-padding bias artifact.
        # Select channel with highest RMS for gate checks.
        # With parabolic dishes at 45/135/225/315 deg, the best channel
        # depends on source direction — channel 0 may not face the source.
        if audio.ndim > 1:
            rms_per_ch = np.sqrt(np.mean(audio ** 2, axis=0))
            best_ch = int(np.argmax(rms_per_ch))
            mono = audio[:, best_ch]
        else:
            mono = audio
        if not self._has_cry_band_energy(mono):
            return False, 0.0, True

        # Spectral flatness gate: reject broadband noise (crickets, water, etc.)
        # before expensive neural network inference. Baby cries are tonal (low flatness).
        if not self._has_tonal_content(mono):
            return False, 0.0, True

        # Preprocess
        waveform = self.preprocess_audio(audio)
        spectrogram = self.audio_to_spectrogram(waveform)

        # Predict — torch.no_grad() saves ~2x memory and 10-30% latency on Pi
        # since this model is never trained at inference time.
        with torch.no_grad():
            if use_tta:
                logits = self.predict_with_tta(spectrogram)
            else:
                logits = self.model(spectrogram)

            # Apply post-hoc calibration temperature before softmax
            logits = logits / self.calibration_temperature
            probs = torch.softmax(logits, dim=1)

        cry_prob = probs[0, 1].item()
        is_cry = cry_prob > self.detection_threshold

        return is_cry, cry_prob, False

    def _sliding_window_detect(
        self, audio: np.ndarray, use_tta: bool = True
    ) -> Tuple[bool, float]:
        """
        Slide a 1-second window over audio, pad each to DURATION, and return
        the best detection.

        Quick detection processes 1-second stream chunks that get zero-padded
        to DURATION (3 s) by preprocess_audio.  The zero-padding creates a
        bimodal spectrogram whose per-sample z-score normalization produces
        strong contrast that the model relies on.  Full DURATION-length
        windows lack this padding, so normalization flattens the signal and
        confidence drops to ~50 % even on clear cries.

        This method reproduces the same 1 s + 2 s-padding input format used
        by quick detection, scanning the context buffer with 50 % overlap and
        returning the **maximum** cry confidence.

        Args:
            audio: Single-channel 1-D audio array (any length).
            use_tta: Whether to apply test-time augmentation.

        Returns:
            Tuple of (is_cry, best_confidence)
        """
        # Use the same chunk size as the low-power quick detection path.
        chunk_len = int(self.chunk_duration * self.config.SAMPLE_RATE)

        # If audio is already short, just run detect_cry (preprocess_audio pads).
        # Strip gate_rejected — _sliding_window_detect returns a 2-tuple.
        if len(audio) <= chunk_len:
            is_cry_short, conf_short, _ = self.detect_cry(audio, use_tta=use_tta)
            return is_cry_short, conf_short

        hop = chunk_len // 2  # 50 % overlap

        with torch.no_grad():
            if use_tta:
                # TTA uses random augmentations, so fall back to per-window calls
                best_conf = 0.0
                for start in range(0, len(audio) - chunk_len + 1, hop):
                    window = audio[start:start + chunk_len]
                    _, conf, _gate = self.detect_cry(window, use_tta=True)
                    if conf > best_conf:
                        best_conf = conf
            else:
                # Batch all windows into a single forward pass for ~6x speedup
                spectrograms = []
                for start in range(0, len(audio) - chunk_len + 1, hop):
                    window = audio[start:start + chunk_len]
                    waveform = self.preprocess_audio(window)
                    spec = self.audio_to_spectrogram(waveform)
                    spectrograms.append(spec)

                if spectrograms:
                    batch = torch.cat(spectrograms, dim=0)
                    logits = self.model(batch)
                    # Apply post-hoc calibration temperature before softmax
                    logits = logits / self.calibration_temperature
                    probs = torch.softmax(logits, dim=1)
                    best_conf = float(probs[:, 1].max().item())
                else:
                    best_conf = 0.0

        is_cry = best_conf > self.detection_threshold
        return is_cry, best_conf

    def confirm_and_filter(self, audio: np.ndarray) -> DetectionResult:
        """
        Confirm detection with TTA and prepare multi-channel audio for localization.

        Uses a sliding-window approach over the context buffer: for each
        selected channel the model scans DURATION-length windows with 50 %
        overlap and keeps the best confidence.  This prevents the confirmation
        from failing simply because the cry was not centred in the first (or
        last) 3 seconds of the 5-second context.

        Args:
            audio: Multi-channel audio numpy array with shape (num_samples, num_channels)

        Returns:
            DetectionResult with preserved multi-channel audio for sound localization
        """
        # Track which channels were selected (used by acoustic validation below)
        best_channels = None

        # Resample context audio to 16 kHz for BCD model inference.
        # Keep the original 48 kHz audio for the DetectionResult (localization).
        audio_16k = self._resample_to_model_rate(audio)

        # Confirm with sliding-window TTA for higher accuracy
        if self.multichannel_detector and audio.ndim > 1:
            # Use 16kHz audio for SNR/health checks since the SNR computer's
            # bandpass filter was designed for 16kHz (SAMPLE_RATE, not CAPTURE_SAMPLE_RATE).
            health_metrics = self.multichannel_detector.health_monitor.get_channel_health(audio_16k)
            for metric in health_metrics:
                if metric.rms < self.multichannel_detector.health_monitor.rms_min_threshold:
                    logging.warning(
                        f"Channel {metric.channel_idx} health issue: Low RMS ({metric.rms:.6f})"
                    )
                if metric.clipping:
                    logging.warning(
                        f"Channel {metric.channel_idx} health issue: Clipping detected"
                    )

            # Select best 2 channels by SNR at 16kHz (matches SNR computer's sample rate)
            best_channels, snr_scores = self.multichannel_detector.select_best_channels(audio_16k, n_channels=2)

            # Run sliding-window detection on each selected channel (16 kHz)
            channel_confidences = []
            for ch in best_channels:
                channel_audio = audio_16k[:, ch]
                _, conf = self._sliding_window_detect(channel_audio, use_tta=True)
                channel_confidences.append(conf)

            # Weighted confidence for logging/observability
            if self.multichannel_detector.voting_strategy == "weighted":
                weights = np.exp(np.array([snr_scores[ch] for ch in best_channels]) / 10.0)
                weights = weights / weights.sum()
                weighted_confidence = float(np.sum(np.array(channel_confidences) * weights))
            elif self.multichannel_detector.voting_strategy == "logical_or":
                weighted_confidence = float(max(channel_confidences))
            else:
                weighted_confidence = float(np.mean(channel_confidences))

            # Use weighted confidence for confirmation decision, matching the
            # quick-detection path.  The old AND gate (both channels >= threshold)
            # was too strict for mic-array audio where per-channel confidence is
            # degraded by room acoustics and speaker playback.
            is_cry = weighted_confidence >= self.confirmation_threshold
            confidence = weighted_confidence

            agreement = (
                1.0 - abs(channel_confidences[0] - channel_confidences[1])
                if len(channel_confidences) == 2 else None
            )

            agreement_str = f", agreement: {agreement:.2%}" if agreement is not None else ""
            logging.info(
                f"Sliding-window confirmation: {is_cry} (conf: {confidence:.2%}, "
                f"weighted: {weighted_confidence:.2%}), "
                f"channels: {best_channels[0]}/{best_channels[1] if len(best_channels) > 1 else '-'}, "
                f"per-channel: {[f'{c:.2%}' for c in channel_confidences]}"
                f"{agreement_str}"
            )
            for metric in health_metrics:
                logging.debug(
                    f"  Channel {metric.channel_idx}: SNR={metric.snr_db:.1f}dB, "
                    f"RMS={metric.rms:.4f}, clipping={metric.clipping}"
                )
        else:
            # Fallback: sliding-window single-channel detection with TTA (16 kHz)
            is_cry, confidence = self._sliding_window_detect(
                audio_16k if audio_16k.ndim == 1 else audio_16k[:, 0],
                use_tta=True,
            )

        # Acoustic feature validation gate: catches FPs that pass all other filters
        # (adult screams with pitch < 300Hz, animal sounds with wrong HNR/centroid,
        # broadband noise with low HNR).
        if is_cry and confidence >= self.confirmation_threshold:
            # Extract best channel mono audio for acoustic validation (16 kHz)
            if audio_16k.ndim > 1:
                # Use the channel with highest SNR for acoustic analysis
                best_ch = best_channels[0] if best_channels is not None else 0
                mono_for_validation = audio_16k[:, best_ch]
            else:
                mono_for_validation = audio_16k

            mono_tensor = torch.from_numpy(mono_for_validation).float()
            try:
                features = self.acoustic_validator.extract_all_features(mono_tensor)
                is_valid, rejection_reason = validate_cry_binary(features, confidence)
                if not is_valid:
                    logging.info(
                        f"Acoustic validation REJECTED: {rejection_reason} "
                        f"(pitch={features['pitch_mean']:.1f}Hz, "
                        f"HNR={features['hnr_mean']:.3f}, "
                        f"duration={features['duration']:.2f}s)"
                    )
                    is_cry = False
                else:
                    logging.info(
                        f"Acoustic validation PASSED: {rejection_reason} "
                        f"(pitch={features['pitch_mean']:.1f}Hz, "
                        f"HNR={features['hnr_mean']:.3f})"
                    )
            except Exception as e:
                # Don't block detection if acoustic validation fails unexpectedly
                logging.warning(f"Acoustic validation error (allowing detection): {e}")

        filtered_audio = None
        cry_regions = None
        if is_cry and confidence >= self.confirmation_threshold:
            # Apply audio filtering for sound localization (uses 16 kHz for model)
            logging.info("Applying audio filtering for sound localization...")

            if audio_16k.ndim > 1:
                audio_tensor = torch.from_numpy(audio_16k).float()
                filtered_tensor, cry_segments, _ = self.audio_filter.isolate_baby_cry_multichannel(
                    audio_tensor,
                    cry_threshold=self.detection_threshold
                )
                filtered_audio = filtered_tensor.numpy()
            else:
                audio_tensor = torch.from_numpy(audio_16k).float()
                filtered_tensor, cry_segments, _ = self.audio_filter.isolate_baby_cry(
                    audio_tensor,
                    cry_threshold=self.detection_threshold
                )
                filtered_audio = filtered_tensor.numpy()

            if cry_segments:
                cry_regions = cry_segments

        # audio_buffer is the original 48 kHz audio for DOAnet localization
        return DetectionResult(
            is_cry=is_cry,
            confidence=confidence,
            timestamp=time.time(),
            audio_buffer=audio,
            filtered_audio=filtered_audio,
            cry_regions=cry_regions,
        )

    def wake_robot(self, detection: DetectionResult):
        """
        Wake robot from low-power mode and send multi-channel data to sound localization.

        Sends raw multi-channel audio with preserved phase relationships for beamforming
        and sound source localization.

        Args:
            detection: Detection result with multi-channel filtered audio
        """
        logging.info(f"BABY CRY DETECTED! Confidence: {detection.confidence:.2%}")
        logging.info(f"Waking robot from low-power mode... ({self.num_channels}-channel audio)")

        self.low_power_mode = False

        # Prepare data for sound localization
        # Send raw multi-channel audio with phase information preserved
        localization_data = {
            'timestamp': detection.timestamp,
            'confidence': detection.confidence,
            'raw_audio': detection.audio_buffer,  # Full multi-channel audio
            'filtered_audio': detection.filtered_audio,  # Multi-channel filtered (cry regions only)
            'sample_rate': self.capture_sample_rate,
            'num_channels': self.num_channels,
            'audio_shape': detection.audio_buffer.shape if detection.audio_buffer is not None else None
        }

        # Send to sound localization process via queue
        try:
            self.detection_queue.put(localization_data, timeout=1.0)
            logging.info(f"Multi-channel audio ({self.num_channels} channels) sent to sound localization")
            if detection.filtered_audio is not None:
                logging.info(f"  Filtered audio shape: {detection.filtered_audio.shape}")
        except queue.Full:
            logging.error("Localization queue full, data not sent")

        # Call user callback if set
        if self.on_cry_detected:
            self.on_cry_detected(detection)

    def reset_to_low_power(self):
        """Reset detector to low-power mode after robot task completion."""
        logging.info("Resetting to low-power listening mode")
        self.low_power_mode = True
        self.audio_buffer.clear()

        # Reset temporal smoother to clear prediction history
        if self.temporal_smoother is not None:
            self.temporal_smoother.reset()
            logging.debug("Temporal smoother reset")
