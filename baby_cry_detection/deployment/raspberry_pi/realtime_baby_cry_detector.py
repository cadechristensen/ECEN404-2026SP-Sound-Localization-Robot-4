"""
Real-Time Baby Cry Detection System for Raspberry Pi 5
Optimized for TI PCM6260-Q1 4-microphone array with low-power listening mode.
Interfaces with sound localization model for robot navigation.

Features:
- Multi-channel detection with adaptive channel selection
- Temporal smoothing to reduce false positives from transient sounds
- Low-power listening mode with wake-on-cry capability
- Phase preservation for sound localization
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import pyaudio
import queue
import threading
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Callable
import multiprocessing as mp

# Insert the project root (two directories up) so that src.* imports resolve
# regardless of whether the script is invoked directly or via the systemd service.
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# src.* modules (live in <project_root>/src/)
from src.model import create_model
from src.audio_filter import BabyCryAudioFilter

# Local deployment modules (live alongside this file)
from config_pi import ConfigPi
from multichannel_detector import create_multichannel_detector
from detection_types import DetectionResult
from audio_buffer import CircularAudioBuffer
from temporal_smoother import TemporalSmoothedDetector



class RealtimeBabyCryDetector:
    """
    Real-time baby cry detection with low-power mode and sound localization integration.
    Designed for Raspberry Pi 5 with TI PCM6260-Q1 microphone array.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[ConfigPi] = None,
        use_tta: bool = False,
        detection_threshold: float = 0.5,
        confirmation_threshold: float = 0.85,
        device: Optional[str] = None,
        audio_device_index: Optional[int] = None,
        num_channels: int = 4,
        enable_multichannel: bool = True,
        multichannel_voting: str = "weighted",
        enable_temporal_smoothing: bool = True,
        temporal_window_size: int = 5,
        temporal_min_consecutive: int = 3,
        temporal_confidence_threshold: float = 0.5
    ):
        """
        Initialize real-time detector.

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
            multichannel_voting: Voting strategy ("weighted" or "logical_or", default: "weighted")
            enable_temporal_smoothing: Enable temporal smoothing to reduce false positives (default: True)
            temporal_window_size: Number of predictions to keep in sliding window (default: 5)
            temporal_min_consecutive: Minimum consecutive high-confidence predictions for alert (default: 3)
            temporal_confidence_threshold: Threshold for high-confidence classification (default: 0.5)
        """
        self.config = config or ConfigPi()
        self.use_tta = use_tta
        self.detection_threshold = detection_threshold
        self.confirmation_threshold = confirmation_threshold
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
        self.audio_filter = BabyCryAudioFilter(self.config, model_path)
        logging.info("Audio filter initialized")

        # Audio processing setup
        self.chunk_duration = 1.0  # Process 1 second chunks in low-power mode
        self.chunk_size = int(self.chunk_duration * self.config.SAMPLE_RATE)
        self.context_duration = 5.0  # Keep 5 seconds of context

        # Circular buffer for multi-channel audio context (preserves phase)
        self.audio_buffer = CircularAudioBuffer(
            max_duration=self.context_duration,
            sample_rate=self.config.SAMPLE_RATE,
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

        # Threading and IPC
        self.audio_queue = queue.Queue(maxsize=100)
        self.detection_queue = mp.Queue(maxsize=10)  # For IPC to localization
        self.is_running = False
        self.low_power_mode = True

        # State tracking
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # Seconds before re-detection

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
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if not isinstance(checkpoint, dict):
            # Raw model object saved directly
            self.model = checkpoint.to(self.device)

        elif checkpoint.get('pi_optimized'):
            # Quantized model: cannot use state_dict, replace model entirely
            self.model = checkpoint['model'].to(self.device)
            logging.info("Loaded Pi-quantized model")

        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Calibrated model: TemperatureScaledModel saved a second temperature
            # on top of BabyCryClassifier's own self.temperature.
            # Fold both into model.temperature so forward() applies the correct scaling.
            if 'temperature' in checkpoint:
                self.model.temperature.data *= checkpoint['temperature']
                logging.info(f"Calibrated temperature {checkpoint['temperature']:.4f} applied")

        else:
            self.model.load_state_dict(checkpoint)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for continuous multi-channel audio capture."""
        if status:
            logging.warning(f"Audio callback status: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Reshape to preserve all channels (num_frames, num_channels)
        # This preserves phase relationships between channels for beamforming
        audio_data = audio_data.reshape(-1, self.num_channels)

        # Add to queue for processing
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            logging.warning("Audio queue full, dropping frame")

        return (in_data, pyaudio.paContinue)

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

        # Ensure correct duration
        target_length = int(self.config.DURATION * self.config.SAMPLE_RATE)
        if len(waveform) < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - len(waveform)))
        elif len(waveform) > target_length:
            waveform = waveform[:target_length]

        return waveform

    def audio_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to mel spectrogram."""
        waveform = waveform.to(self.device).unsqueeze(0)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale and normalize
        mel_spec = torch.log(mel_spec + 1e-9)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(0)  # (1, 1, n_mels, time)

        return mel_spec

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

                # Light noise
                noise = torch.randn_like(aug_spec) * 0.005
                aug_spec = aug_spec + noise

                predictions.append(self.model(aug_spec))

        return torch.mean(torch.stack(predictions), dim=0)

    def detect_cry(self, audio: np.ndarray, use_tta: bool = False) -> Tuple[bool, float]:
        """
        Detect baby cry in audio chunk.

        Args:
            audio: Audio numpy array
            use_tta: Whether to use TTA

        Returns:
            Tuple of (is_cry, confidence)
        """
        # Preprocess
        waveform = self.preprocess_audio(audio)
        spectrogram = self.audio_to_spectrogram(waveform)

        # Predict
        with torch.no_grad():
            if use_tta:
                logits = self.predict_with_tta(spectrogram)
            else:
                logits = self.model(spectrogram)

            probs = torch.softmax(logits, dim=1)
            cry_prob = probs[0, 1].item()
            is_cry = cry_prob > self.detection_threshold

        return is_cry, cry_prob

    def confirm_and_filter(self, audio: np.ndarray) -> DetectionResult:
        """
        Confirm detection with TTA and prepare multi-channel audio for localization.

        Args:
            audio: Multi-channel audio numpy array with shape (num_samples, num_channels)

        Returns:
            DetectionResult with preserved multi-channel audio for sound localization
        """
        # Confirm with TTA for higher accuracy
        if self.multichannel_detector and audio.ndim > 1:
            # Use dual-channel voting with TTA for robust confirmation
            mc_result, health_metrics = self.multichannel_detector.detect_with_health_check(
                audio,
                use_tta=True,
                confidence_threshold=self.confirmation_threshold
            )
            is_cry = mc_result.is_cry
            confidence = mc_result.confidence

            # Log channel health and detection details
            logging.info(
                f"Dual-channel confirmation: {is_cry} (conf: {confidence:.2%}), "
                f"channels: {mc_result.primary_channel}/{mc_result.secondary_channel}, "
                f"agreement: {mc_result.multi_channel_agreement:.2%}"
            )
            for metric in health_metrics:
                logging.debug(
                    f"  Channel {metric.channel_idx}: SNR={metric.snr_db:.1f}dB, "
                    f"RMS={metric.rms:.4f}, clipping={metric.clipping}"
                )
        else:
            # Fallback to single-channel detection with TTA
            is_cry, confidence = self.detect_cry(audio, use_tta=True)

        filtered_audio = None
        if is_cry and confidence >= self.confirmation_threshold:
            # Apply audio filtering for sound localization
            logging.info("Applying audio filtering for sound localization...")

            # For multi-channel audio, preserve all channels and phase relationships
            if audio.ndim > 1:
                # Use the multi-channel aware filter that preserves phase
                audio_tensor = torch.from_numpy(audio).float()
                filtered_tensor, cry_segments, _ = self.audio_filter.isolate_baby_cry_multichannel(
                    audio_tensor,
                    cry_threshold=self.detection_threshold
                )
                filtered_audio = filtered_tensor.numpy()
            else:
                # Fallback for mono audio
                audio_tensor = torch.from_numpy(audio).float()
                filtered_tensor, cry_segments, _ = self.audio_filter.isolate_baby_cry(
                    audio_tensor,
                    cry_threshold=self.detection_threshold
                )
                filtered_audio = filtered_tensor.numpy()

        return DetectionResult(
            is_cry=is_cry,
            confidence=confidence,
            timestamp=time.time(),
            audio_buffer=audio,
            filtered_audio=filtered_audio
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
            'sample_rate': self.config.SAMPLE_RATE,
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

    def process_audio_stream(self):
        """Main processing loop for audio stream with optional temporal smoothing."""
        logging.info("Audio processing thread started")

        while self.is_running:
            try:
                # Get audio chunk from queue (timeout for responsiveness)
                audio_chunk = self.audio_queue.get(timeout=0.5)

                # Add to circular buffer
                self.audio_buffer.add(audio_chunk)

                # Check cooldown
                if time.time() - self.last_detection_time < self.detection_cooldown:
                    continue

                if self.low_power_mode:
                    # Low-power mode: Quick detection without TTA
                    if self.multichannel_detector and audio_chunk.ndim > 1:
                        # Use adaptive dual-channel detection
                        mc_result = self.multichannel_detector.detect_cry_dual_channel(
                            audio_chunk,
                            use_tta=False,
                            confidence_threshold=self.detection_threshold
                        )
                        is_cry = mc_result.is_cry
                        confidence = mc_result.confidence
                        logging.debug(
                            f"Multi-channel detection: channels {mc_result.primary_channel}/{mc_result.secondary_channel}, "
                            f"SNRs: {mc_result.channel_snr_scores[mc_result.primary_channel]:.1f}/"
                            f"{mc_result.channel_snr_scores[mc_result.secondary_channel]:.1f} dB"
                        )
                    else:
                        # Fallback to single-channel detection
                        is_cry, confidence = self.detect_cry(audio_chunk, use_tta=False)

                    # Apply temporal smoothing if enabled
                    should_proceed_to_confirmation = False

                    if self.temporal_smoother is not None:
                        # Feed prediction to temporal smoother
                        smoothed_result = self.temporal_smoother.update(confidence)

                        logging.debug(
                            f"Temporal smoothing: raw={confidence:.2%}, "
                            f"smoothed={smoothed_result.smoothed_confidence:.2%}, "
                            f"consecutive={smoothed_result.consecutive_high_count}/{self.temporal_smoother.min_consecutive}"
                        )

                        # Only proceed if temporal criteria are met
                        if smoothed_result.should_alert:
                            should_proceed_to_confirmation = True
                            logging.info(
                                f"Temporal smoothing criteria met: "
                                f"{smoothed_result.consecutive_high_count} consecutive high-confidence predictions "
                                f"(smoothed confidence: {smoothed_result.smoothed_confidence:.2%})"
                            )
                    else:
                        # No temporal smoothing - use raw detection result
                        should_proceed_to_confirmation = is_cry

                    if should_proceed_to_confirmation:
                        logging.info(f"Potential cry detected (confidence: {confidence:.2%})")

                        # Get context audio (last 3-5 seconds)
                        context_audio = self.audio_buffer.get_last_n_seconds(3.0)

                        # Confirm with TTA and filter
                        detection = self.confirm_and_filter(context_audio)

                        if detection.is_cry and detection.confidence >= self.confirmation_threshold:
                            # Wake robot
                            self.wake_robot(detection)
                            self.last_detection_time = time.time()

                            # Reset temporal smoother after successful detection
                            if self.temporal_smoother is not None:
                                self.temporal_smoother.reset()
                        else:
                            logging.info(f"False positive filtered out (conf: {detection.confidence:.2%})")

                else:
                    # Active mode: Robot is navigating
                    # Continue monitoring but don't wake again
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio processing: {e}", exc_info=True)

    def start(self, stream_audio: bool = True):
        """
        Start real-time detection.

        Args:
            stream_audio: Whether to start audio streaming (set False for testing)
        """
        logging.info("Starting real-time baby cry detector...")
        self.is_running = True

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self.process_audio_stream,
            daemon=True
        )
        self.processing_thread.start()

        if stream_audio:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()

            # List available devices
            if self.audio_device_index is None:
                logging.info("Available audio devices:")
                for i in range(self.audio.get_device_count()):
                    info = self.audio.get_device_info_by_index(i)
                    logging.info(f"  [{i}] {info['name']} (channels: {info['maxInputChannels']})")

            # Open audio stream
            device_index = self.audio_device_index

            try:
                self.stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=self.num_channels,
                    rate=self.config.SAMPLE_RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )

                self.stream.start_stream()
                logging.info(f"Audio stream started (device: {device_index}, channels: {self.num_channels})")

            except Exception as e:
                logging.error(f"Error starting audio stream: {e}")
                logging.info("Detector running in test mode without audio input")

        logging.info("Real-time detector running in LOW-POWER MODE")

    def stop(self):
        """Stop real-time detection."""
        logging.info("Stopping real-time detector...")

        self.is_running = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        if hasattr(self, 'audio'):
            self.audio.terminate()

        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)

        logging.info("Real-time detector stopped")

    def reset_to_low_power(self):
        """Reset detector to low-power mode after robot task completion."""
        logging.info("Resetting to low-power listening mode")
        self.low_power_mode = True
        self.audio_buffer.clear()

        # Reset temporal smoother to clear prediction history
        if self.temporal_smoother is not None:
            self.temporal_smoother.reset()
            logging.debug("Temporal smoother reset")


def main():
    """Command-line interface for real-time detection."""
    parser = argparse.ArgumentParser(description='Real-Time Baby Cry Detection for Raspberry Pi')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device-index', type=int, default=None,
                       help='Audio device index for microphone array')
    parser.add_argument('--channels', type=int, default=4,
                       help='Number of microphone channels (default: 4)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (default: 0.5)')
    parser.add_argument('--confirm-threshold', type=float, default=0.85,
                       help='Confirmation threshold for wake-up (default: 0.85)')
    parser.add_argument('--no-tta', action='store_true',
                       help='Disable TTA for confirmation (faster)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode without audio input')
    parser.add_argument('--multichannel-voting', type=str, default='weighted',
                       choices=['weighted', 'logical_or'],
                       help='Multi-channel voting strategy (default: weighted)')
    parser.add_argument('--disable-multichannel', action='store_true',
                       help='Disable multi-channel detection (use Channel 0 only)')
    parser.add_argument('--debug-channels', action='store_true',
                       help='Enable detailed channel statistics logging')

    # Temporal smoothing arguments
    parser.add_argument('--disable-temporal-smoothing', action='store_true',
                       help='Disable temporal smoothing (alert on single detection)')
    parser.add_argument('--temporal-window', type=int, default=5,
                       help='Temporal smoothing window size (default: 5)')
    parser.add_argument('--temporal-consecutive', type=int, default=3,
                       help='Minimum consecutive high-confidence predictions required (default: 3)')
    parser.add_argument('--temporal-threshold', type=float, default=0.5,
                       help='Confidence threshold for temporal smoothing (default: 0.5)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug_channels else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize detector
    detector = RealtimeBabyCryDetector(
        model_path=args.model,
        use_tta=not args.no_tta,
        detection_threshold=args.threshold,
        confirmation_threshold=args.confirm_threshold,
        device=args.device,
        audio_device_index=args.device_index,
        num_channels=args.channels,
        enable_multichannel=not args.disable_multichannel,
        multichannel_voting=args.multichannel_voting,
        enable_temporal_smoothing=not args.disable_temporal_smoothing,
        temporal_window_size=args.temporal_window,
        temporal_min_consecutive=args.temporal_consecutive,
        temporal_confidence_threshold=args.temporal_threshold
    )

    # Optional: Set callback
    def on_cry_callback(detection: DetectionResult):
        print(f"\n{'='*70}")
        print(f"BABY CRY ALERT!")
        print(f"  Confidence: {detection.confidence:.1%}")
        print(f"  Timestamp: {time.strftime('%H:%M:%S', time.localtime(detection.timestamp))}")
        print(f"  Filtered audio ready for sound localization")
        print(f"{'='*70}\n")

    detector.on_cry_detected = on_cry_callback

    # Start detector
    try:
        detector.start(stream_audio=not args.test_mode)

        print("\n" + "="*70)
        print("Real-Time Baby Cry Detector - ACTIVE")
        print("="*70)
        print(f"Mode: LOW-POWER LISTENING")
        print(f"Microphone Channels: {args.channels}")
        if not args.disable_multichannel and args.channels > 1:
            print(f"Multi-Channel Detection: ENABLED ({args.multichannel_voting} voting)")
        else:
            print(f"Multi-Channel Detection: DISABLED (using Channel 0 only)")
        print(f"Detection Threshold: {args.threshold:.0%}")
        print(f"Confirmation Threshold: {args.confirm_threshold:.0%}")
        if not args.disable_temporal_smoothing:
            print(f"Temporal Smoothing: ENABLED")
            print(f"  - Window Size: {args.temporal_window} predictions")
            print(f"  - Min Consecutive: {args.temporal_consecutive} high-confidence")
            print(f"  - Confidence Threshold: {args.temporal_threshold:.0%}")
        else:
            print(f"Temporal Smoothing: DISABLED")
        print(f"Device: {args.device}")
        if args.debug_channels:
            print(f"Debug Mode: ENABLED (detailed channel stats)")
        print("="*70)
        print("\nPress Ctrl+C to stop\n")

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        detector.stop()
        print("Detector stopped successfully")


if __name__ == "__main__":
    main()
