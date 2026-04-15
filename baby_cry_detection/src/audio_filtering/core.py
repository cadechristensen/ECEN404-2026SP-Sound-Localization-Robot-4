"""
Core BabyCryAudioFilter class - main interface for baby cry detection and isolation.

This module orchestrates all filtering components for professional, modular operation.
Optimized for Raspberry Pi 5 8GB deployment with 4-channel microphone array support.
"""

import os
import logging
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Tuple, List, Optional, Literal
import warnings
# Suppress only torchaudio backend deprecation notices; do not silence all warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

from .filters import SpectralFilters
from .detection import AcousticDetector
from .noise_reduction import NoiseReducer
from .classification import AudioClassifier
from .multichannel import MultichannelProcessor
from .utils import merge_overlapping_segments


class BabyCryAudioFilter:
    """
    Advanced audio filtering system to isolate baby cries from mixed audio.

    Combines spectral filtering, voice activity detection, acoustic feature detection,
    and deep learning classification in a modular architecture.

    Supports multi-channel audio processing with phase preservation for sound localization.
    """

    def __init__(self, config: Optional['Config'] = None, model_path: Optional[str] = None,
                 calibrator_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize the baby cry audio filter.

        Args:
            config: Configuration object
            model_path: Path to trained baby cry detection model
            calibrator_path: Path to confidence calibrator (optional)
            verbose: Enable verbose output (default: True)
        """
        self.verbose = verbose
        from ..config import Config
        from ..data_preprocessing import AudioPreprocessor
        from ..acoustic_features import AcousticFeatureExtractor

        self.config = config if config is not None else Config()
        self.sample_rate = self.config.SAMPLE_RATE

        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(self.config)

        # Initialize acoustic feature extractor
        self.acoustic_extractor = AcousticFeatureExtractor(sample_rate=self.sample_rate)

        # Initialize modular components
        self.spectral_filters = SpectralFilters(
            sample_rate=self.sample_rate,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )

        self.acoustic_detector = AcousticDetector(sample_rate=self.sample_rate)

        self.noise_reducer = NoiseReducer(
            sample_rate=self.sample_rate,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )

        self.classifier = AudioClassifier(
            config=self.config,
            preprocessor=self.preprocessor,
            acoustic_extractor=self.acoustic_extractor
        )

        self.multichannel = MultichannelProcessor(
            sample_rate=self.sample_rate,
            num_channels=4  # 4-channel mic array
        )

        # Initialize transforms
        self._init_transforms()

        # Load trained model for cry detection
        if model_path and os.path.exists(model_path):
            self.classifier.load_model(model_path)

        # Load confidence calibrator
        if calibrator_path and os.path.exists(calibrator_path):
            self.classifier.load_calibrator(calibrator_path, self.config.NUM_CLASSES)

    @property
    def model(self):
        """Backward compatibility: expose classifier's model."""
        return self.classifier.model

    def spectral_filter(self, audio: torch.Tensor) -> torch.Tensor:
        """Backward compatibility wrapper for spectral filtering."""
        return self.spectral_filters.bandpass_filter(audio)

    def voice_activity_detection(self, audio: torch.Tensor) -> torch.Tensor:
        """Backward compatibility wrapper for voice activity detection."""
        return self.noise_reducer.voice_activity_detection(audio)

    def spectral_subtraction(self, audio: torch.Tensor) -> torch.Tensor:
        """Backward compatibility wrapper for spectral subtraction."""
        return self.noise_reducer.spectral_subtraction(audio)

    def compute_acoustic_features(self, audio: torch.Tensor) -> dict:
        """Backward compatibility wrapper for acoustic feature computation."""
        return self.acoustic_detector.compute_all_features(
            audio, frame_length=self.config.N_FFT, hop_length=self.config.HOP_LENGTH
        )

    def classify_audio_segments(self, audio: torch.Tensor,
                               use_acoustic_validation: bool = False) -> list:
        """Backward compatibility wrapper for audio segment classification."""
        return self.classifier.classify_segments(
            audio,
            use_acoustic_validation=use_acoustic_validation
        )

    def _init_transforms(self):
        """Initialize audio processing transforms."""
        # Spectral transforms
        self.stft_transform = T.Spectrogram(
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            power=2.0
        )

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            f_min=self.config.F_MIN,
            f_max=self.config.F_MAX
        )

        # GriffinLim is not called in the current pipeline (phase is preserved via
        # iSTFT); it is kept here for potential offline reconstruction use.
        # n_iter=32 avoids the 3× latency cost of n_iter=100 if it is ever activated.
        self.griffin_lim = T.GriffinLim(
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_iter=32
        )

    def isolate_baby_cry(self, audio: torch.Tensor,
                        cry_threshold: float = 0.5,
                        use_acoustic_features: bool = False) -> Tuple[torch.Tensor, List[Tuple[float, float]], List[Tuple[float, float, float]]]:
        """
        Main function to isolate baby cry from mixed audio.

        Binary classification mode: ML model predictions with optional acoustic validation.

        Args:
            audio: Input audio tensor (mono)
            cry_threshold: Probability threshold for cry detection
            use_acoustic_features: Whether to compute acoustic features for monitoring (default: False)

        Returns:
            Tuple of (isolated_cry_audio, cry_time_segments, all_segments_with_probs)
            all_segments_with_probs contains (start_time, end_time, probability) for cry_percentage calculation
        """
        if self.verbose:
            print("Step 1: Applying spectral filtering...")
        # Step 1: Spectral filtering
        filtered_audio = self.spectral_filters.bandpass_filter(audio)

        if self.verbose:
            print("Step 2: Voice activity detection...")
        # Step 2: Voice activity detection
        vad_mask = self.noise_reducer.voice_activity_detection(filtered_audio)

        # Step 3: Acoustic feature analysis (optional)
        acoustic_segments = []
        if use_acoustic_features:
            if self.verbose:
                print("Step 3: Analyzing acoustic features...")
            features = self._compute_acoustic_features(filtered_audio)
            acoustic_segments = self.acoustic_detector.combine_scores(features)
            if self.verbose:
                print(f"  Found {len(acoustic_segments)} segments from acoustic analysis")

        if self.verbose:
            print("Step 4: Classifying audio segments with ML model and acoustic validation...")
        # Step 4: Classify segments with ML model using RAW audio (same as training)
        ml_segments = self.classifier.classify_segments(
            audio,
            use_acoustic_validation=use_acoustic_features
        )

        # Step 5: Process results - ALWAYS use ML-only scoring for binary classification
        if self.verbose:
            print("Step 5: Processing results...")
            print(f"  Using ML-ONLY scoring (binary classification)")
        segments = [(s, e, p) for s, e, p, meta in ml_segments]

        # Log acoustic features if enabled (for monitoring only)
        if use_acoustic_features and len(acoustic_segments) > 0 and self.verbose:
            print(f"  Acoustic features computed for monitoring (not used in scoring)")

        if self.verbose:
            print("Step 6: Applying spectral subtraction...")
        # Step 6: Noise reduction
        denoised_audio = self.noise_reducer.spectral_subtraction(filtered_audio)

        if self.verbose:
            print("Step 7: Extracting cry segments...")

        logging.debug("Found %d segments to evaluate", len(segments))
        for i, (start, end, prob) in enumerate(segments[:10]):
            logging.debug("  Segment %d: %.2fs-%.2fs, score=%.3f, threshold=%.2f",
                          i + 1, start, end, prob, cry_threshold)

        cry_segments = [(start, end) for start, end, prob in segments if prob >= cry_threshold]
        logging.debug("After filtering with threshold %.2f: %d cry segments",
                      cry_threshold, len(cry_segments))

        # Create mask for cry regions
        cry_mask = torch.zeros_like(audio, dtype=torch.bool)
        for start_time, end_time in cry_segments:
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            cry_mask[start_idx:end_idx] = True

        # Apply masks
        isolated_audio = torch.zeros_like(denoised_audio)
        isolated_audio[cry_mask] = denoised_audio[cry_mask]

        return isolated_audio, cry_segments, segments

    def isolate_baby_cry_multichannel(self, audio: torch.Tensor,
                                     cry_threshold: float = 0.5,
                                     use_acoustic_features: bool = False) -> Tuple[torch.Tensor, List[Tuple[float, float]], List[Tuple[float, float, float]]]:
        """
        Isolate baby cry from multi-channel audio while preserving phase relationships.

        This method processes the primary channel for cry detection but preserves
        all channels for sound localization with intact phase information.

        Binary classification mode: ML model predictions with optional acoustic validation.

        Args:
            audio: Multi-channel input audio tensor with shape (num_samples, num_channels)
            cry_threshold: Probability threshold for cry detection
            use_acoustic_features: Whether to compute acoustic features for monitoring (default: False)

        Returns:
            Tuple of (isolated_multichannel_audio, cry_time_segments, all_segments_with_probs)
            All channels are preserved with phase relationships intact
            all_segments_with_probs contains (start_time, end_time, probability) for cry_percentage calculation
        """
        # Handle both mono and multi-channel input
        if audio.dim() == 1:
            # Single channel - use original method
            return self.isolate_baby_cry(audio, cry_threshold, use_acoustic_features)

        # Multi-channel audio
        num_channels = audio.shape[1] if audio.dim() > 1 else 1

        # Select best channel by cry-band SNR (300-900 Hz)
        best_ch, snr_scores = self.multichannel.select_best_channel_by_snr(audio)
        primary_channel = audio[:, best_ch]

        if self.verbose:
            snr_str = "  ".join(f"Ch{i}: {snr_scores[i]:5.1f} dB" for i in range(num_channels))
            print(f"Step 0: Channel selection by cry-band SNR: {snr_str}")
            print(f"        Selected channel {best_ch}")
            print(f"Step 1: Applying spectral filtering (channel {best_ch})...")
        # Spectral filtering on primary channel
        filtered_primary = self.spectral_filters.bandpass_filter(primary_channel)

        if self.verbose:
            print("Step 2: Voice activity detection...")
        # VAD on primary channel
        vad_mask = self.noise_reducer.voice_activity_detection(filtered_primary)

        # Acoustic feature analysis (optional)
        acoustic_segments = []
        if use_acoustic_features:
            if self.verbose:
                print("Step 3: Analyzing acoustic features (primary channel)...")
            features = self._compute_acoustic_features(filtered_primary)
            acoustic_segments = self.acoustic_detector.combine_scores(features)
            if self.verbose:
                print(f"  Found {len(acoustic_segments)} segments from acoustic analysis")

        if self.verbose:
            print("Step 4: Classifying audio segments with ML model...")
        # Classify segments with ML model using RAW audio (same as training)
        ml_segments = self.classifier.classify_segments(
            primary_channel,
            use_acoustic_validation=use_acoustic_features
        )

        # Process results - ALWAYS use ML-only scoring for binary classification
        if self.verbose:
            print("Step 5: Processing results...")
            print(f"  Using ML-ONLY scoring (binary classification)")
        segments = [(s, e, p) for s, e, p, meta in ml_segments]

        # Log acoustic features if enabled (for monitoring only)
        if use_acoustic_features and len(acoustic_segments) > 0 and self.verbose:
            print(f"  Acoustic features computed for monitoring (not used in scoring)")

        if self.verbose:
            print("Step 6: Applying spectral subtraction (primary channel)...")
        # Noise reduction on primary channel
        denoised_primary = self.noise_reducer.spectral_subtraction(filtered_primary)

        if self.verbose:
            print("Step 7: Extracting cry segments...")

        logging.debug("Found %d segments to evaluate", len(segments))
        for i, (start, end, prob) in enumerate(segments[:10]):
            logging.debug("  Segment %d: %.2fs-%.2fs, score=%.3f, threshold=%.2f",
                          i + 1, start, end, prob, cry_threshold)

        cry_segments = [(start, end) for start, end, prob in segments if prob >= cry_threshold]
        logging.debug("After filtering with threshold %.2f: %d cry segments",
                      cry_threshold, len(cry_segments))

        # Create mask for cry regions
        cry_mask = torch.zeros(len(denoised_primary), dtype=torch.bool)
        for start_time, end_time in cry_segments:
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            cry_mask[start_idx:end_idx] = True

        # Apply mask to ALL channels (preserving phase)
        if self.verbose:
            print(f"Step 8: Applying cry mask to {num_channels} channels...")
        isolated_audio_multichannel = self.multichannel.apply_mask_multichannel(audio, cry_mask)

        return isolated_audio_multichannel, cry_segments, segments

    def _compute_acoustic_features(self, audio: torch.Tensor) -> dict:
        """
        Compute all acoustic features for baby cry detection.

        Args:
            audio: Input audio tensor

        Returns:
            Dictionary containing all acoustic feature scores
        """
        frame_length = self.config.N_FFT
        hop_length = self.config.HOP_LENGTH

        if self.verbose:
            print("  Computing harmonic structure...")
        features = self.acoustic_detector.compute_all_features(audio, frame_length, hop_length)

        if self.verbose:
            print("  Computing energy distribution...")
        energy_scores = self.spectral_filters.analyze_energy_distribution(audio, frame_length, hop_length)
        features['energy_scores'] = energy_scores

        if self.verbose:
            print("  Computing rejection filters...")
        adult_rejection = self.spectral_filters.filter_adult_speech(audio, frame_length, hop_length)
        music_rejection = self.spectral_filters.filter_music(audio, frame_length, hop_length)
        env_rejection = self.spectral_filters.filter_environmental_sounds(audio, frame_length, hop_length)

        features['adult_rejection'] = adult_rejection
        features['music_rejection'] = music_rejection
        features['env_rejection'] = env_rejection

        return features

    def extract_cry_segments_only(self, audio: torch.Tensor,
                                 cry_segments: List[Tuple[float, float]]) -> torch.Tensor:
        """
        Extract and concatenate only the cry segments from audio.

        This method extracts detected cry segments and concatenates them together,
        removing all silence and non-cry portions. Useful for analysis and storage efficiency.

        IMPORTANT: This method merges overlapping segments before extraction to ensure
        the output duration matches the reported cry_duration.

        Args:
            audio: Input audio tensor (can be mono or multi-channel)
                  Shape: (num_samples,) for mono or (num_samples, num_channels) for multi-channel
            cry_segments: List of (start_time, end_time) tuples in seconds

        Returns:
            Concatenated cry-only audio tensor with same shape format as input
            Returns empty tensor if no cry segments detected
        """
        if len(cry_segments) == 0:
            if audio.dim() > 1:
                return torch.zeros((0, audio.shape[1]), dtype=audio.dtype)
            else:
                return torch.zeros(0, dtype=audio.dtype)

        if audio.dim() > 1:
            # Multi-channel
            return self.multichannel.extract_cry_segments_multichannel(
                audio, cry_segments, merge_overlapping_segments
            )

        # Mono
        merged_segments = merge_overlapping_segments(cry_segments)

        extracted_segments = []

        for start_time, end_time in merged_segments:
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)

            start_idx = max(0, start_idx)
            end_idx = min(len(audio), end_idx)

            if start_idx < end_idx:
                segment = audio[start_idx:end_idx]
                extracted_segments.append(segment)

        if not extracted_segments:
            return torch.zeros(0, dtype=audio.dtype)

        concatenated = torch.cat(extracted_segments, dim=0)

        return concatenated

    def process_audio_file(self, input_path: str, output_path: str,
                          cry_threshold: float = 0.5,
                          use_acoustic_features: bool = False,
                          output_mode: Literal["full_length", "cry_only", "both"] = "full_length") -> dict:
        """
        Process an audio file to extract baby cries.

        Binary classification mode: ML model predictions with optional acoustic validation.

        Args:
            input_path: Path to input audio file
            output_path: Path to save filtered audio (for full_length mode)
                        For cry_only mode, "_cry_only" is appended before extension
                        For both mode, both files are saved
            cry_threshold: Probability threshold for cry detection
            use_acoustic_features: Whether to compute acoustic features for monitoring (default: False)
            output_mode: Output format mode:
                        "full_length" - Preserves original length, mutes non-cry (default)
                        "cry_only" - Concatenated cry segments only, removes silence
                        "both" - Saves both versions with different filenames

        Returns:
            Processing results dictionary with keys:
                - All standard keys (input_file, total_duration, etc.)
                - output_files: dict mapping mode to output path(s)
                - cry_only_duration: Duration of cry_only output (if applicable)
        """
        if self.verbose:
            print(f"Processing audio file: {input_path}")
            print(f"Output mode: {output_mode}")
            print(f"Acoustic features: {'ENABLED' if use_acoustic_features else 'DISABLED'}")

        # Load audio
        audio, sample_rate = torchaudio.load(input_path)

        # Preserve multi-channel if present, otherwise use first channel
        if audio.shape[0] > 1:
            # Multi-channel - preserve all channels
            audio = audio.transpose(0, 1)  # (channels, samples) -> (samples, channels)
            if self.verbose:
                print(f"Loaded multi-channel audio: {audio.shape}")
        else:
            # Single channel
            audio = audio[0]
            if self.verbose:
                print(f"Loaded mono audio: {audio.shape}")

        # Resample if necessary
        if sample_rate != self.sample_rate:
            audio = self.multichannel.resample_multichannel(audio, sample_rate, self.sample_rate)

        # Process audio with acoustic features
        if audio.dim() > 1:
            # Multi-channel
            isolated_audio, cry_segments, all_segments = self.isolate_baby_cry_multichannel(
                audio,
                cry_threshold,
                use_acoustic_features=use_acoustic_features
            )
        else:
            # Mono
            isolated_audio, cry_segments, all_segments = self.isolate_baby_cry(
                audio,
                cry_threshold,
                use_acoustic_features=use_acoustic_features
            )

        # Generate output paths based on mode
        output_files = {}
        cry_only_audio = None
        cry_only_duration = 0.0

        # Determine which outputs to save
        save_full_length = output_mode in ["full_length", "both"]
        save_cry_only = output_mode in ["cry_only", "both"]

        # Save full-length filtered audio
        if save_full_length:
            self.multichannel.save_multichannel(isolated_audio, output_path, self.sample_rate)
            output_files["full_length"] = output_path
            if self.verbose:
                print(f"Saved full-length filtered audio to: {output_path}")

        # Save cry-only concatenated audio
        if save_cry_only:
            if len(cry_segments) == 0:
                if self.verbose:
                    print("Warning: No cry segments detected. Cry-only output will be empty.")
                cry_only_duration = 0.0
            else:
                cry_only_audio = self.extract_cry_segments_only(audio, cry_segments)
                cry_only_duration = len(cry_only_audio) / self.sample_rate

                output_path_obj = Path(output_path)
                cry_only_path = output_path_obj.parent / f"{output_path_obj.stem}_cry_only{output_path_obj.suffix}"

                self.multichannel.save_multichannel(cry_only_audio, str(cry_only_path), self.sample_rate)
                output_files["cry_only"] = str(cry_only_path)
                if self.verbose:
                    print(f"Saved cry-only audio to: {cry_only_path} (duration: {cry_only_duration:.2f}s)")

        # Calculate statistics — shape[0] is num_samples for both mono (1D) and multichannel (2D)
        total_duration = audio.shape[0] / self.sample_rate

        # Merge overlapping cry segments before calculating duration
        merged_segments = merge_overlapping_segments(cry_segments)
        cry_duration = sum(end - start for start, end in merged_segments)

        # Calculate duration percentage (time-based)
        duration_percentage = (cry_duration / total_duration) * 100 if total_duration > 0 else 0.0

        # Calculate average model confidence (softmax probability-based)
        if len(all_segments) > 0:
            cry_probs = [prob for start, end, prob in all_segments if prob >= cry_threshold]
            if cry_probs:
                avg_confidence = (sum(cry_probs) / len(cry_probs)) * 100
                min_confidence = min(cry_probs) * 100
                max_confidence = max(cry_probs) * 100
            else:
                avg_confidence = 0.0
                min_confidence = 0.0
                max_confidence = 0.0
        else:
            avg_confidence = 0.0
            min_confidence = 0.0
            max_confidence = 0.0

        # Create segments with probabilities for cry segments only
        cry_segments_with_prob = [(start, end, prob) for start, end, prob in all_segments if prob >= cry_threshold]

        results = {
            'input_file': input_path,
            'output_file': output_path,
            'output_files': output_files,
            'output_mode': output_mode,
            'total_duration': total_duration,
            'cry_duration': cry_duration,
            'cry_only_duration': cry_only_duration,
            'duration_percentage': duration_percentage,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'max_confidence': max_confidence,
            'cry_segments': cry_segments,
            'cry_segments_with_prob': cry_segments_with_prob,
            'num_cry_segments': len(cry_segments),
            'acoustic_features_used': use_acoustic_features
        }

        if self.verbose:
            print(f"Processing complete:")
            print(f"  Total duration: {total_duration:.2f}s")
            print(f"  Cry duration: {cry_duration:.2f}s ({duration_percentage:.1f}% of file)")
            if save_cry_only:
                print(f"  Cry-only duration: {cry_only_duration:.2f}s")
            print(f"  Cry segments found: {len(cry_segments)}")
            print(f"  Average confidence: {avg_confidence:.1f}% (range: {min_confidence:.1f}%-{max_confidence:.1f}%)")

        return results


def create_audio_filter(config = None, model_path: Optional[str] = None,
                       calibrator_path: Optional[str] = None) -> BabyCryAudioFilter:
    """
    Create and return a baby cry audio filter.

    Args:
        config: Configuration object
        model_path: Path to trained model
        calibrator_path: Path to confidence calibrator

    Returns:
        Initialized audio filter
    """
    return BabyCryAudioFilter(config, model_path, calibrator_path)
