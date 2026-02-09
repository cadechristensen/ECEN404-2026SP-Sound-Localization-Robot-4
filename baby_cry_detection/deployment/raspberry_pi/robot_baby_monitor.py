"""
Integrated Robot Baby Monitor System
Combines real-time baby cry detection with sound localization for robot navigation.
Optimized for Raspberry Pi 5 with TI PCM6260-Q1 microphone array.
"""

import os
import sys
import multiprocessing as mp
import logging
import time
import signal
import argparse
from pathlib import Path
from typing import Optional

# Insert the project root (two directories up) so that src.* imports resolve
# in this script and in any modules it imports (e.g. realtime_baby_cry_detector).
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Local deployment modules (live alongside this file)
from realtime_baby_cry_detector import RealtimeBabyCryDetector
from detection_types import DetectionResult
from sound_localization_interface import run_localization_process, LocalizationResult


class RobotBabyMonitor:
    """
    Complete baby monitoring system integrating cry detection and sound localization.
    """

    def __init__(
        self,
        model_path: str,
        audio_device_index: Optional[int] = None,
        num_channels: int = 4,
        detection_threshold: float = 0.5,
        confirmation_threshold: float = 0.85,
        device: str = 'cpu',
        enable_multichannel: bool = True,
        multichannel_voting: str = "weighted",
        enable_temporal_smoothing: bool = True,
        temporal_window_size: int = 5,
        temporal_min_consecutive: int = 3,
        temporal_confidence_threshold: float = 0.5
    ):
        """
        Initialize robot baby monitor.

        Args:
            model_path: Path to trained cry detection model
            audio_device_index: Microphone array device index
            num_channels: Number of microphone channels
            detection_threshold: Initial detection threshold
            confirmation_threshold: Confirmation threshold for wake-up
            device: Processing device (cpu/cuda)
            enable_multichannel: Enable multi-channel detection voting
            multichannel_voting: Voting strategy ('weighted' or 'logical_or')
            enable_temporal_smoothing: Enable temporal smoothing across frames
            temporal_window_size: Number of frames in the temporal smoothing window
            temporal_min_consecutive: Minimum consecutive high-confidence frames required
            temporal_confidence_threshold: Per-frame confidence threshold for temporal smoothing
        """
        self.model_path = model_path
        self.audio_device_index = audio_device_index
        self.num_channels = num_channels
        self.detection_threshold = detection_threshold
        self.confirmation_threshold = confirmation_threshold
        self.device = device
        self.enable_multichannel = enable_multichannel
        self.multichannel_voting = multichannel_voting
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.temporal_window_size = temporal_window_size
        self.temporal_min_consecutive = temporal_min_consecutive
        self.temporal_confidence_threshold = temporal_confidence_threshold

        # IPC queue for detection -> localization
        self.detection_queue = mp.Queue(maxsize=10)

        # Process handles
        self.localization_process = None
        self.detector = None

        logging.info("Robot Baby Monitor initialized")

    def start_localization_process(self):
        """Start sound localization in separate process."""
        logging.info("Starting sound localization process...")

        self.localization_process = mp.Process(
            target=run_localization_process,
            args=(self.detection_queue,),
            daemon=True
        )
        self.localization_process.start()

        logging.info("Sound localization process started")

    def start_cry_detector(self):
        """Start real-time cry detector."""
        logging.info("Starting baby cry detector...")

        self.detector = RealtimeBabyCryDetector(
            model_path=self.model_path,
            use_tta=True,  # Use TTA for confirmation
            detection_threshold=self.detection_threshold,
            confirmation_threshold=self.confirmation_threshold,
            device=self.device,
            audio_device_index=self.audio_device_index,
            num_channels=self.num_channels,
            enable_multichannel=self.enable_multichannel,
            multichannel_voting=self.multichannel_voting,
            enable_temporal_smoothing=self.enable_temporal_smoothing,
            temporal_window_size=self.temporal_window_size,
            temporal_min_consecutive=self.temporal_min_consecutive,
            temporal_confidence_threshold=self.temporal_confidence_threshold
        )

        # Set detection queue for localization
        self.detector.detection_queue = self.detection_queue

        # Optional: Add custom callback
        def on_detection(detection: DetectionResult):
            print(f"\n{'='*70}")
            print(f"BABY CRY DETECTED!")
            print(f"{'='*70}")
            print(f"Confidence: {detection.confidence:.1%}")
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print(f"Status: Waking robot and running localization...")
            print(f"{'='*70}\n")

        self.detector.on_cry_detected = on_detection

        # Start detector
        self.detector.start(stream_audio=True)

        logging.info("Baby cry detector started")

    def start(self):
        """Start the complete monitoring system."""
        print("\n" + "="*70)
        print("ROBOT BABY MONITOR - STARTING")
        print("="*70)
        print(f"Model: {Path(self.model_path).name}")
        print(f"Microphone Channels: {self.num_channels}")
        print(f"Detection Threshold: {self.detection_threshold:.0%}")
        print(f"Confirmation Threshold: {self.confirmation_threshold:.0%}")
        print(f"Processing Device: {self.device}")
        if self.enable_multichannel:
            print(f"Multi-Channel Detection: ENABLED ({self.multichannel_voting} voting)")
        else:
            print(f"Multi-Channel Detection: DISABLED (Channel 0 only)")
        if self.enable_temporal_smoothing:
            print(f"Temporal Smoothing: ENABLED (window={self.temporal_window_size}, "
                  f"consecutive={self.temporal_min_consecutive})")
        else:
            print(f"Temporal Smoothing: DISABLED")
        print("="*70 + "\n")

        # Start localization process
        self.start_localization_process()

        # Small delay to ensure process is ready
        time.sleep(1)

        # Start cry detector
        self.start_cry_detector()

        print("\n" + "="*70)
        print("SYSTEM ACTIVE - LOW POWER LISTENING MODE")
        print("="*70)
        print("Listening for baby cries...")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")

    def stop(self):
        """Stop the monitoring system."""
        logging.info("Stopping Robot Baby Monitor...")

        # Stop detector
        if self.detector:
            self.detector.stop()

        # Terminate localization process
        if self.localization_process and self.localization_process.is_alive():
            self.localization_process.terminate()
            self.localization_process.join(timeout=2.0)

        logging.info("Robot Baby Monitor stopped")

    def run(self):
        """Run the monitoring system."""
        try:
            self.start()

            # Keep running
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("SHUTDOWN INITIATED")
            print("="*70)
            self.stop()
            print("\nSystem stopped successfully\n")


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    print("\n\nReceived shutdown signal...")
    sys.exit(0)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Robot Baby Monitor - Integrated Cry Detection & Localization'
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained baby cry detection model')
    parser.add_argument('--device-index', type=int, default=None,
                       help='Audio device index for microphone array')
    parser.add_argument('--channels', type=int, default=4,
                       help='Number of microphone channels (default: 4)')
    parser.add_argument('--detection-threshold', type=float, default=0.5,
                       help='Detection threshold (default: 0.5)')
    parser.add_argument('--confirmation-threshold', type=float, default=0.85,
                       help='Confirmation threshold for wake-up (default: 0.85)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Processing device (default: cpu)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--multichannel-voting', type=str, default='weighted',
                       choices=['weighted', 'logical_or'],
                       help='Multi-channel voting strategy (default: weighted)')
    parser.add_argument('--disable-multichannel', action='store_true',
                       help='Disable multi-channel detection (use Channel 0 only)')
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
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Verify model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Create and run monitor
    monitor = RobotBabyMonitor(
        model_path=args.model,
        audio_device_index=args.device_index,
        num_channels=args.channels,
        detection_threshold=args.detection_threshold,
        confirmation_threshold=args.confirmation_threshold,
        device=args.device,
        enable_multichannel=not args.disable_multichannel,
        multichannel_voting=args.multichannel_voting,
        enable_temporal_smoothing=not args.disable_temporal_smoothing,
        temporal_window_size=args.temporal_window,
        temporal_min_consecutive=args.temporal_consecutive,
        temporal_confidence_threshold=args.temporal_threshold
    )

    monitor.run()


if __name__ == "__main__":
    main()
