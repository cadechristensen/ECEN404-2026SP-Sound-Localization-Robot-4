"""
Real-Time Baby Cry Detection — CLI entry point.

Parses command-line arguments and wires BabyCryDetector + AudioPipeline
for live operation on the Raspberry Pi.
"""

import os
import sys
import time
import logging
import argparse

# Insert the project root so that src.* imports resolve
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config_pi import ConfigPi
from detector import BabyCryDetector, DetectionResult
from audio_pipeline import AudioPipeline

# Backward-compat alias so existing code that does
#   from realtime_baby_cry_detector import RealtimeBabyCryDetector
# keeps working without changes.
RealtimeBabyCryDetector = BabyCryDetector


def main():
    """Command-line interface for real-time detection."""
    _config = ConfigPi()

    parser = argparse.ArgumentParser(description='Real-Time Baby Cry Detection for Raspberry Pi')
    parser.add_argument('--model', type=str, default=None,
                       help=f'Path to trained model checkpoint (default: from config: {_config.MODEL_PATH})')
    parser.add_argument('--device-index', type=int, default=None,
                       help='Audio device index for microphone array')
    parser.add_argument('--channels', type=int, default=_config.PI_CHANNELS,
                       help=f'Number of microphone channels (default: {_config.PI_CHANNELS})')
    parser.add_argument('--threshold', type=float, default=_config.DETECTION_THRESHOLD,
                       help=f'Detection threshold (default: {_config.DETECTION_THRESHOLD})')
    parser.add_argument('--confirm-threshold', type=float, default=_config.CONFIRMATION_THRESHOLD,
                       help=f'Confirmation threshold for wake-up (default: {_config.CONFIRMATION_THRESHOLD})')
    parser.add_argument('--no-tta', action='store_true',
                       help='Disable TTA for confirmation (faster)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode without audio input')
    parser.add_argument('--multichannel-voting', type=str, default=None,
                       choices=['weighted', 'logical_or'],
                       help='Multi-channel voting strategy (default: from config.PI_MULTICHANNEL_VOTING)')
    parser.add_argument('--disable-multichannel', action='store_true',
                       help='Disable multi-channel detection (use Channel 0 only)')
    parser.add_argument('--debug-channels', action='store_true',
                       help='Enable detailed channel statistics logging')

    # Temporal smoothing arguments
    parser.add_argument('--disable-temporal-smoothing', action='store_true',
                       help='Disable temporal smoothing (alert on single detection)')
    parser.add_argument('--temporal-window', type=int, default=_config.TEMPORAL_WINDOW_SIZE,
                       help=f'Temporal smoothing window size (default: {_config.TEMPORAL_WINDOW_SIZE})')
    parser.add_argument('--temporal-consecutive', type=int, default=_config.TEMPORAL_MIN_CONSECUTIVE,
                       help=f'Minimum consecutive high-confidence predictions required (default: {_config.TEMPORAL_MIN_CONSECUTIVE})')
    parser.add_argument('--temporal-threshold', type=float, default=_config.TEMPORAL_CONFIDENCE_THRESHOLD,
                       help=f'Confidence threshold for temporal smoothing (default: {_config.TEMPORAL_CONFIDENCE_THRESHOLD})')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug_channels else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize detector
    detector = BabyCryDetector(
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

    # Create audio pipeline
    pipeline = AudioPipeline(detector)

    # Start detector
    try:
        pipeline.start(stream_audio=not args.test_mode)

        print("\n" + "="*70)
        print("Real-Time Baby Cry Detector - ACTIVE")
        print("="*70)
        print(f"Mode: LOW-POWER LISTENING")
        print(f"Microphone Channels: {args.channels}")
        if not args.disable_multichannel and args.channels > 1:
            effective_voting = args.multichannel_voting or getattr(
                detector.config, 'PI_MULTICHANNEL_VOTING', 'weighted'
            )
            print(f"Multi-Channel Detection: ENABLED ({effective_voting} voting)")
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
        pipeline.stop()
        print("Detector stopped successfully")


if __name__ == "__main__":
    main()
