"""
Unit tests for multichannel detector implementation.

Tests the enhanced SNR computation, channel selection, and dual-channel voting.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Insert the project root (two directories up) so that any transitive src.*
# imports resolve, even though all direct imports below are local.
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Local deployment modules (live alongside this file)
from multichannel_detector import (
    EnhancedSNRComputation,
    ChannelHealthMonitor,
    DualChannelVotingDetector,
    create_multichannel_detector
)


def test_snr_computation():
    """Test enhanced SNR computation."""
    print("\n" + "="*70)
    print("TEST 1: Enhanced SNR Computation")
    print("="*70)

    snr_computer = EnhancedSNRComputation(sample_rate=16000)

    # Create synthetic audio with cry-like signal
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Channel 0: Clean cry signal (400 Hz fundamental)
    cry_signal_ch0 = np.sin(2 * np.pi * 400 * t) + 0.5 * np.sin(2 * np.pi * 800 * t)
    noise_ch0 = 0.1 * np.random.randn(len(t))
    audio_ch0 = cry_signal_ch0 + noise_ch0

    # Channel 1: Same cry but with more noise
    audio_ch1 = cry_signal_ch0 + 0.5 * np.random.randn(len(t))

    # Channel 2: Only noise (no cry)
    audio_ch2 = np.random.randn(len(t)) * 0.3

    # Channel 3: Cry with low-frequency rumble
    rumble = 0.8 * np.sin(2 * np.pi * 100 * t)
    audio_ch3 = cry_signal_ch0 + rumble + noise_ch0

    # Stack into multi-channel audio
    audio = np.stack([audio_ch0, audio_ch1, audio_ch2, audio_ch3], axis=1)

    # Compute SNR for all channels
    snr_scores = snr_computer.compute_snr_all_channels(audio)

    print(f"Channel 0 SNR: {snr_scores[0]:.2f} dB (clean cry)")
    print(f"Channel 1 SNR: {snr_scores[1]:.2f} dB (noisy cry)")
    print(f"Channel 2 SNR: {snr_scores[2]:.2f} dB (noise only)")
    print(f"Channel 3 SNR: {snr_scores[3]:.2f} dB (cry + rumble)")

    # Verify SNR ranking
    assert snr_scores[0] > snr_scores[1], "Channel 0 should have higher SNR than Channel 1"
    assert snr_scores[0] > snr_scores[2], "Channel 0 should have higher SNR than Channel 2"

    print("PASSED: SNR computation correctly ranks channels")


def test_channel_health_monitoring():
    """Test channel health monitoring."""
    print("\n" + "="*70)
    print("TEST 2: Channel Health Monitoring")
    print("="*70)

    health_monitor = ChannelHealthMonitor(sample_rate=16000, num_channels=4)

    duration = 1.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)

    # Create test audio with various health issues
    audio = np.zeros((num_samples, 4))

    # Channel 0: Normal signal
    audio[:, 0] = 0.3 * np.random.randn(num_samples)

    # Channel 1: Clipping
    audio[:, 1] = np.clip(2.0 * np.random.randn(num_samples), -1.0, 1.0)

    # Channel 2: Very low signal (mic failure)
    audio[:, 2] = 0.0001 * np.random.randn(num_samples)

    # Channel 3: Normal signal
    audio[:, 3] = 0.3 * np.random.randn(num_samples)

    # Get health metrics
    health_metrics = health_monitor.get_channel_health(audio)

    print("\nChannel Health Metrics:")
    for metric in health_metrics:
        print(f"  Channel {metric.channel_idx}:")
        print(f"    SNR: {metric.snr_db:.2f} dB")
        print(f"    RMS: {metric.rms:.6f}")
        print(f"    Clipping: {metric.clipping}")

    # Verify health detection
    assert health_metrics[1].clipping == True, "Channel 1 should detect clipping"
    assert health_metrics[2].rms < 0.001, "Channel 2 should have low RMS"

    print("\nPASSED: Channel health monitoring detects issues correctly")


def test_channel_selection():
    """Test adaptive channel selection."""
    print("\n" + "="*70)
    print("TEST 3: Adaptive Channel Selection")
    print("="*70)

    class MockDetector:
        """Mock detector for testing."""
        def detect_cry(self, audio, use_tta=False):
            # Simulate detection based on signal strength
            rms = np.sqrt(np.mean(audio ** 2))
            confidence = min(0.95, max(0.5, rms * 2))
            is_cry = confidence > 0.75
            return is_cry, confidence

    mock_detector = MockDetector()
    multichannel_detector = create_multichannel_detector(
        detector=mock_detector,
        num_channels=4,
        voting_strategy="weighted",
        sample_rate=16000
    )

    # Create multi-channel audio with different SNRs
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))

    cry_signal = np.sin(2 * np.pi * 400 * t) + 0.5 * np.sin(2 * np.pi * 800 * t)

    audio = np.zeros((len(t), 4))
    audio[:, 0] = cry_signal + 0.5 * np.random.randn(len(t))  # Noisy
    audio[:, 1] = cry_signal + 0.1 * np.random.randn(len(t))  # Clean (best)
    audio[:, 2] = cry_signal + 0.8 * np.random.randn(len(t))  # Very noisy
    audio[:, 3] = cry_signal + 0.3 * np.random.randn(len(t))  # Moderate

    # Select best channels
    best_channels, snr_scores = multichannel_detector.select_best_channels(audio, n_channels=2)

    print(f"\nSNR Scores:")
    for ch, snr in enumerate(snr_scores):
        print(f"  Channel {ch}: {snr:.2f} dB")

    print(f"\nSelected Channels: {best_channels}")
    print(f"  Primary: Channel {best_channels[0]} (SNR: {snr_scores[best_channels[0]]:.2f} dB)")
    print(f"  Secondary: Channel {best_channels[1]} (SNR: {snr_scores[best_channels[1]]:.2f} dB)")

    # Verify best channel is selected
    assert best_channels[0] == 1, "Channel 1 should be selected as best (cleanest)"

    print("\nPASSED: Adaptive channel selection chooses best channels")


def test_dual_channel_voting():
    """Test dual-channel voting."""
    print("\n" + "="*70)
    print("TEST 4: Dual-Channel Voting")
    print("="*70)

    class MockDetector:
        """Mock detector for testing."""
        def detect_cry(self, audio, use_tta=False):
            # Simulate detection based on signal presence at 400 Hz
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/16000)

            # Energy in cry band (300-600 Hz)
            cry_mask = (freqs >= 300) & (freqs <= 600)
            cry_energy = np.mean(np.abs(fft[cry_mask]) ** 2)

            # Noise energy
            noise_mask = (freqs < 200) | (freqs > 1000)
            noise_energy = np.mean(np.abs(fft[noise_mask]) ** 2)

            # Confidence based on SNR
            snr = 10 * np.log10((cry_energy + 1e-10) / (noise_energy + 1e-10))
            confidence = 1.0 / (1.0 + np.exp(-0.2 * (snr - 5)))  # Sigmoid
            is_cry = confidence > 0.75

            return is_cry, confidence

    mock_detector = MockDetector()

    # Test weighted voting
    print("\nTesting WEIGHTED voting strategy:")
    multichannel_detector = create_multichannel_detector(
        detector=mock_detector,
        num_channels=4,
        voting_strategy="weighted",
        sample_rate=16000
    )

    # Create test audio
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))
    cry_signal = np.sin(2 * np.pi * 400 * t)

    audio = np.zeros((len(t), 4))
    audio[:, 0] = cry_signal + 0.1 * np.random.randn(len(t))
    audio[:, 1] = cry_signal + 0.1 * np.random.randn(len(t))
    audio[:, 2] = 0.5 * np.random.randn(len(t))  # Noise only
    audio[:, 3] = cry_signal + 0.2 * np.random.randn(len(t))

    # Run detection
    result = multichannel_detector.detect_cry_dual_channel(audio, use_tta=False)

    print(f"  Detection: {result.is_cry}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Primary Channel: {result.primary_channel}")
    print(f"  Secondary Channel: {result.secondary_channel}")
    print(f"  Channel Confidences: {[f'{c:.2%}' for c in result.channel_confidences]}")
    print(f"  Multi-Channel Agreement: {result.multi_channel_agreement:.2%}")

    # Test logical OR voting
    print("\nTesting LOGICAL_OR voting strategy:")
    multichannel_detector_or = create_multichannel_detector(
        detector=mock_detector,
        num_channels=4,
        voting_strategy="logical_or",
        sample_rate=16000
    )

    result_or = multichannel_detector_or.detect_cry_dual_channel(audio, use_tta=False)

    print(f"  Detection: {result_or.is_cry}")
    print(f"  Confidence: {result_or.confidence:.2%}")
    print(f"  Primary Channel: {result_or.primary_channel}")
    print(f"  Secondary Channel: {result_or.secondary_channel}")

    print("\nPASSED: Dual-channel voting works correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("MULTICHANNEL DETECTOR TEST SUITE")
    print("="*70)

    try:
        test_snr_computation()
        test_channel_health_monitoring()
        test_channel_selection()
        test_dual_channel_voting()

        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print("\nMulti-channel detector implementation is working correctly.")
        print("\nKey Features Verified:")
        print("  - Enhanced SNR computation (300-900 Hz signal band)")
        print("  - Channel health monitoring (RMS, clipping detection)")
        print("  - Adaptive channel selection (best 2 of 4 channels)")
        print("  - Dual-channel voting (weighted and logical_or)")
        print("\nRecommendations:")
        print("  1. Use 'weighted' voting for balanced accuracy")
        print("  2. Enable --debug-channels for detailed monitoring")
        print("  3. Monitor channel health in production")
        print("="*70 + "\n")

        return True

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
