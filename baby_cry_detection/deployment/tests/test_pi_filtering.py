#!/usr/bin/env python3
"""
Test script to verify filtering performance on Raspberry Pi.
Run this on your Pi to check if real-time performance is achievable.
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# This file lives in deployment/tests/ — insert the parent deployment/
# directory so that config_pi and audio_filtering can be imported.
_DEPLOY_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _DEPLOY_DIR not in sys.path:
    sys.path.insert(0, _DEPLOY_DIR)

from config_pi import ConfigPi
from audio_filtering import AudioFilteringPipeline


def test_filtering_performance():
    """Test filtering performance on Pi."""
    print("\n" + "="*60)
    print("RASPBERRY PI FILTERING PERFORMANCE TEST")
    print("="*60)

    config = ConfigPi()
    pipeline = AudioFilteringPipeline(config)

    print("\n1. Configuration:")
    print(f"   High-pass cutoff: {config.HIGHPASS_CUTOFF} Hz")
    print(f"   Band-pass range: {config.BANDPASS_LOW}-{config.BANDPASS_HIGH} Hz")
    print(f"   Spectral subtraction: {config.NOISE_REDUCE_STRENGTH}")
    print(f"   VAD enabled: {config.USE_ADVANCED_FILTERING}")
    print(f"   Deep spectrum: {config.USE_DEEP_SPECTRUM}")

    print("\n2. Running performance test (10 iterations)...")

    # Test with 1 second of audio
    audio = torch.randn(16000)

    # Warmup
    for _ in range(3):
        _ = pipeline.preprocess_audio(audio, apply_filtering=True)

    # Benchmark
    times = []
    for i in range(10):
        start = time.time()
        result = pipeline.preprocess_audio(
            audio,
            apply_vad=True,
            apply_filtering=True
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Iteration {i+1}: {elapsed*1000:.1f}ms")

    # Calculate statistics
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000

    print(f"\n3. Results:")
    print(f"   Average: {avg_time:.1f}ms")
    print(f"   Std Dev: {std_time:.1f}ms")
    print(f"   Min: {min_time:.1f}ms")
    print(f"   Max: {max_time:.1f}ms")

    # Real-time factor
    rtf = avg_time / 1000  # For 1 second of audio
    print(f"\n4. Real-Time Factor: {rtf:.3f}")
    print(f"   (Processing time / Audio duration)")

    # Verdict
    print(f"\n5. Verdict:")
    if avg_time < 50:
        print("   EXCELLENT - Extremely fast!")
        print("   Your Pi can handle real-time processing easily.")
    elif avg_time < 100:
        print("   VERY GOOD - Fast enough for real-time!")
        print("   Your Pi is well-suited for deployment.")
    elif avg_time < 200:
        print("   GOOD - Real-time capable")
        print("   Should work fine for baby monitoring.")
    elif avg_time < 500:
        print("   MARGINAL - Close to real-time limit")
        print("   Consider reducing NOISE_REDUCE_STRENGTH.")
    else:
        print("   TOO SLOW - Not suitable for real-time")
        print("   Try disabling spectral subtraction.")

    # Overhead calculation
    overhead_pct = (avg_time / 1000) * 100
    print(f"\n6. Filtering Overhead: {overhead_pct:.1f}%")
    print(f"   (For 1 second of audio)")

    # Expected total processing time (with model)
    model_time = 200  # Typical model inference time in ms
    total_time = avg_time + model_time
    print(f"\n7. Estimated Total Processing Time:")
    print(f"   Filtering: {avg_time:.1f}ms")
    print(f"   Model inference: ~{model_time}ms (estimated)")
    print(f"   Total: ~{total_time:.1f}ms")

    if total_time < 500:
        print(f"   Total latency < 0.5s - Excellent for baby monitor!")

    print("\n" + "="*60 + "\n")


def test_vad_performance():
    """Test VAD (Voice Activity Detection) performance."""
    print("VAD PERFORMANCE TEST")
    print("="*60)

    from audio_filtering import VoiceActivityDetector

    vad = VoiceActivityDetector(sample_rate=16000)

    # Test VAD on different types of audio
    test_cases = [
        ("Silent audio", torch.zeros(16000)),
        ("Random noise", torch.randn(16000) * 0.1),
        ("Loud signal", torch.randn(16000) * 0.5)
    ]

    for name, audio in test_cases:
        start = time.time()
        activity_mask, confidence = vad.detect_activity(audio)
        elapsed = time.time() - start

        activity_ratio = np.sum(activity_mask) / len(activity_mask)

        print(f"\n{name}:")
        print(f"  Processing time: {elapsed*1000:.1f}ms")
        print(f"  Activity detected: {activity_ratio*100:.1f}%")
        print(f"  Avg confidence: {confidence.mean():.3f}")

    print("\n" + "="*60 + "\n")


def main():
    """Run all tests."""
    try:
        # Test filtering
        test_filtering_performance()

        # Test VAD
        test_vad_performance()

        print("\nTesting complete!")
        print("\nNext steps:")
        print("1. If performance is good, use ConfigPi in your deployment")
        print("2. Update your realtime_baby_cry_detector.py to import ConfigPi")
        print("3. Test with actual model inference")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
