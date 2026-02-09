"""
Raspberry Pi performance diagnostics for the filtering pipeline.
Provides estimates and a summary printout for deployment validation.
"""

try:
    from .config_pi import ConfigPi
except ImportError:
    from config_pi import ConfigPi  # type: ignore


def get_filtering_performance_estimate(config: ConfigPi) -> dict:
    """
    Estimate filtering performance on Raspberry Pi 5.

    Args:
        config: Pi configuration

    Returns:
        Dictionary with performance estimates
    """
    estimates = {
        'high_pass_filter': {
            'time_per_second': '~2ms',
            'overhead': '~0.2%',
            'recommended': True
        },
        'band_pass_filter': {
            'time_per_second': '~2ms',
            'overhead': '~0.2%',
            'recommended': True
        },
        'spectral_subtraction': {
            'time_per_second': '~8ms',
            'overhead': '~0.8%',
            'recommended': True,
            'note': 'Reduced strength (0.3) for speed'
        },
        'vad': {
            'time_per_second': '~3ms',
            'overhead': '~0.3%',
            'recommended': True,
            'benefit': 'Reduces processing by only running model when activity detected'
        },
        'deep_spectrum': {
            'time_per_second': '~200ms',
            'overhead': '~20%',
            'recommended': False,
            'note': 'DISABLED - Too slow for real-time on Pi'
        },
        'total_filtering_overhead': {
            'time_per_second': '~15ms',
            'overhead': '~1.5%',
            'real_time_capable': True,
            'note': 'With basic filters only (high-pass, band-pass, spectral subtraction, VAD)'
        },
        'model_inference': {
            'time_per_second': '~150-250ms',
            'overhead': '~15-25%',
            'note': 'FP32 model inference. Main bottleneck, not filtering.'
        }
    }

    return estimates


def print_pi_filtering_info():
    """Print information about Pi filtering configuration."""
    config = ConfigPi()

    print("\n" + "=" * 70)
    print("RASPBERRY PI 5 FILTERING CONFIGURATION")
    print("=" * 70)

    print("\nENABLED FILTERS (Fast & Recommended):")
    print(f"  High-pass filter: {config.HIGHPASS_CUTOFF} Hz cutoff")
    print(f"  Band-pass filter: {config.BANDPASS_LOW}-{config.BANDPASS_HIGH} Hz")
    print(f"  Spectral subtraction: {config.NOISE_REDUCE_STRENGTH} strength")
    print(f"  Voice Activity Detection: Threshold {config.VAD_ENERGY_THRESHOLD}")
    print(f"  Channels: {config.PI_CHANNELS}")

    print("\nDISABLED FILTERS (Too Slow for Real-Time):")
    print(f"  Deep spectrum features: {config.USE_DEEP_SPECTRUM}")
    print(f"  MFCC deltas: {config.EXTRACT_MFCC_DELTAS}")
    print(f"  Spectral contrast: {config.EXTRACT_SPECTRAL_CONTRAST}")
    print(f"  Chroma features: {config.EXTRACT_CHROMA}")

    print("\nPERFORMANCE ESTIMATES (Raspberry Pi 5):")
    estimates = get_filtering_performance_estimate(config)

    print(f"  Filtering overhead: {estimates['total_filtering_overhead']['time_per_second']}/second")
    print(f"  Model inference: {estimates['model_inference']['time_per_second']}/second")
    print(f"  Real-time capable: {estimates['total_filtering_overhead']['real_time_capable']}")

    print("\nKEY OPTIMIZATIONS:")
    print("  1. VAD gating: Only run model when activity detected (~50% reduction)")
    print("  2. FP32 model: calibrated thresholds tuned to full-precision output")
    print("  3. Lightweight filters: <2% overhead")
    print("  4. Single-threaded: Reduced context switching")

    print("\nEXPECTED PERFORMANCE:")
    print("  Audio chunk: 0.5 seconds")
    print("  Processing time: ~100-150ms")
    print("  Real-time factor: 0.2-0.3 (3-5x faster than real-time)")
    print("  Latency: <200ms (acceptable for baby monitor)")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print_pi_filtering_info()
