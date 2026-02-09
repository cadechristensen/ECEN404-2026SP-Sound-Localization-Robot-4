"""
Comprehensive Audio Quality Analysis for Baby Cry Detection
Analyzes audio characteristics to diagnose low model confidence issues
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import Config
from src.data_preprocessing import AudioPreprocessor
from src.audio_filter import BabyCryAudioFilter


def compute_snr(signal, noise_start=0, noise_duration=0.5, sr=48000):
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.

    Args:
        signal: Audio signal
        noise_start: Start time of noise segment (seconds)
        noise_duration: Duration of noise segment (seconds)
        sr: Sample rate

    Returns:
        SNR in dB
    """
    noise_samples = int(noise_duration * sr)
    noise_segment = signal[:noise_samples]

    noise_power = torch.mean(noise_segment ** 2)
    signal_power = torch.mean(signal ** 2)

    if noise_power == 0:
        return float('inf')

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def analyze_frequency_content(audio, sr, segment_start, segment_end, n_fft=2048):
    """
    Analyze frequency content of a specific segment.

    Args:
        audio: Audio tensor
        sr: Sample rate
        segment_start: Start time in seconds
        segment_end: End time in seconds
        n_fft: FFT size

    Returns:
        Dictionary with frequency analysis results
    """
    start_idx = int(segment_start * sr)
    end_idx = int(segment_end * sr)
    segment = audio[start_idx:end_idx]

    stft = torch.stft(segment, n_fft=n_fft, hop_length=512, return_complex=True)
    magnitude = torch.abs(stft)
    power = magnitude ** 2

    freqs = torch.fft.fftfreq(n_fft, 1/sr)[:n_fft//2 + 1]

    avg_power = power.mean(dim=1)

    total_power = avg_power.sum()

    freq_ranges = {
        '100-300Hz (Adult F0)': (100, 300),
        '300-600Hz (Baby Cry F0)': (300, 600),
        '600-1500Hz (Lower Harmonics)': (600, 1500),
        '1500-3000Hz (Upper Harmonics)': (1500, 3000),
        '3000-8000Hz (High Freq)': (3000, 8000),
    }

    energy_distribution = {}
    for range_name, (f_min, f_max) in freq_ranges.items():
        mask = (freqs >= f_min) & (freqs <= f_max)
        range_energy = avg_power[mask].sum()
        percentage = (range_energy / total_power * 100).item() if total_power > 0 else 0
        energy_distribution[range_name] = percentage

    spectral_centroid = (avg_power * freqs).sum() / (avg_power.sum() + 1e-10)

    fundamental_freq = 0
    f0_mask = (freqs >= 300) & (freqs <= 600)
    if f0_mask.sum() > 0:
        f0_power = avg_power[f0_mask]
        f0_freqs = freqs[f0_mask]
        fundamental_freq = f0_freqs[torch.argmax(f0_power)].item()

    return {
        'energy_distribution': energy_distribution,
        'spectral_centroid': spectral_centroid.item(),
        'fundamental_frequency': fundamental_freq,
        'total_power': total_power.item(),
    }


def analyze_preprocessing_impact(audio, sr, config):
    """
    Analyze how preprocessing affects the audio.

    Args:
        audio: Original audio tensor
        sr: Sample rate
        config: Config object

    Returns:
        Dictionary with preprocessing analysis
    """
    preprocessor = AudioPreprocessor(config)
    audio_filter = BabyCryAudioFilter(config)

    filtered = audio_filter.spectral_filter(audio)

    filtered_denoised = audio_filter.spectral_subtraction(filtered)

    original_rms = torch.sqrt(torch.mean(audio ** 2))
    filtered_rms = torch.sqrt(torch.mean(filtered ** 2))
    denoised_rms = torch.sqrt(torch.mean(filtered_denoised ** 2))

    return {
        'original_rms_db': 20 * np.log10(original_rms.item() + 1e-10),
        'filtered_rms_db': 20 * np.log10(filtered_rms.item() + 1e-10),
        'denoised_rms_db': 20 * np.log10(denoised_rms.item() + 1e-10),
        'filtering_loss_db': 20 * np.log10(filtered_rms.item() + 1e-10) - 20 * np.log10(original_rms.item() + 1e-10),
        'denoising_loss_db': 20 * np.log10(denoised_rms.item() + 1e-10) - 20 * np.log10(filtered_rms.item() + 1e-10),
    }


def analyze_spectrogram_features(audio, sr, config, segment_start, segment_end):
    """
    Analyze mel spectrogram that the model actually sees.

    Args:
        audio: Audio tensor
        sr: Sample rate
        config: Config object
        segment_start: Start time in seconds
        segment_end: End time in seconds

    Returns:
        Dictionary with spectrogram analysis
    """
    start_idx = int(segment_start * sr)
    end_idx = int(segment_end * sr)
    segment = audio[start_idx:end_idx]

    preprocessor = AudioPreprocessor(config)

    target_length = int(config.SAMPLE_RATE * config.DURATION)
    if len(segment) < target_length:
        padding = target_length - len(segment)
        segment = torch.nn.functional.pad(segment, (0, padding))
    else:
        segment = segment[:target_length]

    mel_spec = preprocessor.extract_log_mel_spectrogram(segment)

    return {
        'mel_spec_shape': mel_spec.shape,
        'mel_spec_min': mel_spec.min().item(),
        'mel_spec_max': mel_spec.max().item(),
        'mel_spec_mean': mel_spec.mean().item(),
        'mel_spec_std': mel_spec.std().item(),
        'mel_spec': mel_spec.numpy(),
    }


def compare_cry_vs_noncry_segments(audio, sr, cry_start, cry_end, config):
    """
    Compare crying segment vs non-crying segment characteristics.

    Args:
        audio: Audio tensor
        sr: Sample rate
        cry_start: Start of cry segment (seconds)
        cry_end: End of cry segment (seconds)
        config: Config object

    Returns:
        Comparison dictionary
    """
    cry_segment = audio[int(cry_start * sr):int(cry_end * sr)]

    noncry_end = min(cry_start - 0.5, 3.0)
    noncry_start = max(0, noncry_end - 3.0)
    noncry_segment = audio[int(noncry_start * sr):int(noncry_end * sr)]

    cry_freq = analyze_frequency_content(audio, sr, cry_start, cry_end)
    noncry_freq = analyze_frequency_content(audio, sr, noncry_start, noncry_end)

    cry_rms = torch.sqrt(torch.mean(cry_segment ** 2))
    noncry_rms = torch.sqrt(torch.mean(noncry_segment ** 2))

    return {
        'cry_segment': {
            'duration': cry_end - cry_start,
            'rms_db': 20 * np.log10(cry_rms.item() + 1e-10),
            'frequency_analysis': cry_freq,
        },
        'noncry_segment': {
            'duration': noncry_end - noncry_start,
            'rms_db': 20 * np.log10(noncry_rms.item() + 1e-10),
            'frequency_analysis': noncry_freq,
        },
        'amplitude_ratio_db': 20 * np.log10((cry_rms / (noncry_rms + 1e-10)).item()),
    }


def main():
    audio_path = "examples/audio_test3.wav"

    config = Config()

    print("=" * 80)
    print("AUDIO QUALITY ANALYSIS FOR BABY CRY DETECTION")
    print("=" * 80)
    print(f"\nAnalyzing: {audio_path}")
    print(f"Crying segment: 4.5-9 seconds")
    print(f"Model performance: 37-40% confidence (below 50% threshold)")
    print("=" * 80)

    audio, sr = torchaudio.load(audio_path)

    if audio.shape[0] > 1:
        primary_channel = audio[0]
        print(f"\nMulti-channel audio detected: {audio.shape[0]} channels")
        print(f"Using primary channel (Ch0) for analysis")
    else:
        primary_channel = audio[0]

    if sr != config.SAMPLE_RATE:
        resampler = T.Resample(sr, config.SAMPLE_RATE)
        primary_channel = resampler(primary_channel)
        sr = config.SAMPLE_RATE
        print(f"Resampled to {sr} Hz")

    print("\n" + "=" * 80)
    print("1. BASIC AUDIO CHARACTERISTICS")
    print("=" * 80)

    duration = len(primary_channel) / sr
    rms_level = torch.sqrt(torch.mean(primary_channel ** 2))
    rms_db = 20 * np.log10(rms_level.item() + 1e-10)
    peak_amplitude = torch.max(torch.abs(primary_channel))

    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample Rate: {sr} Hz")
    print(f"RMS Level: {rms_db:.2f} dB")
    print(f"Peak Amplitude: {peak_amplitude:.6f}")
    print(f"Dynamic Range: {20 * np.log10(peak_amplitude.item() / (rms_level.item() + 1e-10)):.2f} dB")

    print("\n" + "=" * 80)
    print("2. SIGNAL-TO-NOISE RATIO ANALYSIS")
    print("=" * 80)

    snr = compute_snr(primary_channel, noise_start=0, noise_duration=0.5, sr=sr)
    print(f"Estimated SNR (using first 0.5s as noise): {snr:.2f} dB")
    print(f"SNR Assessment: ", end="")
    if snr > 20:
        print("EXCELLENT (>20 dB)")
    elif snr > 15:
        print("GOOD (15-20 dB)")
    elif snr > 10:
        print("FAIR (10-15 dB)")
    else:
        print("POOR (<10 dB) - MAY IMPACT DETECTION")

    print("\n" + "=" * 80)
    print("3. CRYING SEGMENT ANALYSIS (4.5-9 seconds)")
    print("=" * 80)

    cry_analysis = analyze_frequency_content(primary_channel, sr, 4.5, 9.0)

    print(f"\nSpectral Centroid: {cry_analysis['spectral_centroid']:.1f} Hz")
    print(f"Dominant Fundamental Frequency: {cry_analysis['fundamental_frequency']:.1f} Hz")
    print(f"Total Power: {cry_analysis['total_power']:.6e}")

    print("\nEnergy Distribution by Frequency Band:")
    for band, energy_pct in cry_analysis['energy_distribution'].items():
        print(f"  {band}: {energy_pct:.1f}%")

    baby_cry_energy = cry_analysis['energy_distribution']['300-600Hz (Baby Cry F0)']
    print(f"\nCRITICAL ASSESSMENT:")
    print(f"  Baby cry fundamental (300-600 Hz): {baby_cry_energy:.1f}%")
    if baby_cry_energy < 20:
        print(f"  WARNING: Low energy in baby cry band (<20%) - LIKELY CAUSE OF LOW CONFIDENCE")
    elif baby_cry_energy < 30:
        print(f"  CAUTION: Moderate energy in baby cry band (20-30%) - May reduce confidence")
    else:
        print(f"  GOOD: Adequate energy in baby cry band (>30%)")

    print("\n" + "=" * 80)
    print("4. PREPROCESSING IMPACT ANALYSIS")
    print("=" * 80)

    preproc_impact = analyze_preprocessing_impact(primary_channel, sr, config)

    print(f"\nRMS Levels Through Pipeline:")
    print(f"  Original Audio: {preproc_impact['original_rms_db']:.2f} dB")
    print(f"  After Spectral Filter (100-3000 Hz): {preproc_impact['filtered_rms_db']:.2f} dB")
    print(f"  After Spectral Subtraction: {preproc_impact['denoising_rms_db']:.2f} dB")

    print(f"\nSignal Loss Analysis:")
    print(f"  Spectral Filtering Loss: {preproc_impact['filtering_loss_db']:.2f} dB")
    print(f"  Spectral Subtraction Loss: {preproc_impact['denoising_loss_db']:.2f} dB")
    print(f"  Total Pipeline Loss: {preproc_impact['filtering_loss_db'] + preproc_impact['denoising_loss_db']:.2f} dB")

    if abs(preproc_impact['filtering_loss_db']) > 6:
        print(f"  WARNING: Spectral filtering causing significant signal loss (>{abs(preproc_impact['filtering_loss_db']):.1f} dB)")
    if abs(preproc_impact['denoising_loss_db']) > 10:
        print(f"  WARNING: Spectral subtraction causing excessive loss (>{abs(preproc_impact['denoising_loss_db']):.1f} dB)")

    print("\n" + "=" * 80)
    print("5. MEL SPECTROGRAM ANALYSIS (MODEL INPUT)")
    print("=" * 80)

    spec_analysis = analyze_spectrogram_features(primary_channel, sr, config, 4.5, 9.0)

    print(f"\nMel Spectrogram Shape: {spec_analysis['mel_spec_shape']}")
    print(f"Value Range: [{spec_analysis['mel_spec_min']:.3f}, {spec_analysis['mel_spec_max']:.3f}]")
    print(f"Mean: {spec_analysis['mel_spec_mean']:.3f}")
    print(f"Std Dev: {spec_analysis['mel_spec_std']:.3f}")

    print("\n" + "=" * 80)
    print("6. CRYING vs NON-CRYING SEGMENT COMPARISON")
    print("=" * 80)

    comparison = compare_cry_vs_noncry_segments(primary_channel, sr, 4.5, 9.0, config)

    print(f"\nCrying Segment (4.5-9s):")
    print(f"  RMS Level: {comparison['cry_segment']['rms_db']:.2f} dB")
    print(f"  Energy in 300-600 Hz: {comparison['cry_segment']['frequency_analysis']['energy_distribution']['300-600Hz (Baby Cry F0)']:.1f}%")
    print(f"  Spectral Centroid: {comparison['cry_segment']['frequency_analysis']['spectral_centroid']:.1f} Hz")

    print(f"\nNon-Crying Segment:")
    print(f"  RMS Level: {comparison['noncry_segment']['rms_db']:.2f} dB")
    print(f"  Energy in 300-600 Hz: {comparison['noncry_segment']['frequency_analysis']['energy_distribution']['300-600Hz (Baby Cry F0)']:.1f}%")

    print(f"\nAmplitude Ratio (Cry / Non-Cry): {comparison['amplitude_ratio_db']:.2f} dB")

    print("\n" + "=" * 80)
    print("7. DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    issues_found = []
    recommendations = []

    if rms_db < -40:
        issues_found.append(f"Very low overall signal level ({rms_db:.1f} dB)")
        recommendations.append("Increase microphone gain or recording level")

    if snr < 15:
        issues_found.append(f"Low SNR ({snr:.1f} dB, target >20 dB)")
        recommendations.append("Reduce background noise or improve microphone placement")

    if baby_cry_energy < 25:
        issues_found.append(f"Insufficient energy in baby cry frequencies ({baby_cry_energy:.1f}%, target >30%)")
        recommendations.append("Cry may be too distant from microphone (background cry)")
        recommendations.append("Consider widening spectral filter range or reducing high-pass cutoff")

    if abs(preproc_impact['denoising_loss_db']) > 10:
        issues_found.append(f"Excessive signal loss from spectral subtraction ({abs(preproc_impact['denoising_loss_db']):.1f} dB)")
        recommendations.append("Reduce spectral subtraction aggressiveness (lower alpha parameter)")
        recommendations.append("Use gentler noise reduction or disable for background cries")

    if comparison['amplitude_ratio_db'] < 6:
        issues_found.append(f"Weak cry-to-background ratio ({comparison['amplitude_ratio_db']:.1f} dB)")
        recommendations.append("Cry is 'background' not 'foreground' - may be fundamentally different from training data")

    cry_spectral_centroid = comparison['cry_segment']['frequency_analysis']['spectral_centroid']
    if cry_spectral_centroid > 1500:
        issues_found.append(f"High spectral centroid ({cry_spectral_centroid:.0f} Hz) suggests noise-dominated signal")
        recommendations.append("Improve noise reduction before classification")

    print(f"\nISSUES IDENTIFIED ({len(issues_found)}):")
    if issues_found:
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("  No major issues detected")

    print(f"\nRECOMMENDATIONS ({len(recommendations)}):")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  Audio quality appears adequate for detection")

    print("\n" + "=" * 80)
    print("8. ROOT CAUSE ANALYSIS")
    print("=" * 80)

    print("\nWhy 98% accurate model gives 37-40% confidence:")
    print()

    if baby_cry_energy < 25 and comparison['amplitude_ratio_db'] < 6:
        print("PRIMARY CAUSE: Background Cry vs Foreground Cry Mismatch")
        print("  - Training data likely contains FOREGROUND cries (close to mic)")
        print("  - Test audio contains BACKGROUND cry (distant from mic)")
        print("  - Spectral signature is fundamentally different:")
        print(f"    * Only {baby_cry_energy:.1f}% energy in baby cry frequencies")
        print(f"    * Cry only {comparison['amplitude_ratio_db']:.1f} dB louder than background")
        print("  - Model correctly recognizes this as 'different' from training data")
        print()
        print("SOLUTION:")
        print("  Option 1: Retrain with background cry samples in training data")
        print("  Option 2: Lower confidence threshold to 30-35% for background detection")
        print("  Option 3: Use multi-stage detection (background boost + classification)")

    elif abs(preproc_impact['denoising_loss_db']) > 10:
        print("PRIMARY CAUSE: Aggressive Preprocessing Degrading Signal")
        print(f"  - Spectral subtraction removing {abs(preproc_impact['denoising_loss_db']):.1f} dB of signal")
        print("  - Background cry's weak signal being treated as noise")
        print("  - Model receiving degraded/incomplete cry features")
        print()
        print("SOLUTION:")
        print("  - Reduce spectral subtraction alpha from 2.0 to 1.0")
        print("  - Increase spectral floor from 0.1 to 0.3")
        print("  - Consider disabling spectral subtraction for low-amplitude signals")

    elif snr < 15:
        print("PRIMARY CAUSE: Insufficient Signal-to-Noise Ratio")
        print(f"  - SNR of {snr:.1f} dB is below recommended 20 dB threshold")
        print("  - Cry signal competing with background noise")
        print("  - Model uncertainty due to noisy input")
        print()
        print("SOLUTION:")
        print("  - Improve recording environment (reduce noise sources)")
        print("  - Use directional microphones or beamforming")
        print("  - Apply adaptive noise cancellation")

    else:
        print("MULTIPLE CONTRIBUTING FACTORS:")
        print("  Audio quality has several moderate issues that compound")
        print("  Review all recommendations above for best results")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
