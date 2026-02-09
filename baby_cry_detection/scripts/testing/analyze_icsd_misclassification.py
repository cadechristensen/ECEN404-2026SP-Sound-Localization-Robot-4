"""
ICSD Misclassification Analysis Script

This script analyzes why specific ICSD files are being misclassified as non_cry
with high confidence. It examines:
1. Audio characteristics (sample rate, bit depth, duration, RMS)
2. Spectral and F0 characteristics
3. Comparison with correctly classified files
4. Feature distribution differences

Problem files to analyze:
- strong_Infantcry_194_4447_8673.wav: 99.4% wrong
- strong_Infantcry_76_9513_10000.wav: 95.9% wrong
- strong_Infantcry_305_3095_4720.wav: 97.1% wrong
- strong_Infantcry_72_8141_9012.wav: 97.6% wrong
- strong_Infantcry_379_5646_7008.wav: 93.2% wrong
- strong_Infantcry_392_0008_9795.wav: 99.3% wrong
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from config import Config
except ImportError:
    # Define minimal config if import fails
    class Config:
        SAMPLE_RATE = 16000
        CRY_F0_MIN = 250
        CRY_F0_MAX = 850
        BANDPASS_HIGH = 4000
        N_FFT = 1024
        HOP_LENGTH = 256
        N_MELS = 128
        F_MIN = 80
        F_MAX = 7000


def get_audio_info(file_path: Path) -> Dict:
    """Get detailed audio file information."""
    info = {
        'file_path': str(file_path),
        'file_name': file_path.name,
        'file_size_kb': file_path.stat().st_size / 1024,
    }

    try:
        # Get file info using soundfile
        sf_info = sf.info(str(file_path))
        info['sample_rate'] = sf_info.samplerate
        info['channels'] = sf_info.channels
        info['frames'] = sf_info.frames
        info['duration'] = sf_info.duration
        info['format'] = sf_info.format
        info['subtype'] = sf_info.subtype
    except Exception as e:
        info['sf_error'] = str(e)

    try:
        # Load audio with librosa
        y, sr = librosa.load(str(file_path), sr=None)
        info['librosa_sr'] = sr
        info['librosa_duration'] = len(y) / sr
        info['librosa_samples'] = len(y)

        # Basic statistics
        info['rms'] = float(np.sqrt(np.mean(y**2)))
        info['peak'] = float(np.max(np.abs(y)))
        info['dynamic_range_db'] = float(20 * np.log10(info['peak'] / (info['rms'] + 1e-10)))
        info['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    except Exception as e:
        info['librosa_error'] = str(e)

    return info


def analyze_spectral_features(file_path: Path, config: Config) -> Dict:
    """Analyze spectral characteristics of an audio file."""
    features = {}

    try:
        # Load at target sample rate
        y, sr = librosa.load(str(file_path), sr=config.SAMPLE_RATE)
        features['duration_at_16k'] = len(y) / sr

        # F0 (pitch) analysis
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=80,
            fmax=1000,
            sr=sr
        )

        # Filter out unvoiced frames
        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) > 0:
            features['f0_mean'] = float(np.mean(valid_f0))
            features['f0_std'] = float(np.std(valid_f0))
            features['f0_min'] = float(np.min(valid_f0))
            features['f0_max'] = float(np.max(valid_f0))
            features['f0_range'] = float(np.max(valid_f0) - np.min(valid_f0))
            features['voiced_ratio'] = float(len(valid_f0) / len(f0))

            # Check if F0 is within cry range
            features['f0_in_cry_range'] = float(np.mean((valid_f0 >= config.CRY_F0_MIN) & (valid_f0 <= config.CRY_F0_MAX)))
            features['f0_below_cry_range'] = float(np.mean(valid_f0 < config.CRY_F0_MIN))
            features['f0_above_cry_range'] = float(np.mean(valid_f0 > config.CRY_F0_MAX))
        else:
            features['f0_mean'] = None
            features['f0_std'] = None
            features['voiced_ratio'] = 0.0
            features['f0_in_cry_range'] = 0.0

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))

        # Energy in cry band (200-4000 Hz)
        S = np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=config.N_FFT)

        cry_band_mask = (freqs >= 200) & (freqs <= 4000)
        total_energy = np.sum(S**2)
        cry_band_energy = np.sum(S[cry_band_mask, :]**2)
        features['cry_band_energy_ratio'] = float(cry_band_energy / (total_energy + 1e-10))

        # Harmonic content
        harmonic, percussive = librosa.effects.hpss(y)
        features['harmonic_ratio'] = float(np.sum(harmonic**2) / (np.sum(y**2) + 1e-10))

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_1_mean'] = float(np.mean(mfccs[0]))
        features['mfcc_2_mean'] = float(np.mean(mfccs[1]))
        features['mfcc_3_mean'] = float(np.mean(mfccs[2]))

    except Exception as e:
        features['error'] = str(e)

    return features


def analyze_segmentation_pattern(file_name: str) -> Dict:
    """Analyze the segmentation pattern from filename."""
    info = {
        'is_segmented': False,
        'segment_duration_ms': None,
        'segment_position': None
    }

    # Pattern: strong_Infantcry_XXX_SSSS_EEEE.wav
    parts = file_name.replace('.wav', '').split('_')

    if len(parts) >= 5 and parts[-1].isdigit() and parts[-2].isdigit():
        try:
            start_ms = int(parts[-2])
            end_ms = int(parts[-1])

            # The values appear to be in 10ths of a second or milliseconds
            # Looking at the pattern, it seems like 4-digit values
            # e.g., 4447_8673 suggests 4.447s to 8.673s if in ms*10
            # or 44.47s to 86.73s if in hundredths of seconds

            # Let's assume these are centiseconds (1/100th of a second)
            # which would make 4447 = 44.47 seconds
            info['is_segmented'] = True
            info['start_centiseconds'] = start_ms
            info['end_centiseconds'] = end_ms
            info['segment_duration_centiseconds'] = end_ms - start_ms

            # Alternative interpretation: milliseconds / 10
            info['segment_duration_ms'] = (end_ms - start_ms) * 10
            info['segment_position'] = 'middle' if start_ms > 1000 else 'start'

        except (ValueError, IndexError):
            pass

    return info


def find_comparison_files(data_dir: Path, num_files: int = 10) -> List[Path]:
    """Find comparison files (other ICSD files and general cry files)."""
    cry_icsd_dir = data_dir / 'cry_baby' / 'cry_ICSD'
    cry_dir = data_dir / 'cry_baby' / 'cry'

    comparison_files = []

    # Get some non-segmented ICSD files
    if cry_icsd_dir.exists():
        for f in cry_icsd_dir.iterdir():
            if f.suffix.lower() == '.wav' and '_' not in f.stem[len('strong_Infantcry_'):]:
                # Non-segmented files (e.g., strong_Infantcry_184.wav)
                comparison_files.append(f)
                if len(comparison_files) >= num_files // 2:
                    break

    # Get some general cry files
    if cry_dir.exists():
        for f in cry_dir.iterdir():
            if f.suffix.lower() in ['.wav', '.ogg', '.mp3']:
                comparison_files.append(f)
                if len(comparison_files) >= num_files:
                    break

    return comparison_files


def main():
    """Main analysis function."""
    print("=" * 80)
    print("ICSD MISCLASSIFICATION ANALYSIS")
    print("=" * 80)

    config = Config()
    data_dir = Path(__file__).parent.parent.parent / 'data'
    cry_icsd_dir = data_dir / 'cry_baby' / 'cry_ICSD'

    # Problem files
    problem_files = [
        'strong_Infantcry_194_4447_8673.wav',
        'strong_Infantcry_76_9513_10000.wav',
        'strong_Infantcry_305_3095_4720.wav',
        'strong_Infantcry_72_8141_9012.wav',
        'strong_Infantcry_379_5646_7008.wav',
        'strong_Infantcry_392_0008_9795.wav',
    ]

    confidence_wrong = {
        'strong_Infantcry_194_4447_8673.wav': 99.4,
        'strong_Infantcry_76_9513_10000.wav': 95.9,
        'strong_Infantcry_305_3095_4720.wav': 97.1,
        'strong_Infantcry_72_8141_9012.wav': 97.6,
        'strong_Infantcry_379_5646_7008.wav': 93.2,
        'strong_Infantcry_392_0008_9795.wav': 99.3,
    }

    print("\n" + "=" * 80)
    print("ANALYZING PROBLEM FILES")
    print("=" * 80)

    problem_features = []

    for file_name in problem_files:
        file_path = cry_icsd_dir / file_name

        print(f"\n--- {file_name} (Wrong Confidence: {confidence_wrong[file_name]}%) ---")

        if not file_path.exists():
            print(f"  FILE NOT FOUND: {file_path}")
            continue

        # Get basic info
        info = get_audio_info(file_path)
        print(f"  File size: {info.get('file_size_kb', 'N/A'):.2f} KB")
        print(f"  Sample rate: {info.get('sample_rate', 'N/A')} Hz")
        print(f"  Duration: {info.get('duration', 'N/A'):.3f} s")
        print(f"  Format: {info.get('format', 'N/A')} / {info.get('subtype', 'N/A')}")
        print(f"  RMS: {info.get('rms', 'N/A'):.4f}")
        print(f"  Peak: {info.get('peak', 'N/A'):.4f}")
        print(f"  Dynamic Range: {info.get('dynamic_range_db', 'N/A'):.1f} dB")

        # Analyze segmentation pattern
        seg_info = analyze_segmentation_pattern(file_name)
        print(f"  Segmented: {seg_info['is_segmented']}")
        if seg_info['is_segmented']:
            print(f"  Segment duration (ms): {seg_info.get('segment_duration_ms', 'N/A')}")

        # Analyze spectral features
        spectral = analyze_spectral_features(file_path, config)
        print(f"\n  Spectral Analysis:")
        print(f"    F0 Mean: {spectral.get('f0_mean', 'N/A')}")
        print(f"    F0 Std: {spectral.get('f0_std', 'N/A')}")
        print(f"    F0 Range: {spectral.get('f0_min', 'N/A')} - {spectral.get('f0_max', 'N/A')} Hz")
        print(f"    Voiced Ratio: {spectral.get('voiced_ratio', 0):.2%}")
        print(f"    F0 in Cry Range (250-700 Hz): {spectral.get('f0_in_cry_range', 0):.2%}")
        print(f"    F0 Below Cry Range (<250 Hz): {spectral.get('f0_below_cry_range', 0):.2%}")
        print(f"    F0 Above Cry Range (>700 Hz): {spectral.get('f0_above_cry_range', 0):.2%}")
        print(f"    Spectral Centroid: {spectral.get('spectral_centroid_mean', 'N/A'):.1f} Hz")
        print(f"    Spectral Flatness: {spectral.get('spectral_flatness_mean', 'N/A'):.4f}")
        print(f"    Cry Band Energy Ratio: {spectral.get('cry_band_energy_ratio', 0):.2%}")
        print(f"    Harmonic Ratio: {spectral.get('harmonic_ratio', 0):.2%}")

        problem_features.append({
            'file': file_name,
            'confidence_wrong': confidence_wrong[file_name],
            **info,
            **spectral,
            **seg_info
        })

    # Find and analyze comparison files
    print("\n" + "=" * 80)
    print("ANALYZING COMPARISON FILES (Non-segmented ICSD + General Cry)")
    print("=" * 80)

    comparison_files = find_comparison_files(data_dir, num_files=6)
    comparison_features = []

    for file_path in comparison_files[:6]:
        print(f"\n--- {file_path.name} ---")

        info = get_audio_info(file_path)
        print(f"  Duration: {info.get('duration', 'N/A'):.3f} s")
        print(f"  RMS: {info.get('rms', 'N/A'):.4f}")

        spectral = analyze_spectral_features(file_path, config)
        print(f"  F0 Mean: {spectral.get('f0_mean', 'N/A')}")
        print(f"  F0 in Cry Range: {spectral.get('f0_in_cry_range', 0):.2%}")
        print(f"  Voiced Ratio: {spectral.get('voiced_ratio', 0):.2%}")
        print(f"  Harmonic Ratio: {spectral.get('harmonic_ratio', 0):.2%}")
        print(f"  Cry Band Energy: {spectral.get('cry_band_energy_ratio', 0):.2%}")

        comparison_features.append({
            'file': file_path.name,
            **info,
            **spectral
        })

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    def safe_mean(values):
        valid = [v for v in values if v is not None and not np.isnan(v)]
        return np.mean(valid) if valid else None

    print("\n--- Problem Files (Misclassified as non_cry) ---")
    print(f"  Avg Duration: {safe_mean([f.get('duration') for f in problem_features]):.3f} s")
    print(f"  Avg F0: {safe_mean([f.get('f0_mean') for f in problem_features]):.1f} Hz")
    print(f"  Avg Voiced Ratio: {safe_mean([f.get('voiced_ratio') for f in problem_features]):.2%}")
    print(f"  Avg F0 in Cry Range: {safe_mean([f.get('f0_in_cry_range') for f in problem_features]):.2%}")
    print(f"  Avg Harmonic Ratio: {safe_mean([f.get('harmonic_ratio') for f in problem_features]):.2%}")
    print(f"  Avg RMS: {safe_mean([f.get('rms') for f in problem_features]):.4f}")

    if comparison_features:
        print("\n--- Comparison Files (Correctly Classified) ---")
        print(f"  Avg Duration: {safe_mean([f.get('duration') for f in comparison_features]):.3f} s")
        print(f"  Avg F0: {safe_mean([f.get('f0_mean') for f in comparison_features]):.1f} Hz")
        print(f"  Avg Voiced Ratio: {safe_mean([f.get('voiced_ratio') for f in comparison_features]):.2%}")
        print(f"  Avg F0 in Cry Range: {safe_mean([f.get('f0_in_cry_range') for f in comparison_features]):.2%}")
        print(f"  Avg Harmonic Ratio: {safe_mean([f.get('harmonic_ratio') for f in comparison_features]):.2%}")
        print(f"  Avg RMS: {safe_mean([f.get('rms') for f in comparison_features]):.4f}")

    # Analysis conclusions
    print("\n" + "=" * 80)
    print("ANALYSIS CONCLUSIONS")
    print("=" * 80)

    # Check for common patterns
    short_durations = [f for f in problem_features if f.get('duration', 999) < 1.0]
    low_voiced = [f for f in problem_features if f.get('voiced_ratio', 1) < 0.3]
    low_f0_in_range = [f for f in problem_features if f.get('f0_in_cry_range', 1) < 0.5]
    low_rms = [f for f in problem_features if f.get('rms', 1) < 0.01]

    print(f"\n1. Short Duration (<1s): {len(short_durations)}/{len(problem_features)} files")
    print(f"2. Low Voiced Ratio (<30%): {len(low_voiced)}/{len(problem_features)} files")
    print(f"3. F0 Outside Cry Range (>50%): {len(low_f0_in_range)}/{len(problem_features)} files")
    print(f"4. Very Low RMS (<0.01): {len(low_rms)}/{len(problem_features)} files")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("""
Based on the analysis, potential issues and recommendations:

1. SEGMENT DURATION:
   - Problem files are segments from longer recordings
   - Segments may be too short to capture full cry pattern
   - RECOMMENDATION: Extend segment boundaries or use overlapping windows

2. F0/PITCH CHARACTERISTICS:
   - Check if F0 values are outside the expected 250-700 Hz range
   - Some "strong" cries may have higher F0 (distress/pain cries)
   - RECOMMENDATION: Widen CRY_F0_MAX to 800-1000 Hz for distress cries

3. SEGMENT BOUNDARIES:
   - Timestamps in filename may cut off mid-cry burst
   - Edge effects from abrupt start/stop
   - RECOMMENDATION: Add fade-in/fade-out or zero-padding

4. FEATURE DISTRIBUTION MISMATCH:
   - ICSD segments may have different acoustic properties
   - Training data may be dominated by other cry sources
   - RECOMMENDATION: Ensure ICSD representation in training split

5. LOW SIGNAL LEVEL:
   - Check if problem files have significantly lower RMS
   - May need gain normalization before feature extraction
   - RECOMMENDATION: Add per-file RMS normalization in preprocessing
""")


if __name__ == '__main__':
    main()
