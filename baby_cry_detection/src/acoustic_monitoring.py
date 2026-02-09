"""
Acoustic Feature Monitoring for Baby Cry Detection.

This module provides monitoring and logging capabilities for acoustic features.
It does NOT affect scoring decisions - it's purely for debugging and analysis.
"""

from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class AcousticAnomaly:
    """Represents an acoustic feature anomaly detected during prediction."""
    timestamp: float
    anomaly_type: str
    description: str
    ml_prediction: float
    acoustic_score: float
    severity: str  # 'info', 'warning', 'error'


class AcousticMonitor:
    """
    Monitor acoustic features during baby cry detection.

    This class tracks acoustic features alongside ML predictions to identify
    potential issues, edge cases, or model weaknesses. It does NOT override
    ML predictions - it only logs anomalies for later analysis.
    """

    def __init__(self, enable_logging: bool = True, log_file: Optional[Path] = None):
        """
        Initialize acoustic monitor.

        Args:
            enable_logging: Whether to log anomalies (default: True)
            log_file: Optional path to log file (default: None = stdout only)
        """
        self.enable_logging = enable_logging
        self.log_file = log_file
        self.anomalies: List[AcousticAnomaly] = []

        # Thresholds for anomaly detection (conservative values)
        self.PITCH_MIN_EXPECTED = 200  # Hz (baby cries usually > 200 Hz)
        self.PITCH_MAX_EXPECTED = 700  # Hz (baby cries usually < 700 Hz)
        self.HNR_MIN_EXPECTED = 0.2    # Harmonic-to-Noise Ratio minimum for cries
        self.ENERGY_MIN_EXPECTED = 0.3 # Energy minimum for cries

    def check_prediction(
        self,
        timestamp: float,
        ml_prediction: float,
        acoustic_features: Optional[Dict] = None,
        threshold: float = 0.5
    ) -> List[AcousticAnomaly]:
        """
        Check ML prediction against acoustic features for anomalies.

        Args:
            timestamp: Time in seconds
            ml_prediction: ML model's cry probability
            acoustic_features: Optional acoustic features dict
            threshold: Detection threshold

        Returns:
            List of detected anomalies
        """
        if not self.enable_logging or acoustic_features is None:
            return []

        anomalies = []
        is_cry_prediction = ml_prediction >= threshold

        # Extract acoustic features
        pitch_mean = acoustic_features.get('pitch_mean', 0)
        hnr_mean = acoustic_features.get('hnr_mean', 0)
        energy_mean = acoustic_features.get('energy_mean', 0)
        acoustic_score = acoustic_features.get('acoustic_score', 0)

        # Check: High-confidence cry with abnormally low pitch
        if is_cry_prediction and ml_prediction > 0.8 and pitch_mean < self.PITCH_MIN_EXPECTED:
            anomaly = AcousticAnomaly(
                timestamp=timestamp,
                anomaly_type='low_pitch_cry',
                description=f'High-confidence cry (ML={ml_prediction:.2f}) with unusually low pitch ({pitch_mean:.1f} Hz < {self.PITCH_MIN_EXPECTED} Hz)',
                ml_prediction=ml_prediction,
                acoustic_score=acoustic_score,
                severity='warning'
            )
            anomalies.append(anomaly)

        # Check: High-confidence cry with abnormally high pitch
        if is_cry_prediction and ml_prediction > 0.8 and pitch_mean > self.PITCH_MAX_EXPECTED:
            anomaly = AcousticAnomaly(
                timestamp=timestamp,
                anomaly_type='high_pitch_cry',
                description=f'High-confidence cry (ML={ml_prediction:.2f}) with unusually high pitch ({pitch_mean:.1f} Hz > {self.PITCH_MAX_EXPECTED} Hz)',
                ml_prediction=ml_prediction,
                acoustic_score=acoustic_score,
                severity='info'
            )
            anomalies.append(anomaly)

        # Check: High-confidence cry with low harmonicity
        if is_cry_prediction and ml_prediction > 0.8 and hnr_mean < self.HNR_MIN_EXPECTED:
            anomaly = AcousticAnomaly(
                timestamp=timestamp,
                anomaly_type='low_harmonicity_cry',
                description=f'High-confidence cry (ML={ml_prediction:.2f}) with low harmonicity (HNR={hnr_mean:.2f} < {self.HNR_MIN_EXPECTED})',
                ml_prediction=ml_prediction,
                acoustic_score=acoustic_score,
                severity='warning'
            )
            anomalies.append(anomaly)

        # Check: High-confidence cry with low energy
        if is_cry_prediction and ml_prediction > 0.8 and energy_mean < self.ENERGY_MIN_EXPECTED:
            anomaly = AcousticAnomaly(
                timestamp=timestamp,
                anomaly_type='low_energy_cry',
                description=f'High-confidence cry (ML={ml_prediction:.2f}) with low energy ({energy_mean:.2f} < {self.ENERGY_MIN_EXPECTED})',
                ml_prediction=ml_prediction,
                acoustic_score=acoustic_score,
                severity='info'
            )
            anomalies.append(anomaly)

        # Check: ML and acoustic features strongly disagree
        if is_cry_prediction and acoustic_score < 0.3:
            disagreement = abs(ml_prediction - acoustic_score)
            if disagreement > 0.5:
                anomaly = AcousticAnomaly(
                    timestamp=timestamp,
                    anomaly_type='ml_acoustic_disagreement',
                    description=f'ML predicts cry (ML={ml_prediction:.2f}) but acoustic features reject it (acoustic={acoustic_score:.2f})',
                    ml_prediction=ml_prediction,
                    acoustic_score=acoustic_score,
                    severity='warning'
                )
                anomalies.append(anomaly)

        # Check: ML rejects but acoustic features suggest cry
        if not is_cry_prediction and acoustic_score > 0.7:
            disagreement = abs(ml_prediction - acoustic_score)
            if disagreement > 0.5:
                anomaly = AcousticAnomaly(
                    timestamp=timestamp,
                    anomaly_type='ml_acoustic_disagreement',
                    description=f'ML rejects cry (ML={ml_prediction:.2f}) but acoustic features suggest it (acoustic={acoustic_score:.2f})',
                    ml_prediction=ml_prediction,
                    acoustic_score=acoustic_score,
                    severity='info'
                )
                anomalies.append(anomaly)

        # Store and log anomalies
        for anomaly in anomalies:
            self.anomalies.append(anomaly)
            self._log_anomaly(anomaly)

        return anomalies

    def _log_anomaly(self, anomaly: AcousticAnomaly):
        """Log an anomaly to console and/or file."""
        if not self.enable_logging:
            return

        severity_prefix = {
            'info': 'INFO',
            'warning': 'WARNING',
            'error': 'ERROR'
        }.get(anomaly.severity, 'INFO')

        message = f"[{severity_prefix}] t={anomaly.timestamp:.2f}s | {anomaly.anomaly_type}: {anomaly.description}"

        if anomaly.severity == 'error':
            logging.error(message)
        elif anomaly.severity == 'warning':
            logging.warning(message)
        else:
            logging.info(message)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def get_anomaly_summary(self) -> Dict:
        """
        Get summary of all detected anomalies.

        Returns:
            Dictionary with anomaly counts and details
        """
        if not self.anomalies:
            return {
                'total_anomalies': 0,
                'by_type': {},
                'by_severity': {},
                'anomalies': []
            }

        by_type = {}
        by_severity = {}

        for anomaly in self.anomalies:
            by_type[anomaly.anomaly_type] = by_type.get(anomaly.anomaly_type, 0) + 1
            by_severity[anomaly.severity] = by_severity.get(anomaly.severity, 0) + 1

        return {
            'total_anomalies': len(self.anomalies),
            'by_type': by_type,
            'by_severity': by_severity,
            'anomalies': [asdict(a) for a in self.anomalies]
        }

    def save_report(self, output_path: Path):
        """
        Save anomaly report to JSON file.

        Args:
            output_path: Path to output file
        """
        summary = self.get_anomaly_summary()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Saved anomaly report to {output_path}")

    def clear(self):
        """Clear all recorded anomalies."""
        self.anomalies = []

    def print_summary(self):
        """Print a human-readable summary of anomalies."""
        summary = self.get_anomaly_summary()

        if summary['total_anomalies'] == 0:
            print("\nNo acoustic anomalies detected.")
            return

        print(f"\n{'=' * 60}")
        print("ACOUSTIC MONITORING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total anomalies detected: {summary['total_anomalies']}")

        if summary['by_severity']:
            print("\nBy Severity:")
            for severity, count in summary['by_severity'].items():
                print(f"  {severity.upper()}: {count}")

        if summary['by_type']:
            print("\nBy Type:")
            for anom_type, count in summary['by_type'].items():
                print(f"  {anom_type}: {count}")

        print(f"\n{'=' * 60}")
        print("RECOMMENDATIONS:")
        print(f"{'=' * 60}")

        if 'low_pitch_cry' in summary['by_type']:
            print("  - Review low-pitch detections - may be adult speech false positives")

        if 'ml_acoustic_disagreement' in summary['by_type']:
            print("  - ML and acoustic features disagree - review these cases manually")
            print("  - Consider retraining model if disagreements are systematic")

        if 'low_harmonicity_cry' in summary['by_type']:
            print("  - Low harmonicity detections may be noise/environmental sounds")

        print(f"\n{'=' * 60}\n")


def create_acoustic_monitoring_features(acoustic_features: Dict) -> Dict:
    """
    Convert raw acoustic features to monitoring-friendly format.

    Args:
        acoustic_features: Raw acoustic features from extractor

    Returns:
        Dict with averaged/summarized features for monitoring
    """
    if not acoustic_features:
        return {}

    import torch

    def safe_mean(tensor):
        if isinstance(tensor, torch.Tensor):
            return float(tensor.mean().item())
        elif isinstance(tensor, (list, np.ndarray)):
            return float(np.mean(tensor))
        return 0.0

    return {
        'pitch_mean': safe_mean(acoustic_features.get('pitch', [])),
        'hnr_mean': safe_mean(acoustic_features.get('hnr', [])),
        'energy_mean': safe_mean(acoustic_features.get('energy_scores', [])),
        'harmonic_mean': safe_mean(acoustic_features.get('harmonic_scores', [])),
        'acoustic_score': safe_mean(acoustic_features.get('harmonic_scores', [])) * 0.6 +
                         safe_mean(acoustic_features.get('energy_scores', [])) * 0.4
    }
