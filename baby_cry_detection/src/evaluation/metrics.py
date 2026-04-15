"""
Metrics calculation module for model evaluation.

This module contains functions for calculating various classification metrics
including accuracy, precision, recall, F1, ROC-AUC, calibration metrics, etc.
Designed for binary classification (non-cry vs cry detection).
"""

from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    brier_score_loss
)
import logging


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_labels: list
) -> Dict:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        class_labels: List of class label names

    Returns:
        Dictionary containing all metrics
    """
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')

    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    for i, label in enumerate(class_labels):
        metrics[f'precision_{label}'] = precision_per_class[i]
        metrics[f'recall_{label}'] = recall_per_class[i]
        metrics[f'f1_score_{label}'] = f1_per_class[i]

    if len(class_labels) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr)

        precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_proba[:, 1])
        metrics['pr_auc'] = auc(recall_pr, precision_pr)
        metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
    else:
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))
            metrics['roc_auc'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
            metrics['average_precision'] = average_precision_score(y_true_bin, y_proba, average='macro')
        except Exception as e:
            logging.warning(f"Could not calculate multi-class ROC/PR metrics: {e}")

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True
    )

    if len(class_labels) == 2:
        # Pull deployment threshold from config so that calibration reporting
        # is always aligned with the actual operating point used in production.
        try:
            from ..config import Config as _Config
            deployment_threshold: Optional[float] = _Config.CONFIDENCE_THRESHOLD
        except Exception:
            deployment_threshold = None

        calibration_metrics = calculate_calibration_metrics(
            y_true, y_proba[:, 1], deployment_threshold=deployment_threshold
        )
        metrics.update(calibration_metrics)

    return metrics


def calculate_deployment_threshold_metrics(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    deployment_threshold: float,
    danger_zone_low: float = 0.7,
) -> Dict:
    """
    Compute calibration-related metrics at the deployment decision threshold.

    Temperature scaling is optimised against NLL at the 0.5 boundary, but
    deployment uses a much higher threshold (typically 0.92).  This function
    gives visibility into how well the model is calibrated *at that threshold*,
    independent of the global ECE.

    Three quantities are reported:

    * **above_threshold_rate** — fraction of all samples whose cry probability
      exceeds ``deployment_threshold``.  Gives a sense of how selective the
      threshold is on the evaluation set.
    * **precision_at_threshold** — of the samples that would be *accepted* by
      the deployment gate (prob >= threshold), what fraction are truly cries?
      This is precision at the deployment operating point.
    * **danger_zone_accuracy** — for samples in (``danger_zone_low``,
      ``deployment_threshold``), what fraction are truly cries?  A low value
      here validates the decision to skip that zone; a high value would suggest
      the threshold could be safely lowered.

    Args:
        y_true: True binary labels (0 = non-cry, 1 = cry).
        y_proba_positive: Predicted probabilities for the positive (cry) class.
        deployment_threshold: The operating threshold used by the detector
            (read from ``config.CONFIDENCE_THRESHOLD``).
        danger_zone_low: Lower bound of the "danger zone" probability interval
            whose accuracy is of interest (default: 0.7).

    Returns:
        Dictionary with keys:

        ``deployment_threshold`` : float
            Echo of the threshold used (for traceability in saved JSON).
        ``above_threshold_count`` : int
            Number of samples with cry probability >= threshold.
        ``above_threshold_rate`` : float
            Fraction of all samples with cry probability >= threshold.
        ``precision_at_threshold`` : float or None
            Fraction of above-threshold samples that are truly cries.
            ``None`` when no sample exceeds the threshold.
        ``danger_zone_low`` : float
            Lower bound of the danger zone interval.
        ``danger_zone_count`` : int
            Number of samples in (danger_zone_low, deployment_threshold).
        ``danger_zone_accuracy`` : float or None
            Fraction of danger-zone samples that are truly cries.
            ``None`` when the danger zone contains no samples.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return {
            'deployment_threshold': float(deployment_threshold),
            'above_threshold_count': 0,
            'above_threshold_rate': 0.0,
            'precision_at_threshold': None,
            'danger_zone_low': float(danger_zone_low),
            'danger_zone_count': 0,
            'danger_zone_accuracy': None,
        }

    above_mask = y_proba_positive >= deployment_threshold
    above_count = int(np.sum(above_mask))
    above_rate = above_count / n_samples

    if above_count > 0:
        precision_at_threshold: Optional[float] = float(
            np.mean(y_true[above_mask])
        )
    else:
        precision_at_threshold = None

    danger_mask = (y_proba_positive >= danger_zone_low) & (
        y_proba_positive < deployment_threshold
    )
    danger_count = int(np.sum(danger_mask))

    if danger_count > 0:
        danger_accuracy: Optional[float] = float(np.mean(y_true[danger_mask]))
    else:
        danger_accuracy = None

    return {
        'deployment_threshold': float(deployment_threshold),
        'above_threshold_count': above_count,
        'above_threshold_rate': float(above_rate),
        'precision_at_threshold': precision_at_threshold,
        'danger_zone_low': float(danger_zone_low),
        'danger_zone_count': danger_count,
        'danger_zone_accuracy': danger_accuracy,
    }


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    n_bins: int = 10,
    deployment_threshold: Optional[float] = None,
) -> Dict:
    """
    Calculate probability calibration metrics for binary classification.

    Calibration measures how well predicted probabilities match actual outcomes.
    A well-calibrated model should have P(y=1|p=0.8) = 0.8.

    When ``deployment_threshold`` is provided, threshold-specific metrics are
    appended under the ``deployment_threshold_metrics`` key.  These metrics
    answer the question: "how well is the model calibrated *at* the operating
    point actually used in production, rather than at the default 0.5 boundary?"

    Args:
        y_true: True binary labels (0 or 1)
        y_proba_positive: Predicted probabilities for the positive class
        n_bins: Number of bins for ECE calculation (default: 10)
        deployment_threshold: Optional operating threshold from
            ``config.CONFIDENCE_THRESHOLD``.  When supplied, per-threshold
            metrics are computed via
            :func:`calculate_deployment_threshold_metrics`.

    Returns:
        Dictionary containing calibration metrics:
            - brier_score: Mean squared error of probability predictions
            - expected_calibration_error: Weighted average of bin calibration errors
            - maximum_calibration_error: Maximum calibration error across bins
            - calibration_data: Detailed per-bin calibration information
            - deployment_threshold_metrics: (only when deployment_threshold is
              given) sub-dict with threshold-specific precision / danger-zone
              accuracy — see :func:`calculate_deployment_threshold_metrics`.
    """
    metrics = {}

    metrics['brier_score'] = float(brier_score_loss(y_true, y_proba_positive))

    ece, mce, calibration_data = calculate_ece(
        y_true, y_proba_positive, n_bins
    )
    metrics['expected_calibration_error'] = float(ece)
    metrics['maximum_calibration_error'] = float(mce)
    metrics['calibration_n_bins'] = n_bins
    metrics['calibration_data'] = calibration_data

    if deployment_threshold is not None:
        metrics['deployment_threshold_metrics'] = (
            calculate_deployment_threshold_metrics(
                y_true,
                y_proba_positive,
                deployment_threshold=deployment_threshold,
            )
        )

    return metrics


def calculate_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, Dict]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE measures the expected difference between predicted confidence and actual
    accuracy across probability bins.

    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class
        n_bins: Number of equal-width bins (default: 10)

    Returns:
        Tuple of (ECE, MCE, calibration_data_dict)
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0, 0.0, {}

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    calibration_data = {
        'bin_edges': bin_boundaries.tolist(),
        'bins': []
    }

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_upper == 1.0:
            in_bin = (y_proba >= bin_lower) & (y_proba <= bin_upper)
        else:
            in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)

        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_proba[in_bin])
            bin_error = np.abs(bin_accuracy - bin_confidence)

            ece += (bin_size / n_samples) * bin_error
            mce = max(mce, bin_error)

            calibration_data['bins'].append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'bin_size': int(bin_size),
                'bin_accuracy': float(bin_accuracy),
                'bin_confidence': float(bin_confidence),
                'bin_error': float(bin_error)
            })
        else:
            calibration_data['bins'].append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'bin_size': 0,
                'bin_accuracy': None,
                'bin_confidence': None,
                'bin_error': None
            })

    return ece, mce, calibration_data
