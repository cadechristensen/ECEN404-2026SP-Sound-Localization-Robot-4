"""
Metrics calculation module for model evaluation.

This module contains functions for calculating various classification metrics
including accuracy, precision, recall, F1, ROC-AUC, calibration metrics, etc.
Designed for binary classification (non-cry vs cry detection).
"""

from typing import Dict, Tuple
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
        calibration_metrics = calculate_calibration_metrics(
            y_true, y_proba[:, 1]
        )
        metrics.update(calibration_metrics)

    return metrics


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Calculate probability calibration metrics for binary classification.

    Calibration measures how well predicted probabilities match actual outcomes.
    A well-calibrated model should have P(y=1|p=0.8) = 0.8.

    Args:
        y_true: True binary labels (0 or 1)
        y_proba_positive: Predicted probabilities for the positive class
        n_bins: Number of bins for ECE calculation (default: 10)

    Returns:
        Dictionary containing calibration metrics:
            - brier_score: Mean squared error of probability predictions
            - expected_calibration_error: Weighted average of bin calibration errors
            - maximum_calibration_error: Maximum calibration error across bins
            - calibration_data: Detailed per-bin calibration information
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
