"""
Visualization module for model evaluation.

This module contains functions for creating various plots and visualizations
including confusion matrices, ROC curves, calibration diagrams, etc.
Designed for binary classification (non-cry vs cry detection).
"""

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, brier_score_loss
)
import logging


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: List[str],
    results_dir: Path
) -> None:
    """
    Plot and save confusion matrix with detailed analysis.

    Args:
        cm: Confusion matrix
        class_labels: List of class label names
        results_dir: Directory to save the plot
    """
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    display_labels = [label.replace('_', ' ').title() for label in class_labels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_labels, yticklabels=display_labels,
                ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    cm_normalized_true = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized_true, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=display_labels, yticklabels=display_labels,
                ax=axes[1])
    axes[1].set_title('Normalized by True Label (Recall)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)

    cm_normalized_pred = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    sns.heatmap(cm_normalized_pred, annot=True, fmt='.2%', cmap='Oranges',
                xticklabels=display_labels, yticklabels=display_labels,
                ax=axes[2])
    axes[2].set_title('Normalized by Predicted Label (Precision)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('True Label', fontsize=12)
    axes[2].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Enhanced confusion matrices saved to {plots_dir}")

    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        logging.info("=== CONFUSION MATRIX ANALYSIS (Binary) ===")
        logging.info(f"True Positives (TP): {tp} - Correctly identified baby cries")
        logging.info(f"True Negatives (TN): {tn} - Correctly identified non-cries")
        logging.info(f"False Positives (FP): {fp} - False alarms (predicted cry, but was non-cry)")
        logging.info(f"False Negatives (FN): {fn} - Missed cries (predicted non-cry, but was cry)")
        logging.info(f"Accuracy: {accuracy:.3f} - Overall correctness")
        logging.info(f"Precision: {precision:.3f} - Of all cry predictions, how many were correct")
        logging.info(f"Recall: {recall:.3f} - Of all actual cries, how many were detected")
        logging.info(f"F1-Score: {f1:.3f} - Balanced measure of precision and recall")
    else:
        accuracy = np.trace(cm) / np.sum(cm)
        logging.info("=== CONFUSION MATRIX ANALYSIS (Multi-Class) ===")
        logging.info(f"Overall Accuracy: {accuracy:.3f}")
        logging.info(f"Class-wise performance:")
        for i, label in enumerate(display_labels):
            class_total = cm[i].sum()
            class_correct = cm[i, i]
            class_recall = class_correct / class_total if class_total > 0 else 0
            logging.info(f"  {label}: {class_correct}/{class_total} correct (recall: {class_recall:.3f})")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_labels: List[str],
    results_dir: Path
) -> None:
    """
    Plot and save ROC curve.

    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        class_labels: List of class label names
        results_dir: Directory to save the plot
    """
    if len(class_labels) != 2:
        logging.warning("ROC curve only applicable for binary classification")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = plots_dir / "roc_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"ROC curve saved to {plot_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_labels: List[str],
    results_dir: Path
) -> None:
    """
    Plot and save Precision-Recall curve.

    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        class_labels: List of class label names
        results_dir: Directory to save the plot
    """
    if len(class_labels) != 2:
        logging.warning("PR curve only applicable for binary classification")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_proba[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f}, AP = {avg_precision:.3f})')

    pos_ratio = np.sum(y_true) / len(y_true)
    plt.axhline(y=pos_ratio, color='red', linestyle='--',
               label=f'Random (AP = {pos_ratio:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = plots_dir / "precision_recall_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Precision-Recall curve saved to {plot_path}")


def plot_training_history(
    history: Dict,
    results_dir: Path
) -> None:
    """
    Plot training history (loss and accuracy).

    Args:
        history: Training history dictionary
        results_dir: Directory to save the plot
    """
    if not history:
        logging.warning("No training history found")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0, 0.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.6, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if 'learning_rates' in history:
        ax3.plot(epochs, history['learning_rates'], 'g-', label='Learning Rate')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = plots_dir / "training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Training history plot saved to {plot_path}")


def plot_class_distribution(
    class_counts: Dict[str, int],
    results_dir: Path,
    dataset_name: str
) -> None:
    """
    Plot class distribution in the dataset.

    Args:
        class_counts: Dictionary of class counts
        results_dir: Directory to save the plot
        dataset_name: Name of the dataset (train/val/test)
    """
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    bars = plt.bar(classes, counts, color=['skyblue', 'lightcoral'])
    plt.title(f'Class Distribution - {dataset_name.title()} Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom')

    plt.tight_layout()

    plot_path = plots_dir / f"class_distribution_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Class distribution plot saved to {plot_path}")


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    results_dir: Path,
    n_bins: int = 10,
    dataset_name: str = "test"
) -> None:
    """
    Plot reliability diagram (calibration curve) for binary classification.

    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Prediction probabilities shape (n_samples, 2)
        results_dir: Directory to save the plot
        n_bins: Number of bins for calibration (default: 10)
        dataset_name: Name for the plot title
    """
    from .metrics import calculate_ece

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_proba_positive = y_proba[:, 1]

    _, _, calibration_data = calculate_ece(y_true, y_proba_positive, n_bins)

    bin_confidences = []
    bin_accuracies = []
    bin_sizes = []
    bin_errors = []

    for bin_info in calibration_data['bins']:
        if bin_info['bin_size'] > 0:
            bin_confidences.append(bin_info['bin_confidence'])
            bin_accuracies.append(bin_info['bin_accuracy'])
            bin_sizes.append(bin_info['bin_size'])
            bin_errors.append(bin_info['bin_error'])

    if not bin_confidences:
        logging.warning("No bins with samples found for reliability diagram")
        return

    bin_confidences = np.array(bin_confidences)
    bin_accuracies = np.array(bin_accuracies)
    bin_sizes = np.array(bin_sizes)

    ece = sum(s * e for s, e in zip(bin_sizes, bin_errors)) / sum(bin_sizes)
    mce = max(bin_errors) if bin_errors else 0.0
    brier = brier_score_loss(y_true, y_proba_positive)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

    ax1.plot(
        bin_confidences, bin_accuracies,
        'o-', color='#1f77b4', linewidth=2, markersize=8,
        label='Model calibration'
    )

    for conf, acc in zip(bin_confidences, bin_accuracies):
        if acc < conf:
            ax1.fill_between(
                [conf - 0.02, conf + 0.02], [acc, acc], [conf, conf],
                color='red', alpha=0.3
            )
        else:
            ax1.fill_between(
                [conf - 0.02, conf + 0.02], [conf, conf], [acc, acc],
                color='blue', alpha=0.3
            )

    ax1.set_xlabel('Mean Predicted Probability (Confidence)', fontsize=12)
    ax1.set_ylabel('Fraction of Positives (Accuracy)', fontsize=12)
    ax1.set_title(
        f'Reliability Diagram - {dataset_name.title()} Set\n'
        f'ECE: {ece:.4f} | MCE: {mce:.4f} | Brier Score: {brier:.4f}',
        fontsize=14, fontweight='bold'
    )
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax2.bar(
        bin_confidences, bin_sizes,
        width=0.08, alpha=0.5, edgecolor='black',
        color='gray', label='Sample distribution'
    )
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Prediction Histogram', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim([0, 1])

    plt.tight_layout()

    plot_path = plots_dir / f"reliability_diagram_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Reliability diagram saved to {plot_path}")


def plot_bootstrap_distributions(
    bootstrap_results: Dict,
    results_dir: Path,
    dataset_name: str = "test"
) -> None:
    """
    Create distribution plots for bootstrap confidence intervals.

    Args:
        bootstrap_results: Dictionary from calculate_bootstrap_confidence_intervals
        results_dir: Directory to save plot
        dataset_name: Name for plot title
    """
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    available_metrics = [m for m in metrics if f'{m}_bootstrap_values' in bootstrap_results]

    if not available_metrics:
        return

    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    confidence_level = bootstrap_results.get('bootstrap_confidence_level', 0.95)
    metric_labels = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'roc_auc': 'ROC-AUC'
    }

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = np.array(bootstrap_results[f'{metric}_bootstrap_values'])
        mean_val = bootstrap_results[f'{metric}_mean']
        std_val = bootstrap_results[f'{metric}_std']
        ci_lower = bootstrap_results[f'{metric}_ci_lower']
        ci_upper = bootstrap_results[f'{metric}_ci_upper']

        ax.hist(values, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')

        x = np.linspace(values.min(), values.max(), 100)
        from scipy import stats
        ax.plot(x, stats.norm.pdf(x, mean_val, std_val), 'r-', lw=2, label='Normal fit')

        ax.axvline(mean_val, color='green', linestyle='--', lw=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(ci_lower, color='orange', linestyle=':', lw=2, label=f'CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.axvline(ci_upper, color='orange', linestyle=':', lw=2)
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='orange')

        ax.set_xlabel(metric_labels.get(metric, metric), fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{metric_labels.get(metric, metric)} Bootstrap Distribution\n'
                    f'Mean: {mean_val:.4f} +/- {std_val:.4f}',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Bootstrap Confidence Intervals - {dataset_name.title()} Set\n'
                f'({bootstrap_results.get("bootstrap_n_resamples", 1000)} resamples, '
                f'{confidence_level*100:.0f}% CI)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plot_path = plots_dir / f"bootstrap_distributions_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Bootstrap distributions saved to {plot_path}")


def plot_bootstrap_summary(
    bootstrap_results: Dict,
    point_estimates: Dict,
    results_dir: Path,
    dataset_name: str = "test"
) -> None:
    """
    Create summary plot comparing point estimates with bootstrap CIs.

    Args:
        bootstrap_results: Dictionary from calculate_bootstrap_confidence_intervals
        point_estimates: Dictionary with point estimate metrics
        results_dir: Directory to save plot
        dataset_name: Name for plot title
    """
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    available_metrics = [m for m in metrics if f'{m}_mean' in bootstrap_results]

    if not available_metrics:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(available_metrics))
    width = 0.35

    point_vals = [point_estimates.get(m, 0) for m in available_metrics]

    boot_means = [bootstrap_results[f'{m}_mean'] for m in available_metrics]
    ci_lowers = [bootstrap_results[f'{m}_ci_lower'] for m in available_metrics]
    ci_uppers = [bootstrap_results[f'{m}_ci_upper'] for m in available_metrics]
    ci_errors = [[m - l for m, l in zip(boot_means, ci_lowers)],
                 [u - m for m, u in zip(boot_means, ci_uppers)]]

    bars1 = ax.bar(x_pos - width/2, point_vals, width, label='Point Estimate',
                  color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, boot_means, width, label='Bootstrap Mean',
                  color='coral', alpha=0.8, yerr=ci_errors, capsize=5,
                  error_kw={'elinewidth': 2, 'capthick': 2})

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Point Estimates vs Bootstrap Confidence Intervals - {dataset_name.title()} Set',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
    ax.legend()
    ax.set_ylim(0.9, 1.01)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, point_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    for bar, mean, lower, upper in zip(bars2, boot_means, ci_lowers, ci_uppers):
        ax.text(bar.get_x() + bar.get_width()/2., upper + 0.003,
               f'{mean:.3f}\n[{lower:.3f}-{upper:.3f}]',
               ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    plot_path = plots_dir / f"bootstrap_summary_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Bootstrap summary saved to {plot_path}")


def plot_model_comparison(
    comparison_results: Dict,
    results_dir: Path
) -> None:
    """
    Create visualization comparing two models.

    Args:
        comparison_results: Dictionary from compare_models_comprehensive
        results_dir: Directory to save plot
    """
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_a_vals = [comparison_results['model_a_metrics'][m] for m in metrics]
    model_b_vals = [comparison_results['model_b_metrics'][m] for m in metrics]

    x_pos = np.arange(len(metrics))
    width = 0.35

    ax = axes[0, 0]
    bars1 = ax.bar(x_pos - width/2, model_a_vals, width,
                   label=comparison_results.get('model_name_a', 'Model A'),
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, model_b_vals, width,
                   label=comparison_results.get('model_name_b', 'Model B'),
                   color='coral', alpha=0.8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, model_a_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, model_b_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax = axes[0, 1]
    if 'mcnemars_test' in comparison_results:
        mcnemar = comparison_results['mcnemars_test']
        categories = ['Both\nCorrect', 'A Correct\nB Wrong', 'B Correct\nA Wrong', 'Both\nWrong']
        values = [
            mcnemar['both_correct'],
            mcnemar['a_correct_b_wrong'],
            mcnemar['b_correct_a_wrong'],
            mcnemar['both_wrong']
        ]
        colors = ['green', 'steelblue', 'coral', 'red']

        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title("McNemar's Test - Prediction Agreement", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.01,
                   str(val), ha='center', va='bottom', fontsize=10)

    ax = axes[1, 0]
    tests = []
    p_values = []
    significant = []

    if 'mcnemars_test' in comparison_results:
        tests.append("McNemar's")
        p_values.append(comparison_results['mcnemars_test']['p_value'])
        significant.append(comparison_results['mcnemars_test']['significant'])

    if 'paired_t_test' in comparison_results:
        tests.append("Paired t-test")
        p_values.append(comparison_results['paired_t_test']['p_value'])
        significant.append(comparison_results['paired_t_test']['significant'])

    if 'wilcoxon_test' in comparison_results:
        tests.append("Wilcoxon")
        p_values.append(comparison_results['wilcoxon_test']['p_value'])
        significant.append(comparison_results['wilcoxon_test']['significant'])

    colors = ['red' if sig else 'green' for sig in significant]
    bars = ax.barh(tests, p_values, color=colors, alpha=0.6)

    ax.axvline(x=0.05, color='black', linestyle='--', linewidth=2, label='alpha=0.05')
    ax.set_xlabel('p-value', fontsize=12)
    ax.set_title('Statistical Significance Tests', fontsize=14, fontweight='bold')
    ax.set_xlim([0, max(0.1, max(p_values) * 1.1)])
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    for bar, pval in zip(bars, p_values):
        ax.text(bar.get_width() + max(p_values)*0.02, bar.get_y() + bar.get_height()/2.,
               f'{pval:.4f}', va='center', fontsize=10)

    ax = axes[1, 1]
    ax.axis('off')

    conclusion_text = comparison_results.get('overall_conclusion', 'No conclusion available')
    ax.text(0.5, 0.5, conclusion_text,
           ha='center', va='center', fontsize=12, wrap=True,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    plot_path = plots_dir / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Model comparison plot saved to {plot_path}")
