"""
Evaluation package for baby cry detection model.

This package provides comprehensive model evaluation capabilities including:
- Metrics calculation (accuracy, precision, recall, F1, ROC-AUC, calibration)
- Statistical analysis (significance tests, bootstrap confidence intervals)
- Visualizations (confusion matrices, ROC curves, calibration diagrams)
- Model comparison tools

Binary classification: non-cry (0) vs cry (1)
"""

from .core import ModelEvaluator
from .metrics import (
    calculate_metrics,
    calculate_calibration_metrics,
    calculate_ece
)
from .analysis import (
    paired_t_test,
    wilcoxon_signed_rank_test,
    mcnemars_test,
    bootstrap_significance_test,
    calculate_bootstrap_confidence_intervals,
    compare_to_random_baseline,
    compare_to_baseline_benchmark
)
from .visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_training_history,
    plot_class_distribution,
    plot_reliability_diagram,
    plot_bootstrap_distributions,
    plot_bootstrap_summary,
    plot_model_comparison
)
from .utils import (
    predict_with_tta,
    EnsembleModel,
    generate_predictions,
    generate_predictions_and_log_errors
)

__all__ = [
    'ModelEvaluator',
    'calculate_metrics',
    'calculate_calibration_metrics',
    'calculate_ece',
    'paired_t_test',
    'wilcoxon_signed_rank_test',
    'mcnemars_test',
    'bootstrap_significance_test',
    'calculate_bootstrap_confidence_intervals',
    'compare_to_random_baseline',
    'compare_to_baseline_benchmark',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_training_history',
    'plot_class_distribution',
    'plot_reliability_diagram',
    'plot_bootstrap_distributions',
    'plot_bootstrap_summary',
    'plot_model_comparison',
    'predict_with_tta',
    'EnsembleModel',
    'generate_predictions',
    'generate_predictions_and_log_errors',
]
