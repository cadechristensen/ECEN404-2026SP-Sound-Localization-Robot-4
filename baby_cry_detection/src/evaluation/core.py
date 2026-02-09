"""
Core evaluation module containing the ModelEvaluator class.

This module provides the main interface for model evaluation,
coordinating metrics calculation, visualization, and statistical analysis.
Designed for binary classification (non-cry vs cry detection).
"""

from typing import Dict, Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import logging

from .metrics import calculate_metrics
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
from .analysis import (
    paired_t_test,
    wilcoxon_signed_rank_test,
    mcnemars_test,
    bootstrap_significance_test,
    calculate_bootstrap_confidence_intervals,
    compare_to_random_baseline,
    compare_to_baseline_benchmark
)
from .utils import (
    generate_predictions,
    generate_predictions_and_log_errors,
    setup_module_aliases
)


class ModelEvaluator:
    """
    Comprehensive model evaluator for baby cry detection.
    Provides various metrics and visualizations for model performance analysis.
    """

    def __init__(self, config, use_tta: bool = False, tta_n_augments: int = 5):
        """
        Initialize the evaluator.

        Args:
            config: Configuration object
            use_tta: Whether to use test-time augmentation
            tta_n_augments: Number of augmentations for TTA
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = list(config.CLASS_LABELS.values())
        self.use_tta = use_tta
        self.tta_n_augments = tta_n_augments

        from ..model import create_model
        self.model = create_model(config).to(self.device)

    def load_model(self, checkpoint_path: Path):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        setup_module_aliases()

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
        else:
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)

        self.model.eval()
        logging.info(f"Model loaded from {checkpoint_path}")

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.

        Args:
            data_loader: DataLoader for the dataset

        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        return generate_predictions(
            self.model,
            data_loader,
            self.device,
            self.use_tta,
            self.tta_n_augments
        )

    def predict_and_log_errors(
        self,
        data_loader: DataLoader,
        dataset_name: str,
        results_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions and log misclassified files.

        Args:
            data_loader: DataLoader for the dataset
            dataset_name: Name of dataset (train/val/test)
            results_dir: Directory to save error log

        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        return generate_predictions_and_log_errors(
            self.model,
            data_loader,
            self.device,
            self.class_labels,
            dataset_name,
            results_dir,
            self.use_tta,
            self.tta_n_augments
        )

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        Returns:
            Dictionary containing all metrics
        """
        return calculate_metrics(y_true, y_pred, y_proba, self.class_labels)

    def paired_t_test(
        self,
        scores_model_a: np.ndarray,
        scores_model_b: np.ndarray,
        metric_name: str = "accuracy",
        alpha: float = 0.05
    ) -> Dict:
        """Perform paired t-test to compare two models."""
        return paired_t_test(scores_model_a, scores_model_b, metric_name, alpha)

    def wilcoxon_signed_rank_test(
        self,
        scores_model_a: np.ndarray,
        scores_model_b: np.ndarray,
        metric_name: str = "accuracy",
        alpha: float = 0.05
    ) -> Dict:
        """Perform Wilcoxon signed-rank test to compare two models."""
        return wilcoxon_signed_rank_test(scores_model_a, scores_model_b, metric_name, alpha)

    def mcnemars_test(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        alpha: float = 0.05,
        correction: bool = True
    ) -> Dict:
        """Perform McNemar's test to compare two models."""
        return mcnemars_test(y_true, y_pred_a, y_pred_b, alpha, correction)

    def bootstrap_significance_test(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        y_proba_a: Optional[np.ndarray] = None,
        y_proba_b: Optional[np.ndarray] = None,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42
    ) -> Dict:
        """Perform bootstrap hypothesis testing to compare two models."""
        return bootstrap_significance_test(
            y_true, y_pred_a, y_pred_b, y_proba_a, y_proba_b,
            n_bootstrap, confidence_level, random_state
        )

    def calculate_bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42
    ) -> Dict:
        """Calculate bootstrap confidence intervals for various metrics."""
        return calculate_bootstrap_confidence_intervals(
            y_true, y_pred, y_proba,
            n_bootstrap, confidence_level, random_state
        )

    def compare_to_random_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """Compare model performance to a random baseline."""
        return compare_to_random_baseline(y_true, y_pred, alpha)

    def compare_to_baseline_benchmark(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_accuracy: float = 0.95,
        alpha: float = 0.05
    ) -> Dict:
        """Compare model performance to a baseline benchmark."""
        return compare_to_baseline_benchmark(y_true, y_pred, baseline_accuracy, alpha)

    def compare_models_comprehensive(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_proba_a: np.ndarray,
        y_pred_b: np.ndarray,
        y_proba_b: np.ndarray,
        model_name_a: str = "Model A",
        model_name_b: str = "Model B",
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform comprehensive statistical comparison between two models.

        Args:
            y_true: True labels
            y_pred_a: Predictions from model A
            y_proba_a: Probabilities from model A
            y_pred_b: Predictions from model B
            y_proba_b: Probabilities from model B
            model_name_a: Name of model A
            model_name_b: Name of model B
            alpha: Significance level

        Returns:
            Comprehensive comparison dictionary
        """
        from sklearn.metrics import accuracy_score, f1_score

        n_samples = len(y_true)

        accuracy_a = accuracy_score(y_true, y_pred_a)
        accuracy_b = accuracy_score(y_true, y_pred_b)
        f1_a = f1_score(y_true, y_pred_a, average='weighted')
        f1_b = f1_score(y_true, y_pred_b, average='weighted')

        correct_a = (y_pred_a == y_true).astype(float)
        correct_b = (y_pred_b == y_true).astype(float)

        prob_scores_a = np.array([
            y_proba_a[i, int(y_true[i])] for i in range(n_samples)
        ])
        prob_scores_b = np.array([
            y_proba_b[i, int(y_true[i])] for i in range(n_samples)
        ])

        results = {
            "comparison_summary": {
                "model_a": model_name_a,
                "model_b": model_name_b,
                "n_samples": n_samples,
                "alpha": alpha
            },
            "model_a_metrics": {
                "name": model_name_a,
                "accuracy": float(accuracy_a),
                "f1_score": float(f1_a)
            },
            "model_b_metrics": {
                "name": model_name_b,
                "accuracy": float(accuracy_b),
                "f1_score": float(f1_b)
            }
        }

        results["mcnemars_test"] = self.mcnemars_test(
            y_true, y_pred_a, y_pred_b, alpha=alpha
        )

        results["paired_t_test"] = self.paired_t_test(
            correct_a, correct_b,
            metric_name="per-sample correctness",
            alpha=alpha
        )

        results["wilcoxon_test"] = self.wilcoxon_signed_rank_test(
            correct_a, correct_b,
            metric_name="per-sample correctness",
            alpha=alpha
        )

        mcnemar_sig = results["mcnemars_test"]["significant"]
        ttest_sig = results["paired_t_test"]["significant"]
        wilcoxon_sig = results["wilcoxon_test"]["significant"]

        significant_tests = sum([mcnemar_sig, ttest_sig, wilcoxon_sig])

        if significant_tests == 0:
            overall_conclusion = (
                f"NO SIGNIFICANT DIFFERENCE between {model_name_a} and {model_name_b}. "
                "All tests agree: cannot conclude one model is better."
            )
        elif significant_tests == 3:
            better = model_name_a if accuracy_a > accuracy_b else model_name_b
            overall_conclusion = (
                f"STRONG EVIDENCE that {better} is significantly better. "
                "All tests show significant difference."
            )
        else:
            tests_significant = []
            if mcnemar_sig:
                tests_significant.append("McNemar's")
            if ttest_sig:
                tests_significant.append("t-test")
            if wilcoxon_sig:
                tests_significant.append("Wilcoxon")

            better = model_name_a if accuracy_a > accuracy_b else model_name_b
            overall_conclusion = (
                f"MIXED EVIDENCE. {', '.join(tests_significant)} show(s) {better} is significantly better, "
                "but not all tests agree. Interpret with caution."
            )

        results["overall_conclusion"] = overall_conclusion
        results["model_name_a"] = model_name_a
        results["model_name_b"] = model_name_b

        return results

    def plot_confusion_matrix(self, cm: np.ndarray, results_dir: Path):
        """Plot and save confusion matrix."""
        plot_confusion_matrix(cm, self.class_labels, results_dir)

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, results_dir: Path):
        """Plot and save ROC curve."""
        plot_roc_curve(y_true, y_proba, self.class_labels, results_dir)

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, results_dir: Path):
        """Plot and save Precision-Recall curve."""
        plot_precision_recall_curve(y_true, y_proba, self.class_labels, results_dir)

    def plot_training_history(self, history: Dict, results_dir: Path):
        """Plot training history."""
        plot_training_history(history, results_dir)

    def plot_class_distribution(self, data_loader: DataLoader, results_dir: Path, dataset_name: str):
        """Plot class distribution in the dataset."""
        class_counts = {label: 0 for label in self.class_labels}

        for batch_data in data_loader:
            if len(batch_data) == 3:
                _, labels, _ = batch_data
            else:
                _, labels = batch_data

            for label in labels:
                class_name = self.class_labels[label.item()]
                class_counts[class_name] += 1

        plot_class_distribution(class_counts, results_dir, dataset_name)

    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        results_dir: Path,
        n_bins: int = 10,
        dataset_name: str = "test"
    ):
        """Plot reliability diagram."""
        plot_reliability_diagram(y_true, y_proba, results_dir, n_bins, dataset_name)

    def plot_bootstrap_distributions(
        self,
        bootstrap_results: Dict,
        results_dir: Path,
        dataset_name: str = "test"
    ):
        """Create distribution plots for bootstrap confidence intervals."""
        plot_bootstrap_distributions(bootstrap_results, results_dir, dataset_name)

    def plot_bootstrap_summary(
        self,
        bootstrap_results: Dict,
        point_estimates: Dict,
        results_dir: Path,
        dataset_name: str = "test"
    ):
        """Create summary plot comparing point estimates with bootstrap CIs."""
        plot_bootstrap_summary(bootstrap_results, point_estimates, results_dir, dataset_name)

    def plot_model_comparison(
        self,
        comparison_results: Dict,
        results_dir: Path
    ):
        """Create visualization comparing two models."""
        plot_model_comparison(comparison_results, results_dir)

    def apply_temperature_scaling(
        self,
        val_loader: DataLoader,
        results_dir: Path,
        save_calibrated_model: bool = True
    ) -> Tuple[Optional[object], dict]:
        """
        Apply temperature scaling calibration to the loaded model.

        Args:
            val_loader: DataLoader for validation set
            results_dir: Directory to save calibration results
            save_calibrated_model: Whether to save the calibrated model

        Returns:
            Tuple of (calibrated_model, calibration_results)
        """
        from ..calibration import TemperatureScaledModel

        logging.info("=" * 60)
        logging.info("APPLYING TEMPERATURE SCALING CALIBRATION")
        logging.info("=" * 60)

        try:
            calibrated_model = TemperatureScaledModel(self.model)
            calibrated_model.to(self.device)

            calibration_results = calibrated_model.calibrate(
                val_loader,
                self.device,
                max_iter=100,
                verbose=True
            )

            self.model = calibrated_model

            calibration_file = results_dir / "temperature_calibration_results.json"
            with open(calibration_file, 'w', encoding='utf-8') as f:
                json.dump(calibration_results, f, indent=2)
            logging.info(f"Calibration results saved to {calibration_file}")

            if save_calibrated_model:
                calibrated_model_path = results_dir / "calibrated_model.pth"
                calibrated_model.save_calibrated_model(str(calibrated_model_path))

            logging.info("=" * 60)
            logging.info("TEMPERATURE SCALING CALIBRATION COMPLETE")
            logging.info(f"  Optimal Temperature: {calibration_results['optimal_temperature']:.4f}")
            logging.info(f"  ECE: {calibration_results['initial_ece']:.4f} -> {calibration_results['final_ece']:.4f}")
            logging.info(f"  MCE: {calibration_results['initial_mce']:.4f} -> {calibration_results['final_mce']:.4f}")
            logging.info("=" * 60)

            return calibrated_model, calibration_results

        except Exception as e:
            logging.error(f"Temperature scaling calibration failed: {e}")
            import traceback
            traceback.print_exc()
            return None, {'error': str(e)}

    def evaluate_model(
        self,
        data_loader: DataLoader,
        results_dir: Path,
        dataset_name: str = "test",
        compute_bootstrap_ci: bool = True,
        n_bootstrap: int = 1000,
        bootstrap_confidence_level: float = 0.95
    ) -> Dict:
        """
        Comprehensive model evaluation with optional bootstrap confidence intervals.

        Args:
            data_loader: DataLoader for evaluation
            results_dir: Directory to save results
            dataset_name: Name of the dataset being evaluated
            compute_bootstrap_ci: Whether to compute bootstrap CIs
            n_bootstrap: Number of bootstrap resamples
            bootstrap_confidence_level: Confidence level for CIs

        Returns:
            Dictionary containing all evaluation metrics
        """
        logging.info(f"Evaluating model on {dataset_name} set...")

        y_pred, y_proba, y_true = self.predict_and_log_errors(data_loader, dataset_name, results_dir)

        metrics = self.calculate_metrics(y_true, y_pred, y_proba)

        cm = np.array(metrics['confusion_matrix'])
        self.plot_confusion_matrix(cm, results_dir)
        self.plot_roc_curve(y_true, y_proba, results_dir)
        self.plot_precision_recall_curve(y_true, y_proba, results_dir)
        self.plot_class_distribution(data_loader, results_dir, dataset_name)

        if len(self.class_labels) == 2:
            self.plot_reliability_diagram(y_true, y_proba, results_dir, dataset_name=dataset_name)

        bootstrap_results = None
        if compute_bootstrap_ci:
            bootstrap_results = self.calculate_bootstrap_confidence_intervals(
                y_true, y_pred, y_proba,
                n_bootstrap=n_bootstrap,
                confidence_level=bootstrap_confidence_level
            )
            bootstrap_summary = {k: v for k, v in bootstrap_results.items()
                               if not k.endswith('_bootstrap_values')}
            metrics['bootstrap_confidence_intervals'] = bootstrap_summary

            self.plot_bootstrap_distributions(bootstrap_results, results_dir, dataset_name)
            self.plot_bootstrap_summary(bootstrap_results, metrics, results_dir, dataset_name)

            bootstrap_file = results_dir / f"bootstrap_results_{dataset_name}.json"
            with open(bootstrap_file, 'w', encoding='utf-8') as f:
                json.dump(bootstrap_results, f, indent=2)
            logging.info(f"Full bootstrap results saved to {bootstrap_file}")

        metrics_file = results_dir / f"metrics_{dataset_name}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, np.floating):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            json.dump(serializable_metrics, f, indent=2)

        logging.info(f"\n{dataset_name.upper()} SET EVALUATION RESULTS:")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
        logging.info(f"F1-Score: {metrics['f1_score']:.4f}")

        if 'roc_auc' in metrics:
            logging.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            logging.info(f"PR AUC: {metrics['pr_auc']:.4f}")

        if 'brier_score' in metrics:
            logging.info(f"Brier Score: {metrics['brier_score']:.4f}")
            logging.info(f"Expected Calibration Error (ECE): {metrics['expected_calibration_error']:.4f}")
            logging.info(f"Maximum Calibration Error (MCE): {metrics['maximum_calibration_error']:.4f}")

        if bootstrap_results and 'accuracy_ci_lower' in bootstrap_results:
            logging.info(f"\nBOOTSTRAP CONFIDENCE INTERVALS ({bootstrap_confidence_level*100:.0f}%):")
            logging.info(f"  Accuracy: {metrics['accuracy']:.4f} [{bootstrap_results['accuracy_ci_lower']:.4f}, {bootstrap_results['accuracy_ci_upper']:.4f}]")
            if 'roc_auc_ci_lower' in bootstrap_results:
                logging.info(f"  ROC AUC: {metrics['roc_auc']:.4f} [{bootstrap_results['roc_auc_ci_lower']:.4f}, {bootstrap_results['roc_auc_ci_upper']:.4f}]")

        return metrics
