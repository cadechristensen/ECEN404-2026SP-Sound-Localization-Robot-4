"""
Main evaluation script for baby cry detection model.

This module provides a clean interface for model evaluation, importing from
the modular evaluation package. It includes standalone functions for common
evaluation tasks and a command-line interface.

Designed for binary classification: non-cry (0) vs cry (1)
"""

import sys
import io

if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import json
from sklearn.metrics import accuracy_score

try:
    from .config import Config
    from .dataset import DatasetManager
    from .evaluation import ModelEvaluator, EnsembleModel
except ImportError:
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from config import Config
    from dataset import DatasetManager
    from evaluation import ModelEvaluator, EnsembleModel


logger = logging.getLogger(__name__)


def evaluate_saved_model(
    model_path: Path,
    config: Optional[Config] = None,
    compute_bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    bootstrap_confidence_level: float = 0.95,
    apply_temperature_scaling: bool = True,
    save_calibrated_model: bool = True
) -> Tuple[Dict, Dict, Dict]:
    """
    Evaluate a saved model with optional bootstrap confidence intervals and
    temperature scaling calibration.

    Args:
        model_path: Path to saved model
        config: Configuration object. Defaults to Config().
        compute_bootstrap_ci: Whether to compute bootstrap CIs (default: True)
        n_bootstrap: Number of bootstrap resamples (default: 1000)
        bootstrap_confidence_level: Confidence level for CIs (default: 0.95)
        apply_temperature_scaling: Whether to apply temperature scaling
            calibration before evaluation (default: False). When True, the
            model will be calibrated on the validation set using temperature
            scaling to improve probability calibration (reduce ECE/MCE).
        save_calibrated_model: Whether to save the calibrated model to disk
            (default: True). Only used if apply_temperature_scaling is True.

    Returns:
        Tuple of (train_metrics, val_metrics, test_metrics) dictionaries.
        If apply_temperature_scaling is True, metrics also include
        'temperature_calibration' with calibration results.
    """
    if config is None:
        config = Config()

    results_dir = config.get_results_dir()

    log_file = results_dir / "logs" / "evaluation.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Use a module-level logger so repeated calls each get their own FileHandler
    # without the root-logger no-op problem of logging.basicConfig (which is a
    # no-op after the first call and would silently write all subsequent eval
    # output to the first log file).
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    # Logger at DEBUG so all messages reach handlers; each handler filters independently.
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_h = logging.FileHandler(log_file)
    file_h.setLevel(logging.DEBUG)       # File captures everything
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler()
    stream_h.setLevel(getattr(logging, config.LOG_LEVEL))  # Terminal shows INFO+ only
    stream_h.setFormatter(fmt)
    logger.addHandler(file_h)
    logger.addHandler(stream_h)

    if compute_bootstrap_ci:
        logger.info(f"Bootstrap CI enabled: {n_bootstrap} resamples, {bootstrap_confidence_level*100:.0f}% CI")

    if apply_temperature_scaling:
        logger.info("Temperature scaling calibration enabled")

    evaluator = ModelEvaluator(config)

    # Load ensemble if enabled and checkpoint metadata exists
    if config.USE_ENSEMBLE:
        results_dir_for_ensemble = model_path.parent
        ensemble_metadata_path = results_dir_for_ensemble / "ensemble_checkpoints.json"
        if ensemble_metadata_path.exists():
            logger.info("USE_ENSEMBLE=True — loading ensemble model from checkpoints")
            ensemble_model = EnsembleModel.from_results_dir(
                results_dir_for_ensemble, config, evaluator.device
            )
            evaluator.model = ensemble_model
        else:
            logger.warning("USE_ENSEMBLE=True but no ensemble_checkpoints.json found — falling back to single model")
            evaluator.load_model(model_path)
    else:
        evaluator.load_model(model_path)

    dataset_manager = DatasetManager(config)
    train_dataset, val_dataset, test_dataset = dataset_manager.prepare_datasets()
    train_loader, val_loader, test_loader = dataset_manager.create_data_loaders(
        train_dataset, val_dataset, test_dataset
    )

    calibration_results = None
    if apply_temperature_scaling:
        _, calibration_results = evaluator.apply_temperature_scaling(
            val_loader, results_dir, save_calibrated_model=save_calibrated_model
        )

    # Bootstrap is intentionally disabled for the training split — it adds
    # significant latency and training-set metrics are not used for model selection.
    train_metrics = evaluator.evaluate_model(
        train_loader, results_dir, "train",
        compute_bootstrap_ci=False,
    )
    val_metrics = evaluator.evaluate_model(
        val_loader, results_dir, "val",
        compute_bootstrap_ci=compute_bootstrap_ci,
        n_bootstrap=n_bootstrap,
        bootstrap_confidence_level=bootstrap_confidence_level
    )
    test_metrics = evaluator.evaluate_model(
        test_loader, results_dir, "test",
        compute_bootstrap_ci=compute_bootstrap_ci,
        n_bootstrap=n_bootstrap,
        bootstrap_confidence_level=bootstrap_confidence_level
    )

    if calibration_results is not None:
        test_metrics['temperature_calibration'] = calibration_results

    history_file = model_path.parent / "training_history.json"
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        evaluator.plot_training_history(history, results_dir)

    logger.info(f"Evaluation completed. Results saved to {results_dir}")

    if compute_bootstrap_ci and 'bootstrap_confidence_intervals' in test_metrics:
        ci = test_metrics['bootstrap_confidence_intervals']
        _nan = float('nan')  # sentinel: missing CI bound is a bug, not zero
        logger.info("\n" + "="*60)
        logger.info(f"FINAL TEST SET RESULTS WITH {bootstrap_confidence_level*100:.0f}% CONFIDENCE INTERVALS")
        logger.info("="*60)
        logger.info(f"Accuracy:  {test_metrics['accuracy']:.4f} [{ci['accuracy_ci_lower']:.4f}, {ci['accuracy_ci_upper']:.4f}]")
        logger.info(f"Precision: {test_metrics['precision']:.4f} [{ci['precision_ci_lower']:.4f}, {ci['precision_ci_upper']:.4f}]")
        logger.info(f"Recall:    {test_metrics['recall']:.4f} [{ci['recall_ci_lower']:.4f}, {ci['recall_ci_upper']:.4f}]")
        logger.info(f"F1-Score:  {test_metrics['f1_score']:.4f} [{ci['f1_score_ci_lower']:.4f}, {ci['f1_score_ci_upper']:.4f}]")
        if 'roc_auc' in test_metrics:
            logger.info(f"ROC-AUC:   {test_metrics['roc_auc']:.4f} [{ci.get('roc_auc_ci_lower', _nan):.4f}, {ci.get('roc_auc_ci_upper', _nan):.4f}]")
        logger.info("="*60)

    if calibration_results is not None and 'optimal_temperature' in calibration_results:
        logger.info("\n" + "="*60)
        logger.info("TEMPERATURE SCALING CALIBRATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Optimal Temperature: {calibration_results['optimal_temperature']:.4f}")
        logger.info(f"ECE: {calibration_results['initial_ece']*100:.2f}% -> {calibration_results['final_ece']*100:.2f}%")
        logger.info(f"MCE: {calibration_results['initial_mce']*100:.2f}% -> {calibration_results['final_mce']*100:.2f}%")
        logger.info(f"ECE Improvement: {calibration_results['ece_improvement_percent']:.1f}%")
        logger.info("="*60)

    return train_metrics, val_metrics, test_metrics


def compare_two_models(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_proba_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_proba_b: np.ndarray,
    model_name_a: str = "Model A",
    model_name_b: str = "Model B",
    results_dir: Optional[Path] = None,
    alpha: float = 0.05,
    config: Optional[Config] = None
) -> Dict:
    """
    Standalone function to compare two models with full statistical analysis.

    This is a convenience wrapper for the comprehensive model comparison that:
    1. Runs all statistical significance tests (McNemar's, t-test, Wilcoxon)
    2. Generates visualizations
    3. Saves results to JSON

    For BINARY CLASSIFICATION: non_cry (0) vs cry (1)

    Usage example:
        >>> y_true = np.array([0, 1, 1, 0, 1, ...])
        >>> y_pred_a = np.array([0, 1, 1, 0, 0, ...])
        >>> y_proba_a = np.array([[0.9, 0.1], [0.2, 0.8], ...])
        >>> y_pred_b = np.array([0, 1, 0, 0, 1, ...])
        >>> y_proba_b = np.array([[0.85, 0.15], [0.3, 0.7], ...])
        >>> results = compare_two_models(
        ...     y_true, y_pred_a, y_proba_a, y_pred_b, y_proba_b,
        ...     model_name_a="ResNet50", model_name_b="MobileNet"
        ... )

    Args:
        y_true: True labels
        y_pred_a: Predictions from model A
        y_proba_a: Probabilities from model A (n_samples, 2)
        y_pred_b: Predictions from model B
        y_proba_b: Probabilities from model B (n_samples, 2)
        model_name_a: Name of model A (for reporting)
        model_name_b: Name of model B (for reporting)
        results_dir: Optional directory to save results and plots
        alpha: Significance level (default: 0.05)
        config: Optional Config object

    Returns:
        Comprehensive comparison dictionary with all test results and conclusions
    """
    if config is None:
        config = Config()

    evaluator = ModelEvaluator(config)

    results = evaluator.compare_models_comprehensive(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_proba_a=y_proba_a,
        y_pred_b=y_pred_b,
        y_proba_b=y_proba_b,
        model_name_a=model_name_a,
        model_name_b=model_name_b,
        alpha=alpha
    )

    results["model_a_vs_random"] = evaluator.compare_to_random_baseline(
        y_true, y_pred_a, alpha=alpha
    )
    results["model_b_vs_random"] = evaluator.compare_to_random_baseline(
        y_true, y_pred_b, alpha=alpha
    )
    results["model_a_vs_baseline"] = evaluator.compare_to_baseline_benchmark(
        y_true, y_pred_a, alpha=alpha
    )
    results["model_b_vs_baseline"] = evaluator.compare_to_baseline_benchmark(
        y_true, y_pred_b, alpha=alpha
    )

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        (results_dir / "plots").mkdir(parents=True, exist_ok=True)

        evaluator.plot_model_comparison(results, results_dir)

        results_file = results_dir / "model_comparison_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Comparison results saved to {results_file}")

    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model A ({model_name_a}): Accuracy = {results['model_a_metrics']['accuracy']:.4f}")
    logger.info(f"Model B ({model_name_b}): Accuracy = {results['model_b_metrics']['accuracy']:.4f}")
    logger.info(f"\nMcNemar's test p-value: {results['mcnemars_test']['p_value']:.4f}")
    logger.info(f"Significant at alpha={alpha}: {results['mcnemars_test']['significant']}")
    logger.info(f"\nConclusion: {results['overall_conclusion']}")
    logger.info("=" * 60)

    return results


def run_significance_tests_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    alpha: float = 0.05,
    config: Optional[Config] = None
) -> Dict:
    """
    Run all significance tests for a single model against baselines.

    This provides a quick summary of how a model compares to:
    1. Random baseline (50% accuracy)
    2. Baseline benchmark (95% accuracy)

    For BINARY CLASSIFICATION: non_cry (0) vs cry (1)

    Args:
        y_true: True labels
        y_pred: Model predictions
        model_name: Name of the model (for reporting)
        alpha: Significance level (default: 0.05)
        config: Optional Config object

    Returns:
        Dictionary with comparison results against all baselines
    """
    if config is None:
        config = Config()

    evaluator = ModelEvaluator(config)
    accuracy = accuracy_score(y_true, y_pred)

    results = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "n_samples": len(y_true),
        "alpha": alpha,
        "vs_random": evaluator.compare_to_random_baseline(y_true, y_pred, alpha),
        "vs_baseline": evaluator.compare_to_baseline_benchmark(y_true, y_pred, alpha=alpha)
    }

    logger.info("\n" + "=" * 60)
    logger.info(f"SIGNIFICANCE TESTS FOR: {model_name}")
    logger.info("=" * 60)
    logger.info(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Sample Size: {len(y_true)}")
    logger.info("-" * 60)
    logger.info("VS RANDOM BASELINE (50%):")
    logger.info(f"  {results['vs_random']['conclusion']}")
    logger.info("-" * 60)
    logger.info("VS BASELINE BENCHMARK (95%):")
    logger.info(f"  {results['vs_baseline']['comparison_summary']}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a baby cry detection model with optional bootstrap confidence intervals"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap confidence interval calculation"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)"
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)"
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    config = Config()
    evaluate_saved_model(
        model_path,
        config,
        compute_bootstrap_ci=not args.no_bootstrap,
        n_bootstrap=args.n_bootstrap,
        bootstrap_confidence_level=args.confidence_level
    )
