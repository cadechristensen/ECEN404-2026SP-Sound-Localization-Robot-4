"""
Statistical analysis module for model evaluation.

This module contains functions for statistical significance testing,
bootstrap confidence intervals, and model comparison.
Designed for binary classification (non-cry vs cry detection).
"""

from typing import Dict, List, Optional
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from tqdm import tqdm
import logging


def paired_t_test(
    scores_model_a: np.ndarray,
    scores_model_b: np.ndarray,
    metric_name: str = "accuracy",
    alpha: float = 0.05
) -> Dict:
    """
    Perform paired t-test to compare two models' per-sample performance.

    Args:
        scores_model_a: Per-sample scores from model A
        scores_model_b: Per-sample scores from model B
        metric_name: Name of the metric being compared
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary containing test results and interpretation
    """
    if len(scores_model_a) != len(scores_model_b):
        raise ValueError(
            f"Score arrays must have same length. "
            f"Got {len(scores_model_a)} and {len(scores_model_b)}"
        )

    n_samples = len(scores_model_a)
    differences = scores_model_a - scores_model_b

    mean_a = float(np.mean(scores_model_a))
    mean_b = float(np.mean(scores_model_b))
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))

    t_stat, p_value = stats.ttest_rel(scores_model_a, scores_model_b)

    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    se_diff = std_diff / np.sqrt(n_samples)
    t_critical = stats.t.ppf(1 - alpha / 2, df=n_samples - 1)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff

    significant = p_value < alpha

    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_interpretation = "negligible"
    elif abs_d < 0.5:
        effect_interpretation = "small"
    elif abs_d < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    if significant:
        better_model = "A" if mean_diff > 0 else "B"
        interpretation = (
            f"SIGNIFICANT DIFFERENCE FOUND (p={p_value:.4f} < {alpha}). "
            f"Model {better_model} performs significantly better on {metric_name}. "
            f"Effect size (Cohen's d={cohens_d:.3f}) is {effect_interpretation}."
        )
    else:
        interpretation = (
            f"NO SIGNIFICANT DIFFERENCE (p={p_value:.4f} >= {alpha}). "
            f"Cannot conclude that either model is better on {metric_name}. "
            f"Effect size (Cohen's d={cohens_d:.3f}) is {effect_interpretation}."
        )

    return {
        "test_name": "Paired t-test",
        "metric": metric_name,
        "n_samples": n_samples,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "effect_size_interpretation": effect_interpretation,
        "alpha": alpha,
        "significant": significant,
        "confidence_interval": {
            "lower": float(ci_lower),
            "upper": float(ci_upper),
            "level": 1 - alpha
        },
        "interpretation": interpretation,
        "assumptions": [
            "Paired observations (same samples evaluated by both models)",
            "Differences are approximately normally distributed",
            "If N > 30, normality assumption is less critical (CLT)"
        ]
    }


def wilcoxon_signed_rank_test(
    scores_model_a: np.ndarray,
    scores_model_b: np.ndarray,
    metric_name: str = "accuracy",
    alpha: float = 0.05
) -> Dict:
    """
    Perform Wilcoxon signed-rank test to compare two models.

    This is the non-parametric alternative to the paired t-test.

    Args:
        scores_model_a: Per-sample scores from model A
        scores_model_b: Per-sample scores from model B
        metric_name: Name of the metric being compared
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary containing test results and interpretation
    """
    if len(scores_model_a) != len(scores_model_b):
        raise ValueError(
            f"Score arrays must have same length. "
            f"Got {len(scores_model_a)} and {len(scores_model_b)}"
        )

    n_samples = len(scores_model_a)
    differences = scores_model_a - scores_model_b

    median_a = float(np.median(scores_model_a))
    median_b = float(np.median(scores_model_b))
    median_diff = float(np.median(differences))

    non_zero_diffs = differences[differences != 0]
    n_non_zero = len(non_zero_diffs)

    if n_non_zero == 0:
        return {
            "test_name": "Wilcoxon signed-rank test",
            "metric": metric_name,
            "n_samples": n_samples,
            "n_non_zero_differences": 0,
            "median_a": median_a,
            "median_b": median_b,
            "median_diff": 0.0,
            "w_statistic": None,
            "p_value": 1.0,
            "rank_biserial_r": 0.0,
            "effect_size_interpretation": "none (identical)",
            "alpha": alpha,
            "significant": False,
            "interpretation": (
                "Models have IDENTICAL predictions on all samples. "
                "No difference to test."
            ),
            "assumptions": [
                "Paired observations",
                "Differences are symmetric around the median"
            ]
        }

    try:
        w_stat, p_value = stats.wilcoxon(
            scores_model_a, scores_model_b,
            alternative='two-sided',
            zero_method='wilcox'
        )
    except ValueError as e:
        logging.warning(f"Wilcoxon test warning: {e}")
        w_stat = 0.0
        p_value = 1.0

    max_w = n_non_zero * (n_non_zero + 1) / 2
    rank_biserial_r = 1 - (2 * w_stat) / max_w if max_w > 0 else 0.0

    abs_r = abs(rank_biserial_r)
    if abs_r < 0.1:
        effect_interpretation = "negligible"
    elif abs_r < 0.3:
        effect_interpretation = "small"
    elif abs_r < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    significant = p_value < alpha

    if significant:
        better_model = "A" if median_diff > 0 else "B"
        interpretation = (
            f"SIGNIFICANT DIFFERENCE FOUND (p={p_value:.4f} < {alpha}). "
            f"Model {better_model} performs significantly better on {metric_name}. "
            f"Effect size (rank-biserial r={rank_biserial_r:.3f}) is {effect_interpretation}."
        )
    else:
        interpretation = (
            f"NO SIGNIFICANT DIFFERENCE (p={p_value:.4f} >= {alpha}). "
            f"Cannot conclude that either model is better on {metric_name}. "
            f"Effect size (rank-biserial r={rank_biserial_r:.3f}) is {effect_interpretation}."
        )

    return {
        "test_name": "Wilcoxon signed-rank test",
        "metric": metric_name,
        "n_samples": n_samples,
        "n_non_zero_differences": n_non_zero,
        "median_a": median_a,
        "median_b": median_b,
        "median_diff": median_diff,
        "w_statistic": float(w_stat),
        "p_value": float(p_value),
        "rank_biserial_r": float(rank_biserial_r),
        "effect_size_interpretation": effect_interpretation,
        "alpha": alpha,
        "significant": significant,
        "interpretation": interpretation,
        "assumptions": [
            "Paired observations",
            "Differences are symmetric around the median"
        ]
    }


def mcnemars_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    alpha: float = 0.05,
    correction: bool = True
) -> Dict:
    """
    Perform McNemar's test to compare two models on the same dataset.

    Args:
        y_true: True labels
        y_pred_a: Predictions from model A
        y_pred_b: Predictions from model B
        alpha: Significance level (default: 0.05)
        correction: Whether to apply continuity correction

    Returns:
        Dictionary containing test results and interpretation
    """
    if len(y_true) != len(y_pred_a) or len(y_true) != len(y_pred_b):
        raise ValueError("All arrays must have the same length")

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    both_correct = np.sum(correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)
    a_correct_b_wrong = np.sum(correct_a & ~correct_b)
    b_correct_a_wrong = np.sum(~correct_a & correct_b)

    n = len(y_true)
    accuracy_a = float(np.mean(correct_a))
    accuracy_b = float(np.mean(correct_b))

    contingency = np.array([[both_correct, a_correct_b_wrong],
                            [b_correct_a_wrong, both_wrong]])

    if a_correct_b_wrong + b_correct_a_wrong == 0:
        return {
            "test_name": "McNemar's test",
            "n_samples": n,
            "accuracy_a": accuracy_a,
            "accuracy_b": accuracy_b,
            "contingency_table": contingency.tolist(),
            "both_correct": int(both_correct),
            "both_wrong": int(both_wrong),
            "a_correct_b_wrong": int(a_correct_b_wrong),
            "b_correct_a_wrong": int(b_correct_a_wrong),
            "chi2_statistic": 0.0,
            "p_value": 1.0,
            "odds_ratio": 1.0,
            "alpha": alpha,
            "correction": correction,
            "significant": False,
            "interpretation": (
                "Models make IDENTICAL predictions on all samples. "
                "Cannot perform McNemar's test."
            )
        }

    if correction:
        chi2 = ((abs(a_correct_b_wrong - b_correct_a_wrong) - 1) ** 2) / (a_correct_b_wrong + b_correct_a_wrong)
    else:
        chi2 = (a_correct_b_wrong - b_correct_a_wrong) ** 2 / (a_correct_b_wrong + b_correct_a_wrong)

    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    odds_ratio = a_correct_b_wrong / b_correct_a_wrong if b_correct_a_wrong > 0 else float('inf')

    significant = p_value < alpha

    if significant:
        if accuracy_a > accuracy_b:
            interpretation = (
                f"SIGNIFICANT DIFFERENCE (p={p_value:.4f} < {alpha}). "
                f"Model A is significantly better. "
                f"Model A correct when B wrong: {a_correct_b_wrong}, "
                f"Model B correct when A wrong: {b_correct_a_wrong}. "
                f"Odds ratio: {odds_ratio:.2f}"
            )
        else:
            interpretation = (
                f"SIGNIFICANT DIFFERENCE (p={p_value:.4f} < {alpha}). "
                f"Model B is significantly better. "
                f"Model A correct when B wrong: {a_correct_b_wrong}, "
                f"Model B correct when A wrong: {b_correct_a_wrong}. "
                f"Odds ratio: {odds_ratio:.2f}"
            )
    else:
        interpretation = (
            f"NO SIGNIFICANT DIFFERENCE (p={p_value:.4f} >= {alpha}). "
            f"Models have similar performance. "
            f"Disagreements: A={a_correct_b_wrong}, B={b_correct_a_wrong}"
        )

    return {
        "test_name": "McNemar's test",
        "n_samples": n,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "contingency_table": contingency.tolist(),
        "both_correct": int(both_correct),
        "both_wrong": int(both_wrong),
        "a_correct_b_wrong": int(a_correct_b_wrong),
        "b_correct_a_wrong": int(b_correct_a_wrong),
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "odds_ratio": float(odds_ratio),
        "alpha": alpha,
        "correction": correction,
        "significant": significant,
        "interpretation": interpretation,
        "assumptions": [
            "Paired observations (same test samples)",
            "Large enough sample size (n > 25)",
            "Sufficient disagreements (n_discordant > 25 for chi-square approximation)"
        ]
    }


def bootstrap_significance_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_proba_a: Optional[np.ndarray] = None,
    y_proba_b: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict:
    """
    Perform bootstrap hypothesis testing to compare two models.

    Args:
        y_true: True labels
        y_pred_a: Predictions from model A
        y_pred_b: Predictions from model B
        y_proba_a: Probabilities from model A (optional)
        y_proba_b: Probabilities from model B (optional)
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level for intervals
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing bootstrap test results
    """
    np.random.seed(random_state)
    n_samples = len(y_true)

    metrics_to_test = ['accuracy', 'precision', 'recall', 'f1_score']

    results = {
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
        'n_samples': n_samples,
        'metrics': {}
    }

    for metric in tqdm(metrics_to_test, desc="Bootstrap significance test"):
        diff_distribution = []

        for _ in range(n_bootstrap):
            indices = resample(range(n_samples), n_samples=n_samples, replace=True)

            y_true_boot = y_true[indices]
            y_pred_a_boot = y_pred_a[indices]
            y_pred_b_boot = y_pred_b[indices]

            if metric == 'accuracy':
                score_a = accuracy_score(y_true_boot, y_pred_a_boot)
                score_b = accuracy_score(y_true_boot, y_pred_b_boot)
            elif metric == 'precision':
                score_a = precision_score(y_true_boot, y_pred_a_boot, average='weighted', zero_division=0)
                score_b = precision_score(y_true_boot, y_pred_b_boot, average='weighted', zero_division=0)
            elif metric == 'recall':
                score_a = recall_score(y_true_boot, y_pred_a_boot, average='weighted', zero_division=0)
                score_b = recall_score(y_true_boot, y_pred_b_boot, average='weighted', zero_division=0)
            elif metric == 'f1_score':
                score_a = f1_score(y_true_boot, y_pred_a_boot, average='weighted', zero_division=0)
                score_b = f1_score(y_true_boot, y_pred_b_boot, average='weighted', zero_division=0)

            diff_distribution.append(score_a - score_b)

        diff_distribution = np.array(diff_distribution)

        alpha = 1 - confidence_level
        ci_lower = np.percentile(diff_distribution, alpha / 2 * 100)
        ci_upper = np.percentile(diff_distribution, (1 - alpha / 2) * 100)
        mean_diff = np.mean(diff_distribution)
        std_diff = np.std(diff_distribution)

        p_value = 2 * min(
            np.mean(diff_distribution <= 0),
            np.mean(diff_distribution >= 0)
        )

        significant = 0 not in (ci_lower, ci_upper) and (ci_lower > 0 or ci_upper < 0)

        results['metrics'][metric] = {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'p_value': float(p_value),
            'significant': significant,
            'confidence_level': confidence_level
        }

    return results


def calculate_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict:
    """
    Calculate bootstrap confidence intervals for various metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level for intervals
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing bootstrap confidence intervals for all metrics
    """
    np.random.seed(random_state)
    n_samples = len(y_true)

    metrics_to_compute = {
        'accuracy': lambda yt, yp, ypr: accuracy_score(yt, yp),
        'precision': lambda yt, yp, ypr: precision_score(yt, yp, average='weighted', zero_division=0),
        'recall': lambda yt, yp, ypr: recall_score(yt, yp, average='weighted', zero_division=0),
        'f1_score': lambda yt, yp, ypr: f1_score(yt, yp, average='weighted', zero_division=0),
    }

    if y_proba.shape[1] == 2:
        metrics_to_compute['roc_auc'] = lambda yt, yp, ypr: roc_auc_score(yt, ypr[:, 1])

    bootstrap_distributions = {metric: [] for metric in metrics_to_compute}

    for _ in tqdm(range(n_bootstrap), desc="Computing bootstrap CIs"):
        indices = resample(range(n_samples), n_samples=n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_proba_boot = y_proba[indices]

        for metric, metric_func in metrics_to_compute.items():
            try:
                value = metric_func(y_true_boot, y_pred_boot, y_proba_boot)
                bootstrap_distributions[metric].append(value)
            except Exception as e:
                logging.debug(f"Bootstrap iteration failed for {metric}: {e}")
                continue

    results = {
        'bootstrap_n_resamples': n_bootstrap,
        'bootstrap_confidence_level': confidence_level,
        'bootstrap_random_state': random_state,
        'n_samples': n_samples
    }

    alpha = 1 - confidence_level
    for metric, distribution in bootstrap_distributions.items():
        if len(distribution) > 0:
            distribution_array = np.array(distribution)
            results[f'{metric}_mean'] = float(np.mean(distribution_array))
            results[f'{metric}_std'] = float(np.std(distribution_array))
            results[f'{metric}_ci_lower'] = float(np.percentile(distribution_array, alpha / 2 * 100))
            results[f'{metric}_ci_upper'] = float(np.percentile(distribution_array, (1 - alpha / 2) * 100))
            results[f'{metric}_bootstrap_values'] = distribution_array.tolist()

    return results


def compare_to_random_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Compare model performance to a random baseline (50% accuracy).

    Args:
        y_true: True labels
        y_pred: Model predictions
        alpha: Significance level

    Returns:
        Dictionary with comparison results
    """
    n = len(y_true)
    model_accuracy = accuracy_score(y_true, y_pred)
    baseline_accuracy = 0.5

    p0 = baseline_accuracy
    p_hat = model_accuracy

    z_score = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    p_value = 1 - stats.norm.cdf(z_score)

    significant = p_value < alpha

    if significant:
        conclusion = (
            f"Model SIGNIFICANTLY OUTPERFORMS random baseline (p={p_value:.4e} < {alpha}). "
            f"Model accuracy: {model_accuracy:.4f} vs baseline: {baseline_accuracy:.4f}."
        )
    else:
        conclusion = (
            f"Model does NOT significantly outperform random baseline (p={p_value:.4e} >= {alpha}). "
            f"Model accuracy: {model_accuracy:.4f} vs baseline: {baseline_accuracy:.4f}."
        )

    return {
        "test_name": "One-sample proportion test (vs random)",
        "model_accuracy": float(model_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
        "n_samples": n,
        "z_score": float(z_score),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": significant,
        "conclusion": conclusion
    }


def compare_to_baseline_benchmark(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_accuracy: float = 0.95,
    alpha: float = 0.05
) -> Dict:
    """
    Compare model performance to a baseline benchmark accuracy.

    Args:
        y_true: True labels
        y_pred: Model predictions
        baseline_accuracy: Baseline accuracy to compare against
        alpha: Significance level

    Returns:
        Dictionary with comparison results
    """
    n = len(y_true)
    model_accuracy = accuracy_score(y_true, y_pred)

    p0 = baseline_accuracy
    p_hat = model_accuracy

    z_score = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    p_value_greater = 1 - stats.norm.cdf(z_score)
    p_value_less = stats.norm.cdf(z_score)

    difference = model_accuracy - baseline_accuracy
    percentage_diff = (difference / baseline_accuracy) * 100

    if model_accuracy >= baseline_accuracy:
        meets_or_exceeds = True
        comparison_summary = (
            f"Model MEETS/EXCEEDS baseline ({model_accuracy:.4f} >= {baseline_accuracy:.4f}). "
            f"Difference: {difference:+.4f} ({percentage_diff:+.2f}%)."
        )
    else:
        meets_or_exceeds = False
        comparison_summary = (
            f"Model BELOW baseline ({model_accuracy:.4f} < {baseline_accuracy:.4f}). "
            f"Difference: {difference:+.4f} ({percentage_diff:+.2f}%)."
        )

    return {
        "test_name": "One-sample proportion test (vs baseline)",
        "model_accuracy": float(model_accuracy),
        "baseline_accuracy": float(baseline_accuracy),
        "accuracy_difference": float(difference),
        "percentage_difference": float(percentage_diff),
        "n_samples": n,
        "z_score": float(z_score),
        "p_value_greater": float(p_value_greater),
        "p_value_less": float(p_value_less),
        "alpha": alpha,
        "meets_or_exceeds_baseline": meets_or_exceeds,
        "comparison_summary": comparison_summary
    }
