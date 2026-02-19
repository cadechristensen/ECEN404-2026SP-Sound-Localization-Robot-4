"""
Confidence Calibration for Baby Cry Detection.

Implements temperature scaling and other calibration techniques to ensure
that model output probabilities accurately reflect true confidence.

Calibrated probabilities help distinguish high-confidence cries from
uncertain detections, reducing false positives.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from sklearn.metrics import log_loss
from scipy.optimize import minimize


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.

    Temperature scaling is a simple post-processing method that divides
    logits by a learned temperature parameter before softmax.

    Lower temperature (T < 1) makes predictions more confident.
    Higher temperature (T > 1) makes predictions less confident.
    """

    def __init__(self, num_classes: int = 2):
        """
        Initialize temperature scaling.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
        # Initialize temperature at 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Model output logits (before softmax)

        Returns:
            Temperature-scaled probabilities
        """
        scaled_logits = logits / self.temperature
        return torch.softmax(scaled_logits, dim=1)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 50, lr: float = 0.01) -> float:
        """
        Fit temperature parameter using validation set.

        Args:
            logits: Model output logits (N, num_classes)
            labels: True labels (N,)
            max_iter: Maximum optimization iterations
            lr: Learning rate for optimization

        Returns:
            Final negative log-likelihood loss
        """
        # Use NLL loss for optimization
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Return final loss
        with torch.no_grad():
            scaled_logits = logits / self.temperature
            final_loss = criterion(scaled_logits, labels).item()

        print(f"Temperature scaling fitted: T = {self.temperature.item():.4f}, Loss = {final_loss:.4f}")
        return final_loss


class ConfidenceCalibrator:
    """
    Calibrate model confidence scores for better reliability.

    Provides multiple calibration methods and confidence metrics.
    """

    def __init__(self, method: str = "temperature"):
        """
        Initialize confidence calibrator.

        Args:
            method: Calibration method ("temperature", "isotonic", "platt")
        """
        self.method = method
        self.calibrator = None

        if method == "temperature":
            self.calibrator = None  # Will be set during fit
        else:
            raise NotImplementedError(f"Calibration method '{method}' not implemented yet")

    def fit_temperature_scaling(self, logits: torch.Tensor, labels: torch.Tensor,
                               num_classes: int = 2) -> TemperatureScaling:
        """
        Fit temperature scaling calibrator.

        Args:
            logits: Model output logits (N, num_classes)
            labels: True labels (N,)
            num_classes: Number of classes

        Returns:
            Fitted TemperatureScaling module
        """
        temp_scaler = TemperatureScaling(num_classes=num_classes)
        temp_scaler.fit(logits, labels)
        self.calibrator = temp_scaler
        return temp_scaler

    def calibrate_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply calibration to model logits.

        Args:
            logits: Model output logits (N, num_classes)

        Returns:
            Calibrated probabilities (N, num_classes)
        """
        if self.calibrator is None:
            # No calibration, just apply softmax
            return torch.softmax(logits, dim=1)

        if self.method == "temperature":
            return self.calibrator(logits)
        else:
            raise NotImplementedError(f"Calibration method '{self.method}' not implemented")

    def save(self, path: str):
        """Save calibrator to file."""
        if self.calibrator is not None:
            torch.save({
                'method': self.method,
                'calibrator_state': self.calibrator.state_dict() if hasattr(self.calibrator, 'state_dict') else None
            }, path)
            print(f"Calibrator saved to {path}")

    def load(self, path: str, num_classes: int = 2):
        """Load calibrator from file."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.method = checkpoint['method']

        if self.method == "temperature":
            self.calibrator = TemperatureScaling(num_classes=num_classes)
            if checkpoint['calibrator_state'] is not None:
                self.calibrator.load_state_dict(checkpoint['calibrator_state'])
            print(f"Calibrator loaded from {path}")


def compute_expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                      num_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy.
    Lower ECE means better calibration.

    Args:
        probs: Predicted probabilities (N,) for binary or (N, C) for multi-class
        labels: True labels (N,)
        num_bins: Number of bins for calibration curve

    Returns:
        Expected Calibration Error (0-1, lower is better)
    """
    # For multi-class, use maximum probability
    if len(probs.shape) > 1:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = probs
        predictions = (probs > 0.5).astype(int)

    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(confidences)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum() / total_samples

        if prop_in_bin > 0:
            # Average confidence in bin
            avg_confidence = confidences[in_bin].mean()

            # Accuracy in bin
            accuracy = (predictions[in_bin] == labels[in_bin]).mean()

            # Add weighted difference to ECE
            ece += prop_in_bin * abs(avg_confidence - accuracy)

    return ece


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilities).

    Lower Brier score indicates better calibration.

    Args:
        probs: Predicted probabilities (N,) for binary or (N, C) for multi-class
        labels: True labels (N,)

    Returns:
        Brier score (0-1, lower is better)
    """
    if len(probs.shape) > 1:
        # Multi-class: convert labels to one-hot
        num_classes = probs.shape[1]
        labels_onehot = np.eye(num_classes)[labels]
        brier = np.mean(np.sum((probs - labels_onehot) ** 2, axis=1))
    else:
        # Binary: simple squared error
        brier = np.mean((probs - labels) ** 2)

    return brier


def threshold_optimization(probs: np.ndarray, labels: np.ndarray,
                          metric: str = "f1") -> Tuple[float, float]:
    """
    Find optimal probability threshold for classification.

    Args:
        probs: Predicted probabilities for positive class (N,)
        labels: True binary labels (N,)
        metric: Metric to optimize ("f1", "precision", "recall", "accuracy")

    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score

    if metric == "f1":
        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            score = f1_score(labels, preds)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    elif metric == "precision" or metric == "recall":
        precision, recall, thresholds = precision_recall_curve(labels, probs)

        if metric == "precision":
            # Find threshold with highest precision while maintaining reasonable recall
            min_recall = 0.5
            valid_idx = recall >= min_recall
            if valid_idx.any():
                best_idx = np.argmax(precision[valid_idx])
                return thresholds[best_idx], precision[valid_idx][best_idx]
            else:
                return 0.5, 0.0

        else:  # recall
            # Find threshold with highest recall while maintaining reasonable precision
            min_precision = 0.5
            valid_idx = precision >= min_precision
            if valid_idx.any():
                best_idx = np.argmax(recall[valid_idx])
                return thresholds[best_idx], recall[valid_idx][best_idx]
            else:
                return 0.5, 0.0

    elif metric == "accuracy":
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            score = accuracy_score(labels, preds)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    else:
        raise ValueError(f"Unknown metric: {metric}")


def analyze_confidence_distribution(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Analyze the distribution of confidence scores.

    Args:
        probs: Predicted probabilities (N,) for positive class
        labels: True binary labels (N,)

    Returns:
        Dictionary with confidence analysis metrics
    """
    # Separate probabilities for positive and negative samples
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    analysis = {
        'mean_confidence_positive': float(np.mean(pos_probs)) if len(pos_probs) > 0 else 0.0,
        'mean_confidence_negative': float(np.mean(neg_probs)) if len(neg_probs) > 0 else 0.0,
        'std_confidence_positive': float(np.std(pos_probs)) if len(pos_probs) > 0 else 0.0,
        'std_confidence_negative': float(np.std(neg_probs)) if len(neg_probs) > 0 else 0.0,
        'high_confidence_correct': float(np.mean(probs[labels == 1] > 0.8)) if len(pos_probs) > 0 else 0.0,
        'high_confidence_incorrect': float(np.mean(probs[labels == 0] > 0.8)) if len(neg_probs) > 0 else 0.0,
        'low_confidence_correct': float(np.mean(probs[labels == 0] < 0.2)) if len(neg_probs) > 0 else 0.0,
        'low_confidence_incorrect': float(np.mean(probs[labels == 1] < 0.2)) if len(pos_probs) > 0 else 0.0,
    }

    # Compute separation (how well separated are positive and negative confidences)
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        separation = abs(np.mean(pos_probs) - np.mean(neg_probs))
        analysis['confidence_separation'] = float(separation)
    else:
        analysis['confidence_separation'] = 0.0

    return analysis


# =============================================================================
# TEMPERATURE SCALED MODEL WRAPPER
# =============================================================================


class TemperatureScaledModel(nn.Module):
    """
    Wrapper that applies temperature scaling to any classification model.

    This class wraps an existing trained model and applies post-hoc temperature
    scaling to calibrate its probability outputs. Temperature scaling is a simple
    but effective calibration method that:

    1. Preserves the model's accuracy (argmax predictions unchanged)
    2. Only modifies the confidence of predictions
    3. Has a single learnable parameter (temperature T)

    The temperature parameter controls prediction confidence:
    - T > 1: Softens probabilities (reduces overconfidence)
    - T < 1: Sharpens probabilities (increases confidence)
    - T = 1: No change (original model output)

    For binary classification, the forward pass returns logits that can be
    passed through softmax (for class probabilities) or used with sigmoid
    on the positive class logit for probability of cry.

    Usage:
        # Load and wrap your trained model
        model = create_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create temperature-scaled wrapper
        calibrated_model = TemperatureScaledModel(model)

        # Calibrate on validation set
        calibration_result = calibrated_model.calibrate(val_loader, device)

        # Use for inference (returns calibrated logits)
        with torch.no_grad():
            logits = calibrated_model(spectrogram)
            probs = torch.softmax(logits, dim=1)
    """

    def __init__(self, model: nn.Module, initial_temperature: float = 1.0):
        """
        Initialize temperature-scaled model wrapper.

        Args:
            model: Trained PyTorch model that outputs logits
            initial_temperature: Initial temperature value (default: 1.0)
        """
        super().__init__()
        self.model = model
        # Temperature parameter - initialized to 1.0 (no scaling)
        # Using log scale for numerical stability during optimization
        self.temperature = nn.Parameter(
            torch.tensor([initial_temperature], dtype=torch.float32)
        )
        self._is_calibrated = False
        self._calibration_history: List[dict] = []

    @property
    def is_calibrated(self) -> bool:
        """Check if the model has been calibrated."""
        return self._is_calibrated

    def get_temperature(self) -> float:
        """Get the current temperature value."""
        return self.temperature.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temperature scaling applied to logits.

        Args:
            x: Input tensor (e.g., mel-spectrogram batch)

        Returns:
            Temperature-scaled logits (not probabilities)
        """
        # Get raw logits from the wrapped model
        logits = self.model(x)

        # Apply temperature scaling
        # Higher T -> softer probabilities, Lower T -> sharper probabilities
        scaled_logits = logits / self.temperature

        return scaled_logits

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get calibrated probability predictions.

        Args:
            x: Input tensor

        Returns:
            Calibrated probabilities (batch_size, num_classes)
        """
        scaled_logits = self.forward(x)
        return torch.softmax(scaled_logits, dim=1)

    def calibrate(
        self,
        val_loader: "torch.utils.data.DataLoader",
        device: torch.device,
        max_iter: int = 100,
        lr: float = 0.01,
        verbose: bool = True
    ) -> dict:
        """
        Calibrate temperature on validation set by minimizing NLL loss.

        This method uses LBFGS optimization to find the optimal temperature
        that minimizes the negative log-likelihood on the validation set.

        Args:
            val_loader: DataLoader for validation set
            device: Device to run calibration on
            max_iter: Maximum LBFGS iterations (default: 100)
            lr: Learning rate for LBFGS (default: 0.01)
            verbose: Print calibration progress (default: True)

        Returns:
            Dictionary containing calibration results:
                - optimal_temperature: Final temperature value
                - initial_nll: NLL before calibration
                - final_nll: NLL after calibration
                - nll_improvement: Relative improvement in NLL
                - initial_ece: ECE before calibration
                - final_ece: ECE after calibration
                - ece_improvement: Relative improvement in ECE
        """
        self.model.eval()

        # Collect all logits and labels from validation set
        all_logits = []
        all_labels = []

        if verbose:
            print("Collecting validation logits for temperature calibration...")

        with torch.no_grad():
            for batch_data in val_loader:
                # Handle different batch formats
                if len(batch_data) == 3:
                    spectrograms, labels, _ = batch_data
                else:
                    spectrograms, labels = batch_data

                spectrograms = spectrograms.to(device)
                # Get raw logits from the model (bypass temperature scaling)
                logits = self.model(spectrograms)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        # Concatenate all batches
        logits_tensor = torch.cat(all_logits, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0).long()

        # Move to device for optimization
        logits_tensor = logits_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        self.temperature = self.temperature.to(device)

        # Calculate initial metrics (before calibration, T=1.0)
        with torch.no_grad():
            initial_probs = torch.softmax(logits_tensor, dim=1).cpu().numpy()
            initial_nll = nn.CrossEntropyLoss()(logits_tensor, labels_tensor).item()

        labels_np = labels_tensor.cpu().numpy()
        initial_ece = compute_expected_calibration_error(initial_probs, labels_np)
        initial_mce = self._compute_mce(initial_probs, labels_np)

        if verbose:
            print(f"Initial metrics (T=1.0):")
            print(f"  NLL: {initial_nll:.4f}")
            print(f"  ECE: {initial_ece:.4f} ({initial_ece * 100:.2f}%)")
            print(f"  MCE: {initial_mce:.4f} ({initial_mce * 100:.2f}%)")

        # Reset temperature to 1.0 before optimization
        self.temperature.data.fill_(1.0)

        # Optimize temperature using LBFGS
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn='strong_wolfe'
        )

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_tensor / self.temperature
            loss = nn.CrossEntropyLoss()(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        if verbose:
            print("\nOptimizing temperature...")

        optimizer.step(closure)

        # Ensure temperature is positive (numerical stability)
        with torch.no_grad():
            self.temperature.data = torch.clamp(self.temperature.data, min=0.01)

        # Calculate final metrics (after calibration)
        with torch.no_grad():
            scaled_logits = logits_tensor / self.temperature
            final_nll = nn.CrossEntropyLoss()(scaled_logits, labels_tensor).item()
            final_probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()

        final_ece = compute_expected_calibration_error(final_probs, labels_np)
        final_mce = self._compute_mce(final_probs, labels_np)

        # Calculate improvements
        nll_improvement = (initial_nll - final_nll) / initial_nll * 100
        ece_improvement = (initial_ece - final_ece) / initial_ece * 100 if initial_ece > 0 else 0
        mce_improvement = (initial_mce - final_mce) / initial_mce * 100 if initial_mce > 0 else 0

        result = {
            'optimal_temperature': self.temperature.item(),
            'initial_nll': initial_nll,
            'final_nll': final_nll,
            'nll_improvement_percent': nll_improvement,
            'initial_ece': initial_ece,
            'final_ece': final_ece,
            'ece_improvement_percent': ece_improvement,
            'initial_mce': initial_mce,
            'final_mce': final_mce,
            'mce_improvement_percent': mce_improvement,
        }

        self._is_calibrated = True
        self._calibration_history.append(result)

        if verbose:
            print(f"\nCalibration complete!")
            print(f"  Optimal temperature: {result['optimal_temperature']:.4f}")
            print(f"  NLL: {initial_nll:.4f} -> {final_nll:.4f} ({nll_improvement:+.2f}%)")
            print(f"  ECE: {initial_ece:.4f} -> {final_ece:.4f} ({ece_improvement:+.2f}%)")
            print(f"  MCE: {initial_mce:.4f} -> {final_mce:.4f} ({mce_improvement:+.2f}%)")

        return result

    def _compute_mce(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error (MCE)."""
        if len(probs.shape) > 1:
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
        else:
            confidences = probs
            predictions = (probs > 0.5).astype(int)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            if bin_upper == 1.0:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(predictions[in_bin] == labels[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_error = np.abs(bin_accuracy - bin_confidence)
                mce = max(mce, bin_error)

        return mce

    def save_calibrated_model(self, save_path: str) -> None:
        """
        Save the calibrated model (including temperature parameter).

        Args:
            save_path: Path to save the model
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'temperature': self.temperature.item(),
            'is_calibrated': self._is_calibrated,
            'calibration_history': self._calibration_history,
        }
        torch.save(save_dict, save_path)
        print(f"Calibrated model saved to {save_path}")

    @classmethod
    def load_calibrated_model(
        cls,
        load_path: str,
        model: nn.Module,
        device: torch.device
    ) -> "TemperatureScaledModel":
        """
        Load a calibrated model from checkpoint.

        Args:
            load_path: Path to the saved calibrated model
            model: Uninitialized model instance (same architecture)
            device: Device to load the model on

        Returns:
            TemperatureScaledModel with loaded weights and temperature
        """
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # Create wrapper and set temperature
        wrapper = cls(model, initial_temperature=checkpoint['temperature'])
        wrapper._is_calibrated = checkpoint.get('is_calibrated', True)
        wrapper._calibration_history = checkpoint.get('calibration_history', [])
        wrapper.to(device)

        print(f"Loaded calibrated model with T={checkpoint['temperature']:.4f}")
        return wrapper

    def eval(self) -> "TemperatureScaledModel":
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "TemperatureScaledModel":
        """Set model training mode (temperature is always in eval mode)."""
        self.model.train(mode)
        return self


def calibrate_model_temperature(
    model: nn.Module,
    val_loader: "torch.utils.data.DataLoader",
    device: torch.device,
    save_path: Optional[str] = None,
    max_iter: int = 100,
    verbose: bool = True
) -> Tuple[TemperatureScaledModel, dict]:
    """
    Convenience function to calibrate a trained model using temperature scaling.

    This function:
    1. Wraps the model in a TemperatureScaledModel
    2. Calibrates the temperature on the validation set
    3. Optionally saves the calibrated model

    Args:
        model: Trained PyTorch model
        val_loader: DataLoader for validation set
        device: Device to run calibration on
        save_path: Optional path to save calibrated model
        max_iter: Maximum LBFGS iterations
        verbose: Print calibration progress

    Returns:
        Tuple of (calibrated_model, calibration_results)

    Example:
        >>> model = create_model(config)
        >>> model.load_state_dict(checkpoint['model_state_dict'])
        >>> calibrated_model, results = calibrate_model_temperature(
        ...     model, val_loader, device,
        ...     save_path="models/calibrated_model.pth"
        ... )
        >>> print(f"ECE improved from {results['initial_ece']:.2%} to {results['final_ece']:.2%}")
    """
    # Ensure model is in eval mode
    model.eval()

    # Create temperature-scaled wrapper
    calibrated_model = TemperatureScaledModel(model)
    calibrated_model.to(device)

    # Calibrate
    calibration_results = calibrated_model.calibrate(
        val_loader, device,
        max_iter=max_iter,
        verbose=verbose
    )

    # Save if path provided
    if save_path is not None:
        calibrated_model.save_calibrated_model(save_path)

    return calibrated_model, calibration_results
