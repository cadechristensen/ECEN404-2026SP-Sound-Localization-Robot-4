"""
Training module for baby cry detection model.
Implements training loop with validation, early stopping, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    # Try new API first (PyTorch 2.0+)
    from torch.amp import autocast, GradScaler
except ImportError:
    # Fall back to old API
    from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

try:
    from .config import Config
    from .model import create_model, count_parameters
    from .dataset import DatasetManager
    from .preprocessed_dataset import PreprocessedDatasetManager
except ImportError:
    import sys
    from pathlib import Path as PathLib
    src_dir = PathLib(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from config import Config  # type: ignore
    from model import create_model, count_parameters  # type: ignore
    from dataset import DatasetManager  # type: ignore
    from preprocessed_dataset import PreprocessedDatasetManager  # type: ignore


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    Focuses learning on hard-to-classify examples (like weak cries).

    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)

    Args:
        alpha: Weighting factor for class balance (0-1)
        gamma: Focusing parameter for hard examples (typically 2.0)
        weight: Class weights tensor
        label_smoothing: QUICK WIN 4 - Label smoothing epsilon (default 0.0)
    """

    def __init__(self, alpha=0.25, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class labels
        """
        # QUICK WIN 4: Apply label smoothing if enabled
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait after last time validation loss improved
            min_delta: Minimum change in validation loss to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False

    def save_checkpoint(self, model: nn.Module):
        """Save model weights."""
        self.best_weights = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    def restore_checkpoint(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict({key: value.to(next(model.parameters()).device)
                                 for key, value in self.best_weights.items()})


class BabyCryTrainer:
    """
    Trainer class for baby cry detection model.
    Handles training, validation, and model checkpointing.
    """

    def __init__(self, config: Config = Config()):
        """
        Initialize the trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

        # Initialize model
        self.model = create_model(config).to(self.device)
        logging.info(f"Model initialized with {count_parameters(self.model):,} trainable parameters")

        # Initialize dataset manager (preprocessed or regular)
        if config.USE_PREPROCESSED:
            logging.info(f"Using preprocessed data from: {config.PREPROCESSED_DIR}")
            # Find the actual preprocessed directory (with hash suffix)
            preprocessed_base = Path(config.PREPROCESSED_DIR)
            if preprocessed_base.exists():
                # Look for subdirectories with hash (config hash)
                hash_dirs = list(preprocessed_base.glob("*"))
                if hash_dirs:
                    actual_dir = hash_dirs[0]  # Use first hash directory
                    logging.info(f"Found preprocessed directory: {actual_dir}")
                    self.dataset_manager = PreprocessedDatasetManager(actual_dir, config)
                else:
                    raise FileNotFoundError(f"No preprocessed data found in {preprocessed_base}")
            else:
                raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_base}")
        else:
            logging.info("Using on-the-fly audio preprocessing")
            self.dataset_manager = DatasetManager(config)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

    def setup_training(self):
        """Set up training components."""
        # Prepare datasets (works for both regular and preprocessed)
        if self.config.USE_PREPROCESSED:
            # Preprocessed dataset manager has different API
            train_indices, val_indices, test_indices = self.dataset_manager.create_splits(
                train_ratio=self.config.TRAIN_RATIO,
                val_ratio=self.config.VAL_RATIO,
                test_ratio=self.config.TEST_RATIO,
                shuffle=True,
                random_seed=42
            )
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.create_datasets(
                train_indices, val_indices, test_indices, augment_train=True
            )

            # Create data loaders manually for preprocessed data
            from torch.utils.data import DataLoader
            try:
                from .dataset import collate_with_indices
            except ImportError:
                from dataset import collate_with_indices  # type: ignore

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True,
                collate_fn=collate_with_indices
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True,
                collate_fn=collate_with_indices
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True,
                collate_fn=collate_with_indices
            )
        else:
            # Regular dataset manager
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.prepare_datasets()
            self.train_loader, self.val_loader, self.test_loader = self.dataset_manager.create_data_loaders(
                self.train_dataset, self.val_dataset, self.test_dataset
            )

        # Set up loss function with Focal Loss (better for hard examples like weak cries)
        class_weights = self.dataset_manager.class_weights.to(self.device)
        focal_alpha = getattr(self.config, 'FOCAL_LOSS_ALPHA', 0.25)
        focal_gamma = getattr(self.config, 'FOCAL_LOSS_GAMMA', 2.0)
        # QUICK WIN 4: Label smoothing
        use_label_smoothing = getattr(self.config, 'USE_LABEL_SMOOTHING', False)
        label_smoothing_eps = getattr(self.config, 'LABEL_SMOOTHING_EPSILON', 0.0) if use_label_smoothing else 0.0
        self.criterion = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            weight=class_weights,
            label_smoothing=label_smoothing_eps
        )
        logging.info(
            f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma}, "
            f"label_smoothing={label_smoothing_eps}) with "
            f"cry_weight_multiplier={getattr(self.config, 'CRY_WEIGHT_MULTIPLIER', 2.0)} "
            f"for enhanced weak cry detection"
        )

        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # Set up learning rate scheduler (more patient, smaller reductions)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,  # Smaller reduction (was 0.5)
            patience=7,  # More patience (was 5)
            min_lr=1e-6,
            threshold=0.001  # Only reduce if improvement < 0.1%
        )

        # Set up early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.PATIENCE,
            min_delta=0.001
        )

        # Set up mixed precision training (automatic mixed precision)
        use_amp = getattr(self.config, 'USE_AMP', True) and torch.cuda.is_available()
        self.use_amp = use_amp
        self.amp_dtype = torch.float16  # Use float16 for AMP

        # Initialize GradScaler with proper device type
        if hasattr(GradScaler, '__init__'):
            try:
                # New API (PyTorch 2.0+)
                self.scaler = GradScaler('cuda', enabled=use_amp)
            except TypeError:
                # Old API fallback
                self.scaler = GradScaler(enabled=use_amp)
        else:
            self.scaler = GradScaler(enabled=use_amp)

        if use_amp:
            logging.info("Mixed precision training (AMP) enabled - expect 2× speedup")
        else:
            logging.info("Mixed precision training disabled (CPU mode or config disabled)")

        logging.info("Training setup completed")

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch_data in enumerate(pbar):
            # Handle both old format (specs, labels) and new format (specs, labels, indices)
            if len(batch_data) == 3:
                spectrograms, labels, _ = batch_data
            else:
                spectrograms, labels = batch_data

            spectrograms = spectrograms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with automatic mixed precision
            # Use device-specific autocast for compatibility
            with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping for stability (more conservative)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            # Update weights with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%',
                'LR': f'{current_lr:.6f}'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples

        return epoch_loss, epoch_acc

    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                # Handle both old format (specs, labels) and new format (specs, labels, indices)
                if len(batch_data) == 3:
                    spectrograms, labels, _ = batch_data
                else:
                    spectrograms, labels = batch_data

                spectrograms = spectrograms.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward pass with automatic mixed precision
                with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(spectrograms)
                    loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct_predictions / total_samples

        return epoch_loss, epoch_acc

    def train(self, results_dir: Path) -> Dict:
        """
        Main training loop.

        Args:
            results_dir: Directory to save results

        Returns:
            Training history dictionary
        """
        logging.info("Starting training...")
        start_time = time.time()

        best_val_loss = float('inf')
        best_model_path = results_dir / "model_best.pth"

        # Track best N checkpoints for ensembling
        last_n_checkpoints = []
        n_checkpoints_to_keep = getattr(self.config, 'ENSEMBLE_N_MODELS', 3)

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()

            # Training phase
            train_loss, train_acc = self.train_epoch()

            # Validation phase
            val_loss, val_acc = self.validate_epoch()

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config.__dict__
                }
                # Save scaler state if using AMP
                if self.use_amp:
                    checkpoint_dict['scaler_state_dict'] = self.scaler.state_dict()
                torch.save(checkpoint_dict, best_model_path)

            # Track top N checkpoints for ensembling throughout training
            # Always save checkpoint and track in list
            checkpoint_path = results_dir / f"model_epoch_{epoch+1}.pth"
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': self.config.__dict__
            }
            # Save scaler state if using AMP
            if self.use_amp:
                checkpoint_dict['scaler_state_dict'] = self.scaler.state_dict()
            torch.save(checkpoint_dict, checkpoint_path)

            last_n_checkpoints.append((checkpoint_path, val_loss))

            # Keep only best N checkpoints
            if len(last_n_checkpoints) > n_checkpoints_to_keep:
                # Sort by validation loss
                last_n_checkpoints.sort(key=lambda x: x[1])
                # Remove worst checkpoint from disk
                worst_checkpoint = last_n_checkpoints.pop()
                if worst_checkpoint[0].exists():
                    worst_checkpoint[0].unlink()
                    logging.info(f"Removed checkpoint: {worst_checkpoint[0].name}")

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Logging
            logging.info(
                f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_acc:.4f} "
                f"LR: {current_lr:.6f} "
                f"Time: {epoch_time:.2f}s"
            )

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                logging.info(f"Early stopping triggered after epoch {epoch+1}")
                break

        # Training completed
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f} seconds")

        # Save training history
        history_path = results_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # QUICK WIN 2: Save ensemble checkpoint metadata
        # List of best checkpoints for easy ensemble loading
        if last_n_checkpoints:
            ensemble_checkpoints = sorted(last_n_checkpoints, key=lambda x: x[1])[:n_checkpoints_to_keep]
            ensemble_metadata = {
                'checkpoint_paths': [str(path.relative_to(results_dir)) for path, _ in ensemble_checkpoints],
                'validation_losses': [float(loss) for _, loss in ensemble_checkpoints],
                'ensemble_size': len(ensemble_checkpoints)
            }
            ensemble_path = results_dir / "ensemble_checkpoints.json"
            with open(ensemble_path, 'w') as f:
                json.dump(ensemble_metadata, f, indent=2)
            logging.info(f"Saved ensemble metadata with {len(ensemble_checkpoints)} checkpoints:")
            for i, (path, loss) in enumerate(ensemble_checkpoints):
                logging.info(f"  {i+1}. {path.name} (val_loss: {loss:.6f})")
        else:
            logging.warning("No ensemble checkpoints saved - training may have stopped too early or an error occurred")

        return self.history

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load scaler state if available and using AMP
        if hasattr(self, 'scaler') and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
                    f"with validation loss {checkpoint['val_loss']:.4f}")

        return checkpoint

    def save_model_for_inference(self, results_dir: Path):
        """
        Save optimized model for inference.

        Args:
            results_dir: Directory to save the model
        """
        # Set model to evaluation mode
        self.model.eval()

        # Save inference model (optimized, no training artifacts)
        inference_path = results_dir / "model_inference.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'class_labels': self.config.CLASS_LABELS
        }, inference_path)

        logging.info(f"Inference model saved to {inference_path}")


def setup_logging(results_dir: Path):
    """
    Set up logging configuration.

    Args:
        results_dir: Directory to save log files
    """
    log_file = results_dir / "logs" / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function."""
    # Initialize configuration
    config = Config()

    # Create results directory
    results_dir = config.get_results_dir()
    setup_logging(results_dir)

    logging.info("Starting baby cry detection training")
    logging.info(f"Results will be saved to: {results_dir}")

    try:
        # Initialize trainer
        trainer = BabyCryTrainer(config)

        # Setup training
        trainer.setup_training()

        # Train model
        history = trainer.train(results_dir)

        # Save inference model
        trainer.save_model_for_inference(results_dir)

        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()