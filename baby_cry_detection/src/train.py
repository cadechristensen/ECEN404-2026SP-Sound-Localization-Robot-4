"""
Training module for baby cry detection model.
Implements training loop with validation, early stopping, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from config import Config  # type: ignore
    from model import create_model, count_parameters  # type: ignore
    from dataset import DatasetManager  # type: ignore
    from preprocessed_dataset import PreprocessedDatasetManager  # type: ignore


def _config_snapshot(config: Config) -> dict:
    """
    Serialize a Config object to a plain dict suitable for checkpoint storage.

    config.__dict__ only captures instance attributes; @property values (e.g.
    CLASS_LABELS, NUM_CLASSES) are silently omitted.  This helper adds them
    explicitly so that saved checkpoints are self-contained.
    """
    d = dict(vars(config))
    for attr in ('CLASS_LABELS', 'NUM_CLASSES'):
        try:
            d[attr] = getattr(config, attr)
        except Exception:
            pass
    return d


class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining.
    Focuses learning on hard-to-classify examples (like weak cries).

    Focal Loss = (1 - pt)^gamma * CE(pt)

    Class weighting is handled via the `weight` parameter passed to nll_loss,
    NOT via a separate alpha. This avoids the confusion of stacking two
    independent weighting mechanisms.

    Args:
        gamma: Focusing parameter for hard examples (typically 2.0)
        weight: Class weights tensor (from inverse-frequency * CRY_WEIGHT_MULTIPLIER)
        label_smoothing: Label smoothing epsilon (default 0.0)
    """

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - raw logits
            targets: (batch_size,) - class labels
        """
        # Compute log_probs once via log_softmax to avoid double softmax.
        # Previously: softmax for pt, then F.cross_entropy internally ran log_softmax again.
        log_probs = F.log_softmax(inputs, dim=1)

        # Derive pt without gradient (focal weight should not backprop through pt)
        with torch.no_grad():
            pt = log_probs.detach().exp().gather(1, targets.unsqueeze(1)).squeeze(1)

        # NLL loss with class weights and manual label smoothing
        nll = F.nll_loss(log_probs, targets, weight=self.weight, reduction='none')
        if self.label_smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=1)
            nll = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss

        focal_loss = (1 - pt) ** self.gamma * nll
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
        """Save model weights — only copies to CPU if restore_best_weights is True."""
        if self.restore_best_weights:
            self.best_weights = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    def restore_checkpoint(self, model: nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict({key: value.to(next(model.parameters()).device)
                                 for key, value in self.best_weights.items()})


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to a batch of spectrograms.

    Mixes pairs of training examples and their labels to create
    virtual training examples, improving generalization on ambiguous
    sounds like baby babbling vs crying.

    Args:
        x: Input spectrograms (batch_size, channels, freq, time)
        y: Labels (batch_size,)
        alpha: Beta distribution parameter (0.2 = mild mixing; 0 disables)

    Returns:
        mixed_x, y_a, y_b, lam, index (permutation indices for weight mixing)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


class BabyCryTrainer:
    """
    Trainer class for baby cry detection model.
    Handles training, validation, and model checkpointing.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the trainer.

        Args:
            config: Configuration object. Defaults to Config().
        """
        if config is None:
            config = Config()
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
                hash_dirs = sorted(preprocessed_base.glob("*"))
                if hash_dirs:
                    if len(hash_dirs) > 1:
                        logging.warning(
                            f"Multiple preprocessed directories found in {preprocessed_base}; "
                            f"using the last (most recent) entry: {hash_dirs[-1].name}"
                        )
                    actual_dir = hash_dirs[-1]  # Use last (most recent) sorted entry
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

            # pin_memory only benefits CUDA DMA transfers; on the Raspberry Pi (CPU-only)
            # it wastes locked memory without any throughput gain and risks OOM.
            _pin_memory = torch.cuda.is_available()

            _persistent = self.config.NUM_WORKERS > 0

            # Phase 3: WeightedRandomSampler for category-aware oversampling.
            # Oversample hard-negative categories (baby_noncry 4x, adult_scream 3x, etc.)
            # so the model sees them more often per epoch.
            sampling_weights = self.dataset_manager.compute_sample_weights(train_indices)
            has_oversampling = any(w > 1.0 for w in sampling_weights)
            if has_oversampling:
                train_sampler = WeightedRandomSampler(
                    weights=sampling_weights,
                    num_samples=len(sampling_weights),
                    replacement=True,
                )
                logging.info(f"WeightedRandomSampler enabled: {sum(1 for w in sampling_weights if w > 1.0)} "
                             f"samples with weight > 1.0 out of {len(sampling_weights)} total")
            else:
                train_sampler = None
                logging.info("No category oversampling needed — all weights are 1.0")

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=(train_sampler is None),  # shuffle and sampler are mutually exclusive
                sampler=train_sampler,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=_pin_memory,
                persistent_workers=_persistent,
                collate_fn=collate_with_indices
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=_pin_memory,
                persistent_workers=_persistent,
                collate_fn=collate_with_indices
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=_pin_memory,
                persistent_workers=_persistent,
                collate_fn=collate_with_indices
            )
        else:
            # Regular dataset manager
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.prepare_datasets()
            self.train_loader, self.val_loader, self.test_loader = self.dataset_manager.create_data_loaders(
                self.train_dataset, self.val_dataset, self.test_dataset
            )

        # Set up loss function
        class_weights = self.dataset_manager.class_weights.to(self.device)
        use_label_smoothing = getattr(self.config, 'USE_LABEL_SMOOTHING', False)
        label_smoothing_eps = getattr(self.config, 'LABEL_SMOOTHING_EPSILON', 0.0) if use_label_smoothing else 0.0
        use_focal = getattr(self.config, 'USE_FOCAL_LOSS', False)

        # Asymmetric label smoothing: different epsilon for cry vs non-cry
        self._use_asymmetric_smoothing = (
            use_label_smoothing
            and hasattr(self.config, 'LABEL_SMOOTHING_CRY')
            and hasattr(self.config, 'LABEL_SMOOTHING_NONCRY')
        )
        if self._use_asymmetric_smoothing:
            self._ls_cry = getattr(self.config, 'LABEL_SMOOTHING_CRY', 0.05)
            self._ls_noncry = getattr(self.config, 'LABEL_SMOOTHING_NONCRY', 0.20)
            # Use no label smoothing in CE — we apply it manually per-sample
            label_smoothing_eps = 0.0
            logging.info(f"Asymmetric label smoothing: cry={self._ls_cry}, non_cry={self._ls_noncry}")

        # Confidence penalty (entropy bonus)
        self._confidence_penalty_beta = getattr(self.config, 'CONFIDENCE_PENALTY_BETA', 0.0)
        if self._confidence_penalty_beta > 0:
            logging.info(f"Confidence penalty (entropy bonus) enabled: beta={self._confidence_penalty_beta}")

        # Phase 3: Check if per-sample loss weighting is active
        has_category_loss_weights = bool(getattr(self.config, 'CATEGORY_LOSS_WEIGHTS', {}))
        # Use reduction='none' when we need per-sample loss weighting or asymmetric smoothing
        loss_reduction = 'none' if (has_category_loss_weights or self._use_asymmetric_smoothing) else 'mean'
        self._use_per_sample_loss = has_category_loss_weights or self._use_asymmetric_smoothing

        if use_focal:
            focal_gamma = 2.0  # Standard FocalLoss gamma (only used if USE_FOCAL_LOSS=True)
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                weight=class_weights,
                label_smoothing=label_smoothing_eps
            )
            logging.info(
                f"Using Focal Loss (gamma={focal_gamma}, "
                f"label_smoothing={label_smoothing_eps}) with "
                f"cry_weight_multiplier={self.config.CRY_WEIGHT_MULTIPLIER}"
            )
            if has_category_loss_weights:
                logging.warning("Per-sample loss weighting not supported with FocalLoss — "
                                "FocalLoss already returns a scalar. Category loss weights will be ignored.")
                self._use_per_sample_loss = False
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing_eps,
                reduction=loss_reduction
            )
            logging.info(
                f"Using CrossEntropyLoss (label_smoothing={label_smoothing_eps}, "
                f"reduction='{loss_reduction}') with "
                f"cry_weight_multiplier={self.config.CRY_WEIGHT_MULTIPLIER}"
            )
            if has_category_loss_weights:
                logging.info(f"Per-sample loss weighting enabled: {self.config.CATEGORY_LOSS_WEIGHTS}")

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

        # Set up learning rate warmup
        warmup_epochs = getattr(self.config, 'WARMUP_EPOCHS', 5)
        self.warmup_epochs = warmup_epochs
        if warmup_epochs > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0 / max(warmup_epochs, 1),
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            logging.info(f"Linear LR warmup enabled for first {warmup_epochs} epochs")
        else:
            self.warmup_scheduler = None

        # Set up early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.PATIENCE,
            min_delta=0.001
        )

        # Set up mixed precision training (automatic mixed precision)
        use_amp = getattr(self.config, 'USE_AMP', True) and torch.cuda.is_available()
        self.use_amp = use_amp
        # float16 is only valid for CUDA AMP; CPU AMP requires bfloat16.
        self.amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        # autocast device_type must match the actual device (cannot use 'cuda' on CPU).
        self._autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize GradScaler — try new PyTorch 2.0+ API first, fall back to old.
        # (hasattr(GradScaler, '__init__') is always True; the try/except is sufficient.)
        try:
            self.scaler = GradScaler('cuda', enabled=use_amp)
        except TypeError:
            # Old API (PyTorch < 2.0) does not accept the device string positional arg
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
            # Handle 4-tuple (Phase 3), 3-tuple, and 2-tuple batch formats
            if len(batch_data) == 4:
                spectrograms, labels, _, sample_weights = batch_data
                sample_weights = sample_weights.to(self.device, non_blocking=True)
            elif len(batch_data) == 3:
                spectrograms, labels, _ = batch_data
                sample_weights = None
            else:
                spectrograms, labels = batch_data
                sample_weights = None

            spectrograms = spectrograms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Apply mixup augmentation with 50% probability (before autocast so the
            # mixing arithmetic runs in full float32 precision on both CPU and GPU).
            mixup_alpha = getattr(self.config, 'MIXUP_ALPHA', 0.2)
            use_mixup = (mixup_alpha > 0) and (np.random.random() < 0.5)
            if use_mixup:
                spectrograms, targets_a, targets_b, lam, mixup_index = mixup_data(
                    spectrograms, labels, alpha=mixup_alpha
                )

            # Forward pass with automatic mixed precision
            # Use device-specific autocast for compatibility
            with autocast(device_type=self._autocast_device, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(spectrograms)
                if use_mixup:
                    per_sample_loss_a = self.criterion(outputs, targets_a)
                    per_sample_loss_b = self.criterion(outputs, targets_b)
                    per_sample_loss = lam * per_sample_loss_a + (1 - lam) * per_sample_loss_b
                else:
                    per_sample_loss = self.criterion(outputs, labels)

                # Asymmetric label smoothing: apply per-sample smoothing correction
                if self._use_asymmetric_smoothing and per_sample_loss.dim() > 0:
                    log_probs = torch.log_softmax(outputs, dim=1)
                    smooth_loss = -log_probs.mean(dim=1)
                    # Per-sample epsilon based on true label (cry=1 gets light, non_cry=0 gets heavy)
                    if use_mixup:
                        # Interpolate epsilon to match the mixed target distribution
                        eps_a = torch.where(targets_a == 1, self._ls_cry, self._ls_noncry)
                        eps_b = torch.where(targets_b == 1, self._ls_cry, self._ls_noncry)
                        eps = lam * eps_a + (1 - lam) * eps_b
                    else:
                        eps = torch.where(labels == 1, self._ls_cry, self._ls_noncry)
                    per_sample_loss = (1 - eps) * per_sample_loss + eps * smooth_loss

                # Phase 3: Apply per-sample loss weighting from category weights
                if self._use_per_sample_loss and sample_weights is not None:
                    if use_mixup:
                        # Combine weights of both mixed samples using the same lambda
                        # weighting used for the inputs and labels
                        mixed_weights = lam * sample_weights + (1 - lam) * sample_weights[mixup_index]
                    else:
                        mixed_weights = sample_weights
                    loss = (per_sample_loss * mixed_weights).mean()
                else:
                    # If criterion already returns scalar (reduction='mean' or FocalLoss)
                    loss = per_sample_loss if per_sample_loss.dim() == 0 else per_sample_loss.mean()

                # Confidence penalty: entropy bonus to discourage extreme overconfidence
                if self._confidence_penalty_beta > 0:
                    probs = torch.softmax(outputs, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                    loss = loss - self._confidence_penalty_beta * entropy

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping for stability (more conservative)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            # Update weights with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics — for accuracy tracking use the original (unmixed) labels
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
        processed_batches = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                if batch_data is None:
                    continue  # All samples in this batch were corrupt
                # Handle 4-tuple, 3-tuple, and 2-tuple batch formats
                if len(batch_data) == 4:
                    spectrograms, labels, _, _ = batch_data
                elif len(batch_data) == 3:
                    spectrograms, labels, _ = batch_data
                else:
                    spectrograms, labels = batch_data

                spectrograms = spectrograms.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward pass with automatic mixed precision
                with autocast(device_type=self._autocast_device, enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(spectrograms)
                    val_loss = self.criterion(outputs, labels)
                    # Reduce if per-sample (validation doesn't use per-sample weighting)
                    if val_loss.dim() > 0:
                        val_loss = val_loss.mean()

                # Statistics
                running_loss += val_loss.item()
                processed_batches += 1
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / processed_batches if processed_batches > 0 else 0.0
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

            # Learning rate scheduling: warmup then ReduceLROnPlateau
            if epoch < self.warmup_epochs and self.warmup_scheduler is not None:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            # Build checkpoint_dict once per epoch; save conditionally to best/ensemble paths
            is_best = val_loss < best_val_loss
            _worst_tracked = max((v for _, v in last_n_checkpoints), default=float('inf'))
            _list_full = len(last_n_checkpoints) >= n_checkpoints_to_keep
            qualifies_for_ensemble = not _list_full or val_loss < _worst_tracked

            if is_best or qualifies_for_ensemble:
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': _config_snapshot(self.config)
                }
                if self.use_amp:
                    checkpoint_dict['scaler_state_dict'] = self.scaler.state_dict()

                if is_best:
                    best_val_loss = val_loss
                    torch.save(checkpoint_dict, best_model_path)

                if qualifies_for_ensemble:
                    # Enforce minimum epoch gap between ensemble members to ensure
                    # diversity.  Adjacent epochs share 99%+ weights, making the
                    # ensemble effectively fewer models.
                    MIN_EPOCH_GAP = 3
                    too_close_idx = None
                    for i, (ckpt_path, ckpt_loss) in enumerate(last_n_checkpoints):
                        # Extract epoch from checkpoint metadata (epoch is 0-indexed in dict)
                        ckpt_epoch = None
                        try:
                            ckpt_state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                            ckpt_epoch = ckpt_state.get('epoch', None)
                        except Exception:
                            pass
                        if ckpt_epoch is not None and abs(epoch - ckpt_epoch) < MIN_EPOCH_GAP:
                            too_close_idx = i
                            break

                    if too_close_idx is not None:
                        # Only keep this checkpoint if it's better than the neighbor
                        neighbor_path, neighbor_loss = last_n_checkpoints[too_close_idx]
                        if val_loss < neighbor_loss:
                            # Replace the neighbor
                            if neighbor_path.exists():
                                neighbor_path.unlink()
                                logging.info(
                                    f"Ensemble: replaced {neighbor_path.name} (epoch gap < {MIN_EPOCH_GAP})"
                                )
                            last_n_checkpoints.pop(too_close_idx)
                        else:
                            # Skip — neighbor is better and too close
                            logging.debug(
                                f"Ensemble: skipping epoch {epoch+1} (too close to existing checkpoint)"
                            )
                            qualifies_for_ensemble = False

                if qualifies_for_ensemble:
                    checkpoint_path = results_dir / f"model_epoch_{epoch+1}.pth"
                    torch.save(checkpoint_dict, checkpoint_path)

                    last_n_checkpoints.append((checkpoint_path, val_loss))

                    if len(last_n_checkpoints) > n_checkpoints_to_keep:
                        last_n_checkpoints.sort(key=lambda x: x[1])
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

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception:
            logging.warning(
                f"Could not load {checkpoint_path} with weights_only=True. "
                "Falling back to weights_only=False — ensure this checkpoint is from a trusted source."
            )
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
            'config': _config_snapshot(self.config),
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