"""
Train the DOAnet CRNN model on hardware-collected sound localization data.

Uses filtered cry-region audio at 48kHz from the BCD pipeline,
with direction and distance labels from metadata.csv.

Usage:
    python train_sl.py --task-id 1 --epochs 300
    python train_sl.py --task-id 1 --epochs 300 --resume models/best_model.h5
"""

import os
import sys
import argparse
import csv
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

import doanet_model
import doanet_parameters
import cls_feature_class

logger = logging.getLogger(__name__)


class SLDataset(Dataset):
    """Dataset for sound localization training from metadata.csv."""

    def __init__(self, data_dir, metadata_csv, params, feat_extractor,
                 scaler=None, augment=False):
        """
        Args:
            data_dir: Directory containing WAV files.
            metadata_csv: Path to metadata.csv with labels.
            params: DOAnet parameters dict.
            feat_extractor: FeatureClass instance for feature extraction.
            scaler: Optional sklearn StandardScaler for normalization.
            augment: Whether to apply data augmentation.
        """
        self.data_dir = data_dir
        self.params = params
        self.feat_extractor = feat_extractor
        self.scaler = scaler
        self.augment = augment
        self.channel_angles = params.get('channel_angles',
                                          {0: 135.0, 1: 315.0, 2: 45.0, 3: 225.0})

        # Load metadata and filter to valid entries
        self.samples = []
        with open(metadata_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only use cry-detected samples with direction labels
                if row.get('is_cry', '').strip().lower() != 'true':
                    continue
                if not row.get('direction_deg', '').strip():
                    continue

                filepath = os.path.join(data_dir, row['filename'])
                if not os.path.exists(filepath):
                    continue

                try:
                    direction = float(row['direction_deg'])
                    distance = float(row.get('distance_ft', '0') or '0')
                except (ValueError, TypeError):
                    continue

                self.samples.append({
                    'filepath': filepath,
                    'direction_deg': direction,
                    'distance_ft': distance,
                    'label': row.get('label', ''),
                })

        logger.info(f"Loaded {len(self.samples)} samples from {metadata_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract features using cls_feature_class
        features = self.feat_extractor.extract_features_for_file(sample['filepath'])

        # Apply normalization
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Pad/truncate to feature_sequence_length
        feat_seq_len = self.params['feature_sequence_length']
        if features.shape[0] < feat_seq_len:
            pad_len = feat_seq_len - features.shape[0]
            features = np.pad(features, ((0, pad_len), (0, 0)),
                              'constant', constant_values=1e-6)
        else:
            features = features[:feat_seq_len]

        # Reshape to (nb_channels, feat_seq_len, nb_mel_bins)
        nb_ch = 10  # 4*64/64 mel + 6*64/64 GCC = 10 channels of 64 bins
        features = features.reshape(feat_seq_len, nb_ch, self.params['nb_mel_bins'])
        features = np.transpose(features, (1, 0, 2))  # (nb_ch, T, mel)

        # Create DOA target: (sin, cos) of direction + distance
        direction_rad = np.radians(sample['direction_deg'])
        target_x = np.cos(direction_rad)
        target_y = np.sin(direction_rad)
        target_dist = sample['distance_ft']

        # Channel rotation augmentation is DISABLED because it only
        # permutes mel channels but not GCC-PHAT channels. The GCC pairs
        # encode inter-mic time delays that must be remapped when channels
        # rotate, otherwise the model receives contradictory features.
        # TODO: implement correct GCC pair permutation + sign flipping,
        #       then re-enable this augmentation.
        # if self.augment and np.random.rand() < 0.5:
        #     features, target_x, target_y = self._rotate_channels(
        #         features, target_x, target_y
        #     )

        # Normalize distance to [0, 1] for tanh output compatibility.
        # DOAnet applies tanh to all outputs, clamping to [-1, 1].
        # sin/cos are naturally in [-1, 1], but distance must be scaled.
        MAX_DISTANCE_FT = 15.0  # Max expected distance in feet
        target_dist_norm = min(target_dist / MAX_DISTANCE_FT, 1.0)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        target_tensor = torch.tensor(
            [target_x, target_y, target_dist_norm], dtype=torch.float32
        )

        return features_tensor, target_tensor

    def _rotate_channels(self, features, target_x, target_y):
        """Rotate channels by 90/180/270 degrees using mic geometry symmetry."""
        rotation = np.random.choice([90, 180, 270])
        rot_rad = np.radians(rotation)

        # Rotate the target direction
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
        new_x = target_x * cos_r - target_y * sin_r
        new_y = target_x * sin_r + target_y * cos_r

        # Rotate mel channels (first 4 of the 10 channels)
        # 90-deg rotation order for channels at 45/135/225/315:
        # Rotate source by +90: mic at phi gets what mic at (phi-90) had
        rotation_maps = {
            90: [2, 0, 3, 1],   # ch0(135)<-ch2(45), ch1(315)<-ch0(135), ...
            180: [3, 2, 1, 0],
            270: [1, 3, 0, 2],
        }
        ch_map = rotation_maps[rotation]

        # Rearrange mel channels (indices 0-3)
        mel_rotated = features[ch_map]

        # GCC-PHAT channels (indices 4-9) represent pairs —
        # rotation changes which pair maps to which, but the GCC values
        # themselves are symmetric. For simplicity, keep GCC as-is.
        # The model will learn to handle rotated mel + original GCC.
        gcc = features[4:]

        features = np.concatenate([mel_rotated, gcc], axis=0)
        return features, new_x, new_y


def compute_scaler(dataset_dir, metadata_csv, params, feat_extractor):
    """Compute mean/std normalization from all training features."""
    from sklearn.preprocessing import StandardScaler

    all_features = []
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(dataset_dir, row['filename'])
            if not os.path.exists(filepath):
                continue
            try:
                feats = feat_extractor.extract_features_for_file(filepath)
                all_features.append(feats)
            except Exception as e:
                logger.warning(f"Skipping {filepath}: {e}")

    if not all_features:
        raise RuntimeError("No features extracted — check data directory")

    all_features = np.concatenate(all_features, axis=0)
    scaler = StandardScaler().fit(all_features)
    logger.info(f"Scaler computed from {all_features.shape[0]} frames")
    return scaler


def train(args):
    """Main training loop."""
    params = doanet_parameters.get_params(args.task_id)
    params.update({
        'nb_cnn2d_filt': 128, 'rnn_size': 256,
        'self_attn': True, 'unique_classes': 1,  # Single source (baby)
    })

    data_dir = args.data_dir
    metadata_csv = os.path.join(data_dir, 'metadata.csv')

    if not os.path.exists(metadata_csv):
        print(f"ERROR: {metadata_csv} not found. Use record.py to collect data first.")
        sys.exit(1)

    # Feature extractor
    feat_extractor = cls_feature_class.FeatureClass(params)

    # Compute or load normalization scaler
    scaler_path = os.path.join(args.model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path) and not args.recompute_scaler:
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    else:
        logger.info("Computing feature normalization scaler...")
        scaler = compute_scaler(data_dir, metadata_csv, params, feat_extractor)
        os.makedirs(args.model_dir, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

    # Datasets (separate train/val to avoid augmenting validation data)
    train_dataset = SLDataset(
        data_dir=data_dir,
        metadata_csv=metadata_csv,
        params=params,
        feat_extractor=feat_extractor,
        scaler=scaler,
        augment=True,
    )

    if len(train_dataset) == 0:
        print("ERROR: No valid training samples found in metadata.csv")
        sys.exit(1)

    val_dataset = SLDataset(
        data_dir=data_dir,
        metadata_csv=metadata_csv,
        params=params,
        feat_extractor=feat_extractor,
        scaler=scaler,
        augment=False,
    )

    # Train/val split (80/20)
    n_val = max(1, len(train_dataset) // 5)
    n_train = len(train_dataset) - n_val
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(train_dataset)), [n_train, n_val]
    )
    train_set = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_set = torch.utils.data.Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(train_set, batch_size=params['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'],
                            shuffle=False, num_workers=0)

    # Model
    nb_ch = 10
    data_in = (params['batch_size'], nb_ch,
               params['feature_sequence_length'], params['nb_mel_bins'])
    # Output: 3 values per source (x, y, confidence) * unique_classes
    data_out = [params['batch_size'], params['label_sequence_length'],
                params['unique_classes'] * 3]

    model = doanet_model.CRNN(data_in, data_out, params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss: MSE for direction (x, y) + distance
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'],
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        logger.info(f"Resumed from {args.resume}")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in range(params['nb_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward — model outputs DOA predictions per frame
            # We use the mean prediction across frames as the final output
            doa_out, activity_out = model(features)

            # Average DOA across time frames for single-source prediction
            # doa_out shape: (batch, label_seq_len, unique_classes * 3)
            doa_mean = doa_out.mean(dim=1)  # (batch, 3)

            # Target: (x, y, distance)
            loss = criterion(doa_mean, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_angle_errors = []
        val_dist_errors = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                doa_out, activity_out = model(features)
                doa_mean = doa_out.mean(dim=1)

                loss = criterion(doa_mean, targets)
                val_loss += loss.item()

                # Compute angle error
                pred_x = doa_mean[:, 0].cpu().numpy()
                pred_y = doa_mean[:, 1].cpu().numpy()
                true_x = targets[:, 0].cpu().numpy()
                true_y = targets[:, 1].cpu().numpy()

                pred_deg = np.degrees(np.arctan2(pred_y, pred_x)) % 360
                true_deg = np.degrees(np.arctan2(true_y, true_x)) % 360

                angle_diff = np.abs(pred_deg - true_deg)
                angle_diff = np.minimum(angle_diff, 360 - angle_diff)
                val_angle_errors.extend(angle_diff.tolist())

                # Distance error
                pred_dist = doa_mean[:, 2].cpu().numpy()
                true_dist = targets[:, 2].cpu().numpy()
                val_dist_errors.extend(np.abs(pred_dist - true_dist).tolist())

        val_loss /= max(len(val_loader), 1)
        mean_angle_err = np.mean(val_angle_errors) if val_angle_errors else 0
        mean_dist_err = np.mean(val_dist_errors) if val_dist_errors else 0

        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch+1}/{params['nb_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Angle Err: {mean_angle_err:.1f} deg | Dist Err: {mean_dist_err:.2f} ft"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(args.model_dir, 'best_sl_model.h5')
            torch.save(model.state_dict(), best_path)
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 50:
                logger.info("Early stopping triggered")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(args.model_dir, 'best_sl_model.h5')}")


def main():
    parser = argparse.ArgumentParser(
        description='Train DOAnet CRNN on hardware sound localization data'
    )
    parser.add_argument('--task-id', default='1', help='Parameter set (default: 1)')
    parser.add_argument('--data-dir', default='sl_training_data/',
                        help='Training data directory with metadata.csv')
    parser.add_argument('--model-dir', default='models/',
                        help='Output directory for model checkpoints')
    parser.add_argument('--resume', default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--recompute-scaler', action='store_true',
                        help='Force recomputation of normalization scaler')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    train(args)


if __name__ == '__main__':
    main()
