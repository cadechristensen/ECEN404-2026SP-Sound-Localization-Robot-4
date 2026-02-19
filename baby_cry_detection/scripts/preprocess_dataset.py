"""
Offline preprocessing pipeline for baby cry detection.
Pre-computes spectrograms and saves them to disk for fast training.

Usage:
    python scripts/preprocess_dataset.py --output data/processed/v1 --config base
    python scripts/preprocess_dataset.py --output data/processed/v1 --dry-run  # Preview only
"""

import sys
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from tqdm import tqdm
import logging

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from config import Config
from data_preprocessing import AudioPreprocessor, collect_audio_files


def compute_config_hash(config: Config) -> str:
    """
    Compute deterministic hash of preprocessing configuration.
    This ensures cache invalidation when preprocessing parameters change.

    Args:
        config: Configuration object

    Returns:
        16-character hex hash string
    """
    # Extract only preprocessing-relevant parameters
    config_dict = {
        'SAMPLE_RATE': config.SAMPLE_RATE,
        'DURATION': config.DURATION,
        'N_MELS': config.N_MELS,
        'N_FFT': config.N_FFT,
        'HOP_LENGTH': config.HOP_LENGTH,
        'F_MIN': config.F_MIN,
        'F_MAX': config.F_MAX,
        # Note: augmentation parameters NOT included (applied on-the-fly)
    }

    # Create deterministic JSON string
    config_str = json.dumps(config_dict, sort_keys=True)

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:16]


def preprocess_dataset(
    config: Config,
    output_dir: Path,
    dry_run: bool = False,
    overwrite: bool = False
) -> dict:
    """
    Pre-compute spectrograms for all audio files in the dataset.

    Args:
        config: Configuration object
        output_dir: Output directory for preprocessed data
        dry_run: If True, only preview what would be processed
        overwrite: If True, overwrite existing preprocessed files

    Returns:
        Dictionary with preprocessing statistics
    """
    # Initialize preprocessor (no augmentation for preprocessing)
    preprocessor = AudioPreprocessor(config, use_advanced_filtering=False)

    # Collect all audio files
    logging.info("Collecting audio files...")
    audio_files = collect_audio_files(config.DATA_DIR, config.SUPPORTED_FORMATS, multi_class=config.MULTI_CLASS_MODE)

    if not audio_files:
        raise ValueError(f"No audio files found in {config.DATA_DIR}")

    logging.info(f"Found {len(audio_files)} audio files")

    # Compute config hash for versioning
    config_hash = compute_config_hash(config)
    logging.info(f"Preprocessing config hash: {config_hash}")

    # Create output directory structure
    output_dir = output_dir / config_hash
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save preprocessing config for validation
        config_path = output_dir / "config.json"
        config_dict = {
            'SAMPLE_RATE': config.SAMPLE_RATE,
            'DURATION': config.DURATION,
            'N_MELS': config.N_MELS,
            'N_FFT': config.N_FFT,
            'HOP_LENGTH': config.HOP_LENGTH,
            'F_MIN': config.F_MIN,
            'F_MAX': config.F_MAX,
            'config_hash': config_hash,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logging.info(f"Saved preprocessing config to {config_path}")

    # Create manifest for file mapping
    manifest = {
        'files': [],
        'config_hash': config_hash,
        'total_files': len(audio_files),
        'label_counts': {}
    }

    # Statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'total_time': 0
    }

    # Process each audio file
    logging.info(f"{'[DRY RUN] ' if dry_run else ''}Processing audio files...")

    start_time = time.time()

    for file_path, label in tqdm(audio_files, desc="Preprocessing"):
        try:
            # Generate unique filename based on original file path
            # Use relative path from DATA_DIR to preserve directory structure
            rel_path = file_path.relative_to(config.DATA_DIR)

            # Create hash of the relative path to avoid filename collisions
            path_hash = hashlib.md5(str(rel_path).encode()).hexdigest()[:12]

            # Output filename: <hash>_<label>.npy
            output_filename = f"{path_hash}_{label}.npy"
            output_path = output_dir / output_filename

            # Check if already processed
            if output_path.exists() and not overwrite:
                stats['skipped'] += 1

                # Add to manifest
                manifest['files'].append({
                    'original_path': str(rel_path),
                    'preprocessed_path': output_filename,
                    'label': label
                })

                continue

            if dry_run:
                logging.info(f"Would process: {rel_path} -> {output_filename}")
                stats['processed'] += 1
                continue

            # Process audio file
            spectrogram = preprocessor.process_audio_file(file_path)

            # Save as numpy array (memory-mapped for fast loading)
            np.save(output_path, spectrogram.numpy())

            # Add to manifest
            manifest['files'].append({
                'original_path': str(rel_path),
                'preprocessed_path': output_filename,
                'label': label,
                'shape': list(spectrogram.shape)
            })

            # Update label counts
            manifest['label_counts'][label] = manifest['label_counts'].get(label, 0) + 1

            stats['processed'] += 1

        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")
            stats['failed'] += 1

    stats['total_time'] = time.time() - start_time

    # Save manifest
    if not dry_run:
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logging.info(f"Saved manifest to {manifest_path}")

    # Log statistics
    logging.info("\n" + "="*60)
    logging.info("PREPROCESSING STATISTICS")
    logging.info("="*60)
    logging.info(f"Total files: {len(audio_files)}")
    logging.info(f"Processed: {stats['processed']}")
    logging.info(f"Skipped (already exists): {stats['skipped']}")
    logging.info(f"Failed: {stats['failed']}")
    logging.info(f"Total time: {stats['total_time']:.2f} seconds")
    if stats['processed'] > 0:
        logging.info(f"Average time per file: {stats['total_time'] / stats['processed']:.3f} seconds")

    if not dry_run:
        logging.info(f"\nPreprocessed data saved to: {output_dir}")

        # Calculate storage size
        total_size = sum(f.stat().st_size for f in output_dir.glob("*.npy"))
        logging.info(f"Total storage used: {total_size / (1024**2):.2f} MB")

        # Label distribution
        if manifest['label_counts']:
            logging.info("\nLabel distribution:")
            for label, count in sorted(manifest['label_counts'].items()):
                logging.info(f"  {label}: {count}")

    logging.info("="*60)

    return stats


def validate_preprocessed_data(preprocessed_dir: Path, config: Config) -> bool:
    """
    Validate preprocessed data against current config.

    Args:
        preprocessed_dir: Directory containing preprocessed data
        config: Current configuration

    Returns:
        True if valid, False otherwise
    """
    # Check if config file exists
    config_path = preprocessed_dir / "config.json"
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        return False

    # Load saved config
    with open(config_path, 'r') as f:
        saved_config = json.load(f)

    # Compute current config hash
    current_hash = compute_config_hash(config)
    saved_hash = saved_config.get('config_hash')

    if current_hash != saved_hash:
        logging.warning(f"Config hash mismatch!")
        logging.warning(f"  Saved:   {saved_hash}")
        logging.warning(f"  Current: {current_hash}")
        logging.warning("Preprocessing parameters have changed. Consider re-preprocessing.")
        return False

    # Check if manifest exists
    manifest_path = preprocessed_dir / "manifest.json"
    if not manifest_path.exists():
        logging.error(f"Manifest file not found: {manifest_path}")
        return False

    logging.info("Preprocessed data validation: PASSED")
    logging.info(f"  Config hash: {current_hash}")
    logging.info(f"  Preprocessed on: {saved_config.get('timestamp', 'unknown')}")

    return True


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(
        description="Pre-process audio dataset for fast training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess dataset with default config
  python scripts/preprocess_dataset.py --output data/processed/v1

  # Preview what would be processed (dry-run)
  python scripts/preprocess_dataset.py --output data/processed/v1 --dry-run

  # Overwrite existing preprocessed files
  python scripts/preprocess_dataset.py --output data/processed/v1 --overwrite

  # Validate existing preprocessed data
  python scripts/preprocess_dataset.py --validate data/processed/v1/<hash>
        """
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/v1',
        help='Output directory for preprocessed data (default: data/processed/v1)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview preprocessing without actually processing files'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing preprocessed files'
    )

    parser.add_argument(
        '--validate',
        type=str,
        help='Validate existing preprocessed data directory'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize config
    config = Config()

    # Validation mode
    if args.validate:
        validate_dir = Path(args.validate)
        if not validate_dir.exists():
            logging.error(f"Directory not found: {validate_dir}")
            sys.exit(1)

        is_valid = validate_preprocessed_data(validate_dir, config)
        sys.exit(0 if is_valid else 1)

    # Preprocessing mode
    output_dir = Path(args.output)

    logging.info("="*60)
    logging.info("BABY CRY DETECTION - OFFLINE PREPROCESSING")
    logging.info("="*60)
    logging.info(f"Data directory: {config.DATA_DIR}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info(f"Overwrite existing: {args.overwrite}")
    logging.info("="*60)

    try:
        stats = preprocess_dataset(
            config=config,
            output_dir=output_dir,
            dry_run=args.dry_run,
            overwrite=args.overwrite
        )

        if stats['failed'] > 0:
            logging.warning(f"Completed with {stats['failed']} failures")
            sys.exit(1)
        else:
            logging.info("Preprocessing completed successfully!")
            sys.exit(0)

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
