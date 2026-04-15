#!/usr/bin/env python3
"""
Main execution script for baby cry detection model.
Provides a comprehensive interface to train, evaluate, and analyze the model.
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import torch
from datetime import datetime

# Add project root to Python path (for src package)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Config
from src.train import BabyCryTrainer
from src.evaluate import ModelEvaluator, run_significance_tests_summary
from src.evaluation import EnsembleModel
from src.dataset import DatasetManager
from src.preprocessed_dataset import PreprocessedDatasetManager
from src.inference import BabyCryPredictor
from src.utils import (
    setup_logging, save_experiment_config, print_system_info,
    AudioVisualizer, DataAnalyzer
)


def train_model(config: Config, results_dir: Path, checkpoint_path: Path = None):
    """
    Train the baby cry detection model.

    Args:
        config: Configuration object
        results_dir: Directory to save results
        checkpoint_path: Optional path to checkpoint to resume from
    """
    logging.info("=" * 60)
    model_name = "CNN-TRANSFORMER"

    if checkpoint_path:
        logging.info(f"RESUMING {model_name} BABY CRY DETECTION MODEL TRAINING")
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        logging.info(f"STARTING {model_name} BABY CRY DETECTION MODEL TRAINING")
    logging.info("=" * 60)

    # Print system information
    print_system_info()

    # Save experiment configuration
    save_experiment_config(config, results_dir)

    # Initialize trainer
    trainer = BabyCryTrainer(config)

    # Setup training components
    trainer.setup_training()

    # Load checkpoint if provided
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)

    # Create dataset analysis (skip for preprocessed data to avoid file access issues)
    if not config.USE_PREPROCESSED:
        analyzer = DataAnalyzer(config)
        analyzer.create_dataset_report(trainer.dataset_manager.audio_files, results_dir)
    else:
        logging.info("Skipping dataset analysis - using preprocessed data")

    # Create sample visualizations
    visualizer = AudioVisualizer(config)
    visualizer.create_sample_visualizations(trainer.train_loader, results_dir)

    # Train the model
    history = trainer.train(results_dir)

    # Save inference-ready model
    trainer.save_model_for_inference(results_dir)

    logging.info("Training completed successfully!")
    logging.info(f"Results saved to: {results_dir}")

    return history


def evaluate_model(
    config: Config,
    model_path: Path,
    results_dir: Path = None,
    compute_bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    bootstrap_confidence_level: float = 0.95,
    run_significance_tests: bool = False,
    baseline_accuracy: float = 0.95,
    apply_temperature_scaling: bool = True
):
    """
    Evaluate a trained model.

    Args:
        config: Configuration object
        model_path: Path to the trained model
        results_dir: Directory to save evaluation results (if None, saves in training directory)
        compute_bootstrap_ci: Whether to compute bootstrap CIs (default: True)
        n_bootstrap: Number of bootstrap resamples (default: 1000)
        bootstrap_confidence_level: Confidence level for CIs (default: 0.95)
        run_significance_tests: Whether to run statistical significance tests (default: False)
        baseline_accuracy: Baseline benchmark accuracy for comparison (default: 0.95)
        apply_temperature_scaling: Whether to apply temperature scaling calibration (default: True)
    """
    logging.info("=" * 60)
    logging.info("STARTING MODEL EVALUATION")
    logging.info("=" * 60)

    # Determine where to save evaluation results
    if results_dir is None:
        # Save in training directory under 'evaluations/' subdirectory
        training_dir = model_path.parent
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = training_dir / "evaluations" / f"eval_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "plots").mkdir(exist_ok=True)
        (results_dir / "logs").mkdir(exist_ok=True)

        # Setup logging now that results_dir is created
        from src.utils import setup_logging
        setup_logging(results_dir, "INFO")
        logging.info("Baby Cry Detection System - Starting...")
        logging.info(f"Action: evaluate")
        logging.info(f"Saving evaluation results in training directory: {results_dir}")
    else:
        logging.info(f"Saving evaluation results to: {results_dir}")

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Load ensemble if enabled and checkpoint metadata exists
    if config.USE_ENSEMBLE:
        results_dir_for_ensemble = model_path.parent
        ensemble_metadata_path = results_dir_for_ensemble / "ensemble_checkpoints.json"
        if ensemble_metadata_path.exists():
            logging.info("USE_ENSEMBLE=True — loading ensemble model from checkpoints")
            ensemble_model = EnsembleModel.from_results_dir(
                results_dir_for_ensemble, config, evaluator.device
            )
            evaluator.model = ensemble_model
        else:
            logging.warning("USE_ENSEMBLE=True but no ensemble_checkpoints.json found — falling back to single model")
            evaluator.load_model(model_path)
    else:
        evaluator.load_model(model_path)

    # Prepare dataset — must mirror the same split path used during training so
    # the test indices are identical and there is no train/test overlap.
    if config.USE_PREPROCESSED:
        preprocessed_base = Path(config.PREPROCESSED_DIR)
        hash_dirs = sorted(preprocessed_base.glob("*/")) if preprocessed_base.exists() else []
        if not hash_dirs:
            raise FileNotFoundError(
                f"No preprocessed data found in {preprocessed_base}. "
                "Run preprocess_dataset.py first, or set config.USE_PREPROCESSED=False."
            )
        if len(hash_dirs) > 1:
            logging.warning(
                f"Multiple preprocessed directories found in {preprocessed_base}; "
                f"using the last (most recent) entry: {hash_dirs[-1].name}"
            )
        actual_dir = hash_dirs[-1]  # Use last (most recent) sorted entry — matches train.py
        logging.info(f"Evaluation using preprocessed data from: {actual_dir}")
        dataset_manager = PreprocessedDatasetManager(actual_dir, config)
        train_indices, val_indices, test_indices = dataset_manager.create_splits(
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO,
        )
        train_dataset, val_dataset, test_dataset = dataset_manager.create_datasets(
            train_indices, val_indices, test_indices, augment_train=False
        )
        # Use num_workers=0 for evaluation to avoid multiprocessing issues on Windows
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
        )
    else:
        dataset_manager = DatasetManager(config)
        train_dataset, val_dataset, test_dataset = dataset_manager.prepare_datasets()
        # Use num_workers=0 for evaluation to avoid multiprocessing issues on Windows
        train_loader, val_loader, test_loader = dataset_manager.create_data_loaders(
            train_dataset, val_dataset, test_dataset, num_workers=0
        )

    # Apply temperature scaling if requested
    if apply_temperature_scaling:
        logging.info("Applying temperature scaling calibration on validation set...")
        _, calibration_results = evaluator.apply_temperature_scaling(
            val_loader,
            results_dir,
            save_calibrated_model=True
        )
        logging.info(f"Temperature scaling complete. Optimal temperature: {calibration_results['optimal_temperature']:.4f}")
        logging.info(f"ECE improvement: {calibration_results['initial_ece']:.2%} -> {calibration_results['final_ece']:.2%}")

    # Evaluate on all splits
    logging.info("Evaluating on training set...")
    train_metrics = evaluator.evaluate_model(
        train_loader, results_dir, "train",
        compute_bootstrap_ci=compute_bootstrap_ci,
        n_bootstrap=n_bootstrap,
        bootstrap_confidence_level=bootstrap_confidence_level
    )

    logging.info("Evaluating on validation set...")
    val_metrics = evaluator.evaluate_model(
        val_loader, results_dir, "val",
        compute_bootstrap_ci=compute_bootstrap_ci,
        n_bootstrap=n_bootstrap,
        bootstrap_confidence_level=bootstrap_confidence_level
    )

    logging.info("Evaluating on test set...")
    test_metrics = evaluator.evaluate_model(
        test_loader, results_dir, "test",
        compute_bootstrap_ci=compute_bootstrap_ci,
        n_bootstrap=n_bootstrap,
        bootstrap_confidence_level=bootstrap_confidence_level
    )

    # Load and plot training history if available
    history_file = model_path.parent / "training_history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        evaluator.plot_training_history(history, results_dir)

    # Run statistical significance tests if requested
    significance_results = None
    if run_significance_tests:
        logging.info("=" * 60)
        logging.info("RUNNING STATISTICAL SIGNIFICANCE TESTS")
        logging.info("=" * 60)

        # Get predictions for significance tests (reuse test set predictions)
        y_pred, y_proba, y_true = evaluator.predict(test_loader)

        # Run significance tests summary
        significance_results = run_significance_tests_summary(
            y_true, y_pred, model_name="Baby Cry Detector"
        )

        # Compare to baseline benchmark
        baseline_result = evaluator.compare_to_baseline_benchmark(
            y_true, y_pred, baseline_accuracy=baseline_accuracy
        )
        logging.info("")
        logging.info("BASELINE BENCHMARK COMPARISON:")
        logging.info(f"  {baseline_result['comparison_summary']}")
        logging.info(f"  p-value: {baseline_result['p_value']:.6f}")
        logging.info(f"  Significant: {baseline_result['significant']}")

        # Compare to random baseline
        random_result = evaluator.compare_to_random_baseline(y_true, y_pred)
        logging.info("")
        logging.info("RANDOM BASELINE COMPARISON:")
        logging.info(f"  {random_result['conclusion']}")
        logging.info(f"  p-value: {random_result['p_value']:.6f}")

        # Save significance test results
        significance_results = {
            'baseline_comparison': baseline_result,
            'random_baseline_comparison': random_result
        }

        sig_results_path = results_dir / "significance_tests.json"
        with open(sig_results_path, 'w') as f:
            json.dump(significance_results, f, indent=2, default=str)
        logging.info(f"Significance test results saved to: {sig_results_path}")

    # Create comprehensive evaluation summary with training run metadata
    summary = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'training_directory': str(model_path.parent),
        'model_path': str(model_path),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'significance_tests': significance_results
    }

    # Load experiment config from training run if available
    experiment_config_path = model_path.parent / "experiment_config.json"
    if experiment_config_path.exists():
        with open(experiment_config_path, 'r') as f:
            summary['training_config'] = json.load(f)

    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logging.info("Evaluation completed successfully!")
    logging.info(f"Results saved to: {results_dir}")

    return summary


def analyze_data(config: Config, results_dir: Path):
    """
    Analyze the dataset without training.

    Args:
        config: Configuration object
        results_dir: Directory to save analysis results
    """
    logging.info("=" * 60)
    logging.info("STARTING DATASET ANALYSIS")
    logging.info("=" * 60)

    # Initialize dataset manager
    dataset_manager = DatasetManager(config)

    # Collect audio files
    from src.data_preprocessing import collect_audio_files
    audio_files = collect_audio_files(config.DATA_DIR, config.SUPPORTED_FORMATS)

    if not audio_files:
        logging.error(f"No audio files found in {config.DATA_DIR}")
        return

    # Create comprehensive analysis
    analyzer = DataAnalyzer(config)
    analyzer.create_dataset_report(audio_files, results_dir)

    # Prepare datasets for visualization
    train_dataset, val_dataset, test_dataset = dataset_manager.prepare_datasets()
    train_loader, val_loader, test_loader = dataset_manager.create_data_loaders(
        train_dataset, val_dataset, test_dataset
    )

    # Create visualizations
    visualizer = AudioVisualizer(config)
    visualizer.create_sample_visualizations(train_loader, results_dir)

    logging.info("Dataset analysis completed successfully!")
    logging.info(f"Results saved to: {results_dir}")


def test_model_architecture(config: Config):
    """
    Test model architecture without training.

    Args:
        config: Configuration object
    """
    logging.info("=" * 60)
    logging.info("TESTING MODEL ARCHITECTURE")
    logging.info("=" * 60)

    from src.model import create_model, model_summary, count_parameters

    # Create model
    model = create_model(config)

    # Calculate input shape
    time_steps = int(config.DURATION * config.SAMPLE_RATE // config.HOP_LENGTH) + 1
    input_shape = (1, config.N_MELS, time_steps)

    # Print model summary
    model_summary(model, input_shape)

    # Test forward pass
    dummy_input = torch.randn(4, *input_shape)
    try:
        output = model(dummy_input)
        logging.info(f"Forward pass successful: {dummy_input.shape} -> {output.shape}")
    except Exception as e:
        logging.error(f"Forward pass failed: {e}")

    # Test model components
    try:
        cnn_features, transformer_features = model.get_feature_maps(dummy_input)
        logging.info(f"Feature extraction successful:")
        logging.info(f"CNN features: {cnn_features.shape}")
        logging.info(f"Transformer features: {transformer_features.shape}")
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")

    logging.info("Model architecture test completed!")


def predict_audio(config: Config, model_path: Path, audio_file: Path, threshold: float = None):
    """
    Predict on a single audio file.

    Args:
        config: Configuration object
        model_path: Path to trained model
        audio_file: Path to audio file to predict on
        threshold: Optional confidence threshold
    """
    logging.info("=" * 60)
    logging.info("BABY CRY PREDICTION")
    logging.info("=" * 60)

    # Initialize predictor
    predictor = BabyCryPredictor(config)
    predictor.load_model(model_path)

    # Make prediction
    if threshold is not None:
        result = predictor.predict_with_threshold(audio_file, threshold)
    else:
        result = predictor.predict(audio_file)

    # Display results
    logging.info("=" * 60)
    logging.info("PREDICTION RESULTS")
    logging.info("=" * 60)
    logging.info(f"File: {result['file_path']}")
    logging.info(f"Prediction: {result['predicted_label'].upper()}")
    logging.info(f"Confidence: {result['confidence']:.2f}%")
    logging.info(f"")
    logging.info(f"Probabilities:")
    logging.info(f"  Non-Cry: {result['probabilities']['non_cry']:.2f}%")
    logging.info(f"  Cry:     {result['probabilities']['cry']:.2f}%")

    if 'threshold_decision' in result:
        logging.info(f"")
        logging.info(f"Threshold Decision: {result['threshold_decision']}")
        logging.info(f"Message: {result['threshold_message']}")

    logging.info("=" * 60)

    return result


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Baby Cry Detection Model - Training and Evaluation Pipeline"
    )

    parser.add_argument(
        'action',
        choices=['train', 'evaluate', 'analyze', 'test', 'predict'],
        help='Action to perform: train model, evaluate existing model, analyze data, test architecture, or predict on audio file'
    )

    parser.add_argument(
        '--model-path',
        type=Path,
        help='Path to model file (required for evaluate action, optional for train action to resume from checkpoint)'
    )

    parser.add_argument(
        '--config-path',
        type=Path,
        help='Path to custom configuration file'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Custom output directory (default: auto-generated timestamped directory)'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Path to data directory (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training/evaluation (overrides config)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate for training (overrides config)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )

    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for computation'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )

    parser.add_argument(
        '--audio-file',
        type=Path,
        help='Path to audio file for prediction (required for predict action)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        help='Confidence threshold for prediction (default: from config)'
    )

    # Bootstrap confidence interval arguments
    parser.add_argument(
        '--no-bootstrap',
        action='store_true',
        help='Disable bootstrap CI calculation'
    )

    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap resamples (default: 1000)'
    )

    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for intervals (default: 0.95)'
    )

    # Statistical significance test arguments
    parser.add_argument(
        '--significance-tests',
        action='store_true',
        help='Run statistical significance tests (compare to baseline 95%% and random baseline)'
    )

    parser.add_argument(
        '--baseline-accuracy',
        type=float,
        default=0.95,
        help='Baseline benchmark accuracy for comparison (default: 0.95)'
    )

    # Temperature scaling argument (on by default — critical for calibration)
    parser.add_argument(
        '--no-temperature-scaling',
        action='store_true',
        help='Skip temperature scaling calibration (not recommended — calibration is critical for threshold-based decisions)'
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Config()

    # Load custom configuration if provided
    if args.config_path and args.config_path.exists():
        try:
            with open(args.config_path, 'r') as f:
                custom_config = json.load(f)
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            logging.info(f"Loaded custom configuration from {args.config_path}")
        except Exception as e:
            logging.warning(f"Failed to load custom config: {e}")

    # Override config with command line arguments
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    # Set device
    if args.device == 'cpu':
        config.DEVICE = 'cpu'
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            config.DEVICE = 'cuda'
        else:
            logging.warning("CUDA requested but not available, falling back to CPU")
            config.DEVICE = 'cpu'

    # Create results directory (skip for evaluate if no --output-dir specified)
    if args.output_dir:
        results_dir = args.output_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "plots").mkdir(exist_ok=True)
        (results_dir / "logs").mkdir(exist_ok=True)
    elif args.action == 'evaluate':
        # For evaluate without --output-dir, don't create results_dir yet
        # It will be created inside the training directory by evaluate_model()
        results_dir = None
    else:
        # Use mode-specific directory naming for train/analyze/test
        mode_map = {
            'train': 'train',
            'analyze': 'analyze',
            'test': 'test'
        }
        results_dir = config.get_results_dir(mode=mode_map.get(args.action, 'train'))

    # Setup logging (skip for evaluate without results_dir - it will setup logging inside evaluate_model)
    if results_dir is not None:
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(results_dir, log_level)
        logging.info("Baby Cry Detection System - Starting...")
        logging.info(f"Action: {args.action}")
        logging.info(f"Results directory: {results_dir}")

    try:
        if args.action == 'train':
            # Check if data directory exists
            if not config.DATA_DIR.exists():
                logging.error(f"Data directory not found: {config.DATA_DIR}")
                sys.exit(1)

            # Check if resuming from checkpoint
            checkpoint_path = None
            if args.model_path:
                if not args.model_path.exists():
                    logging.error(f"Checkpoint file not found: {args.model_path}")
                    sys.exit(1)
                checkpoint_path = args.model_path

            train_model(config, results_dir, checkpoint_path)

        elif args.action == 'evaluate':
            if not args.model_path:
                logging.error("Model path required for evaluation")
                parser.print_help()
                sys.exit(1)

            if not args.model_path.exists():
                logging.error(f"Model file not found: {args.model_path}")
                sys.exit(1)

            # If --output-dir was specified, use it; otherwise save in training directory
            eval_results_dir = results_dir if args.output_dir else None
            evaluate_model(
                config,
                args.model_path,
                eval_results_dir,
                compute_bootstrap_ci=not args.no_bootstrap,
                n_bootstrap=args.n_bootstrap,
                bootstrap_confidence_level=args.confidence_level,
                run_significance_tests=args.significance_tests,
                baseline_accuracy=args.baseline_accuracy,
                apply_temperature_scaling=not args.no_temperature_scaling
            )

        elif args.action == 'analyze':
            if not config.DATA_DIR.exists():
                logging.error(f"Data directory not found: {config.DATA_DIR}")
                sys.exit(1)

            analyze_data(config, results_dir)

        elif args.action == 'test':
            test_model_architecture(config)

        elif args.action == 'predict':
            if not args.model_path:
                logging.error("Model path required for prediction")
                parser.print_help()
                sys.exit(1)

            if not args.model_path.exists():
                logging.error(f"Model file not found: {args.model_path}")
                sys.exit(1)

            if not args.audio_file:
                logging.error("Audio file required for prediction")
                parser.print_help()
                sys.exit(1)

            if not args.audio_file.exists():
                logging.error(f"Audio file not found: {args.audio_file}")
                sys.exit(1)

            predict_audio(config, args.model_path, args.audio_file, args.threshold)

        logging.info("=" * 60)
        logging.info("EXECUTION COMPLETED SUCCESSFULLY!")
        logging.info("=" * 60)

    except KeyboardInterrupt:
        logging.info("Execution interrupted by user")
        sys.exit(130)

    except Exception as e:
        logging.error(f"Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
