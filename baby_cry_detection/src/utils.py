"""
Utility functions for baby cry detection project.
Contains helper functions for visualization, logging, and data analysis.
"""

import torch
import numpy as np

# Use non-interactive matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
import logging
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

try:
    from .config import Config
except ImportError:
    import sys
    from pathlib import Path as PathLib
    src_dir = PathLib(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from config import Config  # type: ignore


class AudioVisualizer:
    """
    Utility class for visualizing audio data and model features.
    """

    def __init__(self, config: Config = Config()):
        """
        Initialize the audio visualizer.

        Args:
            config: Configuration object
        """
        self.config = config
        plt.style.use('default')  # Set consistent plot style

    def plot_waveform(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        title: str = "Audio Waveform",
        save_path: Optional[Path] = None
    ):
        """
        Plot audio waveform.

        Args:
            waveform: Audio waveform data
            sample_rate: Sample rate of the audio
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()

        plt.figure(figsize=(12, 4))
        time_axis = np.linspace(0, len(waveform) / sample_rate, len(waveform))
        plt.plot(time_axis, waveform, linewidth=0.5)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    def plot_spectrogram(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        hop_length: int,
        title: str = "Spectrogram",
        save_path: Optional[Path] = None
    ):
        """
        Plot spectrogram.

        Args:
            spectrogram: Spectrogram data
            sample_rate: Sample rate of the audio
            hop_length: Hop length used for STFT
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.numpy()

        # Remove channel dimension if present
        if spectrogram.ndim == 3:
            spectrogram = spectrogram[0]

        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    def plot_feature_comparison(
        self,
        features_list: List[Tuple[Union[np.ndarray, torch.Tensor], str]],
        sample_rate: int,
        hop_length: int,
        save_path: Optional[Path] = None
    ):
        """
        Plot multiple spectrograms for comparison.

        Args:
            features_list: List of (spectrogram, title) tuples
            sample_rate: Sample rate of the audio
            hop_length: Hop length used for STFT
            save_path: Path to save the plot (optional)
        """
        n_plots = len(features_list)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

        if n_plots == 1:
            axes = [axes]

        for i, (spectrogram, title) in enumerate(features_list):
            if isinstance(spectrogram, torch.Tensor):
                spectrogram = spectrogram.numpy()

            # Remove channel dimension if present
            if spectrogram.ndim == 3:
                spectrogram = spectrogram[0]

            im = librosa.display.specshow(
                spectrogram,
                sr=sample_rate,
                hop_length=hop_length,
                x_axis='time',
                y_axis='mel',
                cmap='viridis',
                ax=axes[i]
            )
            axes[i].set_title(title)
            plt.colorbar(im, ax=axes[i], format='%+2.0f dB')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    def create_sample_visualizations(self, data_loader, results_dir: Path, n_samples: int = 4):
        """
        Create visualizations for sample audio files from each class.

        Args:
            data_loader: DataLoader containing the dataset
            results_dir: Directory to save visualizations
            n_samples: Number of samples to visualize per class
        """
        try:
            from .data_preprocessing import AudioPreprocessor
        except ImportError:
            from data_preprocessing import AudioPreprocessor  # type: ignore

        preprocessor = AudioPreprocessor(self.config)
        samples_per_class = {label: [] for label in self.config.CLASS_LABELS.values()}

        # Collect samples from each class
        for batch_data in data_loader:
            # Handle both old format (specs, labels) and new format (specs, labels, indices)
            if len(batch_data) == 3:
                spectrograms, labels, _ = batch_data
            else:
                spectrograms, labels = batch_data

            for i, label in enumerate(labels):
                class_name = self.config.CLASS_LABELS[label.item()]
                if len(samples_per_class[class_name]) < n_samples:
                    samples_per_class[class_name].append(spectrograms[i])

            # Check if we have enough samples for all classes
            if all(len(samples) >= n_samples for samples in samples_per_class.values()):
                break

        # Create visualizations for each class
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        for class_name, samples in samples_per_class.items():
            if not samples:
                continue

            # Create subplot grid
            fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
            if len(samples) == 1:
                axes = axes.reshape(2, 1)

            for i, spectrogram in enumerate(samples):
                # Plot spectrogram
                spec_data = spectrogram[0].numpy()  # Remove channel dimension

                im1 = librosa.display.specshow(
                    spec_data,
                    sr=self.config.SAMPLE_RATE,
                    hop_length=self.config.HOP_LENGTH,
                    x_axis='time',
                    y_axis='mel',
                    cmap='viridis',
                    ax=axes[0, i]
                )
                axes[0, i].set_title(f'{class_name.title()} Sample {i+1}')
                plt.colorbar(im1, ax=axes[0, i], format='%+2.0f dB')

                # Plot frequency distribution (average over time)
                freq_profile = np.mean(spec_data, axis=1)
                axes[1, i].plot(freq_profile)
                axes[1, i].set_title(f'Frequency Profile {i+1}')
                axes[1, i].set_xlabel('Mel Bin')
                axes[1, i].set_ylabel('Average Power (dB)')
                axes[1, i].grid(True, alpha=0.3)

            plt.suptitle(f'{class_name.title()} Class Samples', fontsize=16)
            plt.tight_layout()

            save_path = plots_dir / f"sample_visualizations_{class_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        logging.info(f"Sample visualizations saved to {plots_dir}")


class DataAnalyzer:
    """
    Utility class for analyzing dataset statistics and characteristics.
    """

    def __init__(self, config: Config = Config()):
        """
        Initialize the data analyzer.

        Args:
            config: Configuration object
        """
        self.config = config

    def analyze_audio_files(self, audio_files: List[Tuple[Path, str]]) -> Dict:
        """
        Analyze basic statistics of audio files in the dataset.

        Args:
            audio_files: List of (file_path, label) tuples

        Returns:
            Dictionary containing dataset statistics
        """
        try:
            from .data_preprocessing import AudioPreprocessor
        except ImportError:
            from data_preprocessing import AudioPreprocessor  # type: ignore

        preprocessor = AudioPreprocessor(self.config)
        stats = {
            'total_files': len(audio_files),
            'class_distribution': {},
            'duration_stats': [],
            'sample_rate_stats': [],
            'file_size_stats': []
        }

        # Analyze each file
        for file_path, label in audio_files:
            try:
                # Count class distribution
                if label not in stats['class_distribution']:
                    stats['class_distribution'][label] = 0
                stats['class_distribution'][label] += 1

                # Load audio for duration analysis
                waveform, original_sr = preprocessor.load_audio(file_path)
                duration = len(waveform) / preprocessor.sample_rate

                stats['duration_stats'].append(duration)
                stats['sample_rate_stats'].append(original_sr)
                stats['file_size_stats'].append(file_path.stat().st_size)

            except Exception as e:
                logging.warning(f"Failed to analyze {file_path}: {e}")

        # Calculate summary statistics
        if stats['duration_stats']:
            stats['duration_summary'] = {
                'mean': float(np.mean(stats['duration_stats'])),
                'std': float(np.std(stats['duration_stats'])),
                'min': float(np.min(stats['duration_stats'])),
                'max': float(np.max(stats['duration_stats'])),
                'median': float(np.median(stats['duration_stats']))
            }

        if stats['file_size_stats']:
            stats['file_size_summary'] = {
                'mean_mb': float(np.mean(stats['file_size_stats']) / 1024 / 1024),
                'total_mb': float(np.sum(stats['file_size_stats']) / 1024 / 1024)
            }

        # Unique sample rates
        stats['unique_sample_rates'] = list(set(stats['sample_rate_stats']))

        return stats

    def create_dataset_report(self, audio_files: List[Tuple[Path, str]], results_dir: Path):
        """
        Create a comprehensive dataset analysis report.

        Args:
            audio_files: List of (file_path, label) tuples
            results_dir: Directory to save the report
        """
        logging.info("Analyzing dataset...")

        stats = self.analyze_audio_files(audio_files)

        # Save statistics to JSON
        stats_path = results_dir / "logs" / "dataset_statistics.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Create visualizations
        self._plot_class_distribution(stats['class_distribution'], results_dir)
        self._plot_duration_distribution(stats['duration_stats'], results_dir)

        # Create text report
        report_path = results_dir / "logs" / "dataset_report.txt"
        with open(report_path, 'w') as f:
            f.write("BABY CRY DETECTION DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total Files: {stats['total_files']}\n\n")

            f.write("Class Distribution:\n")
            for class_name, count in stats['class_distribution'].items():
                percentage = count / stats['total_files'] * 100
                f.write(f"  {class_name}: {count} files ({percentage:.1f}%)\n")
            f.write("\n")

            if 'duration_summary' in stats:
                f.write("Duration Statistics:\n")
                dur = stats['duration_summary']
                f.write(f"  Mean: {dur['mean']:.2f} seconds\n")
                f.write(f"  Std: {dur['std']:.2f} seconds\n")
                f.write(f"  Range: {dur['min']:.2f} - {dur['max']:.2f} seconds\n")
                f.write(f"  Median: {dur['median']:.2f} seconds\n\n")

            if 'file_size_summary' in stats:
                f.write("File Size Statistics:\n")
                f.write(f"  Average file size: {stats['file_size_summary']['mean_mb']:.2f} MB\n")
                f.write(f"  Total dataset size: {stats['file_size_summary']['total_mb']:.2f} MB\n\n")

            f.write("Sample Rates:\n")
            for sr in sorted(stats['unique_sample_rates']):
                f.write(f"  {sr} Hz\n")

        logging.info(f"Dataset analysis report saved to {report_path}")

    def _plot_class_distribution(self, class_dist: Dict[str, int], results_dir: Path):
        """Plot class distribution."""
        plt.figure(figsize=(8, 6))
        classes = list(class_dist.keys())
        counts = list(class_dist.values())

        colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow']
        bars = plt.bar(classes, counts, color=colors[:len(classes)])

        plt.title('Class Distribution in Dataset')
        plt.xlabel('Class')
        plt.ylabel('Number of Files')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')

        plt.tight_layout()

        plot_path = results_dir / "plots" / "dataset_class_distribution.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_duration_distribution(self, durations: List[float], results_dir: Path):
        """Plot duration distribution."""
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.hist(durations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Audio Duration Distribution')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Number of Files')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.boxplot(durations)
        plt.title('Audio Duration Box Plot')
        plt.ylabel('Duration (seconds)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = results_dir / "plots" / "duration_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def setup_logging(results_dir: Path, log_level: str = "INFO"):
    """
    Set up comprehensive logging for the project.

    Args:
        results_dir: Directory to save log files
        log_level: Logging level
    """
    # Create logs directory
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "main.log"),
            logging.StreamHandler()
        ]
    )

    # Set up separate loggers for different components
    loggers = ['train', 'evaluate', 'data', 'model']
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        handler = logging.FileHandler(logs_dir / f"{logger_name}.log")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)


def save_experiment_config(config: Config, results_dir: Path):
    """
    Save experiment configuration to results directory.

    Args:
        config: Configuration object
        results_dir: Directory to save configuration
    """
    config_dict = {}
    for attr in dir(config):
        if not attr.startswith('_') and not callable(getattr(config, attr)):
            value = getattr(config, attr)
            if isinstance(value, Path):
                value = str(value)
            config_dict[attr] = value

    config_path = results_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    logging.info(f"Experiment configuration saved to {config_path}")


def print_system_info():
    """Print system information for debugging."""
    import platform
    import psutil

    logging.info("System Information:")
    logging.info(f"  Platform: {platform.platform()}")
    logging.info(f"  Python: {platform.python_version()}")
    logging.info(f"  CPU: {psutil.cpu_count()} cores")
    logging.info(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    if torch.cuda.is_available():
        logging.info(f"  GPU: {torch.cuda.get_device_name()}")
        logging.info(f"  CUDA: {torch.version.cuda}")
    else:
        logging.info("  GPU: Not available")


if __name__ == "__main__":
    # Test utility functions
    config = Config()

    # Test logging setup
    results_dir = Path("test_results")
    setup_logging(results_dir)

    logging.info("Testing utility functions...")
    print_system_info()

    # Test configuration saving
    save_experiment_config(config, results_dir)

    logging.info("Utility functions test completed!")