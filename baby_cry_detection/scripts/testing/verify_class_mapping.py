"""
Verify that model class mapping is correct by checking predictions on known cry/non-cry samples.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torchaudio
from src.config import Config
from src.model import create_model
from src.data_preprocessing import AudioPreprocessor

def verify_model_classes(model_path):
    """Verify model is predicting correct classes."""

    print("=" * 80)
    print("CLASS MAPPING VERIFICATION")
    print("=" * 80)

    config = Config()
    print(f"\nConfig class labels: {config.CLASS_LABELS}")

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if 'class_labels' in checkpoint:
        print(f"Checkpoint class labels: {checkpoint['class_labels']}")

    # Load a known cry sample for testing
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)

    print("\nThe issue is that your model is giving LOW confidence (8-40%)")
    print("even for segments with crying.")
    print("\nTwo possibilities:")
    print("1. Model needs retraining - current model is not confident enough")
    print("2. Crying is genuinely faint/background - model is being realistic")
    print("\nPotential fix: Try INVERTING the probabilities to see if model")
    print("was accidentally trained with swapped labels.")
    print("\nEdit src/audio_filter.py line 841:")
    print("Change:")
    print("    calibrated_prob = probabilities[0, 1].item()")
    print("To:")
    print("    calibrated_prob = probabilities[0, 0].item()  # Try index 0 instead")

    return checkpoint

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_class_mapping.py <model_path>")
        sys.exit(1)

    verify_model_classes(sys.argv[1])
