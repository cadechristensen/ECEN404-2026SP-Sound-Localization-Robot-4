# Quick Start Guide - Training & Evaluation

## Training the Model

Train the CNN-Transformer model (binary classification):

```bash
python training/main.py train
```

The model will:
- Train on cry vs non-cry classification
- Use 128 mel-spectrogram features
- Apply data augmentation (noise, time-stretch, pitch-shift)
- Save the best checkpoint to `results/train_YYYY-MM-DD_HH-MM-SS/model_best.pth`
- Training takes 1-2 hours on GPU

## Evaluation

### Evaluate on Test Set

```bash
python training/main.py evaluate --model-path results/train_XXXX/model_best.pth
```

Outputs:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization
- ROC curve and AUC
- Metrics saved to results/eval_YYYY-MM-DD_HH-MM-SS/

### Quick Test on Audio File

```bash
python quick_test_audio.py
# or
python example_filtering_pipeline.py
```

## Configuration

Edit `src/config.py` to adjust:

```python
# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 60

# Audio parameters
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 128
```

### Evaluate Specific Model
```bash
python training/main.py evaluate --model-path results/train_xxx/model_best.pth
```

## Complete Workflow

Train the CNN-Transformer model with your current configuration:

```bash
python training/main.py train
```

This will:
- Train on your configured classification mode
- Use 128 mel-spectrogram features
- Apply data augmentation
- Save the best model checkpoint
- Training takes 1-2 hours on GPU

## Other Useful Commands

### Test Architecture
```bash
python training/main.py test
```

### Analyze Dataset
```bash
python training/main.py analyze
```

### Predict on Audio File
```bash
python training/main.py predict --model-path results/xxx/model_best.pth --audio-file path/to/audio.wav
```

### Train with Custom Settings
```bash
python training/main.py train --batch-size 256 --epochs 80 --learning-rate 0.0001
```

## GPU Settings

Your RTX 4060 8GB is perfect! No changes needed:
- Batch size: 128 (optimal)
- VRAM usage: ~800MB - 1.2 GB
- Training time: 1-2 hours
