# Inference Model Guide

## What is the Inference Model?

The **inference model** is an optimized version of your trained model designed specifically for making predictions (not training). It's significantly smaller than the full model and works on both CPU and GPU.

> [!warning]
> As of Run 16 (2026-03-14), the current deployment uses the **full model** (112 MB, d_model=384) rather than the inference model. The inference model needs to be regenerated for the current architecture. See note below.

## Model Comparison

| Model Type                       | Size    | Speed    | Use Case                                                                          |
| -------------------------------- | ------- | -------- | --------------------------------------------------------------------------------- |
| **model_best.pth** (Run 16)      | 112 MB  | Moderate | **CURRENT DEPLOYMENT** - Full checkpoint, d_model=384, ensemble (epochs 95/88/85) |
| **model_inference.pth** (Run 16) | TBD     | Faster   | Needs regeneration - would strip optimizer state from current model               |
| **model_inference.pth** (older)  | 20.7 MB | Faster   | OUTDATED - d_model=256, do not use                                                |

## What's Different?

### Full Model (`model_best.pth`)
Contains:
- Model weights
- Optimizer state
- Training history
- Scheduler state
- Validation metrics
- All training artifacts

### Inference Model (`model_inference.pth`)
Contains ONLY:
- Model weights (state_dict)
- Config
- Class labels

> [!tip] Result
> Same accuracy, 60% smaller file, faster loading!

## Training vs Inference

### Training Phase
- Teaching the model with examples
- Updates weights/parameters
- Slow, GPU-intensive
- Happens once (or periodically)

### Inference Phase
- **Using the trained model to make predictions**
- Weights are frozen (no updates)
- Fast and efficient
- Happens constantly in production

## Using the Inference Model

### For Evaluation
```bash
python -m src.evaluate \
    --checkpoint results/train_XXXX/model_inference.pth \
    --split test
```

### For Raspberry Pi Deployment
```bash
scp results/train_XXXX/model_inference.pth pi@raspberrypi.local:~/baby_monitor/
```

### For Real-Time Detection
```bash
python scripts/testing/test_my_audio.py my_audio.wav \
    --model results/train_XXXX/model_inference.pth \
    --threshold 0.85
```

## Benefits for Raspberry Pi

1. **Smaller File Size**: Inference model would reduce ~112 MB to ~85-90 MB
2. **Faster Loading**: Less data to read from disk
3. **Lower Memory**: Reduced RAM usage on Pi
4. **Same Accuracy**: No loss in prediction quality (Run 16: 97.93% test accuracy)
5. **Works on CPU**: Perfect for Raspberry Pi deployment

> [!note] Current Status (Run 16)
> The deployed model is 112 MB (d_model=384). An inference-optimized version would save ~20-25%, but is not yet generated. The full model works well on Pi 5 CPU with acceptable latency.

## How It's Created

During training, the inference model is automatically saved:
```python
# From src/train.py
def save_model_for_inference(self, results_dir: Path):
    # Set model to evaluation mode
    self.model.eval()

    # Save only what's needed for inference
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'config': self.config.__dict__,
        'class_labels': self.config.CLASS_LABELS
    }, results_dir / "model_inference.pth")
```

## When to Use Each Model

### Use `model_best.pth` when:
- Resuming training
- Fine-tuning the model
- Need full training state

### Use `model_inference.pth` when:
- Deploying to Raspberry Pi
- Running evaluations
- Real-time detection
- Production use
- Testing with new audio

## Removed: Quantized Models

Previously, this project supported quantized models (model_quantized.pth) which were:
- 17 MB (even smaller)
- CPU-only
- Had accuracy issues
- Complex loading requirements

> [!info] Decision
> Use inference model instead for simplicity and better accuracy.

## Summary

For your Raspberry Pi baby cry detector:

**Current (Run 16, 2026-03-14):**
- Use **`model_best_2026-03-14_run16_d384.pth`** (112 MB)
- 97.93% test accuracy, 1.12% ECE
- Ensemble of 3 epochs (95, 88, 85)
- Inference latency: 100-200ms on Pi 5 CPU

**For Future Optimization:**
- Generate `model_inference.pth` from Run 16 (would save 20-25%)
- Alternative: Retrain with smaller d_model=256 for 60% smaller models (if accuracy acceptable)
