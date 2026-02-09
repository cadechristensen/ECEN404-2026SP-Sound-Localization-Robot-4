# Inference Model Guide

## What is the Inference Model?

The **inference model** is an optimized version of your trained model designed specifically for making predictions (not training). It's **60% smaller** than the full model and works on both CPU and GPU.

## Model Comparison

| Model Type | Size | Speed | Use Case |
|------------|------|-------|----------|
| **model_best.pth** | 63 MB | Slower | Full checkpoint with training state, optimizer, history |
| **model_inference.pth** | 25 MB | Faster | ✅ **RECOMMENDED** - Production deployment |

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

**Result**: Same accuracy, 60% smaller file, faster loading!

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
    --checkpoint results/train_2025-11-04_01-02-34/model_inference.pth \
    --split test
```

### For Raspberry Pi Deployment
```powershell
.\deployment\deploy_to_pi.ps1 -PiHost "pi@raspberrypi.local" -ModelPath "results/train_2025-11-04_01-02-34/model_inference.pth"
```

### For Real-Time Detection
```bash
python scripts/testing/test_my_audio.py \
    --checkpoint results/train_2025-11-04_01-02-34/model_inference.pth \
    --audio-file "your_audio.wav"
```

## Benefits for Raspberry Pi

1. **Smaller File Size**: 25 MB vs 63 MB = Faster transfer over network
2. **Faster Loading**: Less data to read from disk
3. **Lower Memory**: Reduced RAM usage on Pi
4. **Same Accuracy**: No loss in prediction quality
5. **Works on CPU**: Perfect for Raspberry Pi deployment

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
- ✅ Deploying to Raspberry Pi
- ✅ Running evaluations
- ✅ Real-time detection
- ✅ Production use
- ✅ Testing with new audio

## Removed: Quantized Models

Previously, this project supported quantized models (model_quantized.pth) which were:
- 17 MB (even smaller)
- CPU-only
- Had accuracy issues
- Complex loading requirements

**Decision**: Use inference model instead for simplicity and better accuracy.

## Summary

For your Raspberry Pi baby cry detector, **always use `model_inference.pth`**:
- ✅ 60% smaller than full model
- ✅ Same accuracy
- ✅ Faster loading
- ✅ Works on CPU and GPU
- ✅ Simple to use
- ✅ No special loading required
