# Model Evaluation Guide

Complete guide for evaluating your CNN-Transformer baby cry detection model.

## Quick Evaluation

### Evaluate Latest Model

```bash
python training/main.py evaluate --model-path results/train_XXXX/model_best.pth
```

This will:
- Load your trained model
- Evaluate on test set
- Calculate accuracy, precision, recall, F1-score
- Generate confusion matrix and ROC curve
- Save results to `results/eval_YYYY-MM-DD_HH-MM-SS/`

### Quick Test on Single Audio File

```bash
python quick_test_audio.py
```

or with examples:

```bash
python example_filtering_pipeline.py
```

## What Gets Evaluated

### Metrics Calculated

The evaluation script calculates:

**Overall Metrics:**
- [OK] Accuracy
- [OK] Precision
- [OK] Recall
- [OK] F1 Score
- [OK] ROC-AUC (for binary classification)

**Per-Class Metrics:**
- [OK] Per-class precision
- [OK] Per-class recall
- [OK] Per-class F1 score
- [OK] Per-class support (number of samples)

**Visualizations:**
- [OK] Confusion matrix
- [OK] ROC curves (binary mode)
- [OK] Precision-Recall curves (binary mode)

### Output Files

After evaluation, you'll get:

```
results/eval_YYYY-MM-DD_HH-MM-SS/
├── evaluation_metrics.json             # All metrics in JSON
├── confusion_matrix.png                # Confusion matrix plot
├── roc_curve.png                       # ROC curve
└── logs/
    └── evaluation.log                  # Detailed logs
```

## Test-Time Augmentation (TTA)

TTA can boost accuracy by 0.5-1% by averaging predictions from multiple augmented versions:

**How it works:**
1. Original spectrogram → prediction
2. Time-shifted spectrogram → prediction
3. Noisy spectrogram → prediction
4. ... (5 augmentations total)
5. Average all predictions → final result

**Usage:**
```bash
python training/main.py evaluate --model-path results/train_XXXX/model_best.pth --use-tta
```

**Trade-off:**
- [BENEFIT] Higher accuracy (+0.5-1%)
- [COST] 5x slower evaluation (still fast, ~seconds instead of milliseconds)

## Example Evaluation Session

```bash
# Step 1: Check if training is complete
ls results/train_*/model_best.pth

# Step 2: Evaluate model
python training/main.py evaluate --model-path results/train_YYYY-MM-DD_HH-MM-SS/model_best.pth

# Output will show:
# Accuracy:     0.9583 (95.83%)
# Precision:    0.9602
# Recall:       0.9565
# F1-Score:     0.9583

# Step 3: If you want even higher accuracy, use TTA
python training/main.py evaluate --model-path results/train_YYYY-MM-DD_HH-MM-SS/model_best.pth --use-tta

# Output:
# Accuracy:     0.9650 (96.50%)  # +0.67% boost from TTA!
```

## GPU Memory for Evaluation

Evaluation uses much less memory than training:

| Task | CNN-Transformer VRAM |
|------|----------------------|
| Training | ~276 MB |
| Evaluation | ~50 MB |
| Evaluation with TTA | ~100 MB |

**Your RTX 4060 8GB:** Perfect for evaluation (even with TTA)

## Troubleshooting

### "Checkpoint not found"
```bash
# List available checkpoints
ls results/train_*/model_best.pth

# Use full path
python training/main.py evaluate --model-path results/train_2024-01-15_10-30-00/model_best.pth
```

### "Out of memory during evaluation"
Evaluation should never OOM on 8GB GPU, but if it does:
```bash
# Reduce batch size (won't affect accuracy, just slower)
# Edit src/config.py temporarily:
BATCH_SIZE = 32  # Down from 128
```

### "Different number of samples"
If you're getting unexpected sample counts:
```bash
# Delete the cached dataset splits and regenerate
rm -rf data/processed/*
python training/main.py analyze  # This regenerates splits
```

## Next Steps After Evaluation

Based on your evaluation results:

### If Accuracy >= 95%
1. Your model is production-ready
2. Consider deployment to Raspberry Pi
3. Integrate sound localization
4. Test on real-world data

### If Accuracy 90-95%
1. Review training results and logs
2. Consider increasing training epochs
3. Try adjusting hyperparameters in src/config.py
4. Collect more diverse training data

### If Accuracy < 90%
1. Check data quality and balance
2. Verify preprocessing settings
3. Review training loss curves
4. Consider adding data augmentation

## Files Used for Evaluation

1. [OK] `src/evaluate.py` - Evaluation module
2. [OK] `training/main.py` - Training orchestration
3. [OK] `src/inference_with_filtering.py` - Inference with filtering
4. [OK] `EVALUATION_GUIDE.md` - This guide

All tools are ready to use!
