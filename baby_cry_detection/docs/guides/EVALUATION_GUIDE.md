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
python scripts/testing/test_my_audio.py my_audio.wav --threshold 0.85 --plot
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
results/train_YYYY-MM-DD_HH-MM-SS/evaluations/eval_YYYY-MM-DD_HH-MM-SS/
├── metrics_train.json
├── metrics_val.json
├── metrics_test.json
├── bootstrap_results_*.json
├── misclassified_*.json
├── temperature_calibration_results.json
└── calibrated_model.pth
```

## Test-Time Augmentation (TTA) — Optional

> [!info] Not Used in Deployment
> TTA is available as an optional evaluation flag but is **not used** in the deployment pipeline. The model's 97.93% accuracy at the 0.92 threshold is sufficient without TTA.

TTA can boost accuracy by 0.5-1% by averaging predictions from multiple augmented versions. Available via:

```bash
python training/main.py evaluate --model-path results/train_XXXX/model_best.pth --use-tta
```

## Example Evaluation Session

```bash
# Step 1: Check if training is complete
ls results/train_*/model_best.pth

# Step 2: Evaluate model
python training/main.py evaluate --model-path results/train_YYYY-MM-DD_HH-MM-SS/model_best.pth

# Output will show:
# Accuracy:     0.9793 (97.93%)
# Precision:    0.9793
# Recall:       0.9793
# F1-Score:     0.9793

# Step 3 (optional): TTA for slightly higher accuracy (not used in deployment)
# python training/main.py evaluate --model-path results/train_YYYY-MM-DD_HH-MM-SS/model_best.pth --use-tta
```

## GPU Memory for Evaluation

Evaluation uses much less memory than training:

| Task       | CNN-Transformer VRAM |
| ---------- | -------------------- |
| Training   | ~380 MB              |
| Evaluation | ~60 MB               |

> [!tip] GPU Compatibility
> Your RTX 4060 8GB: Perfect for evaluation

## Troubleshooting

> [!warning]- Checkpoint not found
> ```bash
> # List available checkpoints
> ls results/train_*/model_best.pth
>
> # Use full path
> python training/main.py evaluate --model-path results/train_2024-01-15_10-30-00/model_best.pth
> ```

> [!warning]- Out of memory during evaluation
> Evaluation should never OOM on 8GB GPU, but if it does:
> ```bash
> # Reduce batch size (won't affect accuracy, just slower)
> # Edit src/config.py temporarily:
> BATCH_SIZE = 32  # Down from 128
> ```

> [!warning]- Different number of samples
> If you're getting unexpected sample counts:
> ```bash
> # Delete the cached dataset splits and regenerate
> rm -rf data/processed/*
> python training/main.py analyze  # This regenerates splits
> ```

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
3. Try adjusting hyperparameters in [[Sound-Localization-Claude/baby_cry_detection/src/config.py|config.py]]
4. Collect more diverse training data

### If Accuracy < 90%
1. Check data quality and balance
2. Verify preprocessing settings
3. Review training loss curves
4. Consider adding data augmentation

## Files Used for Evaluation

1. [OK] [[Sound-Localization-Claude/baby_cry_detection/src/evaluate.py|evaluate.py]] - Evaluation module
2. [OK] [[Sound-Localization-Claude/baby_cry_detection/training/main.py|main.py]] - Training orchestration
3. [OK] `EVALUATION_GUIDE.md` - This guide

All tools are ready to use!
