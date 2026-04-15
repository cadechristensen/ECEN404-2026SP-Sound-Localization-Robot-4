# Deployment Models

Model checkpoint files for Raspberry Pi deployment. All `.pth` files are git-ignored.

## Model Files

| File | Description |
|------|-------------|
| `model_best_2026-03-14_run16_d384.pth` | **Primary deployment model.** FP32, 112 MB, d_model=384, Run 16 ensemble. Use this for production. |
| `model_best.pth` | Symlink to current primary model (112 MB, Run 16). |
| `model_inference.pth` | Outdated inference model (20.7 MB, d_model=256). Not recommended; use full 112 MB model. |
| `calibrated_model.pth` | Outdated calibrated model (20.7 MB, d_model=256). Superseded by Run 16 ensemble. |
| `model_best_copy.pth` | Backup of model_best.pth. |
| `model_best_copy2.pth` | Backup of model_best.pth. |
| `model_best_copy3.pth` | Backup of model_best.pth. |

## Usage

```bash
# Run detector (uses MODEL_PATH from config_pi.py by default)
cd deployment
python3 realtime_baby_cry_detector.py --device-index 2

# Test with a specific model override
python3 tests/test_baby_cry_detector.py --audio <file.wav> --model models/model_best.pth
```

## Notes

> [!info]
> All models are FP32 (no quantization). Detection thresholds are set in `config_pi.py` (0.92 detection, 0.92 confirmation) and can be overridden via CLI (`--threshold`, `--confirmation-threshold`). Run 16 ensemble achieves 97.93% test accuracy with 1.12% ECE.

> [!info]
> Run 16 ensemble uses post-hoc temperature scaling (stored separately) for calibrated confidence scores.

> [!note]
> Copy new models from `results/train_*/model_best.pth` after training.
