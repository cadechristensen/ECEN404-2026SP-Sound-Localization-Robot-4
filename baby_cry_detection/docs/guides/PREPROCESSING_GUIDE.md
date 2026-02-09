# Offline Preprocessing Guide

This guide explains how to use the offline preprocessing pipeline for faster training iterations.

## Why Use Preprocessing?

**Benefits:**
- **3-5× faster training**: Spectrograms are computed once and loaded from disk
- **Faster experimentation**: Try different hyperparameters without re-computing spectrograms
- **Reduced CPU usage**: Focus compute on training, not preprocessing
- **Versioned datasets**: Each preprocessing config gets its own cached version

**When to use:**
- Running multiple training experiments
- Hyperparameter tuning
- Quick iteration cycles

**When NOT to use:**
- First time setup (just train normally)
- Frequently changing preprocessing parameters
- Very limited disk space (<2GB free)

---

## Quick Start

### Step 1: Preprocess Your Dataset

```bash
# Preprocess all audio files (takes 5-10 minutes for 9k files)
python scripts/preprocess_dataset.py --output data/processed/v1
```

This will:
1. Load all audio files from `data/`
2. Compute mel spectrograms
3. Save as `.npy` files in `data/processed/v1/<hash>/`
4. Create a manifest for fast loading

**Expected output:**
```
Found 9264 audio files
Preprocessing config hash: f8494771d0d9d2bd
Processing audio files...
100%|████████████████| 9264/9264 [05:23<00:00, 28.6it/s]

PREPROCESSING STATISTICS
========================
Total files: 9264
Processed: 9264
Total time: 323.45 seconds
Average time per file: 0.035 seconds
Total storage used: 436.21 MB

Label distribution:
  cry: 4632
  non_cry: 4632
```

### Step 2: Enable Preprocessed Data in Config

Edit `src/config.py`:

```python
# Preprocessed data configuration
USE_PREPROCESSED = True  # Changed from False
PREPROCESSED_DIR = Path("data/processed/v1")  # Matches --output above
```

### Step 3: Train Normally

```bash
# Training now uses preprocessed data automatically
python src/train.py
```

You should see:
```
Using preprocessed data from: data/processed/v1
Found preprocessed directory: data/processed/v1/f8494771d0d9d2bd
Loaded manifest with 9264 files
```

**Training speedup:**
- Epoch 1 (without preprocessing): ~4-5 minutes
- Epoch 1 (with preprocessing): ~1-2 minutes
- **2-3× faster per epoch!**

---

## Advanced Usage

### Preview Before Processing (Dry Run)

See what will be processed without actually doing it:

```bash
python scripts/preprocess_dataset.py --output data/processed/v1 --dry-run
```

### Overwrite Existing Preprocessed Data

If you already preprocessed but want to redo it:

```bash
python scripts/preprocess_dataset.py --output data/processed/v1 --overwrite
```

### Validate Preprocessed Data

Check if your preprocessed data matches current config:

```bash
python scripts/preprocess_dataset.py --validate data/processed/v1/f8494771d0d9d2bd
```

This will warn you if preprocessing parameters have changed.

---

## How It Works

### Config Hash System

The preprocessing script creates a **hash** of your preprocessing parameters:

```python
# Parameters that affect preprocessing
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 0
F_MAX = 8000
```

If any of these change, you get a **new hash** and need to re-preprocess.

### Directory Structure

```
data/
├── cry_baby/           # Original audio files
│   ├── cry/
│   └── non_cry/
└── processed/
    └── v1/
        └── f8494771d0d9d2bd/   # Hash of config
            ├── config.json      # Saved preprocessing config
            ├── manifest.json    # File mapping
            ├── bd27483ac762_cry.npy
            ├── f5da9453eed6_cry.npy
            └── ...              # 9264 .npy files
```

### Manifest Structure

`manifest.json` contains:

```json
{
  "files": [
    {
      "original_path": "cry_baby/cry/sample.wav",
      "preprocessed_path": "bd27483ac762_cry.npy",
      "label": "cry",
      "shape": [128, 95]
    },
    ...
  ],
  "config_hash": "f8494771d0d9d2bd",
  "total_files": 9264,
  "label_counts": {
    "cry": 4632,
    "non_cry": 4632
  }
}
```

---

## Storage Requirements

**Per file:**
- Original audio (compressed): ~5-50 KB
- Preprocessed spectrogram (`.npy`): ~47 KB

**For 9,264 files:**
- Original audio: ~50-100 MB (compressed)
- Preprocessed data: **~436 MB** (uncompressed)

**Total overhead: ~350-400 MB**

---

## Troubleshooting

### Error: "No preprocessed data found"

**Solution:** Make sure you ran the preprocessing script first:

```bash
python scripts/preprocess_dataset.py --output data/processed/v1
```

### Error: "Config hash mismatch"

**Cause:** You changed preprocessing parameters (SAMPLE_RATE, N_MELS, etc.)

**Solution:** Re-run preprocessing with `--overwrite`:

```bash
python scripts/preprocess_dataset.py --output data/processed/v1 --overwrite
```

Or create a new version:

```bash
python scripts/preprocess_dataset.py --output data/processed/v2
```

Then update config:

```python
PREPROCESSED_DIR = Path("data/processed/v2")
```

### Training is slower with preprocessing

**Possible causes:**
1. **HDD instead of SSD**: Preprocessed data works best on SSD
2. **First epoch**: OS is caching files, subsequent epochs will be faster
3. **Too many workers**: Try reducing `NUM_WORKERS` in config

**Check:**
```bash
# Monitor disk I/O during training
# If disk is bottleneck, preprocessed data won't help much
```

### Out of disk space

**Solution 1:** Delete old preprocessed versions:

```bash
rm -rf data/processed/v1/old_hash/
```

**Solution 2:** Preprocess to external drive:

```bash
python scripts/preprocess_dataset.py --output /path/to/external/drive/processed/v1
```

Then update config:

```python
PREPROCESSED_DIR = Path("/path/to/external/drive/processed/v1")
```

---

## Switching Between Regular and Preprocessed

### Use Preprocessed Data

`src/config.py`:
```python
USE_PREPROCESSED = True
PREPROCESSED_DIR = Path("data/processed/v1")
```

### Use Regular On-the-Fly Processing

`src/config.py`:
```python
USE_PREPROCESSED = False  # Back to original behavior
```

No other changes needed! Training script automatically handles both modes.

---

## Performance Comparison

### Without Preprocessing (Original)
```
Epoch 1: 240 seconds (4 min)
Epoch 2: 240 seconds
...
Total (80 epochs): 320 minutes (5.3 hours)
```

### With Preprocessing
```
Preprocessing: 323 seconds (5.4 min) - ONE TIME
Epoch 1: 120 seconds (2 min)
Epoch 2: 120 seconds
...
Total (80 epochs): 165 minutes (2.75 hours)
```

**Net savings: 155 minutes (2.6 hours) per training run**

**For 10 training runs: 26 hours saved!**

---

## Best Practices

1. **Preprocess once, train many**: Use one preprocessed version for all your experiments
2. **Version your datasets**: Use `v1`, `v2`, etc. for different preprocessing configs
3. **Validate before training**: Run `--validate` to catch config mismatches early
4. **Keep preprocessing config stable**: Once you preprocess, avoid changing SAMPLE_RATE, N_MELS, etc.
5. **Use SSD for preprocessed data**: HDD will bottleneck loading

---

## FAQ

**Q: Can I still use data augmentation with preprocessed data?**

A: Currently, preprocessing saves base spectrograms without augmentation. On-the-fly augmentation at the spectrogram level could be added but is not yet implemented. For now, preprocessing works best when you don't need heavy augmentation.

**Q: What happens if I change BATCH_SIZE or LEARNING_RATE?**

A: No problem! These don't affect preprocessing. You can change training hyperparameters freely.

**Q: Can I preprocess a subset of data?**

A: Not directly, but you can:
1. Move subset to a temporary directory
2. Point `DATA_DIR` to that directory
3. Preprocess as normal

**Q: Does this work on Windows?**

A: Yes! The script uses `pathlib.Path` which works cross-platform.

**Q: How do I delete old preprocessed data?**

A:
```bash
# Windows
rmdir /s data\processed\v1\old_hash

# Linux/Mac
rm -rf data/processed/v1/old_hash/
```

---

## See Also

- [Training Guide](TRAINING_GUIDE_95PERCENT.md) - How to train the model
- [Quick Start](QUICK_START.md) - Getting started tutorial
- [README](README.md) - Project overview
