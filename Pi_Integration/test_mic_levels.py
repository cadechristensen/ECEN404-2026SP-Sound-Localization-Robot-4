"""Quick test to compare mic levels between arecord (S32_LE) and PyAudio (Float32)."""

import numpy as np
from record_samtry import record_audio, find_device_by_name

idx = find_device_by_name("TI USB Audio")
audio, sr = record_audio(idx, 3, 4, 48000)
for ch in range(4):
    print(
        f"CH{ch}: peak={np.max(np.abs(audio[:, ch])):.4f}, rms={np.sqrt(np.mean(audio[:, ch] ** 2)):.4f}"
    )
