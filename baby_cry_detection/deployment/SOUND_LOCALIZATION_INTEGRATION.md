# Sound Localization Integration Guide

## Data Flow to Sound Localization

The refactored detector now sends properly formatted multi-channel audio data to the sound localization module via multiprocessing queue.

## Receiving Localization Data

### Data Structure
When a baby cry is detected, the `detection_queue` receives this dictionary:

```python
localization_data = {
    'timestamp': float,                    # Unix timestamp of detection
    'confidence': float,                   # Cry confidence (0.0-1.0)
    'raw_audio': np.ndarray,              # Shape: (num_samples, 4) - unfiltered
    'filtered_audio': np.ndarray,         # Shape: (num_samples, 4) - cry regions only
    'sample_rate': int,                   # Always 16000 Hz (or configured rate)
    'num_channels': int,                  # Always 4 for PCM6260-Q1
    'audio_shape': tuple                  # (num_samples, 4)
}
```

### Example Receiver Code
```python
import multiprocessing as mp
import numpy as np

def sound_localization_process(detection_queue):
    """Worker process for sound localization."""
    while True:
        try:
            localization_data = detection_queue.get(timeout=5.0)

            # Extract data
            timestamp = localization_data['timestamp']
            confidence = localization_data['confidence']
            raw_audio = localization_data['raw_audio']           # (N, 4)
            filtered_audio = localization_data['filtered_audio'] # (N, 4)
            sample_rate = localization_data['sample_rate']       # 16000
            num_channels = localization_data['num_channels']     # 4

            print(f"Cry detected at {timestamp} (confidence: {confidence:.1%})")
            print(f"Audio shape: {raw_audio.shape}")
            print(f"Channels: {num_channels}")

            # Process with localization algorithm
            direction = estimate_cry_direction(raw_audio, sample_rate, num_channels)
            print(f"Estimated direction: {direction} degrees")

        except mp.queues.Empty:
            continue
```

## Using Multi-Channel Audio for Localization

### 1. Time Difference of Arrival (TDOA) Localization

TDOA is the simplest and most effective for 4-channel arrays.

```python
import scipy.signal
import numpy as np

def compute_tdoa(audio, sample_rate, ref_channel=0):
    """
    Compute Time Difference of Arrival between channels.

    Args:
        audio: Shape (num_samples, 4)
        sample_rate: Sample rate in Hz
        ref_channel: Reference channel (0-3)

    Returns:
        tdoa_samples: List of TDOA in samples relative to ref_channel
    """
    tdoa_samples = [0]  # Reference channel has 0 TDOA

    for ch in range(1, audio.shape[1]):
        # Cross-correlation between channels
        xcorr = scipy.signal.correlate(
            audio[:, ref_channel],
            audio[:, ch],
            mode='same'
        )

        # Find peak (maximum correlation)
        lag = scipy.signal.argrelextrema(xcorr, np.greater)[0]
        if len(lag) > 0:
            peak_lag = lag[np.argmax(xcorr[lag])]
        else:
            peak_lag = np.argmax(xcorr)

        # Convert to relative lag
        center = len(xcorr) // 2
        tdoa = peak_lag - center
        tdoa_samples.append(tdoa)

    # Convert to time
    tdoa_time = np.array(tdoa_samples) / sample_rate
    return tdoa_samples, tdoa_time

# Usage
raw_audio = localization_data['raw_audio']  # (N, 4)
sample_rate = localization_data['sample_rate']
tdoa_samples, tdoa_time = compute_tdoa(raw_audio, sample_rate)
print(f"TDOA (samples): {tdoa_samples}")
print(f"TDOA (ms): {tdoa_time * 1000}")
```

### 2. GCC-PHAT (Generalized Cross-Correlation with Phase Transform)

Better for noisy environments:

```python
def gcc_phat(audio, sample_rate, ref_channel=0):
    """
    GCC-PHAT for improved TDOA estimation in noisy conditions.

    Args:
        audio: Shape (num_samples, 4)
        sample_rate: Sample rate in Hz
        ref_channel: Reference channel (0-3)

    Returns:
        lags: Time lags in samples
        tdoa_estimates: TDOA for each non-reference channel
    """
    tdoa_estimates = [0]

    for ch in range(1, audio.shape[1]):
        if ch == ref_channel:
            continue

        # Compute FFT
        X_ref = np.fft.rfft(audio[:, ref_channel], axis=0)
        X_ch = np.fft.rfft(audio[:, ch], axis=0)

        # Cross power spectrum
        Pxy = X_ref * np.conj(X_ch)

        # Normalize by magnitude (phase transform)
        Pxy_normalized = Pxy / (np.abs(Pxy) + 1e-8)

        # IFFT to get GCC function
        gcc = np.fft.irfft(Pxy_normalized, axis=0)

        # Find peak
        lag = np.argmax(gcc[:audio.shape[0]//2])  # Only search first half

        # Adjust for circular shift
        if lag > audio.shape[0] // 4:
            lag = lag - audio.shape[0]

        tdoa_estimates.append(lag)

    return tdoa_estimates

# Usage
tdoa = gcc_phat(raw_audio, sample_rate)
print(f"GCC-PHAT TDOA: {tdoa}")
```

### 3. Beamforming - Direct Sound to Cry Source

Focus on the cry source direction:

```python
def delay_and_sum_beamformer(audio, sample_rate, tdoa_samples, num_mics=4):
    """
    Delay-and-sum beamformer to focus on source at estimated angle.

    Args:
        audio: Shape (num_samples, 4) - multi-channel audio
        sample_rate: Sample rate in Hz
        tdoa_samples: TDOA for each channel relative to reference
        num_mics: Number of microphones (4)

    Returns:
        beamformed_output: 1D array of beamformed audio
    """
    # Delay each channel by its TDOA
    delayed_audio = np.zeros_like(audio)

    for ch in range(num_mics):
        delay = abs(tdoa_samples[ch])

        if tdoa_samples[ch] > 0:
            # Shift forward
            delayed_audio[delay:, ch] = audio[:-delay, ch]
        else:
            # Shift backward
            delayed_audio[:len(audio)+tdoa_samples[ch], ch] = audio[-tdoa_samples[ch]:, ch]

    # Sum across channels
    beamformed = np.mean(delayed_audio, axis=1)

    return beamformed

# Usage
tdoa = gcc_phat(raw_audio, sample_rate)
beamformed = delay_and_sum_beamformer(raw_audio, sample_rate, tdoa)
print(f"Beamformed audio shape: {beamformed.shape}")
```

### 4. Source Localization from Microphone Array Geometry

For a known 4-microphone array geometry:

```python
def estimate_direction_from_tdoa(tdoa_time, mic_positions, speed_of_sound=343):
    """
    Estimate sound source direction from TDOA and known mic positions.

    Args:
        tdoa_time: TDOA in seconds for each channel (relative to reference)
        mic_positions: List of (x, y) coordinates for each mic
        speed_of_sound: Sound speed in m/s (343 at 20°C)

    Returns:
        direction_deg: Direction angle in degrees (0=front, 90=right)
    """
    # Convert TDOA to distances
    distances = np.array(tdoa_time) * speed_of_sound

    # This is a simplified version - for accurate results, use:
    # - Trilateral/Trilateration algorithms
    # - Particle filter or Kalman filter for tracking
    # - Least squares optimization

    # Simple approximation: use first two channels
    if len(distances) >= 2:
        distance_diff = distances[1] - distances[0]

        # Mic separation distance (for linear array)
        mic_sep = np.linalg.norm(
            np.array(mic_positions[1]) - np.array(mic_positions[0])
        )

        # Angle from distance difference
        if abs(distance_diff) < mic_sep:
            angle_rad = np.arcsin(distance_diff / mic_sep)
            angle_deg = np.degrees(angle_rad)
            return angle_deg

    return 0.0

# Example for square 4-mic array (10cm x 10cm)
mic_positions = [
    (0.00, 0.00),   # Mic 0: front-left
    (0.10, 0.00),   # Mic 1: front-right
    (0.00, 0.10),   # Mic 2: back-left
    (0.10, 0.10),   # Mic 3: back-right
]

sample_rate = 16000
tdoa_samples, tdoa_time = compute_tdoa(raw_audio, sample_rate)
direction = estimate_direction_from_tdoa(tdoa_time, mic_positions)
print(f"Estimated direction: {direction:.1f} degrees")
```

## Complete Integration Example

```python
import multiprocessing as mp
import numpy as np
from scipy import signal

def sound_localization_worker(detection_queue):
    """Complete sound localization process."""

    # Microphone array geometry (PCM6260-Q1 typical layout)
    mic_positions = [
        (0.0, 0.0),    # Ch 0: front-left
        (0.08, 0.0),   # Ch 1: front-right
        (0.0, 0.08),   # Ch 2: back-left
        (0.08, 0.08),  # Ch 3: back-right
    ]

    print("Sound localization worker started...")
    print(f"Microphone array: {len(mic_positions)} mics")

    while True:
        try:
            loc_data = detection_queue.get(timeout=5.0)

            # Get audio
            raw_audio = loc_data['raw_audio']  # (N, 4)
            sample_rate = loc_data['sample_rate']
            confidence = loc_data['confidence']

            print(f"\n{'='*60}")
            print(f"CRY DETECTION RECEIVED")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Audio shape: {raw_audio.shape}")
            print(f"  Duration: {raw_audio.shape[0]/sample_rate:.2f}s")
            print(f"  Channels: {raw_audio.shape[1]}")

            # TDOA estimation
            print("\nComputing TDOA...")
            tdoa_samples = [0]
            tdoa_times = [0.0]

            for ch in range(1, raw_audio.shape[1]):
                # Cross-correlation
                xcorr = signal.correlate(
                    raw_audio[:, 0],
                    raw_audio[:, ch],
                    mode='same'
                )
                lag = np.argmax(xcorr) - len(xcorr)//2
                tdoa_samples.append(lag)
                tdoa_times.append(lag / sample_rate * 1000)  # ms

            print(f"  TDOA (ms): {tdoa_times}")

            # Simple direction estimate
            if abs(tdoa_times[1]) > 0.5:
                mic_dist = 0.08  # 8cm spacing
                sound_speed = 343  # m/s
                distance_diff = (tdoa_times[1] / 1000) * sound_speed

                if abs(distance_diff) < mic_dist:
                    angle = np.degrees(np.arcsin(distance_diff / mic_dist))
                    print(f"  Estimated direction: {angle:.1f}° from front")

            # Beamform audio
            print("Beamforming audio to cry source...")
            delayed = np.zeros_like(raw_audio)
            for ch in range(raw_audio.shape[1]):
                delay = abs(tdoa_samples[ch])
                if delay > 0:
                    if tdoa_samples[ch] > 0:
                        delayed[delay:, ch] = raw_audio[:-delay, ch]
                    else:
                        delayed[:-delay, ch] = raw_audio[delay:, ch]
                else:
                    delayed[:, ch] = raw_audio[:, ch]

            beamformed = np.mean(delayed, axis=1)
            print(f"  Beamformed audio SNR improvement: ~3-6dB expected")

            # Ready for robot control
            print("\nLocalization complete, ready to navigate!")
            print(f"{'='*60}\n")

        except mp.queues.Empty:
            continue
        except Exception as e:
            print(f"Error in localization: {e}")

# Run in main
if __name__ == "__main__":
    detection_queue = mp.Queue(maxsize=10)

    # Start localization worker
    loc_process = mp.Process(
        target=sound_localization_worker,
        args=(detection_queue,),
        daemon=True
    )
    loc_process.start()

    # ... start detector with this queue ...
```

## Key Points for Implementation

### Audio Characteristics
- **Sample rate:** 16000 Hz (configurable in Config)
- **Channels:** 4 (preserved throughout)
- **Bit depth:** 32-bit float
- **Alignment:** All channels perfectly time-aligned
- **Phase:** Preserved for beamforming

### Processing Considerations
- Filtered audio has cry regions only (zeros elsewhere)
- Raw audio includes background noise
- Use filtered for cry detection algorithms
- Use raw for beamforming (contains noise but preserves structure)

### Timing
- Data arrives on detection, not continuously
- Timestamp indicates when cry started
- Audio buffer duration: 3-5 seconds of context
- Processing must be fast enough for real-time navigation

### Error Handling
- Check audio shape before processing
- Verify num_channels == 4
- Handle empty/short audio gracefully
- Log all processing steps

## Next Steps

1. Implement your sound localization algorithm
2. Integrate with robot navigation
3. Test TDOA accuracy with known sound sources
4. Calibrate microphone array geometry if needed
5. Add adaptive beamforming for improved tracking
