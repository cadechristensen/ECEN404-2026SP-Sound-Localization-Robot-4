#!/bin/bash
# Verification script for Raspberry Pi deployment
# Run from the deployment/raspberry_pi/ directory after cloning the repo
#
# Usage:
#   bash verify_raspberry_pi_files.sh

echo "=============================================================================="
echo "RASPBERRY PI DEPLOYMENT -- FILE VERIFICATION"
echo "=============================================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || {
    echo "ERROR: Cannot change to script directory"
    exit 1
}

echo "Checking directory: $(pwd)"
echo ""

# ---- Core detection scripts ----
echo "Core Detection Scripts:"
echo "-----------------------"
core_files=(
    "realtime_baby_cry_detector.py"
    "robot_baby_monitor.py"
    "multichannel_detector.py"
    "multichannel_recorder.py"
    "temporal_smoother.py"
)
missing_core=0
for file in "${core_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  [OK]      $file ($size)"
    else
        echo "  [MISSING] $file"
        missing_core=$((missing_core + 1))
    fi
done
echo ""

# ---- Supporting modules ----
echo "Supporting Modules:"
echo "-------------------"
support_files=(
    "audio_buffer.py"
    "audio_filtering.py"
    "config_pi.py"
    "detection_types.py"
    "sound_localization_interface.py"
    "pi_diagnostics.py"
    "pi_setup.py"
)
missing_support=0
for file in "${support_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  [OK]      $file ($size)"
    else
        echo "  [MISSING] $file"
        missing_support=$((missing_support + 1))
    fi
done
echo ""

# ---- Requirements and docs ----
echo "Requirements and Documentation:"
echo "--------------------------------"
other_files=(
    "requirements-pi.txt"
    "README_FILTERING.md"
    "PI_DEPLOYMENT_STEPS.md"
)
for file in "${other_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  [OK]      $file"
    else
        echo "  [MISSING] $file"
    fi
done
echo ""

# ---- Model file (search common locations) ----
# Current deployed model: calibrated_model.pth (FP32, no quantization).
# Candidates are checked in order of preference. Legacy quantized and
# best-checkpoint paths are retained as fallbacks only.
echo "Model File:"
echo "-----------"
model_found=0
for candidate in \
    "../calibrated_model.pth" \
    "../../deployment/calibrated_model.pth" \
    "../../model_quantized.pth" \
    "../../model_best.pth" \
    "../model_quantized.pth" \
    "model_quantized.pth" \
    "model_best.pth"; do
    if [ -f "$candidate" ]; then
        size=$(ls -lh "$candidate" | awk '{print $5}')
        echo "  [OK]      $candidate ($size)"
        model_found=1
        break
    fi
done
if [ $model_found -eq 0 ]; then
    echo "  [INFO]    No model file found in common locations."
    echo "            Expected location: ../calibrated_model.pth"
    echo "            Copy the calibrated FP32 model to deployment/calibrated_model.pth"
    echo "            before running this script."
fi
echo ""

# ---- Python import tests ----
echo "Python Import Tests:"
echo "--------------------"
python3 -c "from config_pi import ConfigPi; print('  [OK]      config_pi')" 2>/dev/null || echo "  [FAIL]    config_pi"
python3 -c "from audio_filtering import AudioFilteringPipeline; print('  [OK]      audio_filtering')" 2>/dev/null || echo "  [FAIL]    audio_filtering"
python3 -c "from multichannel_detector import DualChannelVotingDetector; print('  [OK]      multichannel_detector')" 2>/dev/null || echo "  [FAIL]    multichannel_detector"
python3 -c "from temporal_smoother import TemporalSmoothedDetector; print('  [OK]      temporal_smoother')" 2>/dev/null || echo "  [FAIL]    temporal_smoother"
python3 -c "from audio_buffer import CircularAudioBuffer; print('  [OK]      audio_buffer')" 2>/dev/null || echo "  [FAIL]    audio_buffer"
python3 -c "from detection_types import DetectionResult; print('  [OK]      detection_types')" 2>/dev/null || echo "  [FAIL]    detection_types"
python3 -c "import torch; print(f'  [OK]      PyTorch {torch.__version__}')" 2>/dev/null || echo "  [FAIL]    PyTorch not installed"
python3 -c "import pyaudio; print('  [OK]      PyAudio')" 2>/dev/null || echo "  [FAIL]    PyAudio not installed"
python3 -c "import librosa; print('  [OK]      librosa')" 2>/dev/null || echo "  [FAIL]    librosa not installed"
python3 -c "import scipy; print('  [OK]      scipy')" 2>/dev/null || echo "  [FAIL]    scipy not installed"
echo ""

# ---- Audio devices ----
echo "Audio Devices (input only):"
echo "---------------------------"
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
found = False
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'  [{i}] {info[\"name\"]} -- {info[\"maxInputChannels\"]} input channels')
        found = True
if not found:
    print('  No input devices detected.')
p.terminate()
" 2>/dev/null || echo "  Unable to list audio devices (PyAudio not installed)"
echo ""

# ---- Summary ----
total_missing=$((missing_core + missing_support))
echo "=============================================================================="
if [ $total_missing -gt 0 ]; then
    echo "VERIFICATION INCOMPLETE -- $total_missing file(s) missing"
    echo "=============================================================================="
    exit 1
fi

echo "VERIFICATION PASSED"
echo "=============================================================================="
echo ""
echo "To run the detector (replace 2 with your audio device index):"
echo ""
echo "  python3 realtime_baby_cry_detector.py \\"
echo "      --model ../calibrated_model.pth \\"
echo "      --device-index 2 \\"
echo "      --channels 4 \\"
echo "      --enable-multichannel \\"
echo "      --multichannel-voting weighted \\"
echo "      --enable-temporal-smoothing"
echo ""
echo "To run the full system with sound localization:"
echo ""
echo "  python3 robot_baby_monitor.py \\"
echo "      --model ../calibrated_model.pth \\"
echo "      --device-index 2 \\"
echo "      --enable-multichannel \\"
echo "      --enable-temporal-smoothing"
echo ""
echo "=============================================================================="
