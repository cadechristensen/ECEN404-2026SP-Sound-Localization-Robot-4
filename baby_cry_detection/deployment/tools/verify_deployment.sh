#!/bin/bash
# Verification script for Raspberry Pi deployment
# Run from the baby_cry_detection/deployment/ directory after cloning the repo
#
# Usage:
#   bash tools/verify_deployment.sh

echo "=============================================================================="
echo "RASPBERRY PI DEPLOYMENT -- FILE VERIFICATION"
echo "=============================================================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DEPLOY_DIR" || {
    echo "ERROR: Cannot change to deployment directory"
    exit 1
}

echo "Checking directory: $(pwd)"
echo ""

# ---- Core detection scripts ----
echo "Core Detection Scripts:"
echo "-----------------------"
core_files=(
    "realtime_baby_cry_detector.py"
    "multichannel_detector.py"
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

# ---- Tools ----
echo "Tools:"
echo "------"
tool_files=(
    "tools/pi_setup.py"
    "tools/pi_diagnostics.py"
    "tools/multichannel_recorder.py"
)
for file in "${tool_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  [OK]      $file"
    else
        echo "  [MISSING] $file"
    fi
done
echo ""

# ---- Tests ----
echo "Tests:"
echo "------"
test_files=(
    "tests/test_baby_cry_detector.py"
    "tests/test_multichannel_detector.py"
    "tests/test_pi_filtering.py"
)
for file in "${test_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  [OK]      $file"
    else
        echo "  [MISSING] $file"
    fi
done
echo ""

# ---- Requirements and docs ----
echo "Requirements and Documentation:"
echo "--------------------------------"
other_files=(
    "requirements-pi.txt"
    "docs/DEPLOYMENT_GUIDE.md"
    "docs/FILTERING_REFERENCE.md"
)
for file in "${other_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  [OK]      $file"
    else
        echo "  [MISSING] $file"
    fi
done
echo ""

# ---- Model file (search models/ directory) ----
echo "Model File:"
echo "-----------"
model_found=0
# Check config MODEL_PATH first, then fallback candidates
config_model=$(python3 -c "from config_pi import ConfigPi; print(ConfigPi.MODEL_PATH)" 2>/dev/null)
for candidate in \
    "$config_model" \
    "models/model_best.pth" \
    "models/calibrated_model.pth" \
    "models/model_inference.pth"; do
    if [ -n "$candidate" ] && [ -f "$candidate" ]; then
        size=$(ls -lh "$candidate" | awk '{print $5}')
        echo "  [OK]      $candidate ($size)"
        model_found=1
        break
    fi
done
if [ $model_found -eq 0 ]; then
    echo "  [INFO]    No model file found in models/ directory."
    echo "            Set MODEL_PATH in config_pi.py or copy a model to deployment/models/"
    echo "            before running the detector."
fi
echo ""

# ---- Python import tests ----
echo "Python Import Tests:"
echo "--------------------"
python3 -c "import sys; sys.path.insert(0, '.'); from config_pi import ConfigPi; print('  [OK]      config_pi')" 2>/dev/null || echo "  [FAIL]    config_pi"
python3 -c "import sys; sys.path.insert(0, '.'); from audio_filtering import AudioFilteringPipeline; print('  [OK]      audio_filtering')" 2>/dev/null || echo "  [FAIL]    audio_filtering"
python3 -c "import sys; sys.path.insert(0, '.'); from multichannel_detector import DualChannelVotingDetector; print('  [OK]      multichannel_detector')" 2>/dev/null || echo "  [FAIL]    multichannel_detector"
python3 -c "import sys; sys.path.insert(0, '.'); from temporal_smoother import TemporalSmoothedDetector; print('  [OK]      temporal_smoother')" 2>/dev/null || echo "  [FAIL]    temporal_smoother"
python3 -c "import sys; sys.path.insert(0, '.'); from audio_buffer import CircularAudioBuffer; print('  [OK]      audio_buffer')" 2>/dev/null || echo "  [FAIL]    audio_buffer"
python3 -c "import sys; sys.path.insert(0, '.'); from detection_types import DetectionResult; print('  [OK]      detection_types')" 2>/dev/null || echo "  [FAIL]    detection_types"
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
echo "  cd deployment"
echo "  python3 realtime_baby_cry_detector.py \\"
echo "      --model models/calibrated_model.pth \\"
echo "      --device-index 2 \\"
echo "      --channels 4 \\"
echo "      --enable-multichannel \\"
echo "      --multichannel-voting weighted \\"
echo "      --enable-temporal-smoothing"
echo ""
echo "=============================================================================="
