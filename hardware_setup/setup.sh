#!/usr/bin/env bash
# PCM6260 setup script — runs steps 1-5 from setup.txt in order.
# Stops and reports if anything fails.
#
# Usage:
#   sudo ./setup.sh [gain_register_value]
#
# Example:
#   sudo ./setup.sh 0x50    # set all channels to +20 dB
#   sudo ./setup.sh          # skip gain setting, just verify everything

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

GAIN_VALUE="${1:-}"
I2C_BUS=1
I2C_ADDR=0x48
SYSFS_BASE="/sys/devices/platform/axi/1000120000.pcie/1f00074000.i2c/i2c-1/1-0048"
REGDUMP_PATH="/sys/bus/i2c/devices/1-0048/regdump"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

pass() { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
errors=0

# ============================================================
echo ""
echo "=========================================="
echo "  1. VERIFY HARDWARE"
echo "=========================================="

# Check we're running as root
if [ "$(id -u)" -ne 0 ]; then
    fail "Must run as root (sudo)"
    exit 1
fi
pass "Running as root"

# Check all expected I2C devices on the PCM6260 board
# Each address shows as hex value if unbound, or "UU" if a kernel driver claimed it
I2C_OUTPUT=$(i2cdetect -y "$I2C_BUS" 2>/dev/null)
i2c_missing=0
for addr in 0x40 0x48 0x54 0x65; do
    hex="${addr#0x}"
    if echo "$I2C_OUTPUT" | grep -qE "(^| )($hex|UU)( |$)"; then
        pass "I2C device $addr detected on bus $I2C_BUS"
    else
        fail "I2C device $addr NOT found on bus $I2C_BUS"
        i2c_missing=$((i2c_missing + 1))
    fi
done
if [ "$i2c_missing" -gt 0 ]; then
    echo "       Is the PCM6260 board connected? Check SDA/SCL wiring."
    errors=$((errors + i2c_missing))
fi

# Check sysfs device node exists
if [ -d "$SYSFS_BASE" ]; then
    pass "Sysfs device node exists"
else
    fail "Sysfs device node missing: $SYSFS_BASE"
    echo "       Is the pcmdevice driver loaded? Try: modprobe pcmdevice"
    errors=$((errors + 1))
fi

# Temperature
temp=$(vcgencmd measure_temp 2>/dev/null | grep -oP '[0-9.]+' || echo "")
if [ -n "$temp" ]; then
    # Check if temp is above 80
    if echo "$temp" | awk '{exit ($1 >= 80) ? 0 : 1}'; then
        warn "CPU temperature: ${temp}°C (HIGH)"
    else
        pass "CPU temperature: ${temp}°C"
    fi
else
    warn "Could not read CPU temperature"
fi

if [ "$errors" -gt 0 ]; then
    echo ""
    fail "Hardware check failed with $errors error(s). Fix before continuing."
    exit 1
fi

# ============================================================
echo ""
echo "=========================================="
echo "  2. LOAD FIRMWARE"
echo "=========================================="

if [ -f "$SYSFS_BASE/fwload" ]; then
    # Snapshot dmesg line count so we only inspect lines emitted by THIS load
    pre_count=$(dmesg | wc -l)

    echo 1 > "$SYSFS_BASE/fwload" 2>/dev/null
    sleep 1

    # Only the lines added since we triggered fwload, filtered to driver output
    fw_lines=$(dmesg | tail -n +$((pre_count + 1)) | grep -iE "pcmdevice|pcm6260" || true)

    # Use the positive success signal emitted by pcmdevice-regbin.c.
    # Substring-matching "fail" trips on the benign probe-time message
    # "Looking up irq-gpio property failed -22" (optional DT prop, errno -22 = EINVAL).
    if echo "$fw_lines" | grep -q "Firmware init complete"; then
        pass "Firmware load complete"
    else
        fail "Firmware load may have failed (no 'Firmware init complete' in dmesg):"
        echo "$fw_lines" | sed 's/^/       /'
        errors=$((errors + 1))
    fi
else
    fail "fwload sysfs not found at $SYSFS_BASE/fwload"
    errors=$((errors + 1))
fi

if [ "$errors" -gt 0 ]; then
    echo ""
    fail "Firmware step failed. Fix before continuing."
    exit 1
fi

# ============================================================
echo ""
echo "=========================================="
echo "  3. INSPECT REGISTERS"
echo "=========================================="

if [ -f "$REGDUMP_PATH" ]; then
    echo "0 0x00" > "$REGDUMP_PATH" 2>/dev/null
    reg_output=$(cat "$REGDUMP_PATH" 2>/dev/null)

    if echo "$reg_output" | grep -q "No-"; then
        reg_count=$(echo "$reg_output" | grep -c "No-")
        pass "Register dump readable ($reg_count registers)"
    else
        fail "Register dump returned no data"
        errors=$((errors + 1))
    fi

    # Quick sanity check: SLEEP_ENZ (0x02) bit 0 should be 1 (awake)
    sleep_reg=$(echo "$reg_output" | grep "R0x02:" | grep -oP '0x[0-9A-Fa-f]+$' || echo "")
    if [ -n "$sleep_reg" ]; then
        sleep_val=$((sleep_reg))
        if [ $((sleep_val & 1)) -eq 1 ]; then
            pass "Device is awake (SLEEP_ENZ=1)"
        else
            warn "Device is in sleep mode (SLEEP_ENZ=0)"
        fi
    fi

    # Check PWR_CFG (0x75) — ADC_PDZ bit 6
    pwr_reg=$(echo "$reg_output" | grep "R0x75:" | grep -oP '0x[0-9A-Fa-f]+$' || echo "")
    if [ -n "$pwr_reg" ]; then
        pwr_val=$((pwr_reg))
        if [ $((pwr_val & 0x40)) -ne 0 ]; then
            pass "ADCs powered up (ADC_PDZ=1)"
        else
            warn "ADCs powered down (ADC_PDZ=0)"
        fi
        if [ $((pwr_val & 0x80)) -ne 0 ]; then
            pass "MICBIAS powered up"
        else
            warn "MICBIAS powered down"
        fi
    fi

    # Run the decoder script if available
    if [ -f "$SCRIPT_DIR/decode_regdump.py" ]; then
        pass "Register decoder available at decode_regdump.py"
        echo "       Run 'sudo python3 $SCRIPT_DIR/decode_regdump.py' for full decode"
    else
        warn "decode_regdump.py not found — skipping detailed decode"
    fi
else
    fail "regdump sysfs not found at $REGDUMP_PATH"
    errors=$((errors + 1))
fi

if [ "$errors" -gt 0 ]; then
    echo ""
    fail "Register check failed. Fix before continuing."
    exit 1
fi

# ============================================================
echo ""
echo "=========================================="
echo "  4. SET GAIN"
echo "=========================================="

if [ -n "$GAIN_VALUE" ]; then
    reg_sysfs="$SYSFS_BASE/reg"
    if [ -f "$reg_sysfs" ]; then
        gain_errors=0
        for r in 0x3d 0x42 0x47 0x4c; do
            if echo "0 0x00 $r $GAIN_VALUE" > "$reg_sysfs" 2>/dev/null; then
                :
            else
                fail "Failed to write gain $GAIN_VALUE to register $r"
                gain_errors=$((gain_errors + 1))
            fi
        done
        if [ "$gain_errors" -eq 0 ]; then
            # Decode the gain value
            gain_dec=$(printf "%d" "$GAIN_VALUE")
            gain_db=$((gain_dec >> 2))
            pass "All 4 channels set to $GAIN_VALUE (+${gain_db} dB)"
        else
            errors=$((errors + gain_errors))
        fi
    else
        fail "reg sysfs not found at $reg_sysfs"
        errors=$((errors + 1))
    fi
else
    warn "No gain value provided — skipping (usage: sudo ./setup.sh 0x50)"
fi

# ============================================================
echo ""
echo "=========================================="
echo "  5. FIND AUDIO DEVICE"
echo "=========================================="

# Check ALSA capture devices
alsa_output=$(arecord -l 2>/dev/null || true)
if echo "$alsa_output" | grep -qi "card\|T20\|TI\|PCM"; then
    pass "ALSA capture device found:"
    echo "$alsa_output" | grep -i "card" | sed 's/^/       /'
else
    fail "No ALSA capture device found"
    echo "       Expected to see TI USB Audio / T20"
    errors=$((errors + 1))
fi

# Check PyAudio device (non-critical)
if command -v python3 &>/dev/null; then
    pa_output=$(python3 -c "
import pyaudio
pa = pyaudio.PyAudio()
found = False
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    if d['maxInputChannels'] > 0 and 'TI' in d['name']:
        print(f\"       index={i}: {d['name']} ({d['maxInputChannels']} ch)\")
        found = True
if not found:
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d['maxInputChannels'] > 0:
            print(f\"       index={i}: {d['name']} ({d['maxInputChannels']} ch)\")
pa.terminate()
" 2>/dev/null || true)

    if [ -n "$pa_output" ]; then
        pass "PyAudio input device(s):"
        echo "$pa_output"
    else
        warn "PyAudio could not list devices (pyaudio may not be installed)"
    fi
fi

# ============================================================
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="

if [ "$errors" -gt 0 ]; then
    fail "$errors error(s) detected. Review above."
    exit 1
else
    pass "All checks passed. Ready to record."
    echo ""
    echo "  Quick start:"
    echo "    arecord -D hw:T20 -c 8 -r 48000 -f S32_LE -d 5 test.wav"
    echo "    python Pi_Integration/record.py --device-name \"TI USB Audio\" --duration 10 --label throwaway -q"
    echo "    python Pi_Integration/main.py --device-name \"TI USB Audio\" -q"
    echo ""
fi
