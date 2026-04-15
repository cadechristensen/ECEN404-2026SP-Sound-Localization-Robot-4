# Hardware Setup — PCM6260 Mic Array

Raspberry Pi 5 bring-up scripts for the TI PCM6260-Q1 four-channel microphone array (4× AOM-5024L-HD-R). Run once on a fresh Pi before any cry detection or sound localization code — these verify the I2C bus, load firmware into the codec, inspect the register state, and set the per-channel microphone gain.

> **Run order:** always run the hardware setup **before** launching `realtime_baby_cry_detector.py` or `Pi_Integration/main.py`. If the codec firmware isn't loaded or the gain is left at the default, the mic array records silence or severely clipped audio.

---

## Files

| File | Purpose |
|------|---------|
| `setup.txt` | Step-by-step manual reference — full command list with explanations |
| `setup.sh` | Automation script — verify I2C bus, load firmware, inspect registers, set gain |
| `decode_regdump.py` | Human-readable decoder for the PCM6260 Page 0 register dump |

---

## Quick Start

Run the automated script as root. An optional argument sets the analog-gain register value for all four channels.

```bash
sudo ./setup.sh                # verify hardware + firmware; do not touch gain
sudo ./setup.sh 0x50           # verify + set all channels to +20 dB
sudo ./setup.sh 0x7C           # verify + set all channels to +31 dB (max)
```

The script prints `[OK]` / `[FAIL]` / `[WARN]` lines for every check. Any failure prints a diagnostic hint and aborts so you don't end up with half-configured hardware.

### Inspecting register state manually

```bash
# Dump page 0 of the codec to sysfs
echo "0 0x00" | sudo tee /sys/bus/i2c/devices/1-0048/regdump

# Decode into human-readable form
sudo python3 decode_regdump.py
```

`decode_regdump.py` interprets the format fields (ASI format, word length, sample rate, BCLK ratio, MCLK frequency, etc.) instead of leaving you to read raw hex.

---

## Expected I2C Devices

On bus 1 the PCM6260 board registers four I2C endpoints. `setup.sh` checks all of them:

| Address | Role |
|---------|------|
| `0x40` | Board housekeeping |
| `0x48` | PCM6260-Q1 codec (main) |
| `0x54` | Board EEPROM |
| `0x65` | Clock / PLL controller |

If any of these are missing, check the ribbon cable between the Pi and the PCM6260 carrier board, and make sure I2C is enabled in `raspi-config`.

---

## Related Documentation

- [Project Overview](../README.md) — system architecture and top-level quick start
- [Deployment Guide](../baby_cry_detection/deployment/docs/DEPLOYMENT_GUIDE.md) — the Pi deployment checklist calls out hardware setup as prerequisite step 1
- Upstream kernel driver: `pcmdevice-linux-driver/` (separate repo) — provides the sysfs interface these scripts talk to
