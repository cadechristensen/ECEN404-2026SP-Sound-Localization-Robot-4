#!/usr/bin/env python3
"""PCM6260-Q1 Page 0 register decoder.

Reads live from the driver's regdump sysfs and prints human-readable
descriptions of every known register on page 0.

Usage (on the Pi):
    sudo python3 decode_regdump.py
"""

import subprocess
import sys

REGDUMP_PATH = "/sys/bus/i2c/devices/1-0048/regdump"
DEV_NO = 0
PAGE = 0

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

ASI_FORMAT = {0: "TDM", 1: "I2S", 2: "LJ (left-justified)", 3: "Reserved"}
ASI_WLEN = {0: "16-bit", 1: "20-bit", 2: "24-bit", 3: "32-bit"}
POLARITY = {0: "Default", 1: "Inverted"}

FS_RATE = {
    0: "7.35/8 kHz", 1: "14.7/16 kHz", 2: "22.05/24 kHz", 3: "29.4/32 kHz",
    4: "44.1/48 kHz", 5: "88.2/96 kHz", 6: "176.4/192 kHz", 7: "352.8/384 kHz",
    8: "705.6/768 kHz",
}

FS_BCLK_RATIO = {
    0: "16", 1: "24", 2: "32", 3: "48", 4: "64", 5: "96", 6: "128",
    7: "192", 8: "256", 9: "384", 10: "512", 11: "1024", 12: "2048",
    14: "144",
}

MCLK_FREQ = {
    0: "12 MHz", 1: "12.288 MHz", 2: "13 MHz", 3: "16 MHz",
    4: "19.2 MHz", 5: "19.68 MHz", 6: "24 MHz", 7: "24.576 MHz",
}

MCLK_RATIO = {
    0: "64", 1: "256", 2: "384", 3: "512", 4: "768", 5: "1024",
    6: "1536", 7: "2304",
}

GPIO_CFG = {
    0: "Disabled", 1: "GPO", 2: "IRQ output", 3: "SDOUT2",
    7: "ADC power-down input", 8: "MICBIAS_EN input", 9: "GPI",
    10: "MCLK input", 11: "SDIN (daisy-chain)",
}

GPIO_DRV = {
    0: "Hi-Z", 1: "Active low / Active high", 2: "Active low / Weak high",
    3: "Active low / Hi-Z", 4: "Weak low / Active high", 5: "Hi-Z / Active high",
}

GPI_CFG = {
    0: "Disabled", 7: "ADC power-down input", 8: "MICBIAS_EN input",
    9: "GPI", 10: "MCLK input", 11: "SDIN (daisy-chain)",
}

INPUT_TYPE = {0: "Microphone", 1: "Line"}
INPUT_SRC = {0: "Analog differential", 1: "Analog single-ended"}
INPUT_COUPLING = {0: "AC-coupled", 1: "DC-coupled"}
INPUT_RANGE = {0: "Low swing (2 Vrms diff)", 1: "High swing (14.14 Vpp diff)"}
PGA_CFG = {0: "High SNR", 2: "High CMRR"}
AGC_EN = {0: "Disabled", 1: "Enabled"}

DECI_FILT = {0: "Linear phase", 1: "Low latency", 2: "Ultra-low latency"}
CH_SUM = {
    0: "Disabled", 1: "2-ch: (CH1+CH2)/2, (CH3+CH4)/2",
    2: "4-ch: (CH1+CH2+CH3+CH4)/4",
}
HPF_SEL = {
    0: "Programmable IIR (custom)", 1: "0.00025*Fs (12 Hz @ 48k)",
    2: "0.002*Fs (96 Hz @ 48k)", 3: "0.008*Fs (384 Hz @ 48k)",
}
BIQUAD_CFG = {0: "None (disabled)", 1: "1 per channel", 2: "2 per channel", 3: "3 per channel"}

SHDN_CFG = {
    0: "DREG powers down immediately", 1: "DREG active until timeout then off",
    2: "DREG active until clean shutdown",
}
DREG_KA = {0: "30 ms", 1: "25 ms", 2: "10 ms", 3: "5 ms"}
VREF_QCHG = {0: "3.5 ms", 1: "10 ms", 2: "50 ms", 3: "100 ms"}

MODE_STS = {4: "Sleep/shutdown", 6: "Active (ADCs off)", 7: "Active (ADCs on)"}

INT_EVENT = {0: "Active-low, assert until cleared", 1: "Pulse low 2 ms",
             2: "Pulse low 4 ms", 3: "Pulse low 8 ms"}

# ---------------------------------------------------------------------------
# Decoder helpers
# ---------------------------------------------------------------------------

def bits(val, hi, lo):
    """Extract bit field [hi:lo] from val."""
    mask = (1 << (hi - lo + 1)) - 1
    return (val >> lo) & mask


def decode_gain(val):
    """Decode CHx_CFG1 analog gain register."""
    g = bits(val, 7, 2)
    if g <= 42:
        return f"+{g} dB"
    return f"Reserved ({g})"


def decode_dvol(val):
    """Decode CHx_CFG2 digital volume register."""
    if val == 0:
        return "Muted"
    db = (val - 201) * 0.5
    return f"{db:+.1f} dB"


def decode_gcal(val):
    """Decode CHx_CFG3 gain calibration [7:4]."""
    g = bits(val, 7, 4)
    db = (g - 8) * 0.1
    return f"{db:+.1f} dB"


def decode_slot(val):
    """Decode ASI_CHx slot assignment."""
    output = "SDOUT2" if bits(val, 6, 6) else "SDOUT"
    slot = bits(val, 5, 0)
    if slot < 32:
        slot_str = f"TDM slot {slot} / I2S left slot {slot}"
    else:
        slot_str = f"TDM slot {slot} / I2S right slot {slot - 32}"
    return f"{output}, {slot_str}"


# ---------------------------------------------------------------------------
# Register definitions: addr -> (name, decoder_func)
# decoder_func(val) -> list of (field_name, decoded_string)
# ---------------------------------------------------------------------------

def _decode_page_cfg(v):
    return [("PAGE", str(v))]

def _decode_sw_reset(v):
    return [("SW_RESET", "Reset" if v & 1 else "Normal")]

def _decode_sleep_cfg(v):
    return [
        ("VREF_QCHG", VREF_QCHG.get(bits(v,4,3), "?")),
        ("I2C_BRDCAST_EN", "Enabled" if bits(v,2,2) else "Disabled"),
        ("SLEEP_ENZ", "Awake" if v & 1 else "Sleep"),
    ]

def _decode_shdn_cfg(v):
    return [
        ("SHDNZ_CFG", SHDN_CFG.get(bits(v,3,2), "Reserved")),
        ("DREG_KA_TIME", DREG_KA.get(bits(v,1,0), "?")),
    ]

def _decode_asi_cfg0(v):
    return [
        ("ASI_FORMAT", ASI_FORMAT.get(bits(v,7,6), "?")),
        ("ASI_WLEN", ASI_WLEN.get(bits(v,5,4), "?")),
        ("FSYNC_POL", POLARITY.get(bits(v,3,3), "?")),
        ("BCLK_POL", POLARITY.get(bits(v,2,2), "?")),
        ("TX_EDGE", "Inverted" if bits(v,1,1) else "Default"),
        ("TX_FILL", "Hi-Z" if v & 1 else "Drive 0"),
    ]

def _decode_asi_cfg1(v):
    return [
        ("TX_LSB", "Half-cycle + Hi-Z" if bits(v,7,7) else "Full cycle"),
        ("TX_KEEPER", str(bits(v,6,5))),
        ("TX_OFFSET", f"{bits(v,4,0)} BCLK cycles"),
    ]

def _decode_asi_cfg2(v):
    return [
        ("ASI_DAISY", "Daisy-chain" if bits(v,7,7) else "Common bus"),
        ("ASI_ERR", "Disabled" if bits(v,5,5) else "Enabled"),
        ("ASI_ERR_RCOV", "Disabled" if bits(v,4,4) else "Auto resume"),
    ]

def _make_asi_ch_decoder(ch):
    def dec(v):
        return [(f"CH{ch}_SLOT", decode_slot(v))]
    return dec

def _decode_mst_cfg0(v):
    return [
        ("MST_SLV_CFG", "Master" if bits(v,7,7) else "Slave"),
        ("AUTO_CLK_CFG", "Disabled (custom)" if bits(v,6,6) else "Auto"),
        ("AUTO_MODE_PLL_DIS", "PLL disabled" if bits(v,5,5) else "PLL enabled"),
        ("BCLK_FSYNC_GATE", "Gated" if bits(v,4,4) else "Not gated"),
        ("FS_MODE", "44.1 kHz base" if bits(v,3,3) else "48 kHz base"),
        ("MCLK_FREQ_SEL", MCLK_FREQ.get(bits(v,2,0), "?")),
    ]

def _decode_mst_cfg1(v):
    return [
        ("FS_RATE", FS_RATE.get(bits(v,7,4), "Reserved")),
        ("FS_BCLK_RATIO", FS_BCLK_RATIO.get(bits(v,3,0), "Reserved")),
    ]

def _decode_asi_sts(v):
    fs = bits(v,7,4)
    ratio = bits(v,3,0)
    return [
        ("FS_RATE_STS", FS_RATE.get(fs, "Invalid") if fs != 15 else "Invalid"),
        ("FS_RATIO_STS", FS_BCLK_RATIO.get(ratio, "Invalid") if ratio != 15 else "Invalid"),
    ]

def _decode_clk_src(v):
    return [
        ("DIS_PLL_SLV_CLK_SRC", "MCLK" if bits(v,7,7) else "BCLK"),
        ("MCLK_FREQ_SEL_MODE", "MCLK_RATIO_SEL" if bits(v,6,6) else "MCLK_FREQ_SEL"),
        ("MCLK_RATIO_SEL", MCLK_RATIO.get(bits(v,5,3), "?")),
    ]

def _make_gpio_cfg_decoder(gpio_num):
    def dec(v):
        return [
            (f"GPIO{gpio_num}_CFG", GPIO_CFG.get(bits(v,7,4), f"Reserved ({bits(v,7,4)})")),
            (f"GPIO{gpio_num}_DRV", GPIO_DRV.get(bits(v,2,0), f"Reserved ({bits(v,2,0)})")),
        ]
    return dec

def _make_gpi_cfg_decoder(gpi_num):
    def dec(v):
        return [(f"GPI{gpi_num}_CFG", GPI_CFG.get(bits(v,7,4), f"Reserved ({bits(v,7,4)})"))]
    return dec

def _decode_gpio_val(v):
    return [
        ("GPIO1_VAL", str(bits(v,7,7))),
        ("GPIO2_VAL", str(bits(v,6,6))),
        ("GPIO3_VAL", str(bits(v,5,5))),
    ]

def _decode_gpio_mon(v):
    return [
        ("GPIO1_MON", str(bits(v,7,7))),
        ("GPIO2_MON", str(bits(v,6,6))),
        ("GPIO3_MON", str(bits(v,5,5))),
        ("GPI1_MON", str(bits(v,4,4))),
        ("GPI2_MON", str(bits(v,3,3))),
    ]

def _decode_int_cfg(v):
    return [
        ("INT_POL", "Active high" if bits(v,7,7) else "Active low"),
        ("INT_EVENT", INT_EVENT.get(bits(v,6,5), "?")),
        ("PD_ON_FLT_CFG", str(bits(v,4,3))),
        ("LTCH_READ_CFG", str(bits(v,2,2))),
        ("PD_ON_FLT_RCV_CFG", str(bits(v,1,1))),
        ("LTCH_CLR_ON_READ", str(bits(v,0,0))),
    ]

def _decode_int_mask_bits(v, descriptions):
    """Generic interrupt mask decoder."""
    result = []
    for bit_pos, desc in descriptions.items():
        masked = "Masked" if bits(v, bit_pos, bit_pos) else "Unmasked"
        result.append((desc, masked))
    return result

def _decode_int_mask0(v):
    return _decode_int_mask_bits(v, {
        7: "ASI bus clock error",
        6: "PLL lock",
        5: "Over temperature",
        4: "MICBIAS over current",
    })

def _decode_int_mask1(v):
    return _decode_int_mask_bits(v, {
        7: "CH1 DC fault", 6: "CH2 DC fault", 5: "CH3 DC fault",
        4: "CH4 DC fault", 3: "CH5 DC fault", 2: "CH6 DC fault",
        1: "Short-to-VBAT (VBAT<MBIAS)",
    })

def _decode_int_mask2(v):
    return _decode_int_mask_bits(v, {
        7: "Open inputs", 6: "Inputs shorted", 5: "INxP shorted to GND",
        4: "INxM shorted to GND", 3: "INxP shorted to MICBIAS",
        2: "INxM shorted to MICBIAS", 1: "INxP shorted to VBAT",
        0: "INxM shorted to VBAT",
    })

def _decode_int_ltch0(v):
    faults = []
    if bits(v,7,7): faults.append("ASI bus clock error")
    if bits(v,6,6): faults.append("PLL lock")
    if bits(v,5,5): faults.append("Over temperature")
    if bits(v,4,4): faults.append("MICBIAS over current")
    return [("FAULTS", ", ".join(faults) if faults else "None")]

def _decode_chx_ltch(v):
    faults = []
    for i, ch in enumerate(range(1, 7)):
        if bits(v, 7-i, 7-i):
            faults.append(f"CH{ch}")
    if bits(v,1,1): faults.append("Short-to-VBAT(VBAT<MBIAS)")
    return [("CH_FAULTS", ", ".join(faults) if faults else "None")]

def _make_ch_ltch_decoder(ch):
    def dec(v):
        faults = []
        labels = ["Open input", "Inputs shorted", f"IN{ch}P->GND",
                  f"IN{ch}M->GND", f"IN{ch}P->MICBIAS", f"IN{ch}M->MICBIAS",
                  f"IN{ch}P->VBAT", f"IN{ch}M->VBAT"]
        for i, label in enumerate(labels):
            if bits(v, 7-i, 7-i):
                faults.append(label)
        return [(f"CH{ch}_FAULTS", ", ".join(faults) if faults else "None")]
    return dec

def _decode_int_mask3(v):
    return _decode_int_mask_bits(v, {
        7: "INxP over voltage", 6: "INxM over voltage",
        5: "MICBIAS high current", 4: "MICBIAS low current",
        3: "MICBIAS over voltage",
    })

def _decode_int_ltch1(v):
    faults = []
    for i, ch in enumerate(range(1, 7)):
        if bits(v, 7-i, 7-i):
            faults.append(f"CH{ch} IN{ch}P overvoltage")
    return [("OV_P_FAULTS", ", ".join(faults) if faults else "None")]

def _decode_int_ltch2(v):
    faults = []
    for i, ch in enumerate(range(1, 7)):
        if bits(v, 7-i, 7-i):
            faults.append(f"CH{ch} IN{ch}M overvoltage")
    return [("OV_M_FAULTS", ", ".join(faults) if faults else "None")]

def _decode_int_ltch3(v):
    faults = []
    if bits(v,7,7): faults.append("MICBIAS high current")
    if bits(v,6,6): faults.append("MICBIAS low current")
    if bits(v,5,5): faults.append("MICBIAS over voltage")
    return [("MBIAS_FAULTS", ", ".join(faults) if faults else "None")]

def _decode_mbdiag0(v):
    return [("MBIAS_HIGH_CURR_THRS", f"{v} (raw)")]

def _decode_mbdiag1(v):
    return [("MBIAS_LOW_CURR_THRS", f"{v} (raw)")]

def _decode_mbdiag2(v):
    return [
        ("PD_MBIAS_FAULT1", "PD on fault" if bits(v,7,7) else "No PD"),
        ("PD_MBIAS_FAULT2", "PD on fault" if bits(v,6,6) else "No PD"),
        ("PD_MBIAS_FAULT3", "PD on fault" if bits(v,5,5) else "No PD"),
        ("PD_MBIAS_FAULT4", "PD on fault" if bits(v,4,4) else "No PD"),
    ]

def _decode_bias_cfg(v):
    return [("BIAS_CFG", f"0x{v:02X} (raw)")]

def _make_ch_cfg0_decoder(ch):
    def dec(v):
        return [
            (f"CH{ch}_INTYP", INPUT_TYPE.get(bits(v,7,7), "?")),
            (f"CH{ch}_INSRC", INPUT_SRC.get(bits(v,6,5), f"Reserved ({bits(v,6,5)})")),
            (f"CH{ch}_DC", INPUT_COUPLING.get(bits(v,4,4), "?")),
            (f"CH{ch}_MIC_IN_RANGE", INPUT_RANGE.get(bits(v,3,3), "?")),
            (f"CH{ch}_PGA_CFG", PGA_CFG.get(bits(v,2,1), f"Reserved ({bits(v,2,1)})")),
            (f"CH{ch}_AGCEN", AGC_EN.get(bits(v,0,0), "?")),
        ]
    return dec

def _make_ch_cfg1_decoder(ch):
    def dec(v):
        return [(f"CH{ch}_GAIN", decode_gain(v))]
    return dec

def _make_ch_cfg2_decoder(ch):
    def dec(v):
        return [(f"CH{ch}_DVOL", decode_dvol(v))]
    return dec

def _make_ch_cfg3_decoder(ch):
    def dec(v):
        return [(f"CH{ch}_GCAL", decode_gcal(v))]
    return dec

def _make_ch_cfg4_decoder(ch):
    def dec(v):
        return [(f"CH{ch}_PCAL", f"{v} modclk cycles" if v else "None")]
    return dec

def _decode_diag_cfg0(v):
    result = []
    for i, ch in enumerate(range(1, 7)):
        result.append((f"CH{ch}_DIAG_EN", "Enabled" if bits(v, 7-i, 7-i) else "Disabled"))
    result.append(("INCL_SE_INM", "Included" if bits(v,1,1) else "Excluded"))
    result.append(("INCL_AC_COUP", "Included" if bits(v,0,0) else "Excluded"))
    return result

def _decode_diag_cfg1(v):
    return [
        ("DIAG_SHT_TERM", f"{bits(v,7,4) * 30} mV threshold"),
        ("DIAG_SHT_VBAT_IN", f"{bits(v,3,0) * 30} mV threshold"),
    ]

def _decode_diag_cfg2(v):
    return [
        ("DIAG_SHT_GND", f"{bits(v,7,4) * 60} mV threshold"),
        ("DIAG_SHT_MICBIAS", f"{bits(v,3,0) * 30} mV threshold"),
    ]

REP_RATE = {0: "Continuous", 1: "1 ms", 2: "4 ms", 3: "8 ms"}
FAULT_DBNCE = {0: "16 counts", 1: "8 counts", 2: "4 counts", 3: "None"}

def _decode_diag_cfg3(v):
    return [
        ("REP_RATE", REP_RATE.get(bits(v,7,6), "?")),
        ("FAULT_DBNCE_SEL", FAULT_DBNCE.get(bits(v,3,2), "?")),
        ("VSHORT_DBNCE", "8 counts" if bits(v,1,1) else "16 counts"),
        ("DIAG_2X_THRES", "2x scaled" if bits(v,0,0) else "Normal"),
    ]

def _decode_diag_cfg4(v):
    avg = bits(v,7,6)
    avg_str = {0: "Disabled", 1: "50/50 old/new", 2: "75/25 old/new"}.get(avg, "Reserved")
    return [
        ("DIAG_MOV_AVG", avg_str),
        ("MOV_AVG_DIS_MBIAS_LOAD", "Forced off" if bits(v,5,5) else "As configured"),
        ("MOV_AVG_DIS_TEMP_SENS", "Forced off" if bits(v,4,4) else "As configured"),
    ]

def _decode_dsp_cfg0(v):
    return [
        ("DECI_FILT", DECI_FILT.get(bits(v,5,4), "Reserved")),
        ("CH_SUM", CH_SUM.get(bits(v,3,2), "Reserved")),
        ("HPF_SEL", HPF_SEL.get(bits(v,1,0), "?")),
    ]

def _decode_dsp_cfg1(v):
    return [
        ("DVOL_GANG", "All use CH1 DVOL" if bits(v,7,7) else "Independent"),
        ("BIQUAD_CFG", BIQUAD_CFG.get(bits(v,6,5), "?")),
        ("SOFT_STEP", "Disabled" if bits(v,4,4) else "Enabled"),
        ("AGC_SEL", "Per-channel" if bits(v,3,3) else "Reserved (write 1!)"),
    ]

def _decode_agc_cfg0(v):
    lvl = bits(v,7,4)
    target_db = -6 - (lvl * 2)
    gain = bits(v,3,0)
    max_db = 3 + (gain * 3) if gain <= 13 else "Reserved"
    return [
        ("AGC_LVL", f"{target_db} dB target"),
        ("AGC_MAXGAIN", f"{max_db} dB max" if isinstance(max_db, int) else max_db),
    ]

def _decode_in_ch_en(v):
    result = []
    for i, ch in enumerate(range(1, 7)):
        result.append((f"IN_CH{ch}_EN", "Enabled" if bits(v, 7-i, 7-i) else "Disabled"))
    return result

def _decode_asi_out_ch_en(v):
    result = []
    for i, ch in enumerate(range(1, 7)):
        result.append((f"ASI_OUT_CH{ch}_EN", "Enabled" if bits(v, 7-i, 7-i) else "Tri-state"))
    return result

def _decode_pwr_cfg(v):
    dyn_maxch = {0: "CH1-2", 1: "CH1-4", 2: "CH1-6"}
    return [
        ("MICBIAS_PDZ", "Powered up" if bits(v,7,7) else "Powered down"),
        ("ADC_PDZ", "Powered up" if bits(v,6,6) else "Powered down"),
        ("PLL_PDZ", "Powered up" if bits(v,5,5) else "Powered down"),
        ("DYN_CH_PUPD_EN", "Enabled" if bits(v,4,4) else "Disabled"),
        ("DYN_MAXCH_SEL", dyn_maxch.get(bits(v,3,2), "?")),
    ]

def _decode_dev_sts0(v):
    result = []
    for i, ch in enumerate(range(1, 7)):
        result.append((f"CH{ch}_STATUS", "ON" if bits(v, 7-i, 7-i) else "OFF"))
    return result

def _decode_dev_sts1(v):
    return [
        ("MODE_STS", MODE_STS.get(bits(v,7,5), f"Unknown ({bits(v,7,5)})")),
        ("BOOST_STS", "ON" if bits(v,4,4) else "OFF"),
        ("MBIAS_STS", "ON" if bits(v,3,3) else "OFF"),
        ("CHx_PD_FLT_STS", "Fault PD" if bits(v,2,2) else "OK"),
        ("ALL_CHx_PD_FLT_STS", "Fault PD" if bits(v,1,1) else "OK"),
    ]

def _decode_i2c_cksum(v):
    return [("I2C_CKSUM", f"0x{v:02X}")]


# ---------------------------------------------------------------------------
# Master register map: address -> (name, decoder)
# ---------------------------------------------------------------------------

REGISTER_MAP = {
    0x00: ("PAGE_CFG", _decode_page_cfg),
    0x01: ("SW_RESET", _decode_sw_reset),
    0x02: ("SLEEP_CFG", _decode_sleep_cfg),
    0x05: ("SHDN_CFG", _decode_shdn_cfg),
    0x07: ("ASI_CFG0", _decode_asi_cfg0),
    0x08: ("ASI_CFG1", _decode_asi_cfg1),
    0x09: ("ASI_CFG2", _decode_asi_cfg2),
    0x0B: ("ASI_CH1", _make_asi_ch_decoder(1)),
    0x0C: ("ASI_CH2", _make_asi_ch_decoder(2)),
    0x0D: ("ASI_CH3", _make_asi_ch_decoder(3)),
    0x0E: ("ASI_CH4", _make_asi_ch_decoder(4)),
    0x0F: ("ASI_CH5", _make_asi_ch_decoder(5)),
    0x10: ("ASI_CH6", _make_asi_ch_decoder(6)),
    0x13: ("MST_CFG0", _decode_mst_cfg0),
    0x14: ("MST_CFG1", _decode_mst_cfg1),
    0x15: ("ASI_STS", _decode_asi_sts),
    0x16: ("CLK_SRC", _decode_clk_src),
    0x21: ("GPIO_CFG0", _make_gpio_cfg_decoder(1)),
    0x22: ("GPIO_CFG1", _make_gpio_cfg_decoder(2)),
    0x23: ("GPIO_CFG2", _make_gpio_cfg_decoder(3)),
    0x24: ("GPI_CFG0", _make_gpi_cfg_decoder(1)),
    0x25: ("GPI_CFG1", _make_gpi_cfg_decoder(2)),
    0x26: ("GPIO_VAL", _decode_gpio_val),
    0x27: ("GPIO_MON", _decode_gpio_mon),
    0x28: ("INT_CFG", _decode_int_cfg),
    0x29: ("INT_MASK0", _decode_int_mask0),
    0x2A: ("INT_MASK1", _decode_int_mask1),
    0x2B: ("INT_MASK2", _decode_int_mask2),
    0x2C: ("INT_LTCH0", _decode_int_ltch0),
    0x2D: ("CHx_LTCH", _decode_chx_ltch),
    0x2E: ("CH1_LTCH", _make_ch_ltch_decoder(1)),
    0x2F: ("CH2_LTCH", _make_ch_ltch_decoder(2)),
    0x30: ("CH3_LTCH", _make_ch_ltch_decoder(3)),
    0x31: ("CH4_LTCH", _make_ch_ltch_decoder(4)),
    0x32: ("CH5_LTCH", _make_ch_ltch_decoder(5)),
    0x33: ("CH6_LTCH", _make_ch_ltch_decoder(6)),
    0x34: ("INT_MASK3", _decode_int_mask3),
    0x35: ("INT_LTCH1", _decode_int_ltch1),
    0x36: ("INT_LTCH2", _decode_int_ltch2),
    0x37: ("INT_LTCH3", _decode_int_ltch3),
    0x38: ("MBDIAG_CFG0", _decode_mbdiag0),
    0x39: ("MBDIAG_CFG1", _decode_mbdiag1),
    0x3A: ("MBDIAG_CFG2", _decode_mbdiag2),
    0x3B: ("BIAS_CFG", _decode_bias_cfg),
}

# Channel config registers: 5 regs per channel starting at 0x3C
for _ch in range(1, 7):
    _base = 0x3C + (_ch - 1) * 5
    REGISTER_MAP[_base]     = (f"CH{_ch}_CFG0", _make_ch_cfg0_decoder(_ch))
    REGISTER_MAP[_base + 1] = (f"CH{_ch}_CFG1", _make_ch_cfg1_decoder(_ch))
    REGISTER_MAP[_base + 2] = (f"CH{_ch}_CFG2", _make_ch_cfg2_decoder(_ch))
    REGISTER_MAP[_base + 3] = (f"CH{_ch}_CFG3", _make_ch_cfg3_decoder(_ch))
    REGISTER_MAP[_base + 4] = (f"CH{_ch}_CFG4", _make_ch_cfg4_decoder(_ch))

REGISTER_MAP.update({
    0x64: ("DIAG_CFG0", _decode_diag_cfg0),
    0x65: ("DIAG_CFG1", _decode_diag_cfg1),
    0x66: ("DIAG_CFG2", _decode_diag_cfg2),
    0x67: ("DIAG_CFG3", _decode_diag_cfg3),
    0x68: ("DIAG_CFG4", _decode_diag_cfg4),
    0x6B: ("DSP_CFG0", _decode_dsp_cfg0),
    0x6C: ("DSP_CFG1", _decode_dsp_cfg1),
    0x70: ("AGC_CFG0", _decode_agc_cfg0),
    0x73: ("IN_CH_EN", _decode_in_ch_en),
    0x74: ("ASI_OUT_CH_EN", _decode_asi_out_ch_en),
    0x75: ("PWR_CFG", _decode_pwr_cfg),
    0x76: ("DEV_STS0", _decode_dev_sts0),
    0x77: ("DEV_STS1", _decode_dev_sts1),
    0x7E: ("I2C_CKSUM", _decode_i2c_cksum),
})


# ---------------------------------------------------------------------------
# Read and parse regdump
# ---------------------------------------------------------------------------

def read_regdump():
    """Read page 0 regdump from sysfs. Returns dict {addr_int: value_int}."""
    # Select page 0 on device 0
    subprocess.run(
        f"echo '{DEV_NO} 0x{PAGE:02X}' | sudo tee {REGDUMP_PATH}",
        shell=True, check=True, capture_output=True,
    )

    result = subprocess.run(
        ["cat", REGDUMP_PATH],
        check=True, capture_output=True, text=True,
    )

    regs = {}
    for line in result.stdout.splitlines():
        # Format: No-X:P0xYYR0xZZ:0xAA
        line = line.strip()
        if not line.startswith("No-"):
            continue
        # e.g. "No-0:P0x00R0x3D:0x50"
        try:
            parts = line.split(":")
            reg_part = parts[1]   # "P0x00R0x3D"
            val_part = parts[2]   # "0x50"
            reg_hex = reg_part.split("R")[1]  # "0x3D"
            addr = int(reg_hex, 16)
            val = int(val_part, 16)
            regs[addr] = val
        except (IndexError, ValueError):
            continue
    return regs


def print_decoded(regs):
    """Print decoded register values grouped by section."""
    sections = [
        ("SYSTEM", [0x00, 0x01, 0x02, 0x05]),
        ("ASI BUS CONFIG", [0x07, 0x08, 0x09]),
        ("ASI SLOT ASSIGNMENTS", [0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10]),
        ("CLOCK CONFIG", [0x13, 0x14, 0x15, 0x16]),
        ("GPIO", [0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27]),
        ("INTERRUPTS", [0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x34, 0x35, 0x36, 0x37]),
        ("CHANNEL FAULT STATUS", [0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33]),
        ("MICBIAS DIAGNOSTICS", [0x38, 0x39, 0x3A, 0x3B]),
    ]

    for ch in range(1, 7):
        base = 0x3C + (ch - 1) * 5
        sections.append((f"CHANNEL {ch} CONFIG", [base, base+1, base+2, base+3, base+4]))

    sections.extend([
        ("INPUT DIAGNOSTICS", [0x64, 0x65, 0x66, 0x67, 0x68]),
        ("DSP", [0x6B, 0x6C]),
        ("AGC", [0x70]),
        ("POWER & ENABLE", [0x73, 0x74, 0x75]),
        ("DEVICE STATUS", [0x76, 0x77]),
        ("I2C CHECKSUM", [0x7E]),
    ])

    for section_name, addrs in sections:
        has_data = any(a in regs for a in addrs)
        if not has_data:
            continue

        print(f"\n{'=' * 60}")
        print(f"  {section_name}")
        print(f"{'=' * 60}")

        for addr in addrs:
            if addr not in regs:
                continue
            val = regs[addr]
            if addr in REGISTER_MAP:
                name, decoder = REGISTER_MAP[addr]
                fields = decoder(val)
                print(f"\n  0x{addr:02X}  {name}  = 0x{val:02X}")
                for fname, fval in fields:
                    print(f"         {fname}: {fval}")
            else:
                print(f"\n  0x{addr:02X}  (unknown)  = 0x{val:02X}")


def main():
    print("PCM6260-Q1 Page 0 Register Decoder")
    print("Reading from sysfs...")
    regs = read_regdump()
    if not regs:
        print("ERROR: No registers read. Is the driver loaded?", file=sys.stderr)
        sys.exit(1)
    print(f"Read {len(regs)} registers.")
    print_decoded(regs)


if __name__ == "__main__":
    main()
