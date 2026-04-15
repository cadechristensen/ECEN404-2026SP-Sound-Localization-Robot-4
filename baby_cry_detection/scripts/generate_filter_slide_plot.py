"""Generate a before/after spectrogram comparison plot for the noise filtering slide.

Loads a raw 48 kHz multichannel cry recording and applies the deployment NoiseFilter
(the real filter_for_localization() pipeline: phase-preserving HP + BP + spectral sub).

Note: record_samtry.py saves files named *_filtered.wav but those are actually
unfiltered cry-region extractions (line 384 comment: "Save unfiltered cry regions").
The real filter output only exists in memory and is passed directly to SL.
So we apply the same NoiseFilter here to visualize what SL actually receives.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Add Pi_Integration and deployment dirs to path so we can import real production code
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PI_INTEGRATION_DIR = REPO_ROOT / "Pi_Integration"
DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent / "deployment"
sys.path.insert(0, str(PI_INTEGRATION_DIR))
sys.path.insert(0, str(DEPLOYMENT_DIR))

from audio_filtering import NoiseFilter  # noqa: E402
from FunctionCalls_BCD import BabyCryDetection  # noqa: E402

_RAW_FILE_ENV = "FILTER_SLIDE_RAW_FILE"
_raw = os.environ.get(_RAW_FILE_ENV)
if _raw is None:
    raise SystemExit(
        f"Set {_RAW_FILE_ENV} to a 4-channel 48 kHz .wav path before running this script."
    )
RAW_FILE = Path(_raw)
OUTPUT_PNG = Path(__file__).resolve().parent.parent / "docs" / "filter_before_after.png"
OUTPUT_WAVEFORM_PNG = (
    Path(__file__).resolve().parent.parent / "docs" / "filter_waveform_before_after.png"
)
OUTPUT_FLOW_PNG = Path(__file__).resolve().parent.parent / "docs" / "sl_audio_flow.png"


def compute_stft_magnitude(audio: np.ndarray) -> np.ndarray:
    """Compute a linear-magnitude STFT (db conversion happens later with shared ref)."""
    import librosa

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    stft = librosa.stft(audio.astype(np.float32), n_fft=2048, hop_length=512)
    return np.abs(stft)


def detect_cry_regions_bcd(audio_48k: np.ndarray, sr: int = 48000) -> list[tuple[float, float]]:
    """Run the real production BCD detector to get cry regions.

    Mirrors exactly what Pi_Integration/main.py does:
      1. Resample 48 kHz audio to 16 kHz (model rate)
      2. Call BabyCryDetection.detect_from_audio()
      3. Return detection.cry_regions (list of (start_sec, end_sec) tuples)

    Uses the same default model (config_pi.MODEL_PATH = model_best_2026-03-14_run16_d384.pth).
    """
    import librosa

    print("  Resampling 48 kHz -> 16 kHz for BCD model")
    if audio_48k.ndim == 2:
        # (samples, channels) -> resample each channel
        audio_16k = np.stack(
            [
                librosa.resample(audio_48k[:, c].astype(np.float32), orig_sr=sr, target_sr=16000)
                for c in range(audio_48k.shape[1])
            ],
            axis=1,
        )
    else:
        audio_16k = librosa.resample(audio_48k.astype(np.float32), orig_sr=sr, target_sr=16000)

    print("  Initializing BabyCryDetection (loads production model)")
    bcd = BabyCryDetection(device_index=None)

    print("  Running detect_from_audio()")
    detection = bcd.detect_from_audio(audio_16k)
    print(f"    is_cry={detection.is_cry}, confidence={detection.confidence:.2%}")

    if detection.cry_regions is None:
        return []
    return [(float(s), float(e)) for s, e in detection.cry_regions]


def main() -> None:
    print(f"Loading raw audio: {RAW_FILE.name}")
    audio, sr = sf.read(str(RAW_FILE))
    print(f"  shape={audio.shape}, sr={sr}, duration={len(audio)/sr:.2f}s")

    if sr != 48000:
        raise RuntimeError(f"Expected 48 kHz, got {sr}")
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise RuntimeError(f"Expected 4-channel audio, got shape {audio.shape}")

    print("Applying phase-preserving NoiseFilter (HP 100 Hz + BP 100-3000 Hz + spectral sub)")
    print("  [This is the same filter as bcd.filter_for_localization() — the real SL feed]")
    noise_filter = NoiseFilter(
        sample_rate=48000,
        highpass_cutoff=100,
        bandpass_low=100,
        bandpass_high=3000,
    )
    filtered = noise_filter.filter_audio(audio.astype(np.float32))
    print(f"  filtered shape={filtered.shape}")

    print("Computing STFT magnitudes")
    mag_raw = compute_stft_magnitude(audio)
    mag_filt = compute_stft_magnitude(filtered)

    # Shared dB reference = raw signal's max. This is the key change that makes
    # the filtering visually dramatic — the filtered panel will show real energy
    # loss instead of being re-normalized to its own max.
    import librosa

    shared_ref = mag_raw.max()
    spec_raw_db = librosa.amplitude_to_db(mag_raw, ref=shared_ref)
    spec_filt_db = librosa.amplitude_to_db(mag_filt, ref=shared_ref)
    vmin, vmax = -80.0, 0.0

    print("Running production BCD detector to get real cry regions")
    cry_regions = detect_cry_regions_bcd(audio, sr)
    total_cry = sum(end - start for start, end in cry_regions)
    min_cry_duration = 5.0  # matches Pi_Integration/main.py MIN_CRY_DURATION
    print(f"  found {len(cry_regions)} cry region(s): {cry_regions}")
    print(f"  total cry duration: {total_cry:.2f}s (min required: {min_cry_duration}s)")
    if total_cry < min_cry_duration:
        print(
            f"  WARNING: total cry duration < {min_cry_duration}s — "
            "production system would REJECT this recording"
        )

    print("Rendering figure")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    # White-background color scheme
    TEXT_COLOR = "#1A1A1A"
    SPINE_COLOR = "#888888"
    CUTOFF_COLOR = "#E65100"  # orange for filter cutoff lines
    CRY_REGION_COLOR = "#C62828"  # red for cry region boxes

    titles = ["Raw Audio (Unfiltered)", "After Phase-Preserving Filter (SL input)"]
    specs = [spec_raw_db, spec_filt_db]

    for ax, title, spec in zip(axes, titles, specs):
        import librosa.display

        img = librosa.display.specshow(
            spec,
            sr=sr,
            hop_length=512,
            x_axis="time",
            y_axis="linear",
            ax=ax,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylim(0, sr // 2)  # Explicit Nyquist (24 kHz for 48 kHz audio)
        ax.set_yticks([0, 3000, 6000, 12000, 18000, 24000])
        ax.set_yticklabels(["0", "3k", "6k", "12k", "18k", "24k"])
        ax.set_title(title, color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=11)
        ax.set_ylabel("Frequency (Hz)", color=TEXT_COLOR, fontsize=11)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)

        # Filter cutoffs at 100 Hz (HP) and 3000 Hz (BP upper)
        ax.axhline(y=100, color=CUTOFF_COLOR, linestyle=":", linewidth=1.4, alpha=0.9)
        ax.axhline(y=3000, color=CUTOFF_COLOR, linestyle=":", linewidth=1.4, alpha=0.9)

        # Detected cry regions (red boxes)
        for start_sec, end_sec in cry_regions:
            rect = plt.Rectangle(
                (start_sec, 0),
                end_sec - start_sec,
                sr // 2,
                linewidth=2.0,
                edgecolor=CRY_REGION_COLOR,
                facecolor="none",
                linestyle="-",
            )
            ax.add_patch(rect)

    # Legend (single, placed above the plots)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CUTOFF_COLOR,
            linestyle=":",
            linewidth=1.8,
            label="Filter cutoffs (100 Hz & 3 kHz)",
        ),
        Patch(
            edgecolor=CRY_REGION_COLOR,
            facecolor="none",
            linewidth=2.0,
            label="Detected cry regions",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        labelcolor=TEXT_COLOR,
        fontsize=11,
    )

    # Single colorbar for both
    cbar = fig.colorbar(img, ax=axes, format="%+2.0f dB", pad=0.02, shrink=0.85)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)
    cbar.outline.set_edgecolor(SPINE_COLOR)
    cbar.set_label("Power (dB)", color=TEXT_COLOR, fontsize=11)

    fig.suptitle(
        "Noise Filtering: Phase-Preserving Pipeline (48 kHz, feeds Sound Localization)",
        color=TEXT_COLOR,
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_PNG,
        bbox_inches="tight",
        dpi=150,
        facecolor="#FFFFFF",
    )
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_PNG}")

    # --- Second plot: waveform comparison -----------------------------------
    render_waveform_plot(audio, filtered, sr, cry_regions)

    # --- Third plot: SL audio flow diagram (skip — regenerated separately) ---
    # render_sl_audio_flow_diagram()


def render_waveform_plot(
    raw: np.ndarray,
    filtered: np.ndarray,
    sr: int,
    cry_regions: list[tuple[float, float]],
) -> None:
    """Render a stacked waveform comparison showing raw vs filtered amplitude."""
    print("\nRendering waveform comparison figure")

    # Mono-mix for display
    raw_mono = raw.mean(axis=1) if raw.ndim == 2 else raw
    filt_mono = filtered.mean(axis=1) if filtered.ndim == 2 else filtered

    # Match lengths
    n = min(len(raw_mono), len(filt_mono))
    raw_mono = raw_mono[:n]
    filt_mono = filt_mono[:n]
    times = np.arange(n) / sr

    # Shared y-limits so the amplitude difference is visible
    abs_max = max(np.abs(raw_mono).max(), np.abs(filt_mono).max())
    y_lim = abs_max * 1.05

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    # White-background color scheme
    TEXT_COLOR = "#1A1A1A"
    SPINE_COLOR = "#888888"
    GRID_COLOR = "#CCCCCC"
    CRY_REGION_COLOR = "#C62828"

    panels = [
        ("Raw Audio (Unfiltered)", raw_mono, "#1565C0"),  # blue
        ("After Phase-Preserving Filter (SL input)", filt_mono, "#2E7D32"),  # green
    ]

    for ax, (title, signal, color) in zip(axes, panels):
        ax.plot(times, signal, color=color, linewidth=0.5, alpha=0.9)
        ax.fill_between(times, signal, 0, color=color, alpha=0.15)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel("Amplitude", color=TEXT_COLOR, fontsize=11)
        ax.tick_params(colors=TEXT_COLOR)
        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
        ax.grid(True, alpha=0.3, color=GRID_COLOR)
        ax.axhline(0, color=SPINE_COLOR, linewidth=0.8, alpha=0.5)

        # Mark cry regions
        for start_sec, end_sec in cry_regions:
            ax.axvspan(start_sec, end_sec, alpha=0.12, color=CRY_REGION_COLOR, zorder=0)

        # Annotate with RMS
        rms = np.sqrt((signal**2).mean())
        ax.text(
            0.99,
            0.95,
            f"RMS: {rms:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=TEXT_COLOR,
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="#FFFFFF", edgecolor=SPINE_COLOR, boxstyle="round,pad=0.3"),
        )

    axes[-1].set_xlabel("Time (s)", color=TEXT_COLOR, fontsize=11)

    # Legend
    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(facecolor=CRY_REGION_COLOR, alpha=0.2, label="BCD cry regions")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        labelcolor=TEXT_COLOR,
        fontsize=11,
    )

    fig.suptitle(
        "Noise Filtering: Waveform Comparison (raw vs filtered)",
        color=TEXT_COLOR,
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )
    fig.tight_layout()

    OUTPUT_WAVEFORM_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_WAVEFORM_PNG,
        bbox_inches="tight",
        dpi=150,
        facecolor="#FFFFFF",
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT_WAVEFORM_PNG}")


def render_sl_audio_flow_diagram() -> None:
    """Render a block diagram of the dual-audio Sound Localization pipeline.

    Shows that after BCD detects a cry and extracts the cry regions, the audio
    forks into two parallel processing paths:
      - Filtered audio -> DOAnet -> direction (angle)
      - Raw audio      -> ML distance regressor -> distance (feet)
    Then both results combine into the NAV command sent to the ESP32.
    """
    print("\nRendering SL audio flow diagram")

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # Catppuccin Mocha palette
    BG = "#1E1E2E"
    BOX_BG = "#313244"
    BOX_EDGE = "#585B70"
    TEXT = "#CDD6F4"
    TITLE = "#F5E0DC"
    BLUE = "#89B4FA"  # input / BCD
    GREEN = "#A6E3A1"  # filtered path
    YELLOW = "#F9E2AF"  # raw path
    MAUVE = "#CBA6F7"  # output
    RED = "#F38BA8"  # fork highlight

    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")

    def add_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str = "",
        edge_color: str = BOX_EDGE,
        title_color: str = TEXT,
        box_face: str = BOX_BG,
    ) -> None:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=2.0,
            edgecolor=edge_color,
            facecolor=box_face,
        )
        ax.add_patch(patch)
        if subtitle:
            ax.text(
                x,
                y + 0.15,
                title,
                ha="center",
                va="center",
                color=title_color,
                fontsize=12,
                fontweight="bold",
            )
            ax.text(
                x,
                y - 0.28,
                subtitle,
                ha="center",
                va="center",
                color=TEXT,
                fontsize=9,
                alpha=0.85,
            )
        else:
            ax.text(
                x,
                y,
                title,
                ha="center",
                va="center",
                color=title_color,
                fontsize=12,
                fontweight="bold",
            )

    def add_arrow(x1, y1, x2, y2, color=TEXT, label: str = "", label_side: str = "right") -> None:
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.0,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
        ax.add_patch(arrow)
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            dx = 0.35 if label_side == "right" else -0.35
            ha = "left" if label_side == "right" else "right"
            ax.text(
                mx + dx,
                my,
                label,
                ha=ha,
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
                fontfamily="monospace",
            )

    # --- Stage 1: Raw capture ---------------------------------------------
    add_box(
        x=7,
        y=10,
        w=5.5,
        h=0.9,
        title="4-channel mic array @ 48 kHz",
        subtitle="Raw audio buffer (48 kHz, 4 channels)",
        edge_color=BLUE,
        title_color=BLUE,
    )

    # --- Stage 2: BCD detection -------------------------------------------
    add_box(
        x=7,
        y=8.5,
        w=5.5,
        h=0.9,
        title="Baby Cry Detection (BCD)",
        subtitle="CNN-Transformer @ 16 kHz, confidence ≥ 0.92",
        edge_color=BLUE,
        title_color=BLUE,
    )

    # --- Stage 3: Extract cry regions -------------------------------------
    add_box(
        x=7,
        y=7,
        w=5.5,
        h=0.9,
        title="Extract Cry Regions (48 kHz)",
        subtitle="Concatenate regions, require ≥ 5.0 s total",
        edge_color=BOX_EDGE,
        title_color=TEXT,
    )

    # --- Fork label -------------------------------------------------------
    ax.text(
        7,
        5.85,
        "AUDIO FORKS",
        ha="center",
        va="center",
        color=RED,
        fontsize=11,
        fontweight="bold",
        fontfamily="monospace",
    )
    ax.text(
        7,
        5.55,
        "(two streams, different purposes)",
        ha="center",
        va="center",
        color=RED,
        fontsize=9,
        alpha=0.85,
        fontstyle="italic",
    )

    # --- Stage 4a: Filter (left branch) -----------------------------------
    add_box(
        x=3.5,
        y=4.2,
        w=5.2,
        h=1.0,
        title="Phase-Preserving Filter",
        subtitle="HP 100 Hz  ⟶  BP 100-3000 Hz  ⟶  Spectral Sub",
        edge_color=GREEN,
        title_color=GREEN,
    )

    # --- Stage 4b: Raw pass-through (right branch) ------------------------
    add_box(
        x=10.5,
        y=4.2,
        w=5.2,
        h=1.0,
        title="Raw Audio (no filtering)",
        subtitle="Original 48 kHz, full frequency range",
        edge_color=YELLOW,
        title_color=YELLOW,
    )

    # --- Stage 5a: DOAnet -------------------------------------------------
    add_box(
        x=3.5,
        y=2.5,
        w=5.2,
        h=1.0,
        title="DOAnet (Direction)",
        subtitle="CRNN with attention → angle (degrees)",
        edge_color=GREEN,
        title_color=GREEN,
    )

    # --- Stage 5b: Distance regressor -------------------------------------
    add_box(
        x=10.5,
        y=2.5,
        w=5.2,
        h=1.0,
        title="ML Distance Regressor",
        subtitle="Trained on raw → distance (feet)",
        edge_color=YELLOW,
        title_color=YELLOW,
    )

    # --- Stage 6: NAV output ----------------------------------------------
    add_box(
        x=7,
        y=0.7,
        w=6.5,
        h=1.0,
        title="NAV angle=<deg> dist_ft=<ft>",
        subtitle="→ ESP32 robot via UART",
        edge_color=MAUVE,
        title_color=MAUVE,
    )

    # --- Arrows ------------------------------------------------------------
    # Vertical main pipeline
    add_arrow(7, 9.55, 7, 8.95)
    add_arrow(7, 8.05, 7, 7.45)
    add_arrow(7, 6.55, 7, 6.15)

    # Fork split
    add_arrow(7, 5.3, 3.5, 4.75, color=GREEN, label="filtered", label_side="left")
    add_arrow(7, 5.3, 10.5, 4.75, color=YELLOW, label="raw", label_side="right")

    # Down into DOAnet / distance
    add_arrow(3.5, 3.7, 3.5, 3.05, color=GREEN)
    add_arrow(10.5, 3.7, 10.5, 3.05, color=YELLOW)

    # Converge into NAV
    add_arrow(3.5, 1.95, 6.2, 1.25, color=GREEN, label="angle", label_side="left")
    add_arrow(10.5, 1.95, 7.8, 1.25, color=YELLOW, label="dist_ft", label_side="right")

    # --- Title & caption --------------------------------------------------
    fig.suptitle(
        "Sound Localization Audio Flow: Two Streams, One Decision",
        color=TITLE,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    ax.text(
        7,
        10.75,
        "Filtered audio preserves phase for direction estimation;\n"
        "raw audio retains amplitude envelope for distance regression.",
        ha="center",
        va="center",
        color=TITLE,
        fontsize=10,
        fontstyle="italic",
        alpha=0.9,
    )

    OUTPUT_FLOW_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_FLOW_PNG,
        bbox_inches="tight",
        dpi=150,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"Saved: {OUTPUT_FLOW_PNG}")


if __name__ == "__main__":
    main()
