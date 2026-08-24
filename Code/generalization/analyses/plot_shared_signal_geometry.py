"""Plot the shared-space compactness test and spectrum mediation result."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[2]
RESULT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "shared_signal_geometry"
FIGURE_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "figures"
OUT = FIGURE_DIR / "fig_shared_signal_geometry.png"

R1_COLOR = "#4D5156"
R2_COLOR = "#00897B"
DAY_COLOR = "#F2994A"
MEDIATION_COLOR = "#C44536"
ZERO_COLOR = "#8A9097"


def lookup(frame: pd.DataFrame, metric: str) -> pd.Series:
    row = frame[frame.metric == metric]
    if len(row) != 1:
        raise RuntimeError(f"expected one row for {metric}, found {len(row)}")
    return row.iloc[0]


def day_values(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(";")], dtype=float)


def paired_means(ax, row, title: str, ylabel: str):
    values = [row.r1_mean, row.r2_mean]
    ax.plot([0, 1], values, color="#B8BDC3", linewidth=1.8, zorder=1)
    ax.scatter(0, values[0], s=85, color=R1_COLOR, zorder=2, label="R1")
    ax.scatter(1, values[1], s=85, color=R2_COLOR, zorder=2, label="R2")
    difference = row.r2_minus_r1
    ax.text(
        0.5,
        0.04,
        f"R2 - R1 = {difference:+.3f}\n"
        f"[{row.descriptive_boot_lo:+.3f}, {row.descriptive_boot_hi:+.3f}]",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_xticks([0, 1], ["R1", "R2"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(1.50, 1.70)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


def effect_panel(ax, spectrum: pd.DataFrame):
    metrics = [
        ("movement_residual_power", "Residual power", "higher is less clear"),
        ("movement_encoding_fraction", "Movement fraction", "higher is clearer"),
        ("movement_trace_snr", "Trace SNR", "higher is clearer"),
    ]
    y = np.arange(len(metrics))[::-1]
    for position, (metric, _, _) in zip(y, metrics):
        row = lookup(spectrum, metric)
        ax.hlines(
            position,
            row.descriptive_boot_lo,
            row.descriptive_boot_hi,
            color=R2_COLOR,
            linewidth=3,
        )
        ax.scatter(row.r2_minus_r1, position, s=70, color=R2_COLOR, zorder=3)
        days = day_values(row.three_r2_day_differences)
        ax.scatter(
            days,
            np.full(3, position) + np.linspace(-0.13, 0.13, 3),
            s=22,
            color=DAY_COLOR,
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
    ax.axvline(0, color=ZERO_COLOR, linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"{label}\n({direction})" for _, label, direction in metrics])
    ax.set_xlabel("R2 - R1")
    ax.set_title("C. R2 is not clearer in the shared space", loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.22)


def mediation_panel(ax, mediation: pd.DataFrame):
    names = [
        ("movement_signal_spectrum", "Match signal spectrum"),
        ("movement_residual_spectrum", "Match residual spectrum"),
        ("movement_both_spectra", "Match both spectra"),
    ]
    y = np.arange(len(names))[::-1]
    rows = mediation[mediation.rank_scope == "all_shared_ranks"]
    for position, (mode, _) in zip(y, names):
        row = rows[rows["mode"] == mode].iloc[0]
        ax.hlines(
            position,
            row.descriptive_boot_lo,
            row.descriptive_boot_hi,
            color=MEDIATION_COLOR,
            linewidth=3,
        )
        ax.scatter(
            row.observed_minus_random,
            position,
            s=70,
            color=MEDIATION_COLOR,
            zorder=3,
        )
        days = day_values(row.three_r2_day_adjusted_effects)
        ax.scatter(
            days,
            np.full(3, position) + np.linspace(-0.13, 0.13, 3),
            s=22,
            color=DAY_COLOR,
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
    ax.axvline(0, color=ZERO_COLOR, linestyle="--", linewidth=1)
    ax.set_yticks(y, [label for _, label in names])
    ax.set_xlabel("Gap closure beyond same-strength random axes")
    ax.set_title("D. Spectrum matching does not close the gap", loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.22)
    ax.text(
        0.99,
        0.04,
        "positive = closes gap",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=ZERO_COLOR,
    )


def main():
    spectrum = pd.read_csv(RESULT_DIR / "shared_signal_spectrum_summary.csv")
    mediation = pd.read_csv(RESULT_DIR / "shared_signal_mediation_summary.csv")
    all_results = pd.read_csv(RESULT_DIR / "shared_signal_all.csv")
    pair_scores = all_results.groupby(["r2_session", "r1_session"], sort=False)[
        ["fwd_shared", "rev_shared", "gap_shared"]
    ].mean()
    baseline_gap = pair_scores.gap_shared.mean()

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    paired_means(
        axes[0, 0],
        lookup(spectrum, "movement_signal_effective_rank"),
        "A. Movement signal is not narrower",
        "effective rank (shared rank >= 2)",
    )
    paired_means(
        axes[0, 1],
        lookup(spectrum, "predictive_effective_rank"),
        "B. Predictive readout is broader in R2",
        "predictive effective rank (shared rank >= 2)",
    )
    effect_panel(axes[1, 0], spectrum)
    mediation_panel(axes[1, 1], mediation)
    figure.suptitle(
        "R2 compactness does not explain directional cross-session decoding\n"
        f"calibration-only shared space; 42 session pairs, 1,050 splits; "
        f"baseline shared-only gap = {baseline_gap:+.3f}",
        fontsize=15,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.11,
        top=0.84,
        hspace=0.34,
        wspace=0.44,
    )
    figure.text(
        0.5,
        0.025,
        "Intervals are hierarchical session-day sensitivity intervals from one animal; orange points are the three R2 days.",
        ha="center",
        fontsize=9,
        color="#5F6368",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=220, facecolor="white")
    plt.close(figure)
    print(f"saved {OUT}")


if __name__ == "__main__":
    sys.exit(main())
