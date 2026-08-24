"""Create a slide-ready schematic of pair-specific neural variability matching."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_MPL_CACHE = Path(tempfile.gettempdir()) / "marmoset_matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
FIGURE_DIR = REPO / "Code" / "generalization" / "docs" / "figures"
PNG_OUTPUT = FIGURE_DIR / "fig_neural_variability_matching_flowchart.png"
SVG_OUTPUT = FIGURE_DIR / "fig_neural_variability_matching_flowchart.svg"

NAVY = "#17324D"
BLUE = "#3977B7"
TEAL = "#2A9D8F"
ORANGE = "#F28E2B"
INK = "#27343B"
MUTED = "#667680"
GRID = "#D9E1E6"
PALE = "#F7F9FA"


def gaussian(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Return a unit-height Gaussian density for a schematic distribution."""
    values = np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return values / values.max()


def add_step_badge(fig, x: float, y: float, number: int, color: str) -> None:
    """Add a numbered circular badge in figure coordinates."""
    fig.text(
        x,
        y,
        str(number),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="circle,pad=0.34", facecolor=color, edgecolor="none"),
        zorder=10,
    )


def add_arrow(
    fig,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    connectionstyle: str = "arc3,rad=0",
    linewidth: float = 2.2,
) -> None:
    """Add a flow arrow in figure coordinates."""
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=16,
            color=color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
            zorder=8,
        )
    )


def style_density_axis(ax, title: str) -> None:
    """Apply a minimal, presentation-friendly density-axis style."""
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Pairwise neural trajectory distance", fontsize=10.5, color=INK)
    ax.set_ylabel("Trial-pair density", fontsize=10.5, color=INK)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.spines[["left", "bottom"]].set_linewidth(1.1)
    ax.set_facecolor("white")


def draw_distributions(
    ax,
    r2_mean: float,
    title: str,
    matched: bool,
) -> None:
    """Draw overlapping R1 and R2 variability distributions and their means."""
    x = np.linspace(0, 10, 500)
    r1_mean = 4.25
    r1_sd = 1.15
    r2_sd = 1.08
    r1 = gaussian(x, r1_mean, r1_sd)
    r2 = gaussian(x, r2_mean, r2_sd)
    lower = r1_mean - r1_sd
    upper = r1_mean + r1_sd

    style_density_axis(ax, title)
    ax.axvspan(lower, upper, color=BLUE, alpha=0.10, zorder=0)
    ax.plot(x, r1, color=BLUE, linewidth=3.0, label="R1 anchor", zorder=3)
    ax.fill_between(x, 0, r1, color=BLUE, alpha=0.10, zorder=1)
    ax.plot(x, r2, color=TEAL, linewidth=3.0, label="R2 trimmed", zorder=3)
    ax.fill_between(x, 0, r2, color=TEAL, alpha=0.10, zorder=1)
    ax.axvline(r1_mean, color=BLUE, linestyle="--", linewidth=1.6, zorder=4)
    ax.axvline(r2_mean, color=TEAL, linestyle="--", linewidth=1.6, zorder=4)

    ax.text(
        (lower + upper) / 2,
        1.07,
        "R1 target: μ ± 1 SD",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        0.55,
        0.12,
        "R1 anchor",
        fontsize=9.2,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        8.15,
        0.12,
        "R2 trimmed",
        fontsize=9.2,
        fontweight="bold",
        color=TEAL,
    )
    r2_label = "R2 mean inside band" if matched else "R2 mean above band"
    ax.annotate(
        r2_label,
        xy=(r2_mean, 0.98),
        xytext=(r2_mean + (0.55 if matched else 0.15), 0.72),
        fontsize=9.5,
        fontweight="bold",
        color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL, linewidth=1.4),
        ha="left",
    )


def make_figure() -> None:
    """Render the 16:9 slide as PNG and editable SVG."""
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.axis("off")

    fig.text(
        0.055,
        0.925,
        "H1.1  Neural variability matching",
        fontsize=27,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    fig.text(
        0.055,
        0.865,
        "One R1/R2 session pair shown schematically  •  repeated independently for all 14 × 3 pairs",
        fontsize=12.5,
        color=MUTED,
        va="center",
    )

    # Step 1: make the pairwise distance distribution for each day.
    add_step_badge(fig, 0.075, 0.725, 1, BLUE)
    fig.text(
        0.105,
        0.737,
        "Build day-wise distributions",
        fontsize=13.2,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    fig.text(
        0.075,
        0.675,
        "Phase-normalize each neural\npopulation trajectory",
        fontsize=10.8,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    fig.text(
        0.075,
        0.580,
        "Compute mean squared distance\nfor every unique trial pair",
        fontsize=10.8,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    motif = fig.add_axes([0.073, 0.445, 0.145, 0.09])
    motif.axis("off")
    t = np.linspace(0, 1, 100)
    for phase, color in ((0.0, BLUE), (0.7, TEAL), (1.25, ORANGE)):
        motif.plot(t, 0.5 + 0.22 * np.sin(2 * np.pi * t + phase), color=color, linewidth=2.2)
    motif.set_xlim(0, 1)
    motif.set_ylim(0, 1)

    add_arrow(fig, (0.225, 0.590), (0.285, 0.590), BLUE)

    # Step 2 and 4: before/after overlapping distributions.
    before_ax = fig.add_axes([0.300, 0.535, 0.325, 0.245])
    after_ax = fig.add_axes([0.300, 0.155, 0.325, 0.245])
    draw_distributions(before_ax, r2_mean=6.75, title="2   Before matching", matched=False)
    draw_distributions(after_ax, r2_mean=5.10, title="4   After matching", matched=True)

    # Step 3: the removal/recalculation loop between distributions.
    add_step_badge(fig, 0.315, 0.463, 3, ORANGE)
    add_arrow(fig, (0.650, 0.515), (0.650, 0.415), ORANGE)
    fig.text(
        0.350,
        0.482,
        "Remove 1 R2 trial  →  recompute",
        fontsize=9.5,
        fontweight="bold",
        color=ORANGE,
        ha="left",
        va="center",
    )
    fig.text(
        0.345,
        0.447,
        "High mean: remove largest   •   Low mean: remove smallest",
        fontsize=8.0,
        color=INK,
        ha="left",
        va="center",
    )

    # Right-side stopping rule and decoder output.
    panel = FancyBboxPatch(
        (0.685, 0.205),
        0.270,
        0.555,
        transform=fig.transFigure,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=PALE,
        edgecolor=GRID,
        linewidth=1.4,
        zorder=1,
    )
    fig.add_artist(panel)

    fig.text(0.715, 0.700, "Decision rule", fontsize=14, fontweight="bold", color=NAVY)
    fig.text(
        0.715,
        0.642,
        "Is the trimmed-session mean inside\nthe anchor-session μ ± 1 SD band?",
        fontsize=11,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    fig.text(0.715, 0.535, "NO", fontsize=11.5, fontweight="bold", color=ORANGE)
    fig.text(
        0.750,
        0.535,
        "remove one trial, recompute, repeat",
        fontsize=10.5,
        color=INK,
    )
    fig.text(0.715, 0.460, "YES", fontsize=11.5, fontweight="bold", color=TEAL)
    fig.text(0.755, 0.460, "lock the matched subset", fontsize=10.5, color=INK)
    fig.add_artist(
        Line2D(
            [0.715, 0.925],
            [0.410, 0.410],
            transform=fig.transFigure,
            color=GRID,
            linewidth=1.1,
        )
    )
    fig.text(0.715, 0.360, "Refit on selected trials", fontsize=13, fontweight="bold", color=NAVY)
    fig.text(0.715, 0.310, "PCA  →  CCA  →  Kalman", fontsize=11.5, color=INK)
    fig.text(
        0.715,
        0.255,
        "Score R1→R2 and R2→R1\nusing the same matched session pair",
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    add_arrow(fig, (0.628, 0.275), (0.682, 0.275), TEAL)

    # A concise footer prevents ambiguity about which day is trimmed.
    fig.text(
        0.055,
        0.065,
        "Primary analysis: anchor R1, trim only its paired R2 session",
        fontsize=10.5,
        fontweight="bold",
        color=BLUE,
    )
    fig.text(
        0.525,
        0.065,
        "Reverse sensitivity: anchor R2, trim only R1",
        fontsize=10.5,
        fontweight="bold",
        color=TEAL,
    )
    fig.text(
        0.055,
        0.028,
        "Neural variability selects trials; kinematics enter only as the decoding target.",
        fontsize=9.7,
        color=MUTED,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUTPUT, dpi=220, facecolor="white")
    fig.savefig(SVG_OUTPUT, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print(PNG_OUTPUT)
    print(SVG_OUTPUT)
