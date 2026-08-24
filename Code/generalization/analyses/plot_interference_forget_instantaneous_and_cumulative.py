"""Plot instantaneous and headline-matched cumulative position gaps together.

The two rows intentionally show different estimands:

* instantaneous: correlation across test trials at one 30 ms time bin;
* cumulative: within-trial temporal correlation over the prefix ending at that time.

Only the cumulative row converges to the full-reach headline M2 gap.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "marmoset_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
INSTANTANEOUS_PATH = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "interference_vs_forget_position_time_resolved.csv"
)
CUMULATIVE_PATH = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "interference_vs_forget_position_cumulative_m2.csv"
)
OUT_FIGURE = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "figures"
    / "fig_interference_vs_forget_instantaneous_and_cumulative.png"
)


STYLES = {
    "Original interference random fixed40": {
        "label": "Original interference (fixed 40)",
        "color": "#4C78A8",
        "linestyle": "-",
        "linewidth": 2.6,
    },
    "Forget random fixed40": {
        "label": "Forget control (fixed 40)",
        "color": "#F58518",
        "linestyle": "-",
        "linewidth": 2.6,
    },
    "Forget dropout-clean fixed39": {
        "label": "Forget control (dropout-clean 39)",
        "color": "#E45756",
        "linestyle": "-",
        "linewidth": 2.3,
    },
}

DATE_LABELS = {
    1: "Original 8/28  |  Forget 6/26",
    2: "Original 8/29  |  Forget 6/27",
    3: "Original 8/30  |  Forget 6/28",
}


def add_common_guides(axis: plt.Axes) -> None:
    """Apply identical limits and minimal reference styling to every panel."""
    axis.axhline(0, color="#222222", linewidth=0.9, alpha=0.75, zorder=0)
    axis.set_xlim(30, 510)
    axis.set_xticks([30, 150, 270, 360, 510])
    axis.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    """Assemble the original/forget instantaneous and cumulative comparison."""
    instantaneous = pd.read_csv(INSTANTANEOUS_PATH)
    cumulative = pd.read_csv(CUMULATIVE_PATH)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.8, 8.8),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0.25, "wspace": 0.12},
    )

    for column, day_index in enumerate((1, 2, 3)):
        axes[0, column].set_title(
            f"R2 day {day_index}\n{DATE_LABELS[day_index]}",
            fontsize=11.5,
            weight="bold",
            pad=9,
        )

        for condition, style in STYLES.items():
            instant_series = instantaneous.loc[
                (instantaneous["day_index"] == day_index)
                & (instantaneous["condition"] == condition)
            ].sort_values("time_center_ms")
            axes[0, column].plot(
                instant_series["time_center_ms"],
                instant_series["gap_mean"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=style["label"],
            )

            cumulative_all = cumulative.loc[
                (cumulative["day_index"] == day_index)
                & (cumulative["condition"] == condition)
            ].sort_values("time_end_ms")
            cumulative_visible = cumulative_all.loc[
                cumulative_all["time_end_ms"] <= 510
            ]
            axes[1, column].plot(
                cumulative_visible["time_end_ms"],
                cumulative_visible["gap_mean"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )

        for row in range(2):
            add_common_guides(axes[row, column])

    axes[0, 0].set_ylabel(
        "Instantaneous cross-trial gap\n(single 30 ms bin)", fontsize=11.5
    )
    axes[1, 0].set_ylabel(
        "Cumulative within-trial gap\n(prefix from reach start)", fontsize=11.5
    )
    axes[0, 0].set_ylim(-0.245, 0.405)
    axes[1, 0].set_ylim(-0.185, 0.405)
    for axis in axes[1]:
        axis.set_xlabel("Time after reach start (ms)", fontsize=11)

    condition_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        for style in STYLES.values()
    ]
    fig.legend(
        handles=condition_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        "Original interference vs forget control: instantaneous and cumulative position gap",
        fontsize=15.5,
        weight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.937,
        "Top: cross-trial snapshot (diagnostic only).  "
        "Bottom: headline-matched within-trial prefix correlation.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.025, 0.07, 0.995, 0.91))

    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
