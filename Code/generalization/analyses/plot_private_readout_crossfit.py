"""Plot the cross-fitted Kalman decomposition and read-out interventions."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "manifold_geometry" / "private_readout_crossfit"
SUMMARY = IN_DIR / "mechanism_summary.csv"
BY_DAY = IN_DIR / "mechanism_by_r2_session.csv"
FIGURE = (
    REPO / "Results" / "manifold_geometry" / "figures"
    / "fig_private_readout_crossfit.png"
)


def summary_row(frame, analysis, metric, threshold=None):
    selected = frame[(frame.analysis == analysis) & (frame.metric == metric)]
    if threshold is not None:
        selected = selected[np.isclose(selected.cosine_threshold, threshold)]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {analysis}/{metric}/{threshold}")
    return selected.iloc[0]


def main():
    summary = pd.read_csv(SUMMARY)
    days = pd.read_csv(BY_DAY)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    decomposition = [
        ("Raw directional\ngap", "raw_directional_gap", "#253858"),
        ("Target ceiling\ngap", "target_ceiling_gap", "#8a94a3"),
        ("Transfer-penalty\nasymmetry", "transfer_penalty_asymmetry", "#cf5b3e"),
    ]
    x = np.arange(len(decomposition))
    for index, (label, metric, color) in enumerate(decomposition):
        row = summary_row(summary, "kalman_map_decomposition", metric)
        axes[0].bar(index, row.cluster_mean, width=0.62, color=color, alpha=0.88)
        axes[0].errorbar(
            index,
            row.cluster_mean,
            yerr=[[row.cluster_mean - row.hier_boot_lo],
                  [row.hier_boot_hi - row.cluster_mean]],
            color="#202124",
            capsize=4,
            linewidth=1.3,
        )
        day_values = days[
            (days.analysis == "kalman_map_decomposition")
        ][metric].dropna().to_numpy()
        jitter = np.linspace(-0.09, 0.09, len(day_values))
        axes[0].scatter(
            index + jitter, day_values, s=28, facecolor="white",
            edgecolor="#202124", linewidth=0.9, zorder=4,
        )
    axes[0].axhline(0, color="#202124", linewidth=0.8)
    axes[0].set_xticks(x, [item[0] for item in decomposition])
    axes[0].set_ylabel("Position decoding $R^2$ difference")
    axes[0].set_title("A  Where the Kalman asymmetry enters", loc="left", weight="bold")

    interventions = [
        ("Remove R1-private", "r1_private_excess_over_random", "#cf5b3e", "o"),
        ("Shared-only", "shared_excess_over_random", "#2f7f78", "s"),
        ("Remove R2-private", "r2_private_excess_over_random", "#4d67a8", "^"),
    ]
    thresholds = np.array([0.3, 0.5, 0.7])
    for label, metric, color, marker in interventions:
        means = []
        lower = []
        upper = []
        for threshold in thresholds:
            row = summary_row(
                summary, "kalman_subspace_intervention", metric, threshold
            )
            means.append(row.cluster_mean)
            lower.append(row.cluster_mean - row.hier_boot_lo)
            upper.append(row.hier_boot_hi - row.cluster_mean)
        means = np.asarray(means)
        axes[1].errorbar(
            thresholds, means, yerr=[lower, upper], color=color, marker=marker,
            markersize=6, linewidth=1.8, capsize=3, label=label,
        )
        day_frame = days[days.analysis == "kalman_subspace_intervention"]
        for threshold in thresholds:
            values = day_frame[np.isclose(
                day_frame.cosine_threshold, threshold
            )][metric].dropna().to_numpy()
            axes[1].scatter(
                np.full(len(values), threshold), values, s=16, color=color,
                alpha=0.35, linewidth=0, zorder=2,
            )
    axes[1].axhline(0, color="#202124", linewidth=0.8)
    axes[1].set_xticks(thresholds)
    axes[1].set_xlabel("Principal-cosine threshold")
    axes[1].set_ylabel("Selective change beyond rank-matched random")
    axes[1].set_title("B  Which read-out directions matter", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=9)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d8dce2", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Repeated 5x5 cross-fit; points are the three R2 days",
        y=1.01,
        fontsize=10,
        color="#555b65",
    )
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    print(f"saved {FIGURE}")


if __name__ == "__main__":
    main()
