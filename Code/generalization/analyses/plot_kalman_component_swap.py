"""Plot Kalman component-swap and trial-boundary sensitivity results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "kalman_component_swap"
SUMMARY = IN_DIR / "component_swap_summary.csv"
FIGURE = (
    REPO / "Results" / "workflows" / "manifold_geometry" / "figures"
    / "fig_kalman_component_swap.png"
)
MODES = ["concatenated", "trial_aware"]
MODE_LABELS = {"concatenated": "Legacy concatenated", "trial_aware": "Trial-aware"}
MODE_COLORS = {"concatenated": "#8a94a3", "trial_aware": "#cf5b3e"}


def get_row(summary, analysis, mode, metric, **labels):
    selected = summary[
        (summary.analysis == analysis)
        & (summary.transition_mode == mode)
        & (summary.metric == metric)
    ]
    for name, value in labels.items():
        selected = selected[selected[name] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {analysis}/{mode}/{metric}/{labels}")
    return selected.iloc[0]


def add_interval(axis, x, row, color, marker="o", label=None):
    axis.errorbar(
        x,
        row.cluster_mean,
        yerr=[[row.cluster_mean - row.hier_boot_lo],
              [row.hier_boot_hi - row.cluster_mean]],
        color=color,
        marker=marker,
        markersize=6,
        linewidth=1.8,
        capsize=3,
        label=label,
    )


def main():
    summary = pd.read_csv(SUMMARY)
    fig, axes = plt.subplots(1, 4, figsize=(17.4, 4.4))

    decomposition_metrics = [
        ("Raw gap", "raw_directional_gap"),
        ("Fully adapted\ngap", "fully_adapted_gap"),
        ("Full selective\nrescue", "full_selective_rescue"),
    ]
    x = np.arange(len(decomposition_metrics))
    width = 0.34
    for mode_index, mode in enumerate(MODES):
        offset = (mode_index - 0.5) * width
        means = []
        lower = []
        upper = []
        for _, metric in decomposition_metrics:
            row = get_row(summary, "component_decomposition", mode, metric)
            means.append(row.cluster_mean)
            lower.append(row.cluster_mean - row.hier_boot_lo)
            upper.append(row.hier_boot_hi - row.cluster_mean)
        axes[0].bar(
            x + offset, means, width=width, color=MODE_COLORS[mode],
            alpha=0.9, label=MODE_LABELS[mode],
        )
        axes[0].errorbar(
            x + offset, means, yerr=[lower, upper], fmt="none",
            ecolor="#202124", capsize=3, linewidth=1.1,
        )
    axes[0].set_xticks(x, [label for label, _ in decomposition_metrics])
    axes[0].set_ylabel("Position decoding $R^2$ difference")
    axes[0].set_title("A  Trial-boundary sensitivity", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    components = ["A", "W", "H", "Q"]
    component_labels = [
        "A\ntransition", "W\nprocess noise", "H\nneural map", "Q\nneural noise"
    ]
    x = np.arange(len(components))
    for mode_index, mode in enumerate(MODES):
        offset = (mode_index - 0.5) * 0.16
        for index, component in enumerate(components):
            row = get_row(
                summary, "component_shapley", mode, "selective_shapley",
                component=component,
            )
            add_interval(
                axes[1], index + offset, row, MODE_COLORS[mode],
                marker="s" if mode == "concatenated" else "o",
                label=MODE_LABELS[mode] if index == 0 else None,
            )
    axes[1].set_xticks(x, component_labels)
    axes[1].set_ylabel("Selective rescue allocated by Shapley value")
    axes[1].set_title("B  Which Kalman component matters", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=8)

    groups = [(3, "A+W\nstate model"), (12, "H+Q\nobservation model")]
    x = np.arange(len(groups))
    for mode_index, mode in enumerate(MODES):
        offset = (mode_index - 0.5) * 0.18
        for index, (mask, _) in enumerate(groups):
            row = get_row(
                summary, "component_intervention", mode, "selective_rescue",
                target_mask=mask,
            )
            add_interval(
                axes[2], index + offset, row, MODE_COLORS[mode],
                marker="s" if mode == "concatenated" else "o",
                label=MODE_LABELS[mode] if index == 0 else None,
            )
    axes[2].set_xticks(x, [label for _, label in groups])
    axes[2].set_ylabel("Selective rescue from direct component swap")
    axes[2].set_title("C  Direct grouped interventions", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=8)

    kinematic_metrics = [
        ("Forward", "fwd_transition_penalty", "#cf5b3e"),
        ("Reverse", "rev_transition_penalty", "#4d67a8"),
        ("Forward −\nreverse", "directional_transition_penalty", "#2f7f78"),
    ]
    for index, (label, metric, color) in enumerate(kinematic_metrics):
        selected = summary[
            (summary.analysis == "kinematic_transition")
            & (summary.metric == metric)
        ]
        if len(selected) != 1:
            raise ValueError(f"expected one kinematic row for {metric}")
        row = selected.iloc[0]
        axes[3].bar(index, row.cluster_mean, width=0.62, color=color, alpha=0.88)
        axes[3].errorbar(
            index,
            row.cluster_mean,
            yerr=[[row.cluster_mean - row.hier_boot_lo],
                  [row.hier_boot_hi - row.cluster_mean]],
            color="#202124",
            capsize=3,
            linewidth=1.1,
        )
    axes[3].set_xticks(
        np.arange(len(kinematic_metrics)),
        [label for label, _, _ in kinematic_metrics],
    )
    axes[3].set_ylabel("Normalized one-step error penalty")
    axes[3].set_title("D  Neural-free dynamics control", loc="left", weight="bold")

    for axis in axes:
        axis.axhline(0, color="#202124", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d8dce2", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Repeated 5x5 cross-fit; intervals resample three R2 days descriptively",
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
