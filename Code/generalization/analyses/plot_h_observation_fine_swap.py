"""Plot the pre-specified final decomposition of the Kalman observation map H."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_THIS = Path(__file__).resolve().parent
REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "manifold_geometry" / "h_observation_fine_swap"
SUMMARY = IN_DIR / "h_fine_summary.csv"
FIGURE = (
    REPO / "Results" / "manifold_geometry" / "figures"
    / "fig_h_observation_fine_swap.png"
)
COLORS = {
    "private": "#cf5b3e",
    "rest": "#4d67a8",
    "source": "#2f7f78",
    "target": "#d49a34",
    "full": "#7a5a9e",
    "private_scope": "#cf5b3e",
}


def get_row(summary, analysis, metric, **labels):
    selected = summary[
        (summary.analysis == analysis) & (summary.metric == metric)
    ]
    for name, value in labels.items():
        selected = selected[selected[name] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {analysis}/{metric}/{labels}")
    return selected.iloc[0]


def interval(axis, x, row, color, marker="o", label=None):
    axis.errorbar(
        x,
        row.cluster_mean,
        yerr=[[row.cluster_mean - row.hier_boot_lo],
              [row.hier_boot_hi - row.cluster_mean]],
        fmt=marker,
        color=color,
        ecolor="#202124",
        markersize=6,
        capsize=3,
        linewidth=1.2,
        label=label,
        zorder=3,
    )


def main():
    summary = pd.read_csv(SUMMARY)
    fig, axes = plt.subplots(1, 5, figsize=(20.5, 4.6))

    contexts = [
        ("no_intercept", "source", "Legacy\nsource Q"),
        ("no_intercept", "target", "Legacy\ntarget Q"),
        ("affine", "source", "Affine\nsource Q"),
        ("affine", "target", "Affine\ntarget Q"),
    ]
    width = 0.34
    for component_index, component in enumerate(("private", "rest")):
        offset = (component_index - 0.5) * width
        for index, (family, q_context, _) in enumerate(contexts):
            row = get_row(
                summary,
                "h_partition_shapley",
                "selective_shapley",
                observation_family=family,
                q_context=q_context,
                component=component,
            )
            axes[0].bar(
                index + offset, row.cluster_mean, width=width,
                color=COLORS[component], alpha=0.88,
            )
            interval(axes[0], index + offset, row, COLORS[component])
        axes[0].plot(
            [], [], marker="o", color=COLORS[component], label=component
        )
    axes[0].set_xticks(np.arange(len(contexts)), [item[2] for item in contexts])
    axes[0].set_ylabel("Selective H rescue allocated by Shapley")
    axes[0].set_title("A  Neural-space partition", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    direct_specs = [
        ("h_partition_intervention", "private_delta", "Private\nΔH", "#cf5b3e"),
        ("h_partition_intervention", "random_mean", "Random\nΔH", "#d49a34"),
        (
            "h_partition_random_adjusted", "private_minus_random",
            "Private −\nrandom", "#2f7f78",
        ),
        ("h_partition_intervention", "rest_delta", "Rest\nΔH", "#4d67a8"),
        ("h_partition_intervention", "target_H", "Full\nΔH", "#7a5a9e"),
    ]
    for index, (analysis, condition, label, color) in enumerate(direct_specs):
        row = get_row(
            summary,
            analysis,
            "selective_rescue",
            observation_family="no_intercept",
            q_context="source",
            condition=condition,
        )
        axes[1].bar(index, row.cluster_mean, width=0.66, color=color, alpha=0.88)
        interval(axes[1], index, row, color)
    axes[1].set_xticks(np.arange(len(direct_specs)), [item[2] for item in direct_specs])
    axes[1].set_ylabel("Forward - reverse score rescue")
    axes[1].set_title("B  Direct private intervention", loc="left", weight="bold")

    axis_names = ("x", "y", "z")
    for scope_index, (scope, label, color, marker) in enumerate([
        ("full", "Full ΔH", COLORS["full"], "o"),
        ("private", "Private ΔH", COLORS["private_scope"], "s"),
    ]):
        offset = (scope_index - 0.5) * 0.22
        for index, axis_name in enumerate(axis_names):
            row = get_row(
                summary,
                "h_axis_shapley",
                "selective_shapley",
                observation_family="no_intercept",
                q_context="source",
                delta_scope=scope,
                axis=axis_name,
            )
            interval(
                axes[2], index + offset, row, color, marker,
                label=label if index == 0 else None,
            )
    axes[2].set_xticks(np.arange(3), ["x", "y", "z"])
    axes[2].set_ylabel("Selective H rescue allocated by Shapley")
    axes[2].set_title("C  Kinematic-axis allocation", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=8)

    components = ("b", "H")
    width = 0.34
    for q_index, q_context in enumerate(("source", "target")):
        offset = (q_index - 0.5) * width
        for index, component in enumerate(components):
            row = get_row(
                summary,
                "h_affine_shapley",
                "selective_shapley",
                observation_family="affine",
                q_context=q_context,
                component=component,
            )
            color = COLORS[q_context]
            axes[3].bar(
                index + offset, row.cluster_mean, width=width,
                color=color, alpha=0.88,
            )
            interval(axes[3], index + offset, row, color)
        axes[3].plot(
            [], [], marker="o", color=COLORS[q_context],
            label=f"{q_context.capitalize()} Q",
        )
    axes[3].set_xticks(np.arange(2), ["Intercept b", "Linear H"])
    axes[3].set_ylabel("Selective affine rescue allocated by Shapley")
    axes[3].set_title("D  Offset versus linear map", loc="left", weight="bold")
    axes[3].legend(frameon=False, fontsize=8)

    centering_metrics = [
        ("Forward", "fwd_rescue"),
        ("Reverse", "rev_rescue"),
        ("Selective", "selective_rescue"),
    ]
    centering_conditions = [
        ("Behaviour-only center", "behaviour_center", "#2f7f78", "o"),
        ("Fitted target b", "target_b", "#d49a34", "s"),
    ]
    width = 0.34
    for condition_index, (label, condition, color, marker) in enumerate(
        centering_conditions
    ):
        offset = (condition_index - 0.5) * width
        for index, (_, metric) in enumerate(centering_metrics):
            row = get_row(
                summary,
                "h_intercept_centering",
                metric,
                q_context="source",
                condition=condition,
            )
            axes[4].bar(
                index + offset, row.cluster_mean, width=width,
                color=color, alpha=0.88,
            )
            interval(axes[4], index + offset, row, color, marker)
        axes[4].plot([], [], marker=marker, color=color, label=label)
    axes[4].set_xticks(
        np.arange(len(centering_metrics)), [item[0] for item in centering_metrics]
    )
    axes[4].set_ylabel("Score change from source affine baseline")
    axes[4].set_title("E  State-origin sensitivity", loc="left", weight="bold")
    axes[4].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.axhline(0, color="#202124", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d8dce2", linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Final H decomposition: repeated 5x5 cross-fit; intervals resample three R2 days descriptively",
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
