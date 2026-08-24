"""Plot cross-fitted localization of R1-private read-out directions."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


_THIS = Path(__file__).resolve().parent
REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "private_direction_localization"
FIGURE = (
    REPO / "Results" / "workflows" / "manifold_geometry" / "figures"
    / "fig_private_direction_localization.png"
)
FEATURES = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "speed"]
FEATURE_LABELS = ["pos x", "pos y", "pos z", "vel x", "vel y", "vel z", "speed"]
FEATURE_COLORS = {
    "pos_x": "#cf5b3e",
    "pos_y": "#d49a34",
    "pos_z": "#2f7f78",
    "vel_x": "#4d67a8",
    "vel_y": "#7a5a9e",
    "vel_z": "#78828f",
    "speed": "#2f3237",
}
WINDOW_MARKERS = {"early": "o", "middle": "s", "late": "^"}


def summary_row(summary, analysis, metric, **labels):
    selected = summary[(summary.analysis == analysis) & (summary.metric == metric)]
    for name, value in labels.items():
        selected = selected[selected[name] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {analysis}/{metric}/{labels}")
    return selected.iloc[0]


def add_interval(axis, x, row, color, marker="o"):
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
        zorder=3,
    )


def heatmap(axis, frame, value, title, figure):
    matrix = (
        frame.pivot(index="feature", columns="phase_bin", values=value)
        .reindex(FEATURES)
    )
    bound = np.nanmax(np.abs(matrix.to_numpy()))
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="BrBG",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
    )
    axis.set_yticks(np.arange(len(FEATURES)), FEATURE_LABELS)
    axis.set_xticks([0, 9, 19, 29], ["0", ".31", ".66", "1"])
    axis.set_xlabel("Normalized start-to-peak phase")
    axis.set_title(title, loc="left", weight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
    colorbar.ax.tick_params(labelsize=7)


def main():
    summary = pd.read_csv(IN_DIR / "localization_summary.csv")
    encoding_phase = pd.read_csv(IN_DIR / "encoding_phase_cluster_means.csv")
    error_phase = pd.read_csv(IN_DIR / "error_phase_cluster_means.csv")
    triangle = pd.read_csv(IN_DIR / "triangle_cells.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.0))

    for index, feature in enumerate(FEATURES):
        row = summary_row(
            summary, "private_encoding", "private_excess_random", feature=feature
        )
        axes[0, 0].bar(
            index, row.cluster_mean, width=0.66,
            color=FEATURE_COLORS[feature], alpha=0.9,
        )
        add_interval(axes[0, 0], index, row, FEATURE_COLORS[feature])
    axes[0, 0].set_xticks(np.arange(len(FEATURES)), FEATURE_LABELS, rotation=35)
    axes[0, 0].set_ylabel("Private correlation - rank-matched random")
    axes[0, 0].set_title("A  What: within-trial encoding", loc="left", weight="bold")

    heatmap(
        axes[0, 1], encoding_phase, "private_excess_random",
        "B  When: across-trial encoding", fig,
    )
    heatmap(
        axes[0, 2], error_phase[error_phase.direction == "forward"], "corr_rescue",
        "C  Where ablation restores correlation", fig,
    )

    for window, marker in WINDOW_MARKERS.items():
        selected = triangle[triangle.phase_window == window]
        for feature in FEATURES:
            row = selected[selected.feature == feature].iloc[0]
            axes[1, 0].scatter(
                row.private_excess_random,
                row.selective_corr_rescue,
                s=58,
                marker=marker,
                color=FEATURE_COLORS[feature],
                edgecolor="white",
                linewidth=0.6,
            )
    top_indices = set(triangle.nlargest(2, "private_excess_random").index)
    top_indices.update(triangle.nlargest(2, "selective_corr_rescue").index)
    for index in top_indices:
        row = triangle.loc[index]
        axes[1, 0].annotate(
            f"{row.feature}, {row.phase_window}",
            (row.private_excess_random, row.selective_corr_rescue),
            xytext=(4, 5), textcoords="offset points", fontsize=7,
        )
    for window, marker in WINDOW_MARKERS.items():
        axes[1, 0].scatter([], [], marker=marker, color="#626871", label=window)
    axes[1, 0].axhline(0, color="#202124", linewidth=0.8)
    axes[1, 0].axvline(0, color="#202124", linewidth=0.8)
    axes[1, 0].set_xlabel("Private encoding beyond random")
    axes[1, 0].set_ylabel("Forward - reverse correlation rescue")
    axes[1, 0].set_title("D  Predefined 21-cell closure", loc="left", weight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")

    width = 0.34
    direction_styles = {
        "forward": ("Forward", "#cf5b3e", "o"),
        "reverse": ("Reverse", "#4d67a8", "s"),
    }
    for direction_index, (direction, (label, color, marker)) in enumerate(
        direction_styles.items()
    ):
        offset = (direction_index - 0.5) * width
        for index, feature in enumerate(FEATURES):
            row = summary_row(
                summary,
                "private_ablation_error",
                "corr_rescue",
                direction=direction,
                feature=feature,
            )
            add_interval(axes[1, 1], index + offset, row, color, marker)
        axes[1, 1].plot([], [], marker=marker, color=color, label=label)
    axes[1, 1].set_xticks(np.arange(len(FEATURES)), FEATURE_LABELS, rotation=35)
    axes[1, 1].set_ylabel("Correlation rescue after R1-private removal")
    axes[1, 1].set_title("E  Direction-specific error rescue", loc="left", weight="bold")
    axes[1, 1].legend(frameon=False, fontsize=8)

    trial_metrics = [
        ("Full NMSE", "rho_full_nmse"),
        ("NMSE rescue", "rho_nmse_rescue"),
        ("Correlation rescue", "rho_corr_rescue"),
    ]
    energy_predictors = [
        ("R1-private", "private_energy", "#2f7f78", "o"),
        ("Rank-matched random", "random_private_energy", "#d49a34", "s"),
        ("Full activity", "full_activity_energy", "#78828f", "^"),
    ]
    width = 0.24
    for predictor_index, (predictor_label, predictor, color, marker) in enumerate(
        energy_predictors
    ):
        offset = (predictor_index - 1) * width
        for index, (_, metric) in enumerate(trial_metrics):
            row = summary_row(
                summary,
                "trial_stratification",
                metric,
                direction="forward",
                predictor=predictor,
            )
            axes[1, 2].bar(
                index + offset, row.cluster_mean, width=width,
                color=color, alpha=0.88,
            )
            add_interval(axes[1, 2], index + offset, row, color, marker)
        axes[1, 2].plot(
            [], [], marker=marker, color=color, label=predictor_label
        )
    axes[1, 2].set_xticks(np.arange(len(trial_metrics)), [x[0] for x in trial_metrics])
    axes[1, 2].set_ylabel("Within-pair Spearman rho")
    axes[1, 2].set_title("F  Trial-energy specificity controls", loc="left", weight="bold")
    axes[1, 2].legend(frameon=False, fontsize=8)

    for axis in (axes[0, 0], axes[1, 1], axes[1, 2]):
        axis.axhline(0, color="#202124", linewidth=0.8)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d8dce2", linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)
    fig.suptitle(
        "R1-private localization: repeated 5x5 cross-fit; intervals resample three R2 days descriptively",
        y=1.005,
        fontsize=10,
        color="#555b65",
    )
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    print(f"saved {FIGURE}")


if __name__ == "__main__":
    main()
