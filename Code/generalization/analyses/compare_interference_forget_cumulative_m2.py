"""Plot a metric-matched cumulative headline-gap comparison."""
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
import pandas as pd

THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
ORIGINAL = (
    REPO
    / "Results"
    / "generalization"
    / "interference_position_cumulative_m2_random_fixed40_by_r2.csv"
)
FORGET = (
    REPO
    / "Results"
    / "manifold_geometry"
    / "forget_control_position_cumulative_m2_cells.csv"
)
OUT_CSV = (
    REPO
    / "Results"
    / "manifold_geometry"
    / "interference_vs_forget_position_cumulative_m2.csv"
)
OUT_FIGURE = (
    REPO
    / "Results"
    / "manifold_geometry"
    / "figures"
    / "fig_interference_vs_forget_position_cumulative_m2.png"
)


def main() -> None:
    original = pd.read_csv(ORIGINAL)
    original["r2_date"] = original["r2_date"].astype(str)
    original_dates = sorted(original["r2_date"].unique())
    original["day_index"] = original["r2_date"].map(
        {date: index + 1 for index, date in enumerate(original_dates)}
    )
    original["condition"] = "Original interference random fixed40"
    original = original[
        ["condition", "day_index", "r2_date", "time_bin", "time_end_ms", "gap_mean"]
    ]

    forget = pd.read_csv(FORGET)
    forget["r2_date"] = forget["r2_date"].astype(str)
    forget_dates = sorted(forget["r2_date"].unique())
    forget["day_index"] = forget["r2_date"].map(
        {date: index + 1 for index, date in enumerate(forget_dates)}
    )
    forget["condition"] = forget["analysis_mode"].map(
        {
            "fixed40": "Forget random fixed40",
            "dropout_clean_fixed39": "Forget dropout-clean fixed39",
        }
    )
    forget = forget[
        ["condition", "day_index", "r2_date", "time_bin", "time_end_ms", "gap_mean"]
    ]
    combined = pd.concat([original, forget], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)

    styles = {
        "Original interference random fixed40": ("#4C78A8", "-", 2.5),
        "Forget random fixed40": ("#F58518", "-", 2.5),
        "Forget dropout-clean fixed39": ("#E45756", "--", 2.2),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3), sharex=True, sharey=True)
    for column, day_index in enumerate((1, 2, 3)):
        for condition, (color, linestyle, width) in styles.items():
            all_day = combined.loc[
                (combined["day_index"] == day_index)
                & (combined["condition"] == condition)
            ].sort_values("time_end_ms")
            display = all_day.loc[all_day["time_end_ms"] <= 900]
            axes[column].plot(
                display["time_end_ms"],
                display["gap_mean"],
                color=color,
                linestyle=linestyle,
                linewidth=width,
                label=condition,
            )
            endpoint = all_day.iloc[-1]
            axes[column].axhline(
                endpoint["gap_mean"],
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.65,
            )
        axes[column].set_title(f"R2 day {day_index}", weight="bold")
        axes[column].axhline(0, color="black", linewidth=0.8, alpha=0.7)
        for x in (150, 360, 510):
            axes[column].axvline(x, color="grey", linewidth=0.8, alpha=0.35)
        axes[column].set_xlim(120, 900)
        axes[column].set_xlabel("Prefix included from reach start (ms)")
        axes[column].grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative headline-style directional gap")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Original interference versus forget: cumulative within-trial position gap\n"
        "Dotted horizontals are exact full-reach headline endpoints",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_CSV}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
