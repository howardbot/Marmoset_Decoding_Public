"""Compare original-interference and forget time-resolved position gaps."""
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
ORIGINAL_PATH = (
    REPO
    / "Results"
    / "generalization"
    / "locked_position_time_resolved_ts_random_fixed40.csv"
)
FORGET_PATH = (
    REPO
    / "Results"
    / "manifold_geometry"
    / "forget_control_position_time_resolved_long.csv"
)
OUT_CSV = (
    REPO
    / "Results"
    / "manifold_geometry"
    / "interference_vs_forget_position_time_resolved.csv"
)
OUT_FIGURE = (
    REPO
    / "Results"
    / "manifold_geometry"
    / "figures"
    / "fig_interference_vs_forget_position_time_resolved.png"
)


def pair_directions(frame: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    index = id_columns + ["time_bin", "time_center_ms"]
    paired = frame.pivot(index=index, columns="direction", values="corr_mean").reset_index()
    paired["gap"] = paired["R2->R1"] - paired["R1->R2"]
    paired = paired.sort_values(id_columns + ["time_bin"])
    paired["running_mean_gap"] = paired.groupby(id_columns, sort=False)["gap"].transform(
        lambda values: values.expanding().mean()
    )
    return paired


def prepare_original() -> pd.DataFrame:
    frame = pd.read_csv(ORIGINAL_PATH)
    paired = pair_directions(
        frame,
        ["repeat", "pair_id", "r1_session", "r2_session", "r2_date"],
    )
    dates = sorted(paired["r2_date"].astype(str).unique())
    day_map = {date: index + 1 for index, date in enumerate(dates)}
    paired["r2_date"] = paired["r2_date"].astype(str)
    paired["day_index"] = paired["r2_date"].map(day_map)
    paired["condition"] = "Original interference random fixed40"
    return paired


def prepare_forget() -> pd.DataFrame:
    frame = pd.read_csv(FORGET_PATH)
    frame = frame.loc[
        frame["analysis_mode"].isin(("fixed40", "dropout_clean_fixed39"))
        & frame["time_bin"].between(1, 16, inclusive="both")
    ]
    paired = pair_directions(
        frame,
        ["analysis_mode", "repeat", "pair_id", "r1_session", "r2_session", "r2_date"],
    )
    dates = sorted(paired["r2_date"].astype(str).unique())
    day_map = {date: index + 1 for index, date in enumerate(dates)}
    paired["r2_date"] = paired["r2_date"].astype(str)
    paired["day_index"] = paired["r2_date"].map(day_map)
    paired["condition"] = paired["analysis_mode"].map(
        {
            "fixed40": "Forget random fixed40",
            "dropout_clean_fixed39": "Forget dropout-clean fixed39",
        }
    )
    return paired


def main() -> None:
    paired = pd.concat([prepare_original(), prepare_forget()], ignore_index=True)
    summary = (
        paired.groupby(
            ["condition", "day_index", "r2_date", "time_bin", "time_center_ms"],
            as_index=False,
        )
        .agg(
            gap_mean=("gap", "mean"),
            running_mean_gap=("running_mean_gap", "mean"),
        )
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)

    styles = {
        "Original interference random fixed40": ("#4C78A8", "-", 2.5),
        "Forget random fixed40": ("#F58518", "-", 2.5),
        "Forget dropout-clean fixed39": ("#E45756", "--", 2.2),
    }
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.2), sharex=True, sharey="row")
    for column, day_index in enumerate((1, 2, 3)):
        day = summary.loc[summary["day_index"] == day_index]
        for condition, (color, linestyle, width) in styles.items():
            selected = day.loc[day["condition"] == condition].sort_values("time_bin")
            axes[0, column].plot(
                selected["time_center_ms"],
                selected["gap_mean"],
                color=color,
                linestyle=linestyle,
                linewidth=width,
                label=condition,
            )
            axes[1, column].plot(
                selected["time_center_ms"],
                selected["running_mean_gap"],
                color=color,
                linestyle=linestyle,
                linewidth=width,
            )
        axes[0, column].set_title(f"R2 day {day_index}", weight="bold")
        for row in range(2):
            axes[row, column].axhline(0, color="black", linewidth=0.8, alpha=0.65)
            axes[row, column].axvspan(30, 150, color="grey", alpha=0.06)
            axes[row, column].axvspan(360, 510, color="grey", alpha=0.06)
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_xlim(30, 510)
    axes[0, 0].set_ylabel("Instantaneous directional gap")
    axes[1, 0].set_ylabel("Running mean directional gap")
    for axis in axes[1]:
        axis.set_xlabel("Time after reach-window start (ms)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Original interference versus forget: within-reach position gap",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_CSV}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
