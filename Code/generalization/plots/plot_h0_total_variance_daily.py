"""Plot the H0 total-variance control as a per-session time trend.

This is a plotting-only companion to ``analyses/h0_total_variance_control.py``.  It
uses the already-computed R1->R2 pair rows and therefore does not reload NWB
files or rerun PCA/CCA/decoding.

For each target, session, and random seed:
  * an R1 day's value is its mean training variance across the three R2 partners;
  * an R2 day's value is its mean test variance across the fourteen R1 partners.

The plotted error bars are the SD across the 15 seed-level partner means.  They
measure alignment/subsampling sensitivity, not biological uncertainty.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
INPUT = REPO / "Results" / "workflows" / "manifold_geometry" / "h0_total_variance_control_pairs.csv"
OUTPUT_CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "h0_total_variance_daily.csv"
OUTPUT_FIG = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "figures"
    / "fig_h0_total_variance_daily.png"
)

TARGET_ORDER = ["relative_position", "relative_velocity"]
TARGET_LABELS = {
    "relative_position": "Position",
    "relative_velocity": "Velocity",
}
COLORS = {"R1": "#7f8c8d", "R2": "#e67e22"}


def session_date(session: str) -> str:
    """Extract YYYYMMDD from a canonical TSAL session name."""
    match = re.search(r"(2025\d{4})", str(session))
    if match is None:
        raise ValueError(f"Could not parse date from session name: {session}")
    return match.group(1)


def build_daily_summary(pair_rows: pd.DataFrame) -> pd.DataFrame:
    """Return one row per target/session after partner-then-seed aggregation."""
    forward = pair_rows[pair_rows["direction"] == "R1->R2"].copy()
    if forward.empty:
        raise ValueError("Input contains no R1->R2 rows")

    r1 = forward[
        ["target", "seed", "train_session", "test_session", "train_total_var_before"]
    ].rename(
        columns={
            "train_session": "session",
            "test_session": "partner_session",
            "train_total_var_before": "total_variance",
        }
    )
    r1["epoch"] = "R1"

    r2 = forward[
        ["target", "seed", "test_session", "train_session", "test_total_var"]
    ].rename(
        columns={
            "test_session": "session",
            "train_session": "partner_session",
            "test_total_var": "total_variance",
        }
    )
    r2["epoch"] = "R2"

    daily_rows = pd.concat([r1, r2], ignore_index=True)
    daily_rows["date"] = daily_rows["session"].map(session_date)

    # Average partners first, so each seed contributes one value per session.
    per_seed = (
        daily_rows.groupby(["target", "epoch", "session", "date", "seed"], as_index=False)
        .agg(
            total_variance=("total_variance", "mean"),
            n_partners=("partner_session", "nunique"),
        )
    )

    summary = (
        per_seed.groupby(["target", "epoch", "session", "date"], as_index=False)
        .agg(
            total_variance_mean=("total_variance", "mean"),
            total_variance_sd=("total_variance", "std"),
            n_seeds=("seed", "nunique"),
            n_partners=("n_partners", "max"),
        )
        .sort_values(["target", "date"])
        .reset_index(drop=True)
    )
    return summary


def plot_daily(summary: pd.DataFrame, output: Path) -> None:
    """Plot position and velocity daily trends on a shared variance scale."""
    all_dates = sorted(summary["date"].unique())
    x_by_date = {date: index for index, date in enumerate(all_dates)}
    boundary = max(x_by_date[d] for d in all_dates if d <= "20250813") + 0.5

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, sharey=True)
    for ax, target in zip(axes, TARGET_ORDER):
        target_rows = summary[summary["target"] == target].copy()
        for epoch in ("R1", "R2"):
            rows = target_rows[target_rows["epoch"] == epoch].sort_values("date")
            x = np.array([x_by_date[d] for d in rows["date"]], dtype=float)
            y = rows["total_variance_mean"].to_numpy(float)
            yerr = rows["total_variance_sd"].to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=COLORS[epoch],
                marker="o",
                markersize=6,
                linewidth=2,
                elinewidth=1,
                capsize=3,
                label=epoch,
                zorder=3,
            )

            epoch_mean = float(y.mean())
            ax.hlines(
                epoch_mean,
                x.min() - 0.25,
                x.max() + 0.25,
                color=COLORS[epoch],
                linestyle="--",
                linewidth=1.4,
                alpha=0.8,
            )
            ax.text(
                x.max() + 0.35,
                epoch_mean,
                f"{epoch} mean {epoch_mean:.1f}",
                color=COLORS[epoch],
                va="center",
                fontsize=9,
            )

        ax.axvline(boundary, color="#555555", linestyle=":", linewidth=1.2)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("total variance (trace)")
        ax.set_title(TARGET_LABELS[target], loc="left", fontsize=11, fontweight="bold")
        ax.legend(frameon=False, loc="upper left", ncol=2)

    axes[-1].set_xticks(
        range(len(all_dates)),
        [f"{date[4:6]}/{date[6:8]}" for date in all_dates],
        rotation=45,
        ha="right",
    )
    axes[-1].set_xlabel("session date")
    axes[0].text(
        boundary,
        axes[0].get_ylim()[1],
        "  R1 / R2 boundary",
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.suptitle(
        "Daily total canonical variance in the R1→R2 variance-match control",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        "K=12; each day is averaged across its cross-epoch partners within seed; "
        "error bars = SD across 15 alignment seeds",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pair_rows = pd.read_csv(INPUT)
    summary = build_daily_summary(pair_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    plot_daily(summary, OUTPUT_FIG)

    display = summary.copy()
    display["total_variance_mean"] = display["total_variance_mean"].round(3)
    display["total_variance_sd"] = display["total_variance_sd"].round(3)
    print(display.to_string(index=False))
    print(f"\nsaved {OUTPUT_CSV}")
    print(f"saved {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
