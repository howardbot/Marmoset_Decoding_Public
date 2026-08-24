"""Summarize and plot the original-Kalman private/shared threshold sweep."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from private_readout_crossfit_summary import summarize_pair_metrics
from private_readout_threshold_sweep import THRESHOLDS

REPO = THIS_DIR.parents[2]
IN_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "private_readout_threshold_sweep"
OUT_ALL = IN_DIR / "threshold_sweep_all.csv"
OUT_PAIR = IN_DIR / "threshold_sweep_pair_means.csv"
OUT_SUMMARY = IN_DIR / "threshold_sweep_summary.csv"
OUT_DAY = IN_DIR / "threshold_sweep_by_r2_session.csv"
OUT_TABLE = IN_DIR / "threshold_sweep_table.csv"
OUT_FIGURE = (
    REPO / "Results" / "workflows" / "manifold_geometry" / "figures"
    / "fig_private_readout_threshold_sweep.png"
)
KEYS = ["cosine_threshold", "r1_session", "r2_session"]
METRICS = [
    "fwd_full",
    "rev_full",
    "gap_full",
    "fwd_shared",
    "rev_shared",
    "gap_shared",
    "gap_closure",
    "rank_shared",
    "shared_available",
    "gap_full_available",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=8)
    return parser.parse_args()


def metric_wide(summary):
    values = summary.pivot(index="cosine_threshold", columns="metric", values="cluster_mean")
    lower = summary.pivot(index="cosine_threshold", columns="metric", values="hier_boot_lo")
    upper = summary.pivot(index="cosine_threshold", columns="metric", values="hier_boot_hi")
    return values, lower, upper


def make_figure(summary):
    values, lower, upper = metric_wide(summary)
    thresholds = values.index.to_numpy(dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 8.2), sharex=True)

    full = values.gap_full_available.to_numpy()
    shared = values.gap_shared.to_numpy()
    shared_lo = lower.gap_shared.to_numpy()
    shared_hi = upper.gap_shared.to_numpy()
    axes[0].plot(
        thresholds,
        full,
        "--",
        color="#555B61",
        label="full gap on the same available splits",
    )
    axes[0].plot(thresholds, shared, "o-", color="#C44536", label="shared-only gap")
    axes[0].fill_between(thresholds, shared_lo, shared_hi, color="#C44536", alpha=0.16)
    axes[0].axhline(0, color="#90969C", linewidth=1)
    axes[0].axvline(0.5, color="#00897B", linestyle=":", linewidth=1.5)
    axes[0].set_ylabel("R2->R1 minus R1->R2")
    axes[0].set_title(
        "A. Directional gap survives shared-only restriction",
        loc="left",
        fontweight="bold",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.22)

    axes[1].plot(
        thresholds,
        values.rank_shared,
        "o-",
        color="#2878B5",
        label="mean shared rank",
    )
    axes[1].set_ylabel("mean shared rank", color="#2878B5")
    axes[1].tick_params(axis="y", labelcolor="#2878B5")
    coverage_axis = axes[1].twinx()
    coverage_axis.plot(
        thresholds,
        values.shared_available * 100,
        "s-",
        color="#F2994A",
        label="splits with shared rank >= 1",
    )
    coverage_axis.set_ylabel("available splits (%)", color="#C47716")
    coverage_axis.tick_params(axis="y", labelcolor="#C47716")
    coverage_axis.set_ylim(-3, 103)
    axes[1].axvline(0.5, color="#00897B", linestyle=":", linewidth=1.5)
    axes[1].set_xlabel("principal-cosine threshold")
    axes[1].set_title(
        "B. Higher thresholds reduce shared rank and eventually lose splits",
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(alpha=0.22)
    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = coverage_axis.get_legend_handles_labels()
    axes[1].legend(handles_a + handles_b, labels_a + labels_b, frameon=False, loc="center left")

    figure.suptitle(
        "Original Kalman threshold sweep",
        y=0.965,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.915,
        "Shared-only asymmetry persists across the usable threshold range; "
        "42 session pairs, 5x5 held-out splits",
        ha="center",
        fontsize=10,
        color="#555B61",
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.88,
        bottom=0.10,
        top=0.82,
        hspace=0.30,
    )
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_FIGURE, dpi=220, facecolor="white")
    plt.close(figure)


def main():
    args = parse_args()
    paths = [
        IN_DIR / f"threshold_job_{index}_of_{args.num_jobs}.csv"
        for index in range(args.num_jobs)
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing threshold shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = 42 * 5 * 5 * len(THRESHOLDS)
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(frame)}")
    frame["gap_full_available"] = frame.gap_full.where(frame.shared_available > 0)
    frame.to_csv(OUT_ALL, index=False)
    pair = frame.groupby(KEYS, as_index=False)[METRICS].mean()
    pair.to_csv(OUT_PAIR, index=False)
    day_rows, summary_rows = summarize_pair_metrics(
        pair,
        ["cosine_threshold"],
        METRICS,
        "original_kalman_threshold_sweep",
    )
    days = pd.DataFrame(day_rows)
    summary = pd.DataFrame(summary_rows)
    days.to_csv(OUT_DAY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    values, lower, upper = metric_wide(summary)
    table = pd.DataFrame({
        "cosine_threshold": values.index,
        "mean_shared_rank": values.rank_shared,
        "split_coverage": values.shared_available,
        "fwd_shared": values.fwd_shared,
        "rev_shared": values.rev_shared,
        "gap_shared": values.gap_shared,
        "gap_shared_lo": lower.gap_shared,
        "gap_shared_hi": upper.gap_shared,
        "full_gap_same_splits": values.gap_full_available,
        "gap_closure": values.gap_closure,
    }).reset_index(drop=True)
    table["fraction_closed"] = (
        table.gap_closure / table.full_gap_same_splits
    )
    table["fraction_retained"] = (
        table.gap_shared / table.full_gap_same_splits
    )
    table.to_csv(OUT_TABLE, index=False)
    make_figure(summary)

    print(table.round(4).to_string(index=False))
    print(f"\nsaved {OUT_SUMMARY}\nsaved {OUT_TABLE}\nsaved {OUT_FIGURE}")


if __name__ == "__main__":
    sys.exit(main())
