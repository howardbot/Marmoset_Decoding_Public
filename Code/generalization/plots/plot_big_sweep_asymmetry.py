"""Plot forward/reverse asymmetry robustness across the cross-day sweep.

Each point in the sweep distribution is a parameter combination, not an
independent biological replicate. Directional gaps are paired by the same R1
and R2 sessions before being averaged within a parameter configuration.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "marmoset_matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
GENERALIZATION = THIS.parents[1]
if str(GENERALIZATION) not in sys.path:
    sys.path.insert(0, str(GENERALIZATION))

from big_sweep_phase2_crossday import ANIMAL_SESSIONS


REPO = THIS.parents[3]
RESULT_DIR = REPO / "Results" / "generalization"
FIG_DIR = RESULT_DIR / "figures"

CONFIG_COLUMNS = [
    "bin_size_ms",
    "smoother",
    "lag_ms",
    "decoder",
    "target_mode",
    "history_key",
]
PAIR_COLUMNS = ["r1_session", "r2_session"]
TARGETS = ("relative_position", "relative_velocity")
COLORS = {"kalman": "#2c7fb8", "wiener": "#d95f0e"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal",
        type=str.upper,
        choices=sorted(ANIMAL_SESSIONS),
        default="TS",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def default_paths(animal):
    suffix = "" if animal == "TS" else f"_{animal.lower()}"
    csv_path = RESULT_DIR / f"big_sweep_crossday_long{suffix}.csv"
    figure = FIG_DIR / f"fig_big_sweep_asymmetry{suffix}.png"
    config_csv = RESULT_DIR / f"big_sweep_asymmetry{suffix}_config_summary.csv"
    locked_csv = RESULT_DIR / f"big_sweep_asymmetry{suffix}_locked_pairs.csv"
    return csv_path, figure, config_csv, locked_csv


def build_gap_tables(data, animal):
    r1_sessions, r2_sessions = map(set, ANIMAL_SESSIONS[animal])
    data = data.copy()
    if "animal" in data:
        data = data[data["animal"].eq(animal)].copy()

    forward = data.train_session.isin(r1_sessions) & data.test_session.isin(r2_sessions)
    reverse = data.train_session.isin(r2_sessions) & data.test_session.isin(r1_sessions)
    data = data[forward | reverse].copy()
    forward = forward.loc[data.index]
    data["direction"] = np.where(forward, "R1->R2", "R2->R1")
    data["r1_session"] = np.where(
        data.direction.eq("R1->R2"),
        data.train_session,
        data.test_session,
    )
    data["r2_session"] = np.where(
        data.direction.eq("R1->R2"),
        data.test_session,
        data.train_session,
    )
    data["history_key"] = data.history_ms.fillna(-1).astype(int)

    # For TS pairs involving 0828, prefer the trial-excluded row. Other pairs
    # and all TY pairs have only the include row.
    data["outlier_preference"] = data.outlier_mode.eq("exclude").astype(int)
    data = (
        data.sort_values("outlier_preference", ascending=False)
        .drop_duplicates(CONFIG_COLUMNS + PAIR_COLUMNS + ["direction"], keep="first")
    )

    pair_gaps = (
        data.pivot(
            index=CONFIG_COLUMNS + PAIR_COLUMNS,
            columns="direction",
            values="M2_mean",
        )
        .reset_index()
        .dropna(subset=["R1->R2", "R2->R1"])
    )
    pair_gaps["asymmetry"] = pair_gaps["R2->R1"] - pair_gaps["R1->R2"]
    pair_gaps["pair_performance"] = (
        pair_gaps["R2->R1"] + pair_gaps["R1->R2"]
    ) / 2.0

    config_summary = (
        pair_gaps.groupby(CONFIG_COLUMNS)
        .agg(
            forward_corr=("R1->R2", "mean"),
            reverse_corr=("R2->R1", "mean"),
            asymmetry=("asymmetry", "mean"),
            asymmetry_sd_pairs=("asymmetry", "std"),
            positive_pair_fraction=("asymmetry", lambda values: float((values > 0).mean())),
            pair_performance=("pair_performance", "mean"),
            n_session_pairs=("asymmetry", "size"),
        )
        .reset_index()
    )
    config_summary.insert(0, "animal", animal)
    pair_gaps.insert(0, "animal", animal)
    return pair_gaps, config_summary


def locked_pairs(pair_gaps):
    return pair_gaps[
        pair_gaps.bin_size_ms.eq(30)
        & pair_gaps.smoother.eq("butter_o2")
        & pair_gaps.decoder.eq("kalman")
        & pair_gaps.lag_ms.eq(0)
    ].copy()


def plot_summary(config_summary, locked, animal, output):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))

    for row, target in enumerate(TARGETS):
        target_configs = config_summary[config_summary.target_mode.eq(target)]
        target_locked = locked[locked.target_mode.eq(target)]

        ax = axes[row, 0]
        for decoder in ("kalman", "wiener"):
            values = target_configs.loc[
                target_configs.decoder.eq(decoder), "asymmetry"
            ].to_numpy()
            positive = 100 * np.mean(values > 0)
            ax.hist(
                values,
                bins=28,
                alpha=0.55,
                color=COLORS[decoder],
                label=(
                    f"{decoder}: {positive:.0f}% > 0 "
                    f"(n={len(values)}, median={np.median(values):+.3f})"
                ),
            )
        locked_gap = float(target_locked.asymmetry.mean())
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(
            locked_gap,
            color="#6a3d9a",
            linestyle="--",
            linewidth=2,
            label=f"locked Kalman={locked_gap:+.3f}",
        )
        ax.set_title(f"{target.replace('relative_', '')}: all parameter configs")
        ax.set_xlabel("R2→R1 minus R1→R2 decode correlation")
        ax.set_ylabel("# parameter configurations")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        for decoder in ("kalman", "wiener"):
            decoder_rows = target_configs[target_configs.decoder.eq(decoder)]
            lag_summary = (
                decoder_rows.groupby("lag_ms")
                .asymmetry.agg(
                    mean="mean",
                    q25=lambda values: values.quantile(0.25),
                    q75=lambda values: values.quantile(0.75),
                )
                .reset_index()
            )
            ax.plot(
                lag_summary.lag_ms,
                lag_summary["mean"],
                marker="o",
                markersize=3.5,
                linewidth=2,
                color=COLORS[decoder],
                label=decoder,
            )
            ax.fill_between(
                lag_summary.lag_ms,
                lag_summary.q25,
                lag_summary.q75,
                color=COLORS[decoder],
                alpha=0.16,
            )
        ax.scatter(
            [0],
            [locked_gap],
            marker="D",
            s=55,
            color="#6a3d9a",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="locked Kalman",
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{target.replace('relative_', '')}: gap versus neural lag")
        ax.set_xlabel("lag (ms)")
        ax.set_ylabel("mean directional gap")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)

    n_r1 = locked.r1_session.nunique()
    n_r2 = locked.r2_session.nunique()
    fig.suptitle(
        f"{animal} cross-day asymmetry sweep — positive means R2→R1 > R1→R2\n"
        f"parameter robustness, not independent replicates; "
        f"n(R1)={n_r1}, n(R2)={n_r2}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(config_summary, locked):
    print("\nLocked Kalman configuration (30 ms, butter_o2, lag 0):")
    locked_summary = (
        locked.groupby("target_mode")
        .agg(
            forward_corr=("R1->R2", "mean"),
            reverse_corr=("R2->R1", "mean"),
            asymmetry=("asymmetry", "mean"),
            positive_pairs=("asymmetry", lambda values: int((values > 0).sum())),
            n_pairs=("asymmetry", "size"),
        )
        .reset_index()
    )
    print(locked_summary.round(4).to_string(index=False))

    print("\nFull parameter sweep:")
    sweep_summary = (
        config_summary.groupby(["target_mode", "decoder"])
        .agg(
            n_configs=("asymmetry", "size"),
            mean_gap=("asymmetry", "mean"),
            median_gap=("asymmetry", "median"),
            positive_fraction=("asymmetry", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )
    print(sweep_summary.round(4).to_string(index=False))

    print("\nLag-zero parameter robustness:")
    lag_zero = (
        config_summary[config_summary.lag_ms.eq(0)]
        .groupby(["target_mode", "decoder"])
        .agg(
            n_configs=("asymmetry", "size"),
            mean_gap=("asymmetry", "mean"),
            median_gap=("asymmetry", "median"),
            positive_fraction=("asymmetry", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )
    print(lag_zero.round(4).to_string(index=False))


def main():
    args = parse_args()
    default_csv, default_figure, config_csv, locked_csv = default_paths(args.animal)
    csv_path = args.csv or default_csv
    output = args.output or default_figure

    data = pd.read_csv(csv_path)
    pair_gaps, config_summary = build_gap_tables(data, args.animal)
    locked = locked_pairs(pair_gaps)
    if locked.empty:
        raise RuntimeError("locked configuration is missing from the sweep")

    config_summary.to_csv(config_csv, index=False)
    locked.to_csv(locked_csv, index=False)
    plot_summary(config_summary, locked, args.animal, output)
    print_summary(config_summary, locked)
    print(f"\nsaved {output}")
    print(f"saved {config_csv}")
    print(f"saved {locked_csv}")


if __name__ == "__main__":
    main()
