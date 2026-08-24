"""Resolve locked cross-day position decoding by time bin after movement start.

For each selected animal's R1/R2 session pair and transfer direction, this analysis runs the
same PCA-12, trial-average CCA, lag-0 Kalman pipeline as the locked position
matrix. Decoder performance is then evaluated separately at every 30-ms test
bin. Correlations at a time bin are computed across test trials, because a
single trial contributes only one position sample per coordinate at that bin.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "marmoset_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


THIS = Path(__file__).resolve()
GENERALIZATION = THIS.parents[1]
CODE = THIS.parents[2]
for path in (CODE, GENERALIZATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from big_sweep_phase2_crossday import (  # noqa: E402
    ANIMAL_SESSIONS,
    EXCLUDE_TRIALS,
    K_PCS,
    SMOOTHERS,
    build_cache_entry,
    kalman_fit_predict,
)
from manifold_align import apply_alignment, cca_align  # noqa: E402


REPO = THIS.parents[3]
RESULT_DIR = REPO / "Results/generalization"
FIGURE_DIR = RESULT_DIR / "figures"
BIN_SIZE_MS = 30
TARGET = "relative_position"
SMOOTHER_LABEL = "butter_o2"
SMOOTHER_KW = dict(SMOOTHERS)[SMOOTHER_LABEL]
COORDINATES = ("x", "y", "z")
MIN_TRIALS = 8
DISPLAY_BINS = 17
# The Kording Kalman implementation initializes each test trial from its true
# first state, so bin 0 is not an out-of-sample prediction and must be excluded.
FIRST_EVALUATED_BIN = 1
EARLY_STOP_BIN = 5
LATE_START_BIN = 12


def corr_1d(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return Pearson correlation, or NaN for fewer than two varying values."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    good = np.isfinite(actual) & np.isfinite(predicted)
    if good.sum() < 2:
        return np.nan
    actual = actual[good]
    predicted = predicted[good]
    if np.std(actual) <= 1e-12 or np.std(predicted) <= 1e-12:
        return np.nan
    return float(np.corrcoef(actual, predicted)[0, 1])


def time_resolved_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    meta: pd.DataFrame,
    bin_size_ms: int = BIN_SIZE_MS,
    min_trials: int = MIN_TRIALS,
) -> pd.DataFrame:
    """Measure prediction quality across trials at each within-trial time bin."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    if len(actual) != len(meta):
        raise ValueError("actual, predicted, and meta must have matching rows")
    if actual.ndim != 2 or actual.shape[1] != len(COORDINATES):
        raise ValueError("position arrays must have shape (samples, 3)")
    if min_trials < 2:
        raise ValueError("min_trials must be at least 2")

    rows = []
    for time_bin, indices in meta.groupby("local_bin", sort=True).indices.items():
        indices = np.asarray(indices)
        n_trials = int(meta.iloc[indices]["trial_number"].nunique())
        row: dict[str, float | int] = {
            "time_bin": int(time_bin),
            "time_start_ms": int(time_bin) * bin_size_ms,
            "time_end_ms": (int(time_bin) + 1) * bin_size_ms,
            "time_center_ms": (int(time_bin) + 0.5) * bin_size_ms,
            "n_trials": n_trials,
        }
        correlations = []
        normalized_errors = []
        for dimension, coordinate in enumerate(COORDINATES):
            truth = actual[indices, dimension]
            estimate = predicted[indices, dimension]
            finite = np.isfinite(truth) & np.isfinite(estimate)
            if finite.sum() < min_trials:
                correlation = np.nan
                normalized_error = np.nan
            else:
                correlation = corr_1d(truth[finite], estimate[finite])
                truth_sd = float(np.std(truth[finite], ddof=1))
                rmse = float(np.sqrt(np.mean((estimate[finite] - truth[finite]) ** 2)))
                normalized_error = rmse / truth_sd if truth_sd > 1e-12 else np.nan
            row[f"corr_{coordinate}"] = correlation
            row[f"nrmse_{coordinate}"] = normalized_error
            correlations.append(correlation)
            normalized_errors.append(normalized_error)
        row["corr_mean"] = float(np.nanmean(correlations))
        row["nrmse_mean"] = float(np.nanmean(normalized_errors))
        row["n_dimensions"] = int(np.isfinite(correlations).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def decode_with_predictions(train_cache: dict, test_cache: dict):
    """Run locked CCA+Kalman transfer and retain bin-level predictions."""
    train_weights, test_weights, train_mean, test_mean = cca_align(
        train_cache["traj"], test_cache["traj"]
    )
    train_activity = apply_alignment(
        train_cache["Y_pc"], train_weights, train_mean
    )[:, :K_PCS]
    test_activity = apply_alignment(
        test_cache["Y_pc"], test_weights, test_mean
    )[:, :K_PCS]
    test_meta = test_cache["meta"].reset_index(drop=True)
    actual, predicted = kalman_fit_predict(
        train_cache["X"],
        train_activity,
        test_cache["X"],
        test_activity,
        test_meta,
    )
    return actual, predicted, test_meta


def build_profiles(animal: str) -> pd.DataFrame:
    """Build time-resolved profiles for every cross-epoch session pair."""
    r1_sessions, r2_sessions = ANIMAL_SESSIONS[animal]
    sessions = list(r1_sessions) + list(r2_sessions)
    cache = {}
    for index, session in enumerate(sessions, start=1):
        print(f"[{animal} cache {index}/{len(sessions)}] {session}", flush=True)
        cache[session] = build_cache_entry(
            session,
            BIN_SIZE_MS,
            TARGET,
            SMOOTHER_KW,
            EXCLUDE_TRIALS.get(session, ()),
        )

    rows = []
    pair_id = 0
    for r2_session in r2_sessions:
        for r1_session in r1_sessions:
            pair_id += 1
            directions = (
                ("R1->R2", r1_session, r2_session),
                ("R2->R1", r2_session, r1_session),
            )
            for direction, train_session, test_session in directions:
                actual, predicted, meta = decode_with_predictions(
                    cache[train_session], cache[test_session]
                )
                profile = time_resolved_metrics(actual, predicted, meta)
                profile.insert(0, "test_session", test_session)
                profile.insert(0, "train_session", train_session)
                profile.insert(0, "direction", direction)
                profile.insert(0, "r2_session", r2_session)
                profile.insert(0, "r1_session", r1_session)
                profile.insert(0, "pair_id", pair_id)
                profile.insert(0, "animal", animal)
                rows.append(profile)
            print(
                f"[{animal}] pair {pair_id:02d}/"
                f"{len(r1_sessions) * len(r2_sessions)} complete",
                flush=True,
            )
    return pd.concat(rows, ignore_index=True)


def summarize_windows(profiles: pd.DataFrame) -> pd.DataFrame:
    """Summarize early and late bins, excluding the Kalman initialization bin."""
    display = profiles.loc[
        profiles["time_bin"].between(
            FIRST_EVALUATED_BIN, DISPLAY_BINS - 1, inclusive="both"
        )
    ].copy()
    display["window"] = np.where(
        display["time_bin"] < EARLY_STOP_BIN,
        "early_30_150ms",
        np.where(
            display["time_bin"] >= LATE_START_BIN,
            "late_360_510ms",
            "middle",
        ),
    )
    selected = display.loc[display["window"] != "middle"]
    per_pair = (
        selected.groupby(
            ["r2_session", "r1_session", "direction", "window"],
            as_index=False,
        )[["corr_mean", "nrmse_mean"]]
        .mean()
    )
    summary = (
        per_pair.groupby(["r2_session", "direction", "window"], as_index=False)
        .agg(
            corr_mean=("corr_mean", "mean"),
            corr_sd_across_r1=("corr_mean", "std"),
            nrmse_mean=("nrmse_mean", "mean"),
            n_pairs=("r1_session", "nunique"),
        )
    )
    return summary


def summarize_directional_gaps(profiles: pd.DataFrame) -> pd.DataFrame:
    """Test whether the per-pair directional gap is positive early and late."""
    display = profiles.loc[
        profiles["time_bin"].between(
            FIRST_EVALUATED_BIN, DISPLAY_BINS - 1, inclusive="both"
        )
    ].copy()
    display["window"] = np.where(
        display["time_bin"] < EARLY_STOP_BIN,
        "early_30_150ms",
        np.where(
            display["time_bin"] >= LATE_START_BIN,
            "late_360_510ms",
            "middle",
        ),
    )
    selected = display.loc[display["window"] != "middle"]
    per_pair = (
        selected.groupby(
            ["r2_session", "r1_session", "direction", "window"]
        )["corr_mean"]
        .mean()
        .unstack("direction")
    )
    per_pair["gap"] = per_pair["R2->R1"] - per_pair["R1->R2"]

    rows = []
    for (r2_session, window), table in per_pair.groupby(level=[0, 2]):
        values = table["gap"].dropna().to_numpy(dtype=float)
        test = stats.ttest_1samp(values, 0.0, alternative="greater")
        rows.append(
            {
                "r2_session": r2_session,
                "analysis": "directional_gap",
                "window": window,
                "n_pairs": len(values),
                "mean_gap": float(values.mean()),
                "gap_sd": float(values.std(ddof=1)),
                "t": float(test.statistic),
                "df": len(values) - 1,
                "p_one_sided_gt0": float(test.pvalue),
                "positive_fraction": float(np.mean(values > 0)),
            }
        )

    for r2_session, table in per_pair.groupby(level=0):
        early = table.xs("early_30_150ms", level="window")["gap"]
        late = table.xs("late_360_510ms", level="window")["gap"]
        common = early.index.intersection(late.index)
        values = (late.loc[common] - early.loc[common]).to_numpy(dtype=float)
        test = stats.ttest_1samp(values, 0.0, alternative="greater")
        rows.append(
            {
                "r2_session": r2_session,
                "analysis": "late_minus_early_gap",
                "window": "late_minus_early",
                "n_pairs": len(values),
                "mean_gap": float(values.mean()),
                "gap_sd": float(values.std(ddof=1)),
                "t": float(test.statistic),
                "df": len(values) - 1,
                "p_one_sided_gt0": float(test.pvalue),
                "positive_fraction": float(np.mean(values > 0)),
            }
        )
    return pd.DataFrame(rows)


def mean_and_sem(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().to_numpy(dtype=float)
    if not len(values):
        return np.nan, np.nan
    sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(np.mean(values)), float(sem)


def plot_profiles(profiles: pd.DataFrame, output: Path, animal: str) -> None:
    """Plot correlation, directional gap, and normalized error over time."""
    r2_sessions = list(dict.fromkeys(profiles["r2_session"]))
    colors = {"R1->R2": "#4C78A8", "R2->R1": "#F58518"}
    fig, axes = plt.subplots(
        3,
        len(r2_sessions),
        figsize=(6.3 * len(r2_sessions), 11),
        sharex=True,
        squeeze=False,
    )
    for column, r2_session in enumerate(r2_sessions):
        short_day = r2_session.split("_")[0][-4:]
        day = profiles.loc[
            (profiles["r2_session"] == r2_session)
            & profiles["time_bin"].between(
                FIRST_EVALUATED_BIN, DISPLAY_BINS - 1, inclusive="both"
            )
        ]
        summaries: dict[str, pd.DataFrame] = {}
        for direction in colors:
            direction_rows = day.loc[day["direction"] == direction]
            records = []
            for time_bin, values in direction_rows.groupby("time_bin"):
                corr_mean, corr_sem = mean_and_sem(values["corr_mean"])
                error_mean, error_sem = mean_and_sem(values["nrmse_mean"])
                records.append(
                    {
                        "time_bin": time_bin,
                        "time_center_ms": values["time_center_ms"].iloc[0],
                        "corr_mean": corr_mean,
                        "corr_sem": corr_sem,
                        "nrmse_mean": error_mean,
                        "nrmse_sem": error_sem,
                    }
                )
            summary = pd.DataFrame(records).sort_values("time_bin")
            summaries[direction] = summary
            x = summary["time_center_ms"].to_numpy()
            for row, metric in ((0, "corr"), (2, "nrmse")):
                mean = summary[f"{metric}_mean"].to_numpy()
                sem = summary[f"{metric}_sem"].to_numpy()
                axes[row, column].plot(
                    x, mean, color=colors[direction], linewidth=2, label=direction
                )
                axes[row, column].fill_between(
                    x, mean - sem, mean + sem, color=colors[direction], alpha=0.18
                )

        forward = summaries["R1->R2"].set_index("time_bin")
        reverse = summaries["R2->R1"].set_index("time_bin")
        common = forward.index.intersection(reverse.index)
        x = forward.loc[common, "time_center_ms"].to_numpy()
        gap = (
            reverse.loc[common, "corr_mean"].to_numpy()
            - forward.loc[common, "corr_mean"].to_numpy()
        )
        axes[1, column].plot(x, gap, color="#54A24B", linewidth=2.2)
        axes[1, column].fill_between(x, 0, gap, color="#54A24B", alpha=0.18)

        axes[0, column].set_title(
            f"{animal} R2 {short_day}", fontsize=13, weight="bold"
        )
        axes[0, column].axhline(0, color="black", linewidth=0.8, alpha=0.55)
        axes[1, column].axhline(0, color="black", linewidth=0.8, alpha=0.7)
        axes[0, column].set_ylim(-0.45, 0.85)
        axes[1, column].set_ylim(-0.65, 0.85)
        axes[2, column].set_ylim(bottom=0)
        for row in range(3):
            axes[row, column].grid(alpha=0.22)
            axes[row, column].axvspan(
                FIRST_EVALUATED_BIN * BIN_SIZE_MS,
                EARLY_STOP_BIN * BIN_SIZE_MS,
                color="grey",
                alpha=0.06,
            )
            axes[row, column].axvspan(
                LATE_START_BIN * BIN_SIZE_MS,
                DISPLAY_BINS * BIN_SIZE_MS,
                color="grey",
                alpha=0.06,
            )
        axes[0, column].legend(frameon=False, loc="best")

    axes[0, 0].set_ylabel("Position correlation\n(across test trials)")
    axes[1, 0].set_ylabel("Directional gap\nR2→R1 − R1→R2")
    axes[2, 0].set_ylabel("Normalized RMSE")
    for axis in axes[-1]:
        axis.set_xlabel("Time after movement-window start (ms)")
        axis.set_xlim(FIRST_EVALUATED_BIN * BIN_SIZE_MS, DISPLAY_BINS * BIN_SIZE_MS)
    fig.suptitle(
        f"{animal} time-resolved locked position transfer (30-ms bins)",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.965,
        "Mean ± SEM across session pairs; initial true-state Kalman bin excluded",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animal",
        choices=tuple(sorted(ANIMAL_SESSIONS)),
        default="TY",
    )
    args = parser.parse_args(argv)
    profiles = build_profiles(args.animal)
    summary = summarize_windows(profiles)
    gap_summary = summarize_directional_gaps(profiles)
    suffix = args.animal.lower()
    profiles_path = RESULT_DIR / f"locked_position_time_resolved_{suffix}.csv"
    summary_path = RESULT_DIR / f"locked_position_time_windows_{suffix}.csv"
    gap_path = RESULT_DIR / f"locked_position_time_gap_{suffix}.csv"
    figure_path = FIGURE_DIR / f"fig_locked_position_time_resolved_{suffix}.png"
    profiles.to_csv(profiles_path, index=False)
    summary.to_csv(summary_path, index=False)
    gap_summary.to_csv(gap_path, index=False)
    plot_profiles(profiles, figure_path, args.animal)
    print("\nEarly/late summary:")
    print(summary.round(4).to_string(index=False))
    print("\nDirectional-gap tests:")
    print(gap_summary.round(6).to_string(index=False))
    print(f"\nsaved {profiles_path}")
    print(f"saved {summary_path}")
    print(f"saved {gap_path}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()
