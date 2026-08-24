"""Localize the TS forget-control position gap within the reach.

This applies the locked time-resolved position analysis to the currently
available forget grid (one R1 date by three R2 dates).  At each 30-ms test bin,
position correlation is computed across test trials for x/y/z and then averaged
across coordinates.  The Kalman initialization bin is excluded because it is
initialized from the true test state.

Three analysis modes are exported:

``full``
    Use every decoder-usable trial as a descriptive reference.
``fixed40``
    Match R1 and R2 at 40 trials and refit PCA, CCA, and Kalman in 50 repeats.
``dropout_clean_fixed39``
    Remove start-to-peak trials overlapping ``neural_dropout``, match at 39
    trials, and refit the full pipeline in 50 repeats.

Random-subset quantiles describe sensitivity to trial selection.  They are not
confidence intervals over independent biological sessions: only one forget R1
date is currently available.
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "marmoset_matplotlib")
)
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

THIS = Path(__file__).resolve()
GENERALIZATION = THIS.parents[1]
WHY = THIS.parent
for path in (GENERALIZATION, WHY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from decode_variability_matched_crossday import (  # noqa: E402
    load_raw_session,
    subset_cache,
)
from forget_control_equal_n_crossday import (  # noqa: E402
    R1_SESSIONS,
    R2_SESSIONS,
    SEED,
    date_label,
    dropout_overlap_trials,
    sample_trial_ids,
)
from locked_position_time_resolved import (  # noqa: E402
    DISPLAY_BINS,
    EARLY_STOP_BIN,
    FIRST_EVALUATED_BIN,
    LATE_START_BIN,
    decode_with_predictions,
    time_resolved_metrics,
)

warnings.filterwarnings("ignore")

REPO = THIS.parents[3]
OUT_DIR = REPO / "Results" / "manifold_geometry"
FIGURE_DIR = OUT_DIR / "figures"
TARGET_MODE = "relative_position"
N_REPEATS = 50
N_WORKERS = min(6, max(1, (os.cpu_count() or 2) // 2))
ANALYSIS_MODES = {
    "fixed40": {"n_trials": 40, "exclude_dropout": False},
    "dropout_clean_fixed39": {"n_trials": 39, "exclude_dropout": True},
}

LONG_PATH = OUT_DIR / "forget_control_position_time_resolved_long.csv"
CELL_PATH = OUT_DIR / "forget_control_position_time_resolved_cells.csv"
AGGREGATE_PATH = OUT_DIR / "forget_control_position_time_resolved_aggregate.csv"
WINDOW_PATH = OUT_DIR / "forget_control_position_time_windows.csv"


def available_trial_ids(raw: dict) -> np.ndarray:
    """Return unique decoder-usable trial IDs in chronological order."""
    return np.sort(raw["meta"]["trial_number"].drop_duplicates().to_numpy(int))


def profile_direction(
    train_cache: dict,
    test_cache: dict,
    direction: str,
) -> pd.DataFrame:
    """Decode one direction and retain the locked binwise metrics."""
    actual, predicted, meta = decode_with_predictions(train_cache, test_cache)
    profile = time_resolved_metrics(actual, predicted, meta)
    profile.insert(0, "direction", direction)
    return profile


def add_identifiers(
    profile: pd.DataFrame,
    *,
    analysis_mode: str,
    repeat: int,
    pair_id: int,
    r1_session: str,
    r2_session: str,
    n_r1: int,
    n_r2: int,
) -> pd.DataFrame:
    """Attach session and subset metadata to a directional profile."""
    profile = profile.copy()
    identifiers = {
        "analysis_mode": analysis_mode,
        "repeat": repeat,
        "pair_id": pair_id,
        "r1_session": r1_session,
        "r2_session": r2_session,
        "r1_date": date_label(r1_session),
        "r2_date": date_label(r2_session),
        "n_r1": n_r1,
        "n_r2": n_r2,
    }
    for column, value in reversed(tuple(identifiers.items())):
        profile.insert(0, column, value)
    return profile


def run_full_reference(raw: dict[str, dict], allowed_all: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build the unequal-N full-trial descriptive profiles once."""
    r1_session = R1_SESSIONS[0]
    r1_ids = allowed_all[r1_session]
    r1_cache = subset_cache(raw[r1_session], r1_ids)
    rows = []
    for pair_id, r2_session in enumerate(R2_SESSIONS, start=1):
        r2_ids = allowed_all[r2_session]
        r2_cache = subset_cache(raw[r2_session], r2_ids)
        for direction, train, test in (
            ("R1->R2", r1_cache, r2_cache),
            ("R2->R1", r2_cache, r1_cache),
        ):
            profile = profile_direction(train, test, direction)
            rows.append(
                add_identifiers(
                    profile,
                    analysis_mode="full",
                    repeat=0,
                    pair_id=pair_id,
                    r1_session=r1_session,
                    r2_session=r2_session,
                    n_r1=len(r1_ids),
                    n_r2=len(r2_ids),
                )
            )
    return pd.concat(rows, ignore_index=True)


def run_repeat(
    repeat: int,
    analysis_mode: str,
    raw: dict[str, dict],
    allowed: dict[str, np.ndarray],
    n_trials: int,
) -> pd.DataFrame:
    """Run all three R2 cells for one equal-N random-subset repeat."""
    r1_session = R1_SESSIONS[0]
    rows = []
    for pair_id, r2_session in enumerate(R2_SESSIONS, start=1):
        cell_seed = np.random.SeedSequence([SEED, repeat, pair_id])
        rng1, rng2 = [np.random.default_rng(child) for child in cell_seed.spawn(2)]
        r1_ids = sample_trial_ids(allowed[r1_session], n_trials, rng1)
        r2_ids = sample_trial_ids(allowed[r2_session], n_trials, rng2)
        r1_cache = subset_cache(raw[r1_session], r1_ids)
        r2_cache = subset_cache(raw[r2_session], r2_ids)
        for direction, train, test in (
            ("R1->R2", r1_cache, r2_cache),
            ("R2->R1", r2_cache, r1_cache),
        ):
            profile = profile_direction(train, test, direction)
            rows.append(
                add_identifiers(
                    profile,
                    analysis_mode=analysis_mode,
                    repeat=repeat,
                    pair_id=pair_id,
                    r1_session=r1_session,
                    r2_session=r2_session,
                    n_r1=n_trials,
                    n_r2=n_trials,
                )
            )
    return pd.concat(rows, ignore_index=True)


def paired_profiles(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pair directional correlations and calculate instantaneous/running gaps."""
    display = long_df.loc[
        long_df["time_bin"].between(
            FIRST_EVALUATED_BIN, DISPLAY_BINS - 1, inclusive="both"
        )
    ].copy()
    index = [
        "analysis_mode",
        "repeat",
        "pair_id",
        "r1_session",
        "r2_session",
        "r1_date",
        "r2_date",
        "n_r1",
        "n_r2",
        "time_bin",
        "time_start_ms",
        "time_end_ms",
        "time_center_ms",
    ]
    paired = display.pivot(index=index, columns="direction", values="corr_mean").reset_index()
    paired["gap"] = paired["R2->R1"] - paired["R1->R2"]
    paired = paired.sort_values(
        ["analysis_mode", "repeat", "pair_id", "time_bin"]
    ).reset_index(drop=True)
    paired["running_mean_gap"] = paired.groupby(
        ["analysis_mode", "repeat", "pair_id"], sort=False
    )["gap"].transform(lambda values: values.expanding().mean())
    return paired


def summarize_groups(paired: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize repeat sensitivity at each bin for cells or the three-day mean."""
    rows = []
    for key, group in paired.groupby(group_columns, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        repeat_values = (
            group.groupby("repeat", as_index=False)
            .agg(
                forward=("R1->R2", "mean"),
                reverse=("R2->R1", "mean"),
                gap=("gap", "mean"),
                running_mean_gap=("running_mean_gap", "mean"),
            )
        )
        row = dict(zip(group_columns, key))
        row.update(
            {
                "n_repeats": len(repeat_values),
                "forward_mean": repeat_values["forward"].mean(),
                "reverse_mean": repeat_values["reverse"].mean(),
                "gap_mean": repeat_values["gap"].mean(),
                "gap_q025": repeat_values["gap"].quantile(0.025),
                "gap_q975": repeat_values["gap"].quantile(0.975),
                "positive_gap_fraction": np.mean(repeat_values["gap"] > 0),
                "running_mean_gap_mean": repeat_values["running_mean_gap"].mean(),
                "running_mean_gap_q025": repeat_values["running_mean_gap"].quantile(0.025),
                "running_mean_gap_q975": repeat_values["running_mean_gap"].quantile(0.975),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cells(paired: pd.DataFrame) -> pd.DataFrame:
    """Summarize each R1/R2 cell and time bin across subset repeats."""
    return summarize_groups(
        paired,
        [
            "analysis_mode",
            "pair_id",
            "r1_session",
            "r2_session",
            "r1_date",
            "r2_date",
            "time_bin",
            "time_center_ms",
        ],
    )


def summarize_aggregate(paired: pd.DataFrame) -> pd.DataFrame:
    """Summarize the mean of the three available R2 cells at each time bin."""
    return summarize_groups(
        paired,
        ["analysis_mode", "time_bin", "time_center_ms"],
    )


def window_label(time_bin: int) -> str:
    """Assign a 30-ms bin to the prespecified early, middle, or late window."""
    if time_bin < EARLY_STOP_BIN:
        return "early_30_150ms"
    if time_bin >= LATE_START_BIN:
        return "late_360_510ms"
    return "middle_150_360ms"


def summarize_windows(paired: pd.DataFrame) -> pd.DataFrame:
    """Average instantaneous gaps inside the locked early/middle/late windows."""
    paired = paired.copy()
    paired["window"] = paired["time_bin"].map(window_label)
    repeat_windows = (
        paired.groupby(
            [
                "analysis_mode",
                "repeat",
                "pair_id",
                "r1_session",
                "r2_session",
                "r1_date",
                "r2_date",
                "window",
            ],
            as_index=False,
        )
        .agg(
            forward=("R1->R2", "mean"),
            reverse=("R2->R1", "mean"),
            gap=("gap", "mean"),
        )
    )
    rows = []
    keys = [
        "analysis_mode",
        "pair_id",
        "r1_session",
        "r2_session",
        "r1_date",
        "r2_date",
        "window",
    ]
    for key, group in repeat_windows.groupby(keys, sort=True):
        gaps = group["gap"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_repeats": len(group),
                "forward_mean": group["forward"].mean(),
                "reverse_mean": group["reverse"].mean(),
                "gap_mean": gaps.mean(),
                "gap_q025": np.quantile(gaps, 0.025),
                "gap_q975": np.quantile(gaps, 0.975),
                "positive_gap_fraction": np.mean(gaps > 0),
            }
        )
    return pd.DataFrame(rows)


def plot_mode(cells: pd.DataFrame, analysis_mode: str, output: Path) -> None:
    """Plot direction, instantaneous gap, and running mean gap for each R2 day."""
    selected = cells.loc[cells["analysis_mode"] == analysis_mode]
    dates = [date_label(session) for session in R2_SESSIONS]
    fig, axes = plt.subplots(3, len(dates), figsize=(17.5, 10.5), sharex=True)
    colors = {"forward_mean": "#4C78A8", "reverse_mean": "#F58518"}
    labels = {"forward_mean": "R1→R2", "reverse_mean": "R2→R1"}
    for column, date in enumerate(dates):
        day = selected.loc[selected["r2_date"].astype(str) == date].sort_values(
            "time_bin"
        )
        x = day["time_center_ms"].to_numpy(dtype=float)
        for metric, color in colors.items():
            axes[0, column].plot(x, day[metric], color=color, lw=2.2, label=labels[metric])
        gap = day["gap_mean"].to_numpy(dtype=float)
        gap_lo = day["gap_q025"].to_numpy(dtype=float)
        gap_hi = day["gap_q975"].to_numpy(dtype=float)
        axes[1, column].plot(x, gap, color="#54A24B", lw=2.2)
        axes[1, column].fill_between(x, gap_lo, gap_hi, color="#54A24B", alpha=0.18)
        running = day["running_mean_gap_mean"].to_numpy(dtype=float)
        running_lo = day["running_mean_gap_q025"].to_numpy(dtype=float)
        running_hi = day["running_mean_gap_q975"].to_numpy(dtype=float)
        axes[2, column].plot(x, running, color="#B279A2", lw=2.2)
        axes[2, column].fill_between(
            x, running_lo, running_hi, color="#B279A2", alpha=0.18
        )
        axes[0, column].set_title(f"Forget R2 {date[-4:]}", weight="bold")
        axes[0, column].legend(frameon=False)
        for row in range(3):
            axes[row, column].axhline(0, color="black", lw=0.8, alpha=0.65)
            axes[row, column].axvspan(30, 150, color="grey", alpha=0.06)
            axes[row, column].axvspan(360, 510, color="grey", alpha=0.06)
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_xlim(30, 510)
    axes[0, 0].set_ylabel("Position correlation\nacross test trials")
    axes[1, 0].set_ylabel("Instantaneous gap\nR2→R1 − R1→R2")
    axes[2, 0].set_ylabel("Running mean gap\nfrom 30 ms through current bin")
    for axis in axes[-1]:
        axis.set_xlabel("Time after reach-window start (ms)")
    subtitle = (
        "Mean and 2.5–97.5% random-subset range"
        if analysis_mode != "full"
        else "Full-trial descriptive reference"
    )
    fig.suptitle(
        f"Forget-control time-resolved position transfer: {analysis_mode}\n{subtitle}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run full, fixed-40, and dropout-clean instantaneous time analyses."""
    print("Loading forget position sessions", flush=True)
    sessions = R1_SESSIONS + R2_SESSIONS
    raw = {session: load_raw_session(session, TARGET_MODE) for session in sessions}
    allowed_all = {session: available_trial_ids(raw[session]) for session in sessions}
    dropouts = {session: dropout_overlap_trials(session) for session in sessions}
    allowed_clean = {
        session: np.asarray(
            [trial for trial in allowed_all[session] if trial not in dropouts[session]],
            dtype=int,
        )
        for session in sessions
    }

    rows = [run_full_reference(raw, allowed_all)]
    for analysis_mode, config in ANALYSIS_MODES.items():
        allowed = allowed_clean if config["exclude_dropout"] else allowed_all
        n_trials = config["n_trials"]
        jobs = []
        print(
            f"Running {analysis_mode}: {N_REPEATS} repeats, {n_trials}/{n_trials}",
            flush=True,
        )
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                for repeat in range(N_REPEATS):
                    jobs.append(
                        executor.submit(
                            run_repeat,
                            repeat,
                            analysis_mode,
                            raw,
                            allowed,
                            n_trials,
                        )
                    )
                for completed, future in enumerate(as_completed(jobs), start=1):
                    rows.append(future.result())
                    if completed % 10 == 0 or completed == len(jobs):
                        print(
                            f"  {analysis_mode}: {completed}/{len(jobs)} repeats complete",
                            flush=True,
                        )

    long_df = pd.concat(rows, ignore_index=True)
    paired = paired_profiles(long_df)
    cells = summarize_cells(paired)
    aggregate = summarize_aggregate(paired)
    windows = summarize_windows(paired)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(LONG_PATH, index=False)
    cells.to_csv(CELL_PATH, index=False)
    aggregate.to_csv(AGGREGATE_PATH, index=False)
    windows.to_csv(WINDOW_PATH, index=False)
    for analysis_mode in ("full", *ANALYSIS_MODES):
        plot_mode(
            cells,
            analysis_mode,
            FIGURE_DIR
            / f"fig_forget_control_position_time_resolved_{analysis_mode}.png",
        )

    print("\nEarly/middle/late window summary")
    display = windows.loc[
        windows["analysis_mode"].isin(ANALYSIS_MODES)
    ][
        [
            "analysis_mode",
            "r2_date",
            "window",
            "forward_mean",
            "reverse_mean",
            "gap_mean",
            "gap_q025",
            "gap_q975",
            "positive_gap_fraction",
        ]
    ]
    print(display.round(4).to_string(index=False))
    print(f"\nsaved {LONG_PATH}")
    print(f"saved {CELL_PATH}")
    print(f"saved {AGGREGATE_PATH}")
    print(f"saved {WINDOW_PATH}")


if __name__ == "__main__":
    main()
