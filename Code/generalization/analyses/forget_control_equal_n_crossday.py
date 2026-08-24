"""Run the locked equal-N decoder on the complete TS forget-control grid.

The current grid contains three R1 sessions (2026-06-09 through 2026-06-11)
and three R2 sessions (2026-06-26 through 2026-06-28).  For each target and
each of the nine R1/R2 cells, the script runs two random-subset controls:

``fixed31``
    Use 31 trials from every day.  This is the largest common sample because
    2026-06-10 contains 31 decoder-usable S/F trials.
``dropout_clean_fixed31``
    Exclude every start-to-peak trial overlapping an NWB ``neural_dropout``
    interval, then use 31 trials from every day.  The common clean minimum is
    still 31 because none of the 2026-06-10 trials overlap dropout intervals.

PCA, trial-average CCA, and the lag-0 Kalman decoder are refit inside every
repeat.  Repeat-level quantiles measure random-subset sensitivity; inference
is performed on repeat-averaged biological session cells.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy import stats
from threadpoolctl import threadpool_limits

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from decode_variability_matched_crossday import (
    decode_direction,
    load_raw_session,
    subset_cache,
)
from position_asymmetry_significance import gap_test
from project_config import (
    DATA_DIR,
    FORGET_CONTROL_RESULTS_DIR,
    REPO_ROOT,
    TS_FORGET_R1,
    TS_FORGET_R2,
    session_date,
)

OUT_DIR = FORGET_CONTROL_RESULTS_DIR

R1_SESSIONS = TS_FORGET_R1
R2_SESSIONS = TS_FORGET_R2
TARGET_MODES = ("relative_position", "relative_velocity")
ANALYSIS_MODES = {
    "fixed31": {"n_trials": 31, "exclude_dropout": False},
    "dropout_clean_fixed31": {"n_trials": 31, "exclude_dropout": True},
}
N_REPEATS = 50
SEED = 20260817
N_BOOTSTRAP = 50_000
N_WORKERS = min(6, max(1, (os.cpu_count() or 2) // 2))


def date_label(session: str) -> str:
    """Convert an internal session identifier to its YYYYMMDD date label."""
    return session_date(session)


def available_trial_ids(raw: dict) -> np.ndarray:
    """Return unique decoder-usable trial IDs in their stored order."""
    return raw["meta"]["trial_number"].drop_duplicates().to_numpy(dtype=int)


def sample_trial_ids(
    available: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a sorted, reproducible subset of trials without replacement."""
    available = np.asarray(sorted(set(map(int, available))), dtype=int)
    if len(available) < n_trials:
        raise ValueError(f"requested {n_trials} trials from only {len(available)}")
    return np.sort(rng.choice(available, size=n_trials, replace=False))


def _peak_scalar(value) -> float:
    """Normalize an NWB peak-time cell to one floating-point timestamp."""
    values = np.asarray(value).reshape(-1)
    if not len(values):
        raise ValueError("empty peak_extension_times value")
    return float(values[0])


def dropout_overlap_trials(session: str) -> set[int]:
    """Return S/F reach IDs whose start-to-peak window intersects dropout."""
    path = DATA_DIR / f"{session}_processed.nwb"
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        reach = nwb.intervals["reaching_segments_forget"].to_dataframe()
        reach = reach[reach["result"].astype(str).isin(("S", "F"))]
        dropout = nwb.intervals["neural_dropout"].to_dataframe()
        drop_start = dropout["start_time"].to_numpy(dtype=float)
        drop_stop = dropout["stop_time"].to_numpy(dtype=float)
        overlaps = set()
        for trial_id, row in reach.iterrows():
            start = float(row["start_time"])
            peak = _peak_scalar(row["peak_extension_times"])
            if np.any((drop_start < peak) & (drop_stop > start)):
                overlaps.add(int(trial_id))
    return overlaps


def run_repeat(
    repeat: int,
    seed: int,
    target_mode: str,
    analysis_mode: str,
    raw: dict[str, dict],
    allowed: dict[str, np.ndarray],
    n_trials: int,
) -> pd.DataFrame:
    """Run one equal-N resampling repeat for all nine R1/R2 cells.

    Separate child random generators are used for R1 and R2 so the two trial
    subsets are sampled independently while remaining reproducible.
    """
    rows = []
    for r1_index, r1_session in enumerate(R1_SESSIONS):
        for r2_index, r2_session in enumerate(R2_SESSIONS):
            pair_id = r1_index * len(R2_SESSIONS) + r2_index + 1
            cell_seed = np.random.SeedSequence([seed, repeat, pair_id])
            rng1, rng2 = [
                np.random.default_rng(child) for child in cell_seed.spawn(2)
            ]
            trials1 = sample_trial_ids(allowed[r1_session], n_trials, rng1)
            trials2 = sample_trial_ids(allowed[r2_session], n_trials, rng2)
            cache1 = subset_cache(raw[r1_session], trials1)
            cache2 = subset_cache(raw[r2_session], trials2)
            forward = decode_direction(cache1, cache2)
            reverse = decode_direction(cache2, cache1)
            for direction, score in (("R1->R2", forward), ("R2->R1", reverse)):
                rows.append(
                    {
                        "analysis_mode": analysis_mode,
                        "target_mode": target_mode,
                        "repeat": repeat,
                        "seed": seed,
                        "pair_id": pair_id,
                        "direction": direction,
                        "r1_session": r1_session,
                        "r2_session": r2_session,
                        "r1_date": date_label(r1_session),
                        "r2_date": date_label(r2_session),
                        "n_trials_each": n_trials,
                        "random_corr": score,
                    }
                )
    return pd.DataFrame(rows)


def reference_cells(
    target_mode: str,
    analysis_mode: str,
    raw: dict[str, dict],
    allowed: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Decode each cell once using every allowed trial before equal-N sampling."""
    rows = []
    for r1_index, r1_session in enumerate(R1_SESSIONS):
        r1_cache = subset_cache(raw[r1_session], allowed[r1_session])
        for r2_index, r2_session in enumerate(R2_SESSIONS):
            pair_id = r1_index * len(R2_SESSIONS) + r2_index + 1
            r2_cache = subset_cache(raw[r2_session], allowed[r2_session])
            forward = decode_direction(r1_cache, r2_cache)
            reverse = decode_direction(r2_cache, r1_cache)
            rows.append(
                {
                    "analysis_mode": analysis_mode,
                    "target_mode": target_mode,
                    "pair_id": pair_id,
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "r1_date": date_label(r1_session),
                    "r2_date": date_label(r2_session),
                    "n_r1_allowed": len(allowed[r1_session]),
                    "n_r2_allowed": len(allowed[r2_session]),
                    "reference_forward": forward,
                    "reference_reverse": reverse,
                    "reference_gap": reverse - forward,
                }
            )
    return pd.DataFrame(rows)


def summarize_repeats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Average across session cells within each random-subset repeat."""
    pivot = long_df.pivot_table(
        index="repeat",
        columns="direction",
        values="random_corr",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.rename(
        columns={"R1->R2": "random_forward", "R2->R1": "random_reverse"}
    )
    pivot["random_gap"] = pivot["random_reverse"] - pivot["random_forward"]
    return pivot


def summarize_cells(long_df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Summarize subset sensitivity separately for each biological date pair.

    Quantiles across repeats describe sensitivity to trial selection; they are
    not treated as confidence intervals over independent sessions.
    """
    long_df = long_df.copy()
    long_df["r1_date"] = long_df["r1_date"].astype(str)
    long_df["r2_date"] = long_df["r2_date"].astype(str)
    paired = long_df.pivot(
        index=[
            "repeat",
            "pair_id",
            "r1_session",
            "r2_session",
            "r1_date",
            "r2_date",
        ],
        columns="direction",
        values="random_corr",
    ).reset_index()
    paired["gap"] = paired["R2->R1"] - paired["R1->R2"]
    rows = []
    keys = ["pair_id", "r1_session", "r2_session", "r1_date", "r2_date"]
    for values, group in paired.groupby(keys, sort=True):
        gaps = group["gap"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, values)),
                "n_repeats": len(group),
                "random_forward_mean": group["R1->R2"].mean(),
                "random_reverse_mean": group["R2->R1"].mean(),
                "random_gap_mean": gaps.mean(),
                "random_gap_sd": gaps.std(ddof=1),
                "random_gap_q025": np.quantile(gaps, 0.025),
                "random_gap_q975": np.quantile(gaps, 0.975),
                "positive_gap_fraction": np.mean(gaps > 0),
            }
        )
    return pd.DataFrame(rows).merge(reference, on=keys, validate="one_to_one")


def crossed_session_bootstrap(cells: pd.DataFrame) -> np.ndarray:
    """Independently resample R1 and R2 dates from the complete 3-by-3 grid."""
    matrix = cells.pivot(
        index="r1_session",
        columns="r2_session",
        values="random_gap_mean",
    ).to_numpy(dtype=float)
    expected_shape = (len(R1_SESSIONS), len(R2_SESSIONS))
    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
        raise ValueError(
            f"crossed bootstrap requires a complete {expected_shape} grid; "
            f"received {matrix.shape}"
        )
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        r1_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        r2_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draws[index] = matrix[np.ix_(r1_indices, r2_indices)].mean()
    return draws


def summarize_inference(cells: pd.DataFrame) -> pd.DataFrame:
    """Summarize cell, R1-date, R2-date, and crossed-session inference."""
    bootstrap = crossed_session_bootstrap(cells)
    fraction_le_zero = float(np.mean(bootstrap <= 0))
    fraction_ge_zero = float(np.mean(bootstrap >= 0))
    crossed_p_two_sided = min(
        1.0,
        2.0 * min(fraction_le_zero, fraction_ge_zero),
    )
    rows = []
    for unit, group_column in (
        ("pair_cells", None),
        ("r1_session_means", "r1_session"),
        ("r2_session_means", "r2_session"),
    ):
        gaps = (
            cells["random_gap_mean"].to_numpy(dtype=float)
            if group_column is None
            else cells.groupby(group_column)["random_gap_mean"].mean().to_numpy(
                dtype=float
            )
        )
        try:
            wilcoxon_p_two_sided = float(
                stats.wilcoxon(gaps, alternative="two-sided", method="auto").pvalue
            )
        except ValueError:
            wilcoxon_p_two_sided = np.nan
        rows.append(
            {
                "unit": unit,
                **gap_test(gaps),
                "wilcoxon_p_two_sided": wilcoxon_p_two_sided,
                "crossed_boot_ci95_low": np.quantile(bootstrap, 0.025),
                "crossed_boot_ci95_high": np.quantile(bootstrap, 0.975),
                "crossed_boot_fraction_le0": fraction_le_zero,
                "crossed_boot_p_two_sided": crossed_p_two_sided,
            }
        )
    return pd.DataFrame(rows)


def run_mode(
    target_mode: str,
    analysis_mode: str,
    raw: dict[str, dict],
    dropout_trials: dict[str, set[int]],
    repeats: int,
    workers: int,
    restart: bool,
) -> dict[str, Path]:
    """Execute one target/mode combination and save restartable CSV outputs.

    When requested, dropout-overlapping trials are removed before equal-N
    sampling.  PCA, CCA, and Kalman are then refit inside every repeat.
    """
    specification = ANALYSIS_MODES[analysis_mode]
    n_trials = int(specification["n_trials"])
    allowed = {}
    for session, values in raw.items():
        trials = available_trial_ids(values)
        if specification["exclude_dropout"]:
            trials = np.asarray(
                [trial for trial in trials if trial not in dropout_trials[session]],
                dtype=int,
            )
        if len(trials) < n_trials:
            raise ValueError(
                f"{session} has {len(trials)} allowed trials, fewer than {n_trials}"
            )
        allowed[session] = trials

    target_label = target_mode.replace("relative_", "")
    stem = f"forget_control_{analysis_mode}_{target_label}"
    outputs = {
        "long": OUT_DIR / f"{stem}_long.csv",
        "repeats": OUT_DIR / f"{stem}_repeats.csv",
        "cells": OUT_DIR / f"{stem}_cells.csv",
        "inference": OUT_DIR / f"{stem}_inference.csv",
    }
    if outputs["long"].exists() and not restart:
        long_df = pd.read_csv(outputs["long"])
        long_df["r1_date"] = long_df["r1_date"].astype(str)
        long_df["r2_date"] = long_df["r2_date"].astype(str)
        completed = set(long_df["repeat"].unique())
    else:
        long_df = pd.DataFrame()
        completed = set()
    pending = [repeat for repeat in range(repeats) if repeat not in completed]
    if pending:
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
                futures = {
                    pool.submit(
                        run_repeat,
                        repeat,
                        SEED,
                        target_mode,
                        analysis_mode,
                        raw,
                        allowed,
                        n_trials,
                    ): repeat
                    for repeat in pending
                }
                for future in as_completed(futures):
                    repeat = futures[future]
                    long_df = pd.concat([long_df, future.result()], ignore_index=True)
                    long_df = long_df.sort_values(
                        ["repeat", "pair_id", "direction"]
                    ).reset_index(drop=True)
                    long_df.to_csv(outputs["long"], index=False)
                    print(
                        f"{target_label} {analysis_mode}: repeat {repeat + 1}/"
                        f"{repeats} saved",
                        flush=True,
                    )

    reference = reference_cells(target_mode, analysis_mode, raw, allowed)
    repeat_summary = summarize_repeats(long_df)
    cells = summarize_cells(long_df, reference)
    inference = summarize_inference(cells)
    repeat_summary.to_csv(outputs["repeats"], index=False)
    cells.to_csv(outputs["cells"], index=False)
    inference.to_csv(outputs["inference"], index=False)

    print(f"\n=== {target_label} / {analysis_mode} ===")
    print(
        f"allowed trial counts: "
        + ", ".join(f"{date_label(s)}={len(allowed[s])}" for s in allowed)
    )
    print(
        f"random forward={repeat_summary.random_forward.mean():.4f}, "
        f"reverse={repeat_summary.random_reverse.mean():.4f}, "
        f"gap={repeat_summary.random_gap.mean():+.4f}, "
        f"repeat interval=[{repeat_summary.random_gap.quantile(.025):+.4f}, "
        f"{repeat_summary.random_gap.quantile(.975):+.4f}]"
    )
    print(cells[["r2_date", "random_gap_mean", "positive_gap_fraction"]].to_string(index=False))
    print(inference.round(6).to_string(index=False))
    return outputs


def parse_args(argv=None):
    """Parse repeat count, worker count, targets, and control modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--targets", nargs="+", choices=TARGET_MODES, default=list(TARGET_MODES))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(ANALYSIS_MODES),
        default=list(ANALYSIS_MODES),
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    """Load the forget sessions and run every requested equal-N control."""
    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = list(R1_SESSIONS) + list(R2_SESSIONS)
    dropout_trials = {session: dropout_overlap_trials(session) for session in sessions}
    print(
        "dropout-overlap trials: "
        + ", ".join(
            f"{date_label(session)}={len(dropout_trials[session])}"
            for session in sessions
        ),
        flush=True,
    )
    for target_mode in args.targets:
        print(f"\nLoading {len(sessions)} {target_mode} sessions ...", flush=True)
        raw = {session: load_raw_session(session, target_mode) for session in sessions}
        for analysis_mode in args.modes:
            run_mode(
                target_mode,
                analysis_mode,
                raw,
                dropout_trials,
                args.repeats,
                args.workers,
                args.restart,
            )
    print(f"\nElapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
