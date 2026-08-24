"""Random fixed-40 time localization for the original TS interference grid.

For each of 20 deterministic repeats, independently sample 40 trials from both
sides of every one of the 42 R1/R2 cells.  Refit PCA, CCA, and Kalman and retain
the locked 30-ms time-resolved position correlations.  The resulting subset
sensitivity analysis is directly comparable to the forget fixed-40 analysis.
"""
from __future__ import annotations

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "marmoset_matplotlib")
)

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

THIS = Path(__file__).resolve()
GENERALIZATION = THIS.parents[1]
WHY = THIS.parent
for path in (GENERALIZATION, WHY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from big_sweep_phase2_crossday import SESSIONS_R1, SESSIONS_R2  # noqa: E402
from decode_variability_matched_crossday import (  # noqa: E402
    date_label,
    load_raw_session,
    subset_cache,
)
from locked_position_time_resolved import (  # noqa: E402
    DISPLAY_BINS,
    EARLY_STOP_BIN,
    FIRST_EVALUATED_BIN,
    LATE_START_BIN,
    decode_with_predictions,
    plot_profiles,
    time_resolved_metrics,
)
from random_fixed40_crossday_control import (  # noqa: E402
    SEED,
    available_trial_ids,
    sample_trial_ids,
)

REPO = THIS.parents[3]
RESULT_DIR = REPO / "Results" / "workflows" / "generalization"
FIGURE_DIR = RESULT_DIR / "figures"
TARGET_MODE = "relative_position"
N_TRIALS = 40
N_REPEATS = 20
N_WORKERS = min(8, max(1, (os.cpu_count() or 2) // 2))
OUT_PROFILES = RESULT_DIR / "locked_position_time_resolved_ts_random_fixed40.csv"
OUT_GAPS = RESULT_DIR / "locked_position_time_gap_ts_random_fixed40.csv"
OUT_FIGURE = FIGURE_DIR / "fig_locked_position_time_resolved_ts_random_fixed40.png"


def window_label(time_bin: int) -> str:
    """Map each evaluated bin to the locked early, middle, or late window."""
    if time_bin < EARLY_STOP_BIN:
        return "early_30_150ms"
    if time_bin >= LATE_START_BIN:
        return "late_360_510ms"
    return "middle_150_360ms"


def run_repeat(repeat: int, raw: dict[str, dict]) -> pd.DataFrame:
    """Run one deterministic fixed-40 resample for all R1-by-R2 cells."""
    rows = []
    pair_id = 0
    for r1_index, r1_session in enumerate(SESSIONS_R1):
        for r2_index, r2_session in enumerate(SESSIONS_R2):
            pair_id += 1
            cell_seed = np.random.SeedSequence(
                [SEED, repeat, pair_id, r1_index, r2_index]
            )
            rng1, rng2 = [np.random.default_rng(child) for child in cell_seed.spawn(2)]
            r1_trials = sample_trial_ids(
                available_trial_ids(raw[r1_session]), N_TRIALS, rng1
            )
            r2_trials = sample_trial_ids(
                available_trial_ids(raw[r2_session]), N_TRIALS, rng2
            )
            r1_cache = subset_cache(raw[r1_session], r1_trials)
            r2_cache = subset_cache(raw[r2_session], r2_trials)
            for direction, train_cache, test_cache in (
                ("R1->R2", r1_cache, r2_cache),
                ("R2->R1", r2_cache, r1_cache),
            ):
                actual, predicted, meta = decode_with_predictions(
                    train_cache, test_cache
                )
                profile = time_resolved_metrics(actual, predicted, meta)
                profile = profile.loc[
                    profile["time_bin"].between(
                        FIRST_EVALUATED_BIN,
                        DISPLAY_BINS - 1,
                        inclusive="both",
                    )
                ].copy()
                profile.insert(0, "direction", direction)
                profile.insert(0, "r2_session", r2_session)
                profile.insert(0, "r1_session", r1_session)
                profile.insert(0, "r2_date", date_label(r2_session))
                profile.insert(0, "r1_date", date_label(r1_session))
                profile.insert(0, "pair_id", pair_id)
                profile.insert(0, "repeat", repeat)
                rows.append(profile)
    return pd.concat(rows, ignore_index=True)


def pair_directional_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Average subset repeats within each biological cell and direction."""
    group = [
        "pair_id",
        "r1_session",
        "r2_session",
        "time_bin",
        "time_start_ms",
        "time_end_ms",
        "time_center_ms",
        "direction",
    ]
    averaged = profiles.groupby(group, as_index=False).agg(
        corr_mean=("corr_mean", "mean"),
        nrmse_mean=("nrmse_mean", "mean"),
    )
    averaged.insert(0, "animal", "TS random fixed40")
    return averaged


def summarize_gaps(profiles: pd.DataFrame) -> pd.DataFrame:
    """Estimate window-level gaps using R1 date pairs as biological units.

    Trial-subset repeats are averaged before the one-sample tests, so they do
    not artificially increase the inferential sample size.
    """
    profiles = profiles.copy()
    profiles["window"] = profiles["time_bin"].map(window_label)
    directional = (
        profiles.groupby(
            [
                "repeat",
                "pair_id",
                "r1_session",
                "r2_session",
                "r2_date",
                "window",
                "direction",
            ],
            as_index=False,
        )["corr_mean"]
        .mean()
    )
    paired = directional.pivot(
        index=[
            "repeat",
            "pair_id",
            "r1_session",
            "r2_session",
            "r2_date",
            "window",
        ],
        columns="direction",
        values="corr_mean",
    ).reset_index()
    paired["gap"] = paired["R2->R1"] - paired["R1->R2"]
    biological = (
        paired.groupby(
            ["pair_id", "r1_session", "r2_session", "r2_date", "window"],
            as_index=False,
        )["gap"]
        .mean()
    )
    rows = []
    for (r2_session, r2_date), day in biological.groupby(
        ["r2_session", "r2_date"], sort=True
    ):
        wide = day.pivot(index=["pair_id", "r1_session"], columns="window", values="gap")
        wide["late_minus_early"] = (
            wide["late_360_510ms"] - wide["early_30_150ms"]
        )
        for window in (
            "early_30_150ms",
            "middle_150_360ms",
            "late_360_510ms",
            "late_minus_early",
        ):
            values = wide[window].dropna().to_numpy(dtype=float)
            test = stats.ttest_1samp(values, 0.0, alternative="greater")
            rows.append(
                {
                    "r2_session": r2_session,
                    "r2_date": r2_date,
                    "window": window,
                    "n_r1_pairs": len(values),
                    "mean_gap": values.mean(),
                    "gap_sd": values.std(ddof=1),
                    "t": test.statistic,
                    "p_one_sided_gt0": test.pvalue,
                    "positive_fraction": np.mean(values > 0),
                    "n_subset_repeats": N_REPEATS,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Generate instantaneous fixed-40 profiles, tests, and the summary plot."""
    sessions = tuple(SESSIONS_R1) + tuple(SESSIONS_R2)
    raw = {}
    for index, session in enumerate(sessions, start=1):
        print(f"[TS random fixed40 cache {index}/{len(sessions)}] {session}", flush=True)
        raw[session] = load_raw_session(session, TARGET_MODE)

    rows = []
    print(f"Running {N_REPEATS} random fixed40 repeats", flush=True)
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(run_repeat, repeat, raw) for repeat in range(N_REPEATS)]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                print(f"  repeat {completed}/{N_REPEATS} complete", flush=True)
    profiles = pd.concat(rows, ignore_index=True)
    gaps = summarize_gaps(profiles)
    pair_profiles = pair_directional_profiles(profiles)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(OUT_PROFILES, index=False)
    gaps.to_csv(OUT_GAPS, index=False)
    plot_profiles(pair_profiles, OUT_FIGURE, "TS random fixed40")
    print("\nDirectional-gap summary")
    print(gaps.round(6).to_string(index=False))
    print(f"\nsaved {OUT_PROFILES}")
    print(f"saved {OUT_GAPS}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
