"""Cumulative headline-M2 gap for the original TS interference experiment."""
from __future__ import annotations

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "marmoset_matplotlib")
)

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

from big_sweep_phase2_crossday import SESSIONS_R1, SESSIONS_R2  # noqa: E402
from cumulative_m2_timecourse import cumulative_m2_metrics  # noqa: E402
from decode_variability_matched_crossday import (  # noqa: E402
    date_label,
    load_raw_session,
    subset_cache,
)
from locked_position_time_resolved import decode_with_predictions  # noqa: E402
from random_fixed40_crossday_control import (  # noqa: E402
    SEED,
    available_trial_ids,
    sample_trial_ids,
)

REPO = THIS.parents[3]
OUT_DIR = REPO / "Results" / "generalization"
FIGURE_DIR = OUT_DIR / "figures"
TARGET_MODE = "relative_position"
N_TRIALS = 40
N_REPEATS = 20
N_WORKERS = min(8, max(1, (os.cpu_count() or 2) // 2))
OUT_LONG = OUT_DIR / "interference_position_cumulative_m2_random_fixed40_long.csv"
OUT_CELLS = OUT_DIR / "interference_position_cumulative_m2_random_fixed40_cells.csv"
OUT_BY_R2 = OUT_DIR / "interference_position_cumulative_m2_random_fixed40_by_r2.csv"
OUT_CHECKPOINTS = OUT_DIR / "interference_position_cumulative_m2_random_fixed40_checkpoints.csv"
OUT_FIGURE = FIGURE_DIR / "fig_interference_position_cumulative_m2_random_fixed40.png"


def run_repeat(repeat: int, raw: dict[str, dict], max_bin: int) -> pd.DataFrame:
    """Run one fixed-40 resample across all 42 original experiment cells."""
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
                profile = cumulative_m2_metrics(actual, predicted, meta, max_bin)
                profile.insert(0, "direction", direction)
                profile.insert(0, "r2_date", date_label(r2_session))
                profile.insert(0, "r1_date", date_label(r1_session))
                profile.insert(0, "r2_session", r2_session)
                profile.insert(0, "r1_session", r1_session)
                profile.insert(0, "pair_id", pair_id)
                profile.insert(0, "repeat", repeat)
                rows.append(profile)
    return pd.concat(rows, ignore_index=True)


def pair_profiles(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pair transfer directions and compute the cumulative directional gap."""
    index = [
        "repeat",
        "pair_id",
        "r1_session",
        "r2_session",
        "r1_date",
        "r2_date",
        "time_bin",
        "time_end_ms",
    ]
    paired = long_df.pivot(index=index, columns="direction", values="cumulative_corr").reset_index()
    paired["gap"] = paired["R2->R1"] - paired["R1->R2"]
    return paired


def summarize_cells(paired: pd.DataFrame) -> pd.DataFrame:
    """Average cumulative curves over random subsets within each date pair."""
    keys = [
        "pair_id",
        "r1_session",
        "r2_session",
        "r1_date",
        "r2_date",
        "time_bin",
        "time_end_ms",
    ]
    return (
        paired.groupby(keys, as_index=False)
        .agg(
            forward_mean=("R1->R2", "mean"),
            reverse_mean=("R2->R1", "mean"),
            gap_mean=("gap", "mean"),
            gap_sd_subsets=("gap", "std"),
        )
    )


def summarize_by_r2(cells: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 14 R1 cells separately for each independent R2 date."""
    keys = ["r2_session", "r2_date", "time_bin", "time_end_ms"]
    return (
        cells.groupby(keys, as_index=False)
        .agg(
            n_r1_pairs=("r1_session", "nunique"),
            forward_mean=("forward_mean", "mean"),
            reverse_mean=("reverse_mean", "mean"),
            gap_mean=("gap_mean", "mean"),
            gap_sd_across_r1=("gap_mean", "std"),
        )
    )


def checkpoint_summary(by_r2: pd.DataFrame) -> pd.DataFrame:
    """Extract fixed temporal checkpoints and each curve's final endpoint."""
    rows = []
    for r2_date, group in by_r2.groupby("r2_date"):
        group = group.sort_values("time_end_ms")
        for checkpoint in (150, 360, 510, 900):
            eligible = group.loc[group["time_end_ms"] <= checkpoint]
            row = eligible.iloc[-1] if len(eligible) else group.iloc[0]
            rows.append(
                {
                    "r2_date": r2_date,
                    "checkpoint": f"through_{checkpoint}ms",
                    "time_end_ms": row["time_end_ms"],
                    "forward_mean": row["forward_mean"],
                    "reverse_mean": row["reverse_mean"],
                    "gap_mean": row["gap_mean"],
                }
            )
        row = group.iloc[-1]
        rows.append(
            {
                "r2_date": r2_date,
                "checkpoint": "full_reach_endpoint",
                "time_end_ms": row["time_end_ms"],
                "forward_mean": row["forward_mean"],
                "reverse_mean": row["reverse_mean"],
                "gap_mean": row["gap_mean"],
            }
        )
    return pd.DataFrame(rows)


def validate_endpoints(paired: pd.DataFrame) -> None:
    """Confirm that cumulative endpoints equal the saved fixed-40 scores."""
    endpoint = paired.loc[paired["time_bin"] == paired["time_bin"].max()]
    observed = endpoint.melt(
        id_vars=["repeat", "pair_id"],
        value_vars=["R1->R2", "R2->R1"],
        var_name="direction",
        value_name="cumulative_corr",
    )
    reference = pd.read_csv(
        REPO / "Results" / "manifold_geometry" / "random_fixed40_position_long.csv"
    )
    reference = reference.loc[reference["repeat"] < N_REPEATS][
        ["repeat", "pair_id", "direction", "random_corr"]
    ]
    merged = observed.merge(reference, on=["repeat", "pair_id", "direction"])
    error = np.max(np.abs(merged["cumulative_corr"] - merged["random_corr"]))
    print(f"endpoint validation: max abs error={error:.3e}", flush=True)
    if error > 1e-10:
        raise AssertionError(f"cumulative endpoint mismatch: {error}")


def plot_by_r2(by_r2: pd.DataFrame) -> None:
    """Plot one R1-averaged cumulative gap curve for each R2 date."""
    dates = sorted(by_r2["r2_date"].astype(str).unique())
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True, sharey=True)
    for column, date in enumerate(dates):
        day = by_r2.loc[
            (by_r2["r2_date"].astype(str) == date)
            & (by_r2["time_end_ms"] <= 900)
        ].sort_values("time_end_ms")
        axes[column].plot(
            day["time_end_ms"], day["gap_mean"], color="#4C78A8", linewidth=2.4
        )
        endpoint = by_r2.loc[
            by_r2["r2_date"].astype(str) == date
        ].sort_values("time_end_ms").iloc[-1]
        axes[column].axhline(
            endpoint["gap_mean"], color="#4C78A8", linestyle=":", linewidth=1.2
        )
        axes[column].axhline(0, color="black", linewidth=0.8, alpha=0.65)
        for x in (150, 360, 510):
            axes[column].axvline(x, color="grey", linewidth=0.8, alpha=0.35)
        axes[column].set_title(f"Original R2 {date[-4:]}", weight="bold")
        axes[column].set_xlim(120, 900)
        axes[column].set_xlabel("Prefix included from reach start (ms)")
        axes[column].grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative headline-style gap")
    fig.suptitle(
        "Original interference cumulative within-trial position gap\n"
        "Random fixed40; dotted horizontals are exact full-reach endpoints",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run, validate, summarize, plot, and save the cumulative analysis."""
    sessions = tuple(SESSIONS_R1) + tuple(SESSIONS_R2)
    raw = {}
    for index, session in enumerate(sessions, start=1):
        print(f"[cumulative cache {index}/{len(sessions)}] {session}", flush=True)
        raw[session] = load_raw_session(session, TARGET_MODE)
    max_bin = max(int(raw[session]["meta"]["local_bin"].max()) for session in sessions)
    rows = []
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [
                executor.submit(run_repeat, repeat, raw, max_bin)
                for repeat in range(N_REPEATS)
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                print(f"repeat {completed}/{N_REPEATS}", flush=True)
    long_df = pd.concat(rows, ignore_index=True)
    paired = pair_profiles(long_df)
    validate_endpoints(paired)
    cells = summarize_cells(paired)
    by_r2 = summarize_by_r2(cells)
    checkpoints = checkpoint_summary(by_r2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_LONG, index=False)
    cells.to_csv(OUT_CELLS, index=False)
    by_r2.to_csv(OUT_BY_R2, index=False)
    checkpoints.to_csv(OUT_CHECKPOINTS, index=False)
    plot_by_r2(by_r2)
    print("\nCheckpoint summary")
    print(checkpoints.round(4).to_string(index=False))
    print(f"\nsaved {OUT_LONG}")
    print(f"saved {OUT_CELLS}")
    print(f"saved {OUT_BY_R2}")
    print(f"saved {OUT_CHECKPOINTS}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
