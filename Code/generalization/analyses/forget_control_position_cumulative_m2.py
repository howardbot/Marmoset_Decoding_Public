"""Cumulative headline-M2 position gap for the TS forget control.

Unlike the cross-trial single-bin diagnostic, this analysis computes the same
within-trial temporal correlation as the headline decoder on progressively
longer trial prefixes.  Consequently, every curve endpoint must equal the
saved full-reach directional score and gap.
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

from cumulative_m2_timecourse import cumulative_m2_metrics  # noqa: E402
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
from locked_position_time_resolved import decode_with_predictions  # noqa: E402

REPO = THIS.parents[3]
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry"
FIGURE_DIR = OUT_DIR / "figures"
TARGET_MODE = "relative_position"
N_REPEATS = 50
N_WORKERS = min(6, max(1, (os.cpu_count() or 2) // 2))
ANALYSIS_MODES = {
    "fixed40": {"n_trials": 40, "exclude_dropout": False},
    "dropout_clean_fixed39": {"n_trials": 39, "exclude_dropout": True},
}
OUT_LONG = OUT_DIR / "forget_control_position_cumulative_m2_long.csv"
OUT_CELLS = OUT_DIR / "forget_control_position_cumulative_m2_cells.csv"
OUT_CHECKPOINTS = OUT_DIR / "forget_control_position_cumulative_m2_checkpoints.csv"
OUT_FIGURE = FIGURE_DIR / "fig_forget_control_position_cumulative_m2.png"


def available_ids(raw: dict) -> np.ndarray:
    """Return sorted trial IDs that survived decoder preprocessing."""
    return np.sort(raw["meta"]["trial_number"].drop_duplicates().to_numpy(int))


def directional_profile(
    train_cache: dict,
    test_cache: dict,
    max_bin: int,
    direction: str,
) -> pd.DataFrame:
    """Decode one direction and compute the headline metric at every prefix."""
    actual, predicted, meta = decode_with_predictions(train_cache, test_cache)
    profile = cumulative_m2_metrics(actual, predicted, meta, max_bin)
    profile.insert(0, "direction", direction)
    return profile


def run_repeat(
    repeat: int,
    analysis_mode: str,
    n_trials: int,
    raw: dict[str, dict],
    allowed: dict[str, np.ndarray],
    max_bin: int,
) -> pd.DataFrame:
    """Generate forward and reverse cumulative profiles for one resample."""
    r1_session = R1_SESSIONS[0]
    rows = []
    for pair_id, r2_session in enumerate(R2_SESSIONS, start=1):
        cell_seed = np.random.SeedSequence([SEED, repeat, pair_id])
        rng1, rng2 = [np.random.default_rng(child) for child in cell_seed.spawn(2)]
        r1_trials = sample_trial_ids(allowed[r1_session], n_trials, rng1)
        r2_trials = sample_trial_ids(allowed[r2_session], n_trials, rng2)
        r1_cache = subset_cache(raw[r1_session], r1_trials)
        r2_cache = subset_cache(raw[r2_session], r2_trials)
        for direction, train_cache, test_cache in (
            ("R1->R2", r1_cache, r2_cache),
            ("R2->R1", r2_cache, r1_cache),
        ):
            profile = directional_profile(
                train_cache, test_cache, max_bin, direction
            )
            profile.insert(0, "r2_date", date_label(r2_session))
            profile.insert(0, "r1_date", date_label(r1_session))
            profile.insert(0, "r2_session", r2_session)
            profile.insert(0, "r1_session", r1_session)
            profile.insert(0, "pair_id", pair_id)
            profile.insert(0, "repeat", repeat)
            profile.insert(0, "analysis_mode", analysis_mode)
            rows.append(profile)
    return pd.concat(rows, ignore_index=True)


def pair_profiles(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pair directions within each repeat and calculate ``R2->R1 - R1->R2``."""
    index = [
        "analysis_mode",
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
    """Summarize cumulative profiles across trial-subset repeats per date pair."""
    keys = [
        "analysis_mode",
        "pair_id",
        "r1_session",
        "r2_session",
        "r1_date",
        "r2_date",
        "time_bin",
        "time_end_ms",
    ]
    rows = []
    for key, group in paired.groupby(keys, sort=True):
        gaps = group["gap"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_repeats": len(group),
                "forward_mean": group["R1->R2"].mean(),
                "reverse_mean": group["R2->R1"].mean(),
                "gap_mean": gaps.mean(),
                "gap_q025": np.quantile(gaps, 0.025),
                "gap_q975": np.quantile(gaps, 0.975),
                "positive_gap_fraction": np.mean(gaps > 0),
            }
        )
    return pd.DataFrame(rows)


def checkpoint_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Extract interpretable prefix checkpoints plus the full-reach endpoint."""
    checkpoints = (150, 360, 510, 900)
    rows = []
    for (mode, r2_date), group in cells.groupby(["analysis_mode", "r2_date"]):
        group = group.sort_values("time_end_ms")
        for checkpoint in checkpoints:
            eligible = group.loc[group["time_end_ms"] <= checkpoint]
            row = eligible.iloc[-1] if len(eligible) else group.iloc[0]
            rows.append(
                {
                    "analysis_mode": mode,
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
                "analysis_mode": mode,
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
    """Verify that the last cumulative value reproduces the saved M2 score.

    This assertion protects against accidentally comparing the cumulative
    within-trial metric with the distinct across-trial single-bin metric.
    """
    references = {
        "fixed40": OUT_DIR / "forget_control_fixed40_position_long.csv",
        "dropout_clean_fixed39": OUT_DIR
        / "forget_control_dropout_clean_fixed39_position_long.csv",
    }
    endpoint = paired.loc[paired["time_bin"] == paired["time_bin"].max()].copy()
    cumulative_long = endpoint.melt(
        id_vars=["analysis_mode", "repeat", "pair_id"],
        value_vars=["R1->R2", "R2->R1"],
        var_name="direction",
        value_name="cumulative_corr",
    )
    maximum_error = 0.0
    for mode, path in references.items():
        reference = pd.read_csv(path)[
            ["repeat", "pair_id", "direction", "random_corr"]
        ]
        observed = cumulative_long.loc[cumulative_long["analysis_mode"] == mode]
        merged = observed.merge(reference, on=["repeat", "pair_id", "direction"])
        error = np.max(np.abs(merged["cumulative_corr"] - merged["random_corr"]))
        maximum_error = max(maximum_error, float(error))
        print(f"endpoint validation {mode}: max abs error={error:.3e}", flush=True)
    if maximum_error > 1e-10:
        raise AssertionError(f"cumulative endpoint mismatch: {maximum_error}")


def plot_cells(cells: pd.DataFrame) -> None:
    """Plot cumulative gap curves for each R2 day and dropout-control mode."""
    dates = [date_label(session) for session in R2_SESSIONS]
    styles = {
        "fixed40": ("#F58518", "-", "Forget fixed40"),
        "dropout_clean_fixed39": ("#E45756", "--", "Dropout-clean fixed39"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True, sharey=True)
    for column, date in enumerate(dates):
        for mode, (color, linestyle, label) in styles.items():
            day = cells.loc[
                (cells["analysis_mode"] == mode)
                & (cells["r2_date"].astype(str) == date)
                & (cells["time_end_ms"] <= 900)
            ].sort_values("time_end_ms")
            axes[column].plot(
                day["time_end_ms"],
                day["gap_mean"],
                color=color,
                linestyle=linestyle,
                linewidth=2.3,
                label=label,
            )
            endpoint = cells.loc[
                (cells["analysis_mode"] == mode)
                & (cells["r2_date"].astype(str) == date)
            ].sort_values("time_end_ms").iloc[-1]
            axes[column].axhline(
                endpoint["gap_mean"],
                color=color,
                linestyle=":",
                linewidth=1.1,
                alpha=0.75,
            )
        axes[column].set_title(f"Forget R2 {date[-4:]}", weight="bold")
        axes[column].axhline(0, color="black", linewidth=0.8, alpha=0.65)
        for x in (150, 360, 510):
            axes[column].axvline(x, color="grey", linewidth=0.8, alpha=0.35)
        axes[column].grid(alpha=0.2)
        axes[column].set_xlim(120, 900)
        axes[column].set_xlabel("Prefix included from reach start (ms)")
    axes[0].set_ylabel("Cumulative headline-style gap")
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Forget-control cumulative within-trial position gap\n"
        "Dotted horizontals are exact full-reach headline endpoints",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run cumulative forget-control decoding, validate it, and save outputs."""
    sessions = R1_SESSIONS + R2_SESSIONS
    raw = {session: load_raw_session(session, TARGET_MODE) for session in sessions}
    max_bin = max(int(raw[session]["meta"]["local_bin"].max()) for session in sessions)
    allowed_all = {session: available_ids(raw[session]) for session in sessions}
    dropouts = {session: dropout_overlap_trials(session) for session in sessions}
    allowed_clean = {
        session: np.asarray(
            [trial for trial in allowed_all[session] if trial not in dropouts[session]],
            dtype=int,
        )
        for session in sessions
    }
    rows = []
    for mode, config in ANALYSIS_MODES.items():
        allowed = allowed_clean if config["exclude_dropout"] else allowed_all
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = [
                    executor.submit(
                        run_repeat,
                        repeat,
                        mode,
                        config["n_trials"],
                        raw,
                        allowed,
                        max_bin,
                    )
                    for repeat in range(N_REPEATS)
                ]
                for completed, future in enumerate(as_completed(futures), start=1):
                    rows.append(future.result())
                    if completed % 10 == 0:
                        print(f"{mode}: {completed}/{N_REPEATS}", flush=True)
    long_df = pd.concat(rows, ignore_index=True)
    paired = pair_profiles(long_df)
    validate_endpoints(paired)
    cells = summarize_cells(paired)
    checkpoints = checkpoint_summary(cells)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_LONG, index=False)
    cells.to_csv(OUT_CELLS, index=False)
    checkpoints.to_csv(OUT_CHECKPOINTS, index=False)
    plot_cells(cells)
    print("\nCheckpoint summary")
    print(checkpoints.round(4).to_string(index=False))
    print(f"\nsaved {OUT_LONG}")
    print(f"saved {OUT_CELLS}")
    print(f"saved {OUT_CHECKPOINTS}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
