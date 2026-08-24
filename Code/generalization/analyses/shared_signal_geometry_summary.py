"""Summarize shared-space compactness, clarity, and spectrum mediation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "shared_signal_geometry"
OUT_ALL = OUT_DIR / "shared_signal_all.csv"
OUT_NULL = OUT_DIR / "shared_signal_null_all.csv"
OUT_SPECTRUM = OUT_DIR / "shared_signal_spectrum_summary.csv"
OUT_MEDIATION = OUT_DIR / "shared_signal_mediation_summary.csv"

N_SPLITS = 42 * 5 * 5
N_BOOT = 20_000
N_NULL_WORLDS = 100_000
SEED = 20260715
KEYS = ["r1_session", "r2_session", "repeat", "fold"]
PAIR_KEYS = ["r1_session", "r2_session"]

METRICS = {
    "phase_signal_effective_rank": ("lower", True),
    "phase_signal_top1_fraction": ("higher", True),
    "phase_signal_power": ("context", False),
    "phase_residual_power": ("context", False),
    "phase_reliability": ("higher", False),
    "phase_trace_snr": ("higher", False),
    "phase_snr_sum": ("higher", False),
    "phase_snr_effective_rank": ("lower", True),
    "phase_snr_top1_fraction": ("higher", True),
    "phase_snr_axis1": ("higher", False),
    "movement_signal_effective_rank": ("lower", True),
    "movement_signal_top1_fraction": ("higher", True),
    "movement_signal_power": ("context", False),
    "movement_residual_power": ("lower", False),
    "movement_encoding_fraction": ("higher", False),
    "movement_trace_snr": ("higher", False),
    "movement_snr_sum": ("higher", False),
    "movement_snr_effective_rank": ("lower", True),
    "movement_snr_top1_fraction": ("higher", True),
    "movement_snr_axis1": ("higher", False),
    "predictive_effective_rank": ("lower", True),
    "predictive_rank1_energy": ("higher", True),
    "predictive_corr_rank1": ("context", False),
    "predictive_corr_full": ("context", False),
    "predictive_rank1_fraction": ("higher", True),
    "predictive_cv_r2_rank1": ("context", False),
    "predictive_cv_r2_full": ("context", False),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=8)
    return parser.parse_args()


def hierarchical_interval(pair_values, rng):
    grouped = {
        day: group.to_numpy(dtype=float)
        for day, group in pair_values.groupby(level="r2_session")
    }
    days = np.asarray(list(grouped))
    draws = np.empty(N_BOOT)
    for draw in range(N_BOOT):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        day_means = []
        for day in sampled_days:
            values = grouped[day]
            day_means.append(
                rng.choice(values, size=len(values), replace=True).mean()
            )
        draws[draw] = np.mean(day_means)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def balanced_mean(pair_values):
    return float(pair_values.groupby(level="r2_session").mean().mean())


def day_string(pair_values):
    values = pair_values.groupby(level="r2_session").mean()
    return ";".join(f"{value:+.6f}" for value in values)


def spectrum_summary(frame, rng):
    rows = []
    for metric, (prediction, rank_restricted) in METRICS.items():
        subset = frame.copy()
        if rank_restricted:
            subset = subset[subset.shared_rank >= 2]
        subset = subset.dropna(subset=[f"r1_{metric}", f"r2_{metric}"])
        subset["difference"] = subset[f"r2_{metric}"] - subset[f"r1_{metric}"]
        pair = subset.groupby(PAIR_KEYS)[
            [f"r1_{metric}", f"r2_{metric}", "difference"]
        ].mean()
        difference = pair["difference"]
        lo, hi = hierarchical_interval(difference, rng)
        rows.append({
            "metric": metric,
            "compact_or_clear_prediction": prediction,
            "rank_scope": "rank_ge2" if rank_restricted else "all_shared_ranks",
            "r1_mean": balanced_mean(pair[f"r1_{metric}"]),
            "r2_mean": balanced_mean(pair[f"r2_{metric}"]),
            "r2_minus_r1": balanced_mean(difference),
            "descriptive_boot_lo": lo,
            "descriptive_boot_hi": hi,
            "three_r2_day_differences": day_string(difference),
            "n_splits": len(subset),
            "n_pair_days": len(pair),
            "interpretation": "one animal; hierarchical sensitivity interval",
        })
    return pd.DataFrame(rows)


def split_weights(frame):
    counts_pair = frame.groupby(PAIR_KEYS).size().rename("n_split_pair")
    pair_counts_day = (
        frame[PAIR_KEYS].drop_duplicates().groupby("r2_session").size()
        .rename("n_pair_day")
    )
    indexed = frame.set_index(PAIR_KEYS)
    n_split = counts_pair.reindex(indexed.index).to_numpy()
    n_pair = pair_counts_day.reindex(frame.r2_session).to_numpy()
    n_days = frame.r2_session.nunique()
    return 1.0 / (n_days * n_pair * n_split)


def conditional_null(observed, null_frame, rng):
    observed = observed.sort_values(KEYS).reset_index(drop=True)
    grouped = (
        null_frame.groupby(KEYS, sort=True).selective_change.apply(list)
        .reindex(pd.MultiIndex.from_frame(observed[KEYS]))
    )
    if grouped.isna().any():
        raise RuntimeError("null candidates do not cover observed splits")
    candidates = grouped.tolist()
    counts = np.asarray([len(values) for values in candidates], dtype=int)
    width = int(counts.max())
    values = np.full((len(candidates), width), np.nan)
    for index, choices in enumerate(candidates):
        values[index, :len(choices)] = choices
    weights = split_weights(observed)
    observed_statistic = float(np.sum(observed.observed * weights))
    draws = np.empty(N_NULL_WORLDS)
    rows = np.arange(len(candidates))[None, :]
    chunk = 2_000
    for start in range(0, N_NULL_WORLDS, chunk):
        stop = min(start + chunk, N_NULL_WORLDS)
        choices = (
            rng.random((stop - start, len(candidates))) * counts[None, :]
        ).astype(int)
        draws[start:stop] = np.sum(values[rows, choices] * weights, axis=1)
    p_value = float(
        (1 + np.sum(draws >= observed_statistic)) / (N_NULL_WORLDS + 1)
    )
    return observed_statistic, draws, p_value


def mediation_summary(frame, null_frame, rng):
    rows = []
    for mode in sorted(null_frame["mode"].unique()):
        observed_column = f"selective_{mode}"
        for scope, rank_min in (("all_shared_ranks", 1), ("rank_ge2", 2)):
            observed = frame[frame.shared_rank >= rank_min][KEYS + [observed_column]].dropna()
            observed = observed.rename(columns={observed_column: "observed"})
            eligible_null = null_frame[
                (null_frame["mode"] == mode)
                & (null_frame.shared_rank >= rank_min)
            ]
            null_mean = eligible_null.groupby(KEYS).selective_change.mean()
            adjusted = observed.set_index(KEYS).observed - null_mean
            adjusted_pair = adjusted.groupby(PAIR_KEYS).mean()
            lo, hi = hierarchical_interval(adjusted_pair, rng)
            observed_statistic, draws, p_value = conditional_null(
                observed, eligible_null, rng
            )
            rows.append({
                "mode": mode,
                "rank_scope": scope,
                "observed_selective_gap_closure": observed_statistic,
                "random_axis_null_mean": float(draws.mean()),
                "random_axis_null_p95": float(np.percentile(draws, 95)),
                "observed_minus_random": balanced_mean(adjusted_pair),
                "descriptive_boot_lo": lo,
                "descriptive_boot_hi": hi,
                "conditional_randomization_p_one_sided": p_value,
                "three_r2_day_adjusted_effects": day_string(adjusted_pair),
                "n_splits": len(observed),
                "n_null_worlds": N_NULL_WORLDS,
                "interpretation": (
                    "target-calibrated spectrum mediation in fixed sessions; "
                    "not population inference"
                ),
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    result_paths = [
        OUT_DIR / f"shared_signal_job_{index}_of_{args.num_jobs}.csv"
        for index in range(args.num_jobs)
    ]
    null_paths = [
        OUT_DIR / f"shared_signal_null_job_{index}_of_{args.num_jobs}.csv"
        for index in range(args.num_jobs)
    ]
    missing = [path for path in result_paths + null_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing shared-signal shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    null_frame = pd.concat([pd.read_csv(path) for path in null_paths], ignore_index=True)
    if len(frame) != N_SPLITS or frame[KEYS].drop_duplicates().shape[0] != N_SPLITS:
        raise RuntimeError("shared-signal results do not cover all 1,050 splits")
    frame.to_csv(OUT_ALL, index=False)
    null_frame.to_csv(OUT_NULL, index=False)
    rng = np.random.default_rng(SEED)
    spectrum = spectrum_summary(frame, rng)
    mediation = mediation_summary(frame, null_frame, rng)
    spectrum.to_csv(OUT_SPECTRUM, index=False)
    mediation.to_csv(OUT_MEDIATION, index=False)
    print("SPECTRUM")
    print(spectrum.round(5).to_string(index=False))
    print("\nMEDIATION")
    print(mediation.round(5).to_string(index=False))
    print(f"\nsaved {OUT_SPECTRUM}")
    print(f"saved {OUT_MEDIATION}")


if __name__ == "__main__":
    main()
