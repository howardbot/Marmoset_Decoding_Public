"""Aggregate conditional randomization distributions for private specificity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "private_readout_specificity"
OUT_ALL = OUT_DIR / "private_specificity_all.csv"
OUT_DISTRIBUTION = OUT_DIR / "private_specificity_null_distribution.csv"
OUT_SUMMARY = OUT_DIR / "private_specificity_summary.csv"

N_SPLITS = 42 * 5 * 5
N_WORLDS = 100_000
CHUNK_SIZE = 2_000
SEED = 20260715
SPLIT_KEYS = ["r1_session", "r2_session", "repeat", "fold"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=8)
    return parser.parse_args()


def randomization_distribution(candidate_lists, rng):
    counts = np.asarray([len(values) for values in candidate_lists], dtype=int)
    width = int(counts.max())
    values = np.full((len(candidate_lists), width), np.nan)
    for index, candidates in enumerate(candidate_lists):
        values[index, :len(candidates)] = candidates
    draws = np.empty(N_WORLDS)
    rows = np.arange(len(candidate_lists))[None, :]
    for start in range(0, N_WORLDS, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, N_WORLDS)
        choices = (
            rng.random((stop - start, len(candidate_lists))) * counts[None, :]
        ).astype(int)
        draws[start:stop] = values[rows, choices].mean(axis=1)
    return draws


def summarize_scope(frame, scope, rng):
    observed = frame[
        (frame.null_family == "r1_potent_principal")
        & frame.is_observed_private
    ].set_index(SPLIT_KEYS).selective_effect
    observed_statistic = float(observed.mean())
    rows = []
    distribution_rows = []
    for family in ("r1_potent_principal", "r1_output_null"):
        family_frame = frame[frame.null_family == family]
        grouped = family_frame.groupby(SPLIT_KEYS).selective_effect.apply(list)
        if len(grouped) != len(observed):
            raise RuntimeError(f"Incomplete {family} candidates for {scope}")
        draws = randomization_distribution(grouped.tolist(), rng)
        p_value = float((1 + np.sum(draws >= observed_statistic)) / (N_WORLDS + 1))
        rows.append({
            "scope": scope,
            "null_family": family,
            "observed_private": observed_statistic,
            "null_mean": float(draws.mean()),
            "null_sd": float(draws.std(ddof=1)),
            "null_p95": float(np.percentile(draws, 95)),
            "null_p99": float(np.percentile(draws, 99)),
            "conditional_randomization_p_one_sided": p_value,
            "n_splits": int(len(observed)),
            "n_null_worlds": N_WORLDS,
            "interpretation": (
                "conditional geometric specificity in fixed sessions; "
                "not animal-population inference"
            ),
        })
        distribution_rows.extend({
            "scope": scope,
            "null_family": family,
            "world": index,
            "statistic": value,
            "observed_private": observed_statistic,
        } for index, value in enumerate(draws))
    return rows, distribution_rows


def main():
    args = parse_args()
    paths = [
        OUT_DIR / f"private_specificity_job_{index}_of_{args.num_jobs}.csv"
        for index in range(args.num_jobs)
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing specificity shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame[SPLIT_KEYS].drop_duplicates().shape[0] != N_SPLITS:
        raise RuntimeError("Specificity results do not cover all 1,050 splits")
    if frame.isna().any().any():
        raise RuntimeError("Missing values in specificity results")
    frame.to_csv(OUT_ALL, index=False)

    rng = np.random.default_rng(SEED)
    summary_rows, distribution_rows = summarize_scope(frame, "all_r2_days", rng)
    for r2_session, day_frame in frame.groupby("r2_session"):
        rows, distributions = summarize_scope(day_frame, r2_session, rng)
        summary_rows.extend(rows)
        distribution_rows.extend(distributions)
    summary = pd.DataFrame(summary_rows)
    distribution = pd.DataFrame(distribution_rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    distribution.to_csv(OUT_DISTRIBUTION, index=False)
    print(summary.round(6).to_string(index=False))
    print(f"\nsaved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
