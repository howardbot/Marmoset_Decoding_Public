"""Summarise the trial-aware private read-out Kalman intervention."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from private_readout_crossfit import OUT_DIR
from private_readout_crossfit_summary import (
    add_subspace_contrasts,
    summarize_pair_metrics,
)

OUT_ALL = OUT_DIR / "kalman_subspace_trial_aware_all.csv"
OUT_PAIR = OUT_DIR / "kalman_subspace_trial_aware_pair_means.csv"
OUT_DAY = OUT_DIR / "trial_aware_private_by_r2_session.csv"
OUT_SUMMARY = OUT_DIR / "trial_aware_private_summary.csv"

METRICS = [
    "fwd_full",
    "rev_full",
    "gap_full",
    "fwd_minus_r1_private",
    "rev_minus_r1_private",
    "selective_remove_r1_private",
    "selective_random_ablate_r1",
    "r1_private_excess_over_random",
    "fwd_minus_r2_private",
    "rev_minus_r2_private",
    "selective_remove_r2_private",
    "selective_random_ablate_r2",
    "r2_private_excess_over_random",
    "fwd_shared",
    "rev_shared",
    "selective_shared",
    "shared_excess_over_random",
    "rank_shared",
    "rank_private_r1",
    "rank_private_r2",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = [
        OUT_DIR / f"kalman_subspace_trial_aware_job_{index}_of_{args.num_jobs}.csv"
        for index in range(args.num_jobs)
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trial-aware shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected_rows = 42 * 5 * 5 * 3
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(frame)}")
    if set(frame.transition_mode) != {"trial_aware"}:
        raise RuntimeError("Non-trial-aware rows found in trial-aware shards")

    frame = add_subspace_contrasts(frame)
    frame["gap_after_remove_r1_private"] = (
        frame.rev_minus_r1_private - frame.fwd_minus_r1_private
    )
    frame["gap_after_remove_r2_private"] = (
        frame.rev_minus_r2_private - frame.fwd_minus_r2_private
    )
    metrics = METRICS + [
        "gap_after_remove_r1_private",
        "gap_after_remove_r2_private",
    ]
    frame.to_csv(OUT_ALL, index=False)
    pair = frame.groupby(
        ["cosine_threshold", "r1_session", "r2_session"], as_index=False
    )[metrics].mean()
    pair.to_csv(OUT_PAIR, index=False)

    day_rows, summary_rows = summarize_pair_metrics(
        pair,
        ["cosine_threshold"],
        metrics,
        "trial_aware_kalman_subspace_intervention",
    )
    days = pd.DataFrame(day_rows)
    summary = pd.DataFrame(summary_rows)
    days.to_csv(OUT_DAY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    headline = summary[
        summary.metric.isin([
            "fwd_full",
            "rev_full",
            "gap_full",
            "fwd_minus_r1_private",
            "rev_minus_r1_private",
            "gap_after_remove_r1_private",
            "selective_remove_r1_private",
            "r1_private_excess_over_random",
            "selective_remove_r2_private",
            "r2_private_excess_over_random",
            "rank_shared",
            "rank_private_r1",
        ])
    ]
    print(headline.round(4).to_string(index=False))
    print(f"\nsaved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
