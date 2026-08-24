"""Session-level summary of cross-validated read-out compactness."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "readout_compactness"
OUT_ALL = OUT_DIR / "readout_compactness_all.csv"
OUT_SESSION = OUT_DIR / "readout_compactness_by_session.csv"
OUT_SUMMARY = OUT_DIR / "readout_compactness_summary.csv"

N_SESSIONS = 17
N_MODES = 2
N_REPEATS = 5
N_FOLDS = 5
N_BOOT = 20_000
SEED = 20260715

METRIC_DIRECTIONS = {
    "effective_rank": "lower",
    "rank1_energy": "higher",
    "rank2_energy": "higher",
    "corr_rank1": "higher",
    "corr_rank2": "higher",
    "corr_rank3": "context",
    "corr_best_rank": "lower",
    "corr_rank1_best_fraction": "higher",
    "corr_rank1_minus_rank3": "higher",
    "corr_rank2_minus_rank3": "higher",
    "corr_rank1_fraction": "higher",
    "corr_rank2_fraction": "higher",
    "corr_rank2_gain": "lower",
    "corr_rank3_gain": "lower",
    "cv_r2_rank1": "higher",
    "cv_r2_rank2": "higher",
    "cv_r2_rank3": "context",
    "cv_r2_best_rank": "lower",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=8)
    return parser.parse_args()


def bootstrap_difference(r1, r2, rng):
    indices_r1 = rng.integers(0, len(r1), size=(N_BOOT, len(r1)))
    indices_r2 = rng.integers(0, len(r2), size=(N_BOOT, len(r2)))
    draws = r2[indices_r2].mean(axis=1) - r1[indices_r1].mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def balanced_r1_percentile(r1, r2_mean):
    means = np.asarray([
        np.mean(r1[list(indices)]) for indices in combinations(range(len(r1)), 3)
    ])
    return float(100.0 * np.mean(means <= r2_mean))


def main():
    args = parse_args()
    paths = [
        OUT_DIR / f"readout_compactness_job_{index}_of_{args.num_jobs}.csv"
        for index in range(args.num_jobs)
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing read-out compactness shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = N_SESSIONS * N_MODES * N_REPEATS * N_FOLDS
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(frame)}")
    frame["corr_rank1_fraction"] = frame.corr_rank1 / frame.corr_rank3
    frame["corr_rank2_fraction"] = frame.corr_rank2 / frame.corr_rank3
    frame["corr_rank2_gain"] = frame.corr_rank2 - frame.corr_rank1
    frame["corr_rank3_gain"] = frame.corr_rank3 - frame.corr_rank2
    corr_values = frame[["corr_rank1", "corr_rank2", "corr_rank3"]].to_numpy()
    r2_values = frame[["cv_r2_rank1", "cv_r2_rank2", "cv_r2_rank3"]].to_numpy()
    frame["corr_best_rank"] = np.argmax(corr_values, axis=1) + 1
    frame["cv_r2_best_rank"] = np.argmax(r2_values, axis=1) + 1
    frame["corr_rank1_best_fraction"] = (frame.corr_best_rank == 1).astype(float)
    frame["corr_rank1_minus_rank3"] = frame.corr_rank1 - frame.corr_rank3
    frame["corr_rank2_minus_rank3"] = frame.corr_rank2 - frame.corr_rank3
    frame.to_csv(OUT_ALL, index=False)

    metrics = list(METRIC_DIRECTIONS)
    sessions = frame.groupby(
        ["mode", "epoch", "session"], as_index=False
    )[metrics].mean()
    sessions.to_csv(OUT_SESSION, index=False)

    rng = np.random.default_rng(SEED)
    rows = []
    for mode, mode_frame in sessions.groupby("mode"):
        for metric, prediction in METRIC_DIRECTIONS.items():
            r1 = mode_frame.loc[mode_frame.epoch == "R1", metric].to_numpy()
            r2 = mode_frame.loc[mode_frame.epoch == "R2", metric].to_numpy()
            lo, hi = bootstrap_difference(r1, r2, rng)
            difference = float(r2.mean() - r1.mean())
            if prediction == "higher":
                direction_matches = difference > 0
            elif prediction == "lower":
                direction_matches = difference < 0
            else:
                direction_matches = False
            rows.append({
                "mode": mode,
                "metric": metric,
                "compact_prediction": prediction,
                "r1_mean": float(r1.mean()),
                "r2_mean": float(r2.mean()),
                "r2_minus_r1": difference,
                "session_boot_lo": float(lo),
                "session_boot_hi": float(hi),
                "r2_min": float(r2.min()),
                "r2_max": float(r2.max()),
                "r2_mean_percentile_among_3_r1_subsets": balanced_r1_percentile(
                    r1, float(r2.mean())
                ),
                "direction_matches_compact_prediction": direction_matches,
                "interpretation": "descriptive sessions from one animal; R2 n=3",
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    headline = summary[summary.metric.isin([
        "effective_rank", "rank1_energy", "rank2_energy",
        "corr_best_rank", "corr_rank1_best_fraction",
        "corr_rank1_minus_rank3", "corr_rank2_minus_rank3",
        "corr_rank3_gain", "corr_rank3",
    ])]
    print(headline.round(4).to_string(index=False))
    print(f"\nsaved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
