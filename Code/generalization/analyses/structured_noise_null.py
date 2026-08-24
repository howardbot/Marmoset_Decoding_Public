"""Structured residual-covariance conditioning null for the decode asymmetry.

After CCA alignment, the linear effects of instantaneous position and
finite-difference velocity are regressed out of each session's canonical activity. For each ordered pair,
the positive-semidefinite part of (target residual covariance - train residual
covariance) is injected into the training activity. This preserves correlated,
low-dimensional covariance directions that the original independent white-noise
null omitted. It does not match temporal autocorrelation and is not a full noise
model.

Output: Results/manifold_geometry/structured_noise_null.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from dimension_sweep import align_full
from global_state_bridge import kin_residual, load
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS,
    EXCLUDE_TRIALS,
    kalman_fit_predict,
    m2_per_trial,
)

warnings.filterwarnings("ignore")

K = 12
DIMS = (3, 12)
N_DRAWS = 20
SEED = 20260713
TARGET = "relative_position"

REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "manifold_geometry" / "structured_noise_null.csv"


def positive_covariance_increment(
    train_residual: np.ndarray,
    target_residual: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """PSD part of target covariance minus train covariance."""
    train_cov = np.cov(np.asarray(train_residual, dtype=float), rowvar=False)
    target_cov = np.cov(np.asarray(target_residual, dtype=float), rowvar=False)
    difference = (target_cov - train_cov + (target_cov - train_cov).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(difference)
    eigenvalues = np.where(eigenvalues > tolerance, eigenvalues, 0.0)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def sample_covariance_noise(
    n_rows: int,
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw zero-mean Gaussian rows with a possibly rank-deficient covariance."""
    covariance = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    return rng.standard_normal((n_rows, covariance.shape[0])) @ factor.T


def inject_structured_increment(
    train_activity: np.ndarray,
    train_residual: np.ndarray,
    target_residual: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Add the target-minus-train PSD residual-covariance increment."""
    increment = positive_covariance_increment(train_residual, target_residual)
    noise = sample_covariance_noise(len(train_activity), increment, rng)
    return np.asarray(train_activity, dtype=float) + noise, increment


def decode(train, train_activity, target, target_activity, dims: int) -> float:
    x_eval, prediction = kalman_fit_predict(
        train["Xp"],
        train_activity[:, :dims],
        target["Xp"],
        target_activity[:, :dims],
        target["meta"],
    )
    return m2_per_trial(x_eval, prediction, target["meta"])


def prepare_cache(sessions):
    cache = {
        session: load(session, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    return cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal", choices=sorted(ANIMAL_SESSIONS), default="TS"
    )
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = (
        list(sessions) for sessions in ANIMAL_SESSIONS[args.animal]
    )
    suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
    output_csv = OUT_CSV.with_name(f"{OUT_CSV.stem}{suffix}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cache = prepare_cache(sessions_r1 + sessions_r2)
    ordered_pairs = (
        [(a, b, "R1->R2") for a, b in product(sessions_r1, sessions_r2)]
        + [(b, a, "R2->R1") for a, b in product(sessions_r1, sessions_r2)]
    )
    rows = []
    for pair_index, (train_session, target_session, category) in enumerate(ordered_pairs):
        train = cache[train_session]
        target = cache[target_session]
        alignment_rng = np.random.default_rng(SEED + pair_index * 1000)
        train_aligned, target_aligned = align_full(
            "single_trial",
            K,
            (train["Xp"], train["Ypc"], train["meta"], train["tavg"]),
            (target["Xp"], target["Ypc"], target["meta"], target["tavg"]),
            alignment_rng,
        )
        if train_aligned is None:
            continue
        train_residual = kin_residual(train_aligned, train["Kin"])
        target_residual = kin_residual(target_aligned, target["Kin"])
        increment = positive_covariance_increment(train_residual, target_residual)

        for dims in DIMS:
            baseline = decode(train, train_aligned, target, target_aligned, dims)
            for draw in range(args.draws):
                draw_rng = np.random.default_rng(
                    SEED + pair_index * 1000 + dims * 30 + draw
                )
                structured = train_aligned + sample_covariance_noise(
                    len(train_aligned), increment, draw_rng
                )
                rows.append({
                    "animal": args.animal,
                    "target": TARGET,
                    "pair_category": category,
                    "train_session": train_session,
                    "test_session": target_session,
                    "dims": dims,
                    "draw": draw,
                    "baseline_corr": baseline,
                    "structured_corr": decode(
                        train, structured, target, target_aligned, dims
                    ),
                    "train_residual_trace": np.trace(np.cov(train_residual, rowvar=False)),
                    "target_residual_trace": np.trace(np.cov(target_residual, rowvar=False)),
                    "injected_trace": np.trace(increment),
                })
        if (pair_index + 1) % 14 == 0:
            print(f"completed {pair_index + 1}/{len(ordered_pairs)} ordered pairs", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)
    means = result.groupby(["target", "dims", "pair_category"])[
        ["baseline_corr", "structured_corr"]
    ].mean().unstack("pair_category")
    for metric in ("baseline_corr", "structured_corr"):
        means[(metric, "asymmetry")] = (
            means[(metric, "R2->R1")] - means[(metric, "R1->R2")]
        )
    print("\n" + means.round(3).to_string())
    print(f"\nsaved {output_csv}")


if __name__ == "__main__":
    main()
