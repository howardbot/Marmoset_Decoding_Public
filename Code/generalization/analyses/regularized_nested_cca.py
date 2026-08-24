"""Regularized CCA sensitivity under target-trial nesting.

This is a ridge sweep, not a hyperparameter selection procedure. Every ridge
value is reported. Target-session PCA and CCA use calibration trials only, and
the identical held-out target folds are scored at d=3 and d=12.

Output: Results/workflows/manifold_geometry/regularized_nested_cca.csv
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from nested_cca import (
    NestedCCAAlignment,
    fit_pca_projector,
    fit_phase_matched_cca,
)
from nested_cca_validation import load_session, target_trial_folds
from big_sweep_phase2_crossday import (
    EXCLUDE_TRIALS,
    SESSIONS_R1,
    SESSIONS_R2,
    kalman_fit_predict,
    m2_per_trial,
)

K = 12
DIMS = (3, 12)
RIDGES = (0.0, 0.01, 0.1, 1.0)
N_PHASE_BINS = 30
SEED = 20260713
TARGET = "relative_position"

REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "regularized_nested_cca.csv"


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sessions = SESSIONS_R1 + SESSIONS_R2
    print("loading relative_position sessions ...", flush=True)
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    full_pca = {
        session: fit_pca_projector(cache[session]["Y"], K) for session in sessions
    }
    pairs = (
        [(a, b, "R1->R2") for a, b in product(SESSIONS_R1, SESSIONS_R2)]
        + [(b, a, "R2->R1") for a, b in product(SESSIONS_R1, SESSIONS_R2)]
    )

    rows = []
    for pair_index, (train_session, target_session, category) in enumerate(pairs):
        train = cache[train_session]
        target = cache[target_session]
        train_pca = full_pca[train_session]
        train_pc = train_pca.transform(train["Y"])
        for fold, (calibration_mask, evaluation_mask) in enumerate(
            target_trial_folds(target["meta"])
        ):
            calibration_meta = target["meta"][calibration_mask].reset_index(drop=True)
            evaluation_meta = target["meta"][evaluation_mask].reset_index(drop=True)
            target_pca = fit_pca_projector(target["Y"][calibration_mask], K)
            target_calibration_pc = target_pca.transform(target["Y"][calibration_mask])

            for ridge in RIDGES:
                fit_seed = SEED + pair_index * 100 + fold
                rotations = fit_phase_matched_cca(
                    train_pc,
                    train["meta"],
                    target_calibration_pc,
                    calibration_meta,
                    n_components=K,
                    n_phase_bins=N_PHASE_BINS,
                    rng=np.random.default_rng(fit_seed),
                    ridge=ridge,
                )
                model = NestedCCAAlignment(
                    train_pca=train_pca,
                    target_pca=target_pca,
                    train_rotation=rotations[0],
                    target_rotation=rotations[1],
                    train_cca_mean=rotations[2],
                    target_cca_mean=rotations[3],
                )
                train_aligned = model.transform_train(train["Y"])
                target_aligned = model.transform_target(target["Y"][evaluation_mask])
                for dims in DIMS:
                    x_eval, prediction = kalman_fit_predict(
                        train["X"],
                        train_aligned[:, :dims],
                        target["X"][evaluation_mask],
                        target_aligned[:, :dims],
                        evaluation_meta,
                    )
                    rows.append({
                        "target": TARGET,
                        "pair_category": category,
                        "train_session": train_session,
                        "test_session": target_session,
                        "fold": fold,
                        "ridge": ridge,
                        "dims": dims,
                        "corr": m2_per_trial(x_eval, prediction, evaluation_meta),
                    })
        if (pair_index + 1) % 14 == 0:
            print(f"completed {pair_index + 1}/{len(pairs)} ordered pairs", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    means = result.groupby(["ridge", "dims", "pair_category"])["corr"].mean().unstack()
    means["asymmetry"] = means["R2->R1"] - means["R1->R2"]
    print("\n" + means.round(3).to_string())
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
