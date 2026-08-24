"""Compare transductive and target-trial-nested cross-session decoding.

For each ordered R1/R2 pair and each target-session fold:
  - transductive: target PCA and CCA use all target trials;
  - nested: target PCA and CCA use calibration trials only;
  - both methods are scored on the identical held-out target trials.

The train session is fully available in both conditions. Trial-averaged CCA
(the headline pipeline) and single-trial CCA (the dimensionality sensitivity)
are reported separately at d=3 and d=12.

Output: Results/manifold_geometry/nested_cca_validation.csv
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

import decoder_utils as du
from nested_cca import fit_nested_alignment
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS,
    EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS,
    TRIAL_RESULTS,
    UNIT_QUALITIES,
    filter_trials,
    kalman_fit_predict,
    m2_per_trial,
)

warnings.filterwarnings("ignore")

BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 12
DIMS = (3, 12)
N_PHASE_BINS = 30
N_FOLDS = 5
SEED = 20260713
TARGETS = ("relative_position", "relative_velocity")
ALIGNMENT_MODES = ("average", "single_trial")

REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "manifold_geometry" / "nested_cca_validation.csv"


def load_session(session: str, target: str, exclude=()):
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        movement, neural, meta = du.build_decoder_dataset(
            nwb,
            reach,
            target,
            bin_size=BIN_MS / 1000.0,
            unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS,
            trial_window="start_to_peak",
            **SMOOTHER_KW,
        )
    finally:
        io.close()
    movement, neural, meta = filter_trials(movement, neural, meta, exclude)
    neural = du.smooth_neural_causal(
        neural, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS
    )
    return {"X": movement, "Y": neural, "meta": meta}


def target_trial_folds(meta, n_splits=N_FOLDS, seed=SEED):
    """Yield disjoint calibration/evaluation masks grouped by trial."""
    yield from du.kfold_split_by_trial(meta, n_splits=n_splits, random_seed=seed)


def evaluate_pair(train, target, pair_seed: int, alignment_mode: str) -> list[dict]:
    rows = []
    fold_masks = list(target_trial_folds(target["meta"]))

    # Existing transductive alignment: target PCA and CCA see every target trial.
    transductive = fit_nested_alignment(
        train["Y"],
        train["meta"],
        target["Y"],
        target["meta"],
        n_components=K,
        n_phase_bins=N_PHASE_BINS,
        rng=np.random.default_rng(pair_seed),
        alignment_mode=alignment_mode,
    )
    train_transductive = transductive.transform_train(train["Y"])
    target_transductive = transductive.transform_target(target["Y"])

    for fold, (calibration_mask, evaluation_mask) in enumerate(fold_masks):
        calibration_meta = target["meta"][calibration_mask].reset_index(drop=True)
        evaluation_meta = target["meta"][evaluation_mask].reset_index(drop=True)
        nested = fit_nested_alignment(
            train["Y"],
            train["meta"],
            target["Y"][calibration_mask],
            calibration_meta,
            n_components=K,
            n_phase_bins=N_PHASE_BINS,
            rng=np.random.default_rng(pair_seed + fold + 1),
            alignment_mode=alignment_mode,
        )
        train_nested = nested.transform_train(train["Y"])
        target_nested = nested.transform_target(target["Y"][evaluation_mask])

        for dims in DIMS:
            x_eval, pred_nested = kalman_fit_predict(
                train["X"],
                train_nested[:, :dims],
                target["X"][evaluation_mask],
                target_nested[:, :dims],
                evaluation_meta,
            )
            _, pred_transductive = kalman_fit_predict(
                train["X"],
                train_transductive[:, :dims],
                target["X"][evaluation_mask],
                target_transductive[evaluation_mask, :dims],
                evaluation_meta,
            )
            rows.append({
                "fold": fold,
                "alignment_mode": alignment_mode,
                "dims": dims,
                "n_train_trials": train["meta"]["trial_number"].nunique(),
                "n_calibration_trials": calibration_meta["trial_number"].nunique(),
                "n_evaluation_trials": evaluation_meta["trial_number"].nunique(),
                "nested_corr": m2_per_trial(x_eval, pred_nested, evaluation_meta),
                "transductive_corr": m2_per_trial(x_eval, pred_transductive, evaluation_meta),
            })
    return rows


def stable_pair_seed(base_seed: int, target_index: int, pair_index: int) -> int:
    return int(base_seed + target_index * 100_000 + pair_index * 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=sorted(ANIMAL_SESSIONS), default="TS")
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = ANIMAL_SESSIONS[args.animal]
    output = OUT_CSV if args.animal == "TS" else OUT_CSV.with_name(
        f"{OUT_CSV.stem}_{args.animal.lower()}{OUT_CSV.suffix}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs = (
        [(a, b, "R1->R2") for a, b in product(sessions_r1, sessions_r2)]
        + [(b, a, "R2->R1") for a, b in product(sessions_r1, sessions_r2)]
    )
    all_rows = []
    for target_index, target_name in enumerate(TARGETS):
        print(f"loading {target_name} sessions ...", flush=True)
        sessions = tuple(sessions_r1) + tuple(sessions_r2)
        cache = {
            session: load_session(
                session, target_name, EXCLUDE_TRIALS.get(session, ())
            )
            for session in sessions
        }
        for alignment_mode in ALIGNMENT_MODES:
            for pair_index, (train_session, test_session, category) in enumerate(pairs):
                pair_rows = evaluate_pair(
                    cache[train_session],
                    cache[test_session],
                    stable_pair_seed(SEED, target_index, pair_index),
                    alignment_mode,
                )
                for row in pair_rows:
                    row.update(
                        target=target_name,
                        animal=args.animal,
                        pair_category=category,
                        train_session=train_session,
                        test_session=test_session,
                    )
                    all_rows.append(row)
                if (pair_index + 1) % 14 == 0:
                    print(
                        f"  {target_name}|{alignment_mode}: "
                        f"{pair_index + 1}/{len(pairs)} ordered pairs",
                        flush=True,
                    )

    result = pd.DataFrame(all_rows)
    result.to_csv(output, index=False)
    summary = result.groupby(["target", "alignment_mode", "dims", "pair_category"])[
        ["nested_corr", "transductive_corr"]
    ].mean()
    print("\n" + summary.round(3).to_string())
    asymmetry = result.groupby(["target", "alignment_mode", "dims", "pair_category"])[
        "nested_corr"
    ].mean().unstack("pair_category")
    asymmetry["R2->R1 - R1->R2"] = asymmetry["R2->R1"] - asymmetry["R1->R2"]
    print("\nNested directional contrasts:\n" + asymmetry.round(3).to_string())
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
