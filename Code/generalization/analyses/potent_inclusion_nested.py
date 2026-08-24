"""Three-way cross-fitted potent-inclusion diagnostic.

For each R1/R2 pair, trials from both sessions are divided into disjoint roles:
  1. alignment: fit each PCA basis and the CCA rotations;
  2. potent definition: fit each neural-to-kinematic read-out and shared basis;
  3. evaluation: score the opposite session as a held-out target.

Roles rotate three times. Target evaluation kinematics therefore never define
the alignment or potent subspace. The source-day decoder may use all source
trials because they are training data, not target evaluation labels.

Output: Results/workflows/manifold_geometry/potent_inclusion_nested.csv
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
    three_way_trial_masks,
)
from nested_cca_validation import load_session
from potent_inclusion import decode_corr, ridge_fit, row_basis, shared_basis
from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2

K = 12
N_PHASE_BINS = 30
SEED = 20260713
TARGET = "relative_position"

REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "potent_inclusion_nested.csv"


def fit_role_alignment(a, b, alignment_a, alignment_b, seed: int) -> NestedCCAAlignment:
    pca_a = fit_pca_projector(a["Y"][alignment_a], K)
    pca_b = fit_pca_projector(b["Y"][alignment_b], K)
    pc_a = pca_a.transform(a["Y"][alignment_a])
    pc_b = pca_b.transform(b["Y"][alignment_b])
    meta_a = a["meta"][alignment_a].reset_index(drop=True)
    meta_b = b["meta"][alignment_b].reset_index(drop=True)
    rotations = fit_phase_matched_cca(
        pc_a,
        meta_a,
        pc_b,
        meta_b,
        n_components=K,
        n_phase_bins=N_PHASE_BINS,
        rng=np.random.default_rng(seed),
    )
    return NestedCCAAlignment(
        train_pca=pca_a,
        target_pca=pca_b,
        train_rotation=rotations[0],
        target_rotation=rotations[1],
        train_cca_mean=rotations[2],
        target_cca_mean=rotations[3],
    )


def evaluate_pair(a, b, pair_seed: int) -> list[dict]:
    masks_a = three_way_trial_masks(a["meta"], pair_seed)
    masks_b = three_way_trial_masks(b["meta"], pair_seed + 1)
    rows = []
    for rotation in range(3):
        alignment_a, definition_a, evaluation_a = (
            masks_a[rotation], masks_a[(rotation + 1) % 3], masks_a[(rotation + 2) % 3]
        )
        alignment_b, definition_b, evaluation_b = (
            masks_b[rotation], masks_b[(rotation + 1) % 3], masks_b[(rotation + 2) % 3]
        )
        model = fit_role_alignment(
            a, b, alignment_a, alignment_b, pair_seed + rotation * 10
        )
        aligned_a = model.transform_train(a["Y"])
        aligned_b = model.transform_target(b["Y"])

        basis_a, rank_a = row_basis(ridge_fit(aligned_a[definition_a], a["X"][definition_a]))
        basis_b, rank_b = row_basis(ridge_fit(aligned_b[definition_b], b["X"][definition_b]))
        common, cosines = shared_basis(basis_a, basis_b)
        meta_eval_a = a["meta"][evaluation_a].reset_index(drop=True)
        meta_eval_b = b["meta"][evaluation_b].reset_index(drop=True)
        row = {
            "rotation": rotation,
            "rank_r1": rank_a,
            "rank_r2": rank_b,
            "n_shared": 0 if common is None else common.shape[1],
            "mean_principal_cosine": float(np.mean(cosines[:min(rank_a, rank_b)])),
            "fwd_full": decode_corr(
                aligned_a, a["X"], aligned_b[evaluation_b], b["X"][evaluation_b], meta_eval_b
            ),
            "rev_full": decode_corr(
                aligned_b, b["X"], aligned_a[evaluation_a], a["X"][evaluation_a], meta_eval_a
            ),
        }
        if common is None:
            row["fwd_shared"] = np.nan
            row["rev_shared"] = np.nan
        else:
            row["fwd_shared"] = decode_corr(
                aligned_a, a["X"], aligned_b[evaluation_b], b["X"][evaluation_b],
                meta_eval_b, basis=common
            )
            row["rev_shared"] = decode_corr(
                aligned_b, b["X"], aligned_a[evaluation_a], a["X"][evaluation_a],
                meta_eval_a, basis=common
            )
        rows.append(row)
    return rows


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print("loading relative_position sessions ...", flush=True)
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in SESSIONS_R1 + SESSIONS_R2
    }
    rows = []
    for pair_index, (r1, r2) in enumerate(product(SESSIONS_R1, SESSIONS_R2)):
        for row in evaluate_pair(cache[r1], cache[r2], SEED + pair_index * 100):
            row.update(target=TARGET, r1_session=r1, r2_session=r2)
            rows.append(row)
        if (pair_index + 1) % 14 == 0:
            print(f"completed {pair_index + 1}/42 R1/R2 pairs", flush=True)

    result = pd.DataFrame(rows)
    result["gap_full"] = result.rev_full - result.fwd_full
    result["gap_shared"] = result.rev_shared - result.fwd_shared
    result["selective_gap_change"] = result.gap_shared - result.gap_full
    result.to_csv(OUT_CSV, index=False)
    print("\n" + result[
        ["fwd_full", "rev_full", "fwd_shared", "rev_shared", "gap_full", "gap_shared"]
    ].mean().round(3).to_string())
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
