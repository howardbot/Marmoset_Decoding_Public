"""Manifold geometry analysis: alignment index, dimensionality, Procrustes.

This script asks the Sadtler/Oby/Elsayed question directly on your data:
  "Does R2's neural manifold contain R1's manifold plus extra dimensions?"

For every ordered pair (session_i, session_j):
  1. Fit CCA on each session's canonical reach trajectory (re-uses cross_day
     decoder's per-day PCA + 30-bin canonical reach).
  2. Project full single-trial PC activity from both sessions into the SHARED
     k-dim canonical space (so neuron-identity differences are bridged).
  3. Compute covariance of each session in canonical space.
  4. Report:
       - alignment(j -> basis_i): does session_j's variance fit inside
         session_i's top-d manifold? (Elsayed 2016 normalized index, in [0,1])
       - alignment(i -> basis_j): the reverse direction.
       - outside_var(j vs i): fraction of j's variance OUTSIDE i's top-d
         basis -- the Oby "new patterns" metric.
       - procrustes_disparity on canonical reach trajectories.
       - participation ratio of each session in canonical space.
  5. Per-session: participation ratio in each session's own PC space (k=15)
     and number of components needed for 90% variance.

The headline test for B != A asymmetry is:
   mean alignment(R1 -> basis_R2) > mean alignment(R2 -> basis_R1)
   equivalently: outside_var(R2 vs R1_basis) > outside_var(R1 vs R2_basis)

Outputs (Results/manifold_geometry/):
  - per_session_metrics.csv     : dim, PR, n_units, n_trials per session
  - pairwise_metrics_long.csv   : one row per ordered (i, j) pair
  - alignment_matrix.csv        : matrix of alignment(j -> basis_i)
  - outside_matrix.csv          : matrix of outside_var(j vs basis_i)
  - procrustes_matrix.csv       : matrix of procrustes_disparity
  - epoch_summary.csv           : R1-R1, R1-R2, R2-R1, R2-R2 block means
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parents[1]
_CODE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_THIS_DIR))

from manifold_align import (
    cca_align,
    apply_alignment,
    participation_ratio,
    dim_for_variance,
    top_d_basis,
    alignment_index,
    outside_manifold_variance,
    procrustes_distance,
    heldout_canonical_correlations,
)
from cross_day_decoder import (
    list_sessions,
    session_epoch,
    session_date,
    build_session_cache_entry,
    K_PCS,
)

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
RESULTS_DIR = REPO_ROOT / "Results" / "manifold_geometry"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Effective subspace dimensionality used for alignment index. Elsayed 2016
# uses 8-10 in M1; Sadtler 2014 uses ~10 factors from FA. We default to a
# value smaller than K_PCS (=15) so the question is non-trivial.
D_EFFECTIVE = 8

# Taking data after PCA, W is the CCA alignment matrix, mean is the training data of CCA mean trajectory
def session_covariance_in_canonical(Y_pc, W, mean):
    """Project a session's full single-trial PC activity into canonical space
    using a CCA rotation, then return the covariance matrix and the projected
    activity. Mean-centering matches what cca_align used (trajectory mean)."""
    # Projecting full single-trial PC activity to the canonical space, Y_canon is n_bins x K
    Y_canon = apply_alignment(Y_pc, W, mean)
    # Re-center so covariance reflects spread about its own mean (CCA rotation
    # was fit relative to trajectory mean, not single-trial mean).
    Y_canon = Y_canon - Y_canon.mean(axis=0, keepdims=True)
    # Calculating the covariance matrix
    C = (Y_canon.T @ Y_canon) / max(1, Y_canon.shape[0] - 1)
    return C, Y_canon

# Calculating neural dimensionality and geometry
def per_session_metrics(cache):
    """Per-session geometry in the session's own PC space (no CCA)."""
    rows = []
    for s, data in cache.items():
        Y_pc = data["Y_pc"]
        # Centered PC activity covariance.
        Yc = Y_pc - Y_pc.mean(axis=0, keepdims=True)
        C = (Yc.T @ Yc) / max(1, Yc.shape[0] - 1)
        # get PR and 90% 80% threshold
        pr = participation_ratio(C)
        d90 = dim_for_variance(C, fraction=0.90)
        d80 = dim_for_variance(C, fraction=0.80)
        rows.append({
            "session": s,
            "date": session_date(s),
            "epoch": session_epoch(s),
            "n_units": int(data["PCA_V"].shape[0]),
            "n_bins": int(Y_pc.shape[0]),
            "n_trials": int(data["meta"]["trial_number"].nunique()),
            "k_pcs": int(Y_pc.shape[1]),
            "participation_ratio": pr,
            "dim_for_80pct": d80,
            "dim_for_90pct": d90,
        })
    return pd.DataFrame(rows)

# compare two sessions manifold geometry
def pairwise_metric(cache, train_s, test_s, d_eff=D_EFFECTIVE):
    """Compute alignment / outside / procrustes / cancorr for one ordered pair.

    Convention used here:
      train_s defines the "reference manifold" (its top-d eigenbasis in
      canonical space). test_s is the session whose covariance we test against
      that basis. ``alignment(test -> basis_train)`` answers "does the test
      session's variance live inside the train session's manifold?".
    """
    train_data = cache[train_s]
    test_data = cache[test_s]
    # Fit CCA on canonical reach trajectories (the same step used by the
    # cross-day decoder for cross-day Kalman). This bridges the per-session
    # PCA bases into a shared k-dim space.
    W_tr, W_te, m_tr, m_te = cca_align(train_data["traj"], test_data["traj"])
    tr_c_traj = (train_data["traj"] - m_tr) @ W_tr
    te_c_traj = (test_data["traj"] - m_te) @ W_te
    # Held-out canonical correlations as a diagnostic. The in-sample version
    # (corrcoef of tr_c_traj vs te_c_traj on the 30-bin trajectory with 15 dims) is
    # over-determined and saturates to ~1.0 for ANY pair; this fits CCA on one
    # trial-half and scores on the other, so it collapses past the genuinely-shared
    # dimensions. (Does not affect the Golub triage, which uses alignment/outside/
    # procrustes, not cancorr.)
    cancorrs = heldout_canonical_correlations(train_data, test_data)
    # Project full single-trial PC activity into canonical space and form the
    # covariance for each session.
    C_train, _ = session_covariance_in_canonical(
        train_data["Y_pc"], W_tr, m_tr,
    )
    C_test, _ = session_covariance_in_canonical(
        test_data["Y_pc"], W_te, m_te,
    )
    # Manifold = top-d eigenbasis of the train-session covariance in canonical
    # space. Then ask how much test-session variance fits inside it.
    B_train, _ = top_d_basis(C_train, d_eff)
    align_test_in_train = alignment_index(C_test, B_train)
    outside_test_vs_train = outside_manifold_variance(C_test, B_train)
    # ALSO compute a trajectory-based alignment. This uses the covariance of
    # the trial-averaged canonical reach (30 phase bins x k PCs), which is
    # immune to CCA's per-CC variance scaling because the trajectories are
    # what CCA was fit on. It answers "does session_j's reach-related
    # subspace overlap session_i's reach-related subspace?".
    C_tr_traj = (tr_c_traj - tr_c_traj.mean(0)).T @ (tr_c_traj - tr_c_traj.mean(0))
    C_te_traj = (te_c_traj - te_c_traj.mean(0)).T @ (te_c_traj - te_c_traj.mean(0))
    B_tr_traj, _ = top_d_basis(C_tr_traj, d_eff)
    align_traj_test_in_train = alignment_index(C_te_traj, B_tr_traj)
    outside_traj_test_vs_train = outside_manifold_variance(C_te_traj, B_tr_traj)
    # Procrustes on the CCA-aligned canonical trajectories: anything left here
    # is shape mismatch that the linear CCA rotation could not absorb.
    proc = procrustes_distance(tr_c_traj, te_c_traj)
    # Participation ratios in canonical space (a coordinate-free dim per side).
    pr_train_canon = participation_ratio(C_train)
    pr_test_canon = participation_ratio(C_test)
    return {
        "train_session": train_s,
        "test_session": test_s,
        "train_date": session_date(train_s),
        "test_date": session_date(test_s),
        "train_epoch": session_epoch(train_s),
        "test_epoch": session_epoch(test_s),
        "d_eff": d_eff,
        "cancorr_mean": float(np.nanmean(cancorrs)),
        "cancorr_top": float(cancorrs[0]),
        "align_test_in_train": align_test_in_train,
        "outside_test_vs_train": outside_test_vs_train,
        "align_traj_test_in_train": align_traj_test_in_train,
        "outside_traj_test_vs_train": outside_traj_test_vs_train,
        "procrustes_disparity": proc,
        "pr_train_canonical": pr_train_canon,
        "pr_test_canonical": pr_test_canon,
    }


def main():
    sessions = list_sessions()
    print(f"Found {len(sessions)} sessions")
    n_r1 = sum(1 for s in sessions if session_epoch(s) == "r1")
    n_r2 = sum(1 for s in sessions if session_epoch(s) == "r2")
    print(f"  R1: {n_r1}  R2: {n_r2}")
    print(f"  K_PCS={K_PCS}  D_EFFECTIVE={D_EFFECTIVE}")
    print()

    # ---- 1. build per-session cache (PCA + canonical reach trajectory) ----
    t0 = time.time()
    cache = {}
    failed = []
    for i, s in enumerate(sessions, 1):
        print(f"[cache {i}/{len(sessions)}] {s}")
        try:
            cache[s] = build_session_cache_entry(s)
        except Exception as e:
            print(f"  SKIP {s}: {type(e).__name__}: {e}")
            failed.append(s)
    sessions = [s for s in sessions if s not in failed]
    print(f"\nCache built for {len(sessions)} session(s) in {time.time() - t0:.1f}s\n")

    # ---- 2. per-session geometry ----
    per_sess = per_session_metrics(cache)
    per_sess.to_csv(RESULTS_DIR / "per_session_metrics.csv", index=False)
    print("=== per-session geometry (own PC space) ===")
    print(per_sess.round(3).to_string(index=False))
    print()
    epoch_dim_summary = per_sess.groupby("epoch")[
        ["participation_ratio", "dim_for_80pct", "dim_for_90pct"]
    ].mean().round(3)
    print("=== mean dim per epoch ===")
    print(epoch_dim_summary.to_string())
    print()

    # ---- 3. pairwise alignment / outside / procrustes ----
    records = []
    n = len(sessions)
    for i, train_s in enumerate(sessions):
        for j, test_s in enumerate(sessions):
            if i == j:
                # Self-pairs: CCA is rank-deficient with two identical inputs,
                # skip and let downstream interpret diagonal as NA.
                records.append({
                    "train_session": train_s,
                    "test_session": test_s,
                    "train_date": session_date(train_s),
                    "test_date": session_date(test_s),
                    "train_epoch": session_epoch(train_s),
                    "test_epoch": session_epoch(test_s),
                    "d_eff": D_EFFECTIVE,
                    "cancorr_mean": np.nan,
                    "cancorr_top": np.nan,
                    "align_test_in_train": 1.0,
                    "outside_test_vs_train": 0.0,
                    "align_traj_test_in_train": 1.0,
                    "outside_traj_test_vs_train": 0.0,
                    "procrustes_disparity": 0.0,
                    "pr_train_canonical": np.nan,
                    "pr_test_canonical": np.nan,
                })
                continue
            try:
                rec = pairwise_metric(cache, train_s, test_s, d_eff=D_EFFECTIVE)
            except Exception as e:
                print(f"  pair FAILED {train_s} -> {test_s}: {type(e).__name__}: {e}")
                rec = {
                    "train_session": train_s,
                    "test_session": test_s,
                    "train_date": session_date(train_s),
                    "test_date": session_date(test_s),
                    "train_epoch": session_epoch(train_s),
                    "test_epoch": session_epoch(test_s),
                    "d_eff": D_EFFECTIVE,
                    "cancorr_mean": np.nan,
                    "cancorr_top": np.nan,
                    "align_test_in_train": np.nan,
                    "outside_test_vs_train": np.nan,
                    "align_traj_test_in_train": np.nan,
                    "outside_traj_test_vs_train": np.nan,
                    "procrustes_disparity": np.nan,
                    "pr_train_canonical": np.nan,
                    "pr_test_canonical": np.nan,
                }
            records.append(rec)
        print(f"[pairs {i+1}/{n}] processed train={session_date(train_s)} ({session_epoch(train_s)})")
    long_df = pd.DataFrame(records)
    long_df.to_csv(RESULTS_DIR / "pairwise_metrics_long.csv", index=False)

    # ---- 4. pivot to matrices ----
    def pivot(col):
        return long_df.pivot(
            index="train_date", columns="test_date", values=col,
        )
    pivot("align_test_in_train").to_csv(RESULTS_DIR / "alignment_matrix.csv")
    pivot("outside_test_vs_train").to_csv(RESULTS_DIR / "outside_matrix.csv")
    pivot("align_traj_test_in_train").to_csv(RESULTS_DIR / "alignment_traj_matrix.csv")
    pivot("outside_traj_test_vs_train").to_csv(RESULTS_DIR / "outside_traj_matrix.csv")
    pivot("procrustes_disparity").to_csv(RESULTS_DIR / "procrustes_matrix.csv")
    pivot("cancorr_mean").to_csv(RESULTS_DIR / "cancorr_matrix.csv")

    print("\n=== alignment(test_session -> train_session manifold) ===")
    print(pivot("align_test_in_train").round(3).to_string())
    print("\n=== outside-manifold variance (test vs train basis) ===")
    print(pivot("outside_test_vs_train").round(3).to_string())
    print("\n=== procrustes disparity ===")
    print(pivot("procrustes_disparity").round(4).to_string())

    # ---- 5. epoch-block summary + asymmetry test ----
    nodiag = long_df[long_df["train_session"] != long_df["test_session"]].copy()
    block = (
        nodiag.groupby(["train_epoch", "test_epoch"])[
            ["align_test_in_train", "outside_test_vs_train",
             "align_traj_test_in_train", "outside_traj_test_vs_train",
             "procrustes_disparity", "cancorr_mean"]
        ]
        .mean()
        .round(3)
    )
    block.to_csv(RESULTS_DIR / "epoch_summary.csv")
    print("\n=== epoch-block means (train_epoch, test_epoch) ===")
    print(block.to_string())

    # Asymmetry: A(test in R1_basis) for R1->R2 pairs vs R2->R1 pairs.
    # Headline test mapping to (b) B superset A:
    #   - train=R2, test=R1  -> alignment HIGH  (R2 basis contains R1)
    #   - train=R1, test=R2  -> alignment LOW   (R1 basis misses R2 new dims)
    r1_to_r2 = nodiag[(nodiag.train_epoch == "r1") & (nodiag.test_epoch == "r2")]
    r2_to_r1 = nodiag[(nodiag.train_epoch == "r2") & (nodiag.test_epoch == "r1")]
    print("\n=== headline asymmetry (single-trial canonical covariance) ===")
    print(
        f"  align(R2 test in R1 basis)  mean = "
        f"{r1_to_r2['align_test_in_train'].mean():.3f} "
        f"n_pairs={len(r1_to_r2)}"
    )
    print(
        f"  align(R1 test in R2 basis)  mean = "
        f"{r2_to_r1['align_test_in_train'].mean():.3f} "
        f"n_pairs={len(r2_to_r1)}"
    )
    print(
        f"  outside(R2 vs R1 basis)     mean = "
        f"{r1_to_r2['outside_test_vs_train'].mean():.3f}"
    )
    print(
        f"  outside(R1 vs R2 basis)     mean = "
        f"{r2_to_r1['outside_test_vs_train'].mean():.3f}"
    )
    print("  -> B superset A predicts: align(R1 in R2) > align(R2 in R1).")
    print(
        f"\n=== headline asymmetry (trial-averaged reach trajectory) ==="
    )
    print(
        f"  align_traj(R2 test in R1 basis)  mean = "
        f"{r1_to_r2['align_traj_test_in_train'].mean():.3f}"
    )
    print(
        f"  align_traj(R1 test in R2 basis)  mean = "
        f"{r2_to_r1['align_traj_test_in_train'].mean():.3f}"
    )
    print(
        f"  outside_traj(R2 vs R1)            mean = "
        f"{r1_to_r2['outside_traj_test_vs_train'].mean():.3f}"
    )
    print(
        f"  outside_traj(R1 vs R2)            mean = "
        f"{r2_to_r1['outside_traj_test_vs_train'].mean():.3f}"
    )
    print(
        f"\n=== procrustes (low = same reach shape after CCA) ==="
    )
    print(
        f"  procrustes(R1, R2)  mean = "
        f"{pd.concat([r1_to_r2, r2_to_r1])['procrustes_disparity'].mean():.4f}"
    )
    print(
        f"  procrustes(R1, R1)  mean = "
        f"{nodiag[(nodiag.train_epoch == 'r1') & (nodiag.test_epoch == 'r1')]['procrustes_disparity'].mean():.4f}"
    )
    print(
        f"  procrustes(R2, R2)  mean = "
        f"{nodiag[(nodiag.train_epoch == 'r2') & (nodiag.test_epoch == 'r2')]['procrustes_disparity'].mean():.4f}"
    )

    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
