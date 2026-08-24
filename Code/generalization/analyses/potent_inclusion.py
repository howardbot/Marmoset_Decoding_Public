"""Directional potent-inclusion test: is the R1->R2 / R2->R1 asymmetry caused by
R1 relying on read-out directions that are NULL (unread) in R2?

Design (agreed with Qin after discarding three weaker metrics):
  For every (R1 day a, R2 day b) pair, in the shared CCA-aligned canonical space:
    1. Fit a linear read-out W_a (R1) and W_b (R2): canonical neural -> kinematics.
       Potent subspace = row space of W (the directions the decoder reads).
    2. SHARED potent subspace = the read-out directions the two days agree on
       (principal vectors of B_a vs B_b with cosine > COS_THR). "Private" = one
       day's potent that is the other day's null.
    3. Cross-day decode restricted to the SHARED subspace vs the FULL manifold,
       BOTH directions:
         gap_full   = dec(R2->R1 | full)   - dec(R1->R2 | full)      (the asymmetry)
         gap_shared = dec(R2->R1 | shared) - dec(R1->R2 | shared)
       Hypothesis: gap_shared << gap_full, and it shrinks by R1->R2 RISING (rescued)
       while R2->R1 barely moves.
    4. Built-in control: R2->R1 is restricted the same way. If restriction were pure
       regularisation, both arms would move together; a selective rescue of the
       FAILING arm (R1->R2) is what implicates R1's private (R2-null) read-out.

This first implementation is partly circular: R2 kinematics define the shared
subspace and the same R2 trials are evaluated. See ``potent_inclusion_nested.py``
for the target-trial-separated version.

Config: single-trial phase-matched CCA, K_PCS=12, ridge read-out, both targets.
n(R2)=3 -> report effect sizes, not p-values. Reads NWB -> HatLab env.
Output: Results/manifold_geometry/potent_inclusion.csv (+ figure).
"""
from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from manifold_align import pca_neural
from dimension_sweep import align_full
from big_sweep_phase2_crossday import (
    SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 12
L2 = 1e-2
COS_THR = 0.5          # principal-angle cosine above which a read-out direction is "shared"
SEED = 0
TARGETS = ["relative_position", "relative_velocity"]
REPO_ROOT = _THIS.parents[2]
OUT_CSV = REPO_ROOT / "Results" / "manifold_geometry" / "potent_inclusion.csv"
FIG = REPO_ROOT / "Results" / "manifold_geometry" / "figures" / "fig_potent_inclusion.png"


def load(session, target, exclude=()):
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, target, bin_size=BIN_MS / 1000.0, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    Ysm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS)
    Ypc = pca_neural(Ysm, k=K)[0]
    return X, Ypc, meta, None


def ridge_fit(Y, X):
    """W (d x kin): X ~ Y @ W, ridge."""
    return np.linalg.solve(Y.T @ Y + L2 * np.eye(Y.shape[1]), Y.T @ X)


def row_basis(W, tol=1e-8):
    """Orthonormal basis (K x rank) of row space of W (= potent subspace)."""
    U, S, Vt = np.linalg.svd(W.T, full_matrices=False)   # W.T is (kin x K)
    rank = int((S > tol * S[0]).sum())
    return Vt[:rank].T, rank


def decode_corr(Ytr, Xtr, Yte, Xte, meta_te, basis=None):
    """Ridge read-out fit on train, applied to test; per-trial mean corr over kin dims.
    If basis (K x dS) given, decode inside that subspace only."""
    if basis is not None:
        Ytr, Yte = Ytr @ basis, Yte @ basis
    W = ridge_fit(Ytr, Xtr)
    pred = Yte @ W
    # per-trial correlation, averaged over trials and kin dims
    cors = []
    for _, idx in meta_te.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 4:
            continue
        ck = [np.corrcoef(Xte[idx, d], pred[idx, d])[0, 1] for d in range(Xte.shape[1])]
        cors.append(np.nanmean(ck))
    return float(np.nanmean(cors)) if cors else np.nan


def shared_basis(Ba, Bb, cos_thr=COS_THR):
    """Shared potent subspace: bisectors of principal vectors of Ba vs Bb with
    principal cosine > cos_thr. Returns (K x dS) orthonormal basis + the cosines."""
    M = Ba.T @ Bb                       # (ra x rb)
    U, S, Vt = np.linalg.svd(M)
    keep = S > cos_thr
    if keep.sum() == 0:
        return None, S
    pa = Ba @ U[:, keep]                # A's principal dirs (K x dS)
    pb = Bb @ Vt.T[:, keep]            # B's principal dirs (K x dS)
    bis = pa + pb                       # bisector ~ the common direction
    Q, _ = np.linalg.qr(bis)
    return Q, S


def run_target(target, rng):
    cache = {s: load(s, target, EXCLUDE_TRIALS.get(s, [])) for s in SESSIONS_R1 + SESSIONS_R2}
    rows = []
    for a, b in product(SESSIONS_R1, SESSIONS_R2):
        Ya, Yb = align_full("single_trial", K, cache[a], cache[b], rng)
        if Ya is None:
            continue
        Xa, ma = cache[a][0], cache[a][2]
        Xb, mb = cache[b][0], cache[b][2]
        Ba, ra = row_basis(ridge_fit(Ya, Xa))
        Bb, rb = row_basis(ridge_fit(Yb, Xb))
        S, cosines = shared_basis(Ba, Bb)
        dS = 0 if S is None else S.shape[1]
        row = {
            "r1": a.replace("TSAL", "")[:8], "r2": b.replace("TSAL", "")[:8],
            "rank_R1": ra, "rank_R2": rb, "n_shared": dS,
            "mean_cos": float(np.mean(cosines[:min(ra, rb)])),
            # full-manifold cross-day decode (the standard asymmetry)
            "fwd_full": decode_corr(Ya, Xa, Yb, Xb, mb),          # R1->R2 (failing)
            "rev_full": decode_corr(Yb, Xb, Ya, Xa, ma),          # R2->R1 (holding)
        }
        if S is not None:
            row["fwd_shared"] = decode_corr(Ya, Xa, Yb, Xb, mb, basis=S)   # R1->R2 | shared
            row["rev_shared"] = decode_corr(Yb, Xb, Ya, Xa, ma, basis=S)   # R2->R1 | shared
        else:
            row["fwd_shared"] = row["rev_shared"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tgt in zip(axes, TARGETS):
        g = df[df.target == tgt]
        m = g[["fwd_full", "rev_full", "fwd_shared", "rev_shared"]].mean()
        gap_full = m["rev_full"] - m["fwd_full"]
        gap_shared = m["rev_shared"] - m["fwd_shared"]
        x = np.arange(2)
        ax.bar(x - .18, [m["fwd_full"], m["fwd_shared"]], .36, label="R1→R2 (forward)", color="#e74c3c")
        ax.bar(x + .18, [m["rev_full"], m["rev_shared"]], .36, label="R2→R1 (reverse)", color="#3498db")
        ax.set_xticks(x); ax.set_xticklabels([f"full ({K}d)\ngap={gap_full:+.3f}",
                                              f"shared potent\ngap={gap_shared:+.3f}"])
        ax.set_ylabel("cross-day decode corr")
        ax.set_title(f"{tgt}\nrescue of R1→R2 = {m['fwd_shared']-m['fwd_full']:+.3f}, "
                     f"R2→R1 shift = {m['rev_shared']-m['rev_full']:+.3f}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Current potent-inclusion diagnostic: both directions shift under shared-subspace "
                 "restriction\nthis partly circular implementation does not provide a clean "
                 "private-direction test",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


def main():
    rng = np.random.default_rng(SEED)
    alld = []
    for tgt in TARGETS:
        print(f"\n########## {tgt} (single_trial CCA, K={K}) ##########")
        df = run_target(tgt, rng); df.insert(0, "target", tgt); alld.append(df)
        m = df.mean(numeric_only=True)
        gap_full = m["rev_full"] - m["fwd_full"]
        gap_shared = m["rev_shared"] - m["fwd_shared"]
        print(f"  n pairs={len(df)}  potent rank R1/R2={m['rank_R1']:.1f}/{m['rank_R2']:.1f}  "
              f"shared dims={m['n_shared']:.1f}  mean principal cos={m['mean_cos']:.2f}")
        print(f"  FULL   : R1->R2={m['fwd_full']:.3f}  R2->R1={m['rev_full']:.3f}  gap={gap_full:+.3f}")
        print(f"  SHARED : R1->R2={m['fwd_shared']:.3f}  R2->R1={m['rev_shared']:.3f}  gap={gap_shared:+.3f}")
        print(f"  --> rescue(R1->R2)={m['fwd_shared']-m['fwd_full']:+.3f}   "
              f"shift(R2->R1)={m['rev_shared']-m['rev_full']:+.3f}   "
              f"gap shrink={gap_full-gap_shared:+.3f}")
    out = pd.concat(alld, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}")
    make_figure(out, FIG)


if __name__ == "__main__":
    main()
