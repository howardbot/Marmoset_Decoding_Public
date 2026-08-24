"""Is R2's neural space more "inclusive" than R1's?  (tests cause #2)

The asymmetry (R2->R1 good, R1->R2 bad) would be explained if R1's subspace is
embedded in a broader R2 subspace: R2->R1 interpolates, R1->R2 extrapolates.
The literal test EV(R2 | PCA_R1) vs EV(R1 | PCA_R2) needs a SHARED neuron basis,
which we don't have (units are re-sorted each day). So we test the intent two ways:

(A) Neural dimensionality, per session, in each session's own space (no alignment):
      participation ratio PR = (sum li)^2 / sum li^2, and #PCs for 80/90% variance,
      on the single-trial population AND on the trial-averaged reach "signal".
    R2 broader  <=>  R2 has higher PR / needs more PCs than R1.

(B) Directional cross-EV + principal angles in the CCA-ALIGNED canonical space
    (the space the cross-day decoder actually operates in). For each (R1,R2) pair:
      EV(R2 | top-k PCA of R1)  vs  EV(R1 | top-k PCA of R2),  k = 1..K.
    R2 more inclusive  <=>  EV(R1|PCA_R2) > EV(R2|PCA_R1)  (R1 fits inside R2's
    high-variance directions, but not vice versa). Principal angles give the
    (symmetric) subspace overlap. Caveat: measured post-CCA, i.e. in the decoder's
    working space, not raw neuron space.

Config: locked single-trial phase-matched CCA, K_PCS=15, both targets.
Output: printed table + CSV + figure (Results/manifold_geometry/...).
NOTE: n(R2)=3 -> epoch-level neural-dimensionality stats are n=3 for R2.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from manifold_align import pca_neural, trial_average_pc
from dimension_sweep import align_full
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS, EXCLUDE_TRIALS, N_PHASE_BINS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 15
SEED = 0
TARGET_FOR_ALIGN = "relative_velocity"     # alignment target (decode space); both give same neural
REPO = _THIS.parents[2]
OUT = REPO / "Results" / "manifold_geometry"
FIG = OUT / "figures"


def load(session, exclude=()):
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, TARGET_FOR_ALIGN, bin_size=BIN_MS / 1000.0, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    Ysm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS)
    Ypc = pca_neural(Ysm, k=K)[0]
    return X, Ypc, meta, Ysm


# ---------- Part A: dimensionality in each session's own space ----------
def spectrum_stats(M):
    """PR and #dims for 80/90% variance from data matrix M (samples x features)."""
    Mc = M - M.mean(0)
    lam = np.linalg.eigvalsh(np.cov(Mc.T))
    lam = np.clip(lam[::-1], 0, None)
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    cum = np.cumsum(lam) / lam.sum()
    return pr, int(np.argmax(cum >= 0.8) + 1), int(np.argmax(cum >= 0.9) + 1)


# ---------- Part B: cross-EV + principal angles in aligned space ----------
def top_pca(Y, k):
    Yc = Y - Y.mean(0)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    return Vt[:k].T                       # (K, k) orthonormal


def ev_in(Y, V):
    Yc = Y - Y.mean(0)
    return float(((Yc @ V) ** 2).sum() / (Yc ** 2).sum())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal", choices=sorted(ANIMAL_SESSIONS), default="TS"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = (
        list(sessions) for sessions in ANIMAL_SESSIONS[args.animal]
    )
    all_sessions = sessions_r1 + sessions_r2
    suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
    FIG.mkdir(parents=True, exist_ok=True)
    cache = {
        s: load(s, EXCLUDE_TRIALS.get(s, [])) for s in all_sessions
    }
    rng = np.random.default_rng(SEED)

    # ---- Part A ----
    rowsA = []
    for s in all_sessions:
        X, Ypc, meta, Ysm = cache[s]
        pr, d80, d90 = spectrum_stats(Ysm)                       # single-trial population
        traj = trial_average_pc(Ysm, meta, n_phase_bins=N_PHASE_BINS)
        pr_s, d80_s, _ = spectrum_stats(traj)                    # trial-averaged signal
        rowsA.append(dict(animal=args.animal, session=s[4:12],
                          epoch="R2" if s in sessions_r2 else "R1",
                          n_units=Ysm.shape[1], PR=pr, dim80=d80, dim90=d90,
                          PR_signal=pr_s, dim80_signal=d80_s))
    A = pd.DataFrame(rowsA)
    dimensionality_path = OUT / f"subspace_dimensionality{suffix}.csv"
    A.to_csv(dimensionality_path, index=False)
    print("=== Part A: neural dimensionality (own space) ===")
    print(A.groupby("epoch")[["n_units", "PR", "dim80", "dim90", "PR_signal", "dim80_signal"]]
          .mean().round(2).to_string())

    # ---- Part B ----
    ks = np.arange(1, K + 1)
    ev_r2_given_r1 = {k: [] for k in ks}     # EV(R2 | PCA_R1)
    ev_r1_given_r2 = {k: [] for k in ks}     # EV(R1 | PCA_R2)
    cos_overlap = {k: [] for k in ks}        # mean cos of principal angles
    for a, b in product(sessions_r1, sessions_r2):
        Ya, Yb = align_full("single_trial", K, cache[a], cache[b], rng)
        if Ya is None:
            continue
        for k in ks:
            Va, Vb = top_pca(Ya, k), top_pca(Yb, k)
            ev_r2_given_r1[k].append(ev_in(Yb, Va))
            ev_r1_given_r2[k].append(ev_in(Ya, Vb))
            cos_overlap[k].append(float(np.mean(np.cos(subspace_angles(Va, Vb)))))
    rowsB = []
    for k in ks:
        rowsB.append(dict(animal=args.animal, k=int(k),
                          EV_R2_given_R1=np.mean(ev_r2_given_r1[k]),
                          EV_R1_given_R2=np.mean(ev_r1_given_r2[k]),
                          incl_gap=np.mean(ev_r1_given_r2[k]) - np.mean(ev_r2_given_r1[k]),
                          cos_overlap=np.mean(cos_overlap[k])))
    B = pd.DataFrame(rowsB)
    cross_ev_path = OUT / f"subspace_crossEV{suffix}.csv"
    B.to_csv(cross_ev_path, index=False)
    print("\n=== Part B: cross-EV in CCA-aligned space (mean over R1xR2 pairs) ===")
    print(B.round(3).to_string(index=False))
    g = B[B.k.isin([2, 3, 5])]
    print("\n  inclusion gap = EV(R1|PCA_R2) - EV(R2|PCA_R1)   (positive => R2 more inclusive)")
    for _, r in g.iterrows():
        print(f"    k={int(r.k):2d}:  EV(R2|R1)={r.EV_R2_given_R1:.3f}  EV(R1|R2)={r.EV_R1_given_R2:.3f}"
              f"  gap={r.incl_gap:+.3f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for ep, c in [("R1", "#7f8c8d"), ("R2", "#e74c3c")]:
        sub = A[A.epoch == ep]
        ax[0].scatter([f"{ep}\nPR"] * len(sub), sub.PR, color=c, s=55, alpha=.7, edgecolors="w")
        ax[0].scatter([f"{ep}\ndim80"] * len(sub), sub.dim80, color=c, s=55, alpha=.7, edgecolors="w")
    ax[0].set_title("(A) Neural dimensionality, own space\n(higher = broader)", fontsize=11)
    ax[0].set_ylabel("value"); ax[0].grid(alpha=.3, axis="y")
    ax[1].plot(B.k, B.EV_R2_given_R1, "-o", color="#e74c3c", lw=2, ms=4, label="EV(R2 | PCA_R1)")
    ax[1].plot(B.k, B.EV_R1_given_R2, "-o", color="#3498db", lw=2, ms=4, label="EV(R1 | PCA_R2)")
    ax[1].plot(B.k, B.cos_overlap, "--", color="#888", lw=1.5, label="subspace overlap (cos)")
    ax[1].set_xlabel("# PCs (k) in CCA-aligned space"); ax[1].set_ylabel("cross-explained variance")
    ax[1].set_title("(B) Directional cross-EV\nEV(R1|R2) > EV(R2|R1)  =>  R2 more inclusive", fontsize=11)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    fig.suptitle(
        f"{args.animal} subspace inclusion test (cause #2): "
        f"is R2 broader / does R1 sit inside R2?  n(R2)={len(sessions_r2)}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    figure_path = FIG / f"fig_subspace_inclusion{suffix}.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {dimensionality_path}\nsaved {cross_ev_path}"
          f"\nsaved {figure_path}")


if __name__ == "__main__":
    main()
