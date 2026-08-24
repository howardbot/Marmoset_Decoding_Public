"""L3 — principal angles between the neural PC subspace and the decoder read-out subspace.

Tests Sami's "R2 is closer to read-out space" (compressed-projection): is R2's task-relevant
neural variance packed more tightly into the read-out directions than R1's?

Math (Bjorck-Golub): for two subspaces with orthonormal bases Q_A (n x p), Q_B (n x q),
the singular values of M = Q_A^T Q_B are the cosines of the principal angles; theta_i = arccos(sigma_i).
sigma=1 -> shared direction (0 deg); sigma=0 -> orthogonal (90 deg).

Per session, in that day's K=12 PC space:
  - read-out subspace: least-squares map PC scores -> kinematics gives B (K x n_out);
    QR(B) -> orthonormal Q_B (the output-potent directions in PC space).
  - neural variance subspace: the top-m PC axes = coordinate axes e_1..e_m (already orthonormal).
  - principal angles(top-m PC, read-out); potent fraction = variance inside Q_B / total variance.
Compare R1 (n=14) vs R2 (n=3): smaller angles / higher potent fraction in R2 => closer to read-out.

Secondary (cross-day read-out rotation, in CCA-aligned canonical space): principal angles between
the two days' read-out subspaces AFTER alignment, for R1->R1 / R1->R2 / R2->R1 pairs — a geometric
quantification of read-out remapping (§4). Both days share the canonical space, so the angle is meaningful.

Config: single-trial CCA, K_PCS=12, 0828 trial-41 excluded. Targets: position (headline) + velocity.
Output: Results/workflows/manifold_geometry/readout_geometry_angles.csv (+ figure).
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from manifold_align import pca_neural, trial_average_pc
from dimension_sweep import align_full
from big_sweep_phase2_crossday import (
    SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS, N_PHASE_BINS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 12
M_TOP = 3                     # top-m PC axes = the "neural variance subspace"
SEED = 0
TARGETS = ["relative_position", "relative_velocity"]

REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "readout_geometry_angles.csv"
OUT_CSV_X = REPO / "Results" / "workflows" / "manifold_geometry" / "readout_geometry_crossday.csv"
FIG = REPO / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_readout_geometry_angles.png"


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
    return X, Ypc, meta, trial_average_pc(Ypc, meta, n_phase_bins=N_PHASE_BINS)


def principal_angles(QA, QB):
    """QA (n,p), QB (n,q) orthonormal columns -> principal angles (radians, ascending)."""
    sv = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return np.arccos(np.clip(sv, -1.0, 1.0))


def readout_basis(scores, Y):
    """Least-squares read-out PC-scores -> kinematics; orthonormal basis of its column space."""
    B, *_ = np.linalg.lstsq(scores, Y, rcond=None)   # (K, n_out)
    Q, _ = np.linalg.qr(B)                            # (K, n_out) orthonormal
    return Q


def within_day(target, rng):
    rows = []
    for epoch, sessions in [("R1", SESSIONS_R1), ("R2", SESSIONS_R2)]:
        for s in sessions:
            X, Ypc, meta, _ = load(s, target, EXCLUDE_TRIALS.get(s, []))
            Xc = X - X.mean(0, keepdims=True)
            Yc = Ypc - Ypc.mean(0, keepdims=True)
            Q_read = readout_basis(Yc, Xc)                       # (K, n_out)
            Q_toppc = np.eye(K)[:, :M_TOP]                       # top-m PC axes in PC space
            ang = np.degrees(principal_angles(Q_toppc, Q_read))  # n_out angles
            proj = Yc @ Q_read
            potent_frac = float(proj.var(0).sum() / Yc.var(0).sum())
            rows.append(dict(target=target, epoch=epoch, session=s[4:12],
                             n_out=Q_read.shape[1],
                             mean_angle_deg=float(ang.mean()),
                             min_angle_deg=float(ang.min()),
                             potent_frac=potent_frac))
    return rows


def cross_day(target, rng):
    """Read-out subspace rotation across days, measured in the CCA-aligned canonical space."""
    cache = {s: load(s, target, EXCLUDE_TRIALS.get(s, [])) for s in SESSIONS_R1 + SESSIONS_R2}
    cats = {
        "R1->R1": list(combinations(SESSIONS_R1, 2)),
        "R1->R2": list(product(SESSIONS_R1, SESSIONS_R2)),
        "R2->R1": list(product(SESSIONS_R2, SESSIONS_R1)),
    }
    rows = []
    for cat, pairs in cats.items():
        for a, b in pairs:
            Ya, Yb = align_full("single_trial", K, cache[a], cache[b], rng)
            if Ya is None:
                continue
            Xa = cache[a][0] - cache[a][0].mean(0, keepdims=True)
            Xb = cache[b][0] - cache[b][0].mean(0, keepdims=True)
            Qa = readout_basis(Ya - Ya.mean(0, keepdims=True), Xa)
            Qb = readout_basis(Yb - Yb.mean(0, keepdims=True), Xb)
            ang = np.degrees(principal_angles(Qa, Qb))
            rows.append(dict(target=target, cat=cat, mean_angle_deg=float(ang.mean())))
    return rows


def main():
    FIG.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    wrows, xrows = [], []
    for tgt in TARGETS:
        print(f"\n########## {tgt} ##########")
        wr = within_day(tgt, rng); wrows += wr
        w = pd.DataFrame(wr)
        for ep in ["R1", "R2"]:
            g = w[w.epoch == ep]
            print(f"  within-day [{ep}] n={len(g)}: mean_angle={g.mean_angle_deg.mean():.1f} deg  "
                  f"potent_frac={g.potent_frac.mean():.3f}")
        # R2 closer to read-out?  (smaller angle / higher potent_frac)
        r1, r2 = w[w.epoch == "R1"], w[w.epoch == "R2"]
        print(f"    -> R2 potent_frac {r2.potent_frac.mean():.3f} vs R1 {r1.potent_frac.mean():.3f} "
              f"({'HIGHER (supports)' if r2.potent_frac.mean() > r1.potent_frac.mean() else 'not higher'})")
        print(f"    -> R2 angle {r2.mean_angle_deg.mean():.1f} vs R1 {r1.mean_angle_deg.mean():.1f} deg "
              f"({'SMALLER (supports)' if r2.mean_angle_deg.mean() < r1.mean_angle_deg.mean() else 'not smaller'})")

        xr = cross_day(tgt, rng); xrows += xr
        x = pd.DataFrame(xr)
        print("  cross-day read-out rotation (principal angle between days' read-out subspaces):")
        for cat in ["R1->R1", "R1->R2", "R2->R1"]:
            gg = x[x.cat == cat]
            if len(gg):
                print(f"    {cat}: {gg.mean_angle_deg.mean():.1f} deg  (n={len(gg)})")

    wdf = pd.DataFrame(wrows); xdf = pd.DataFrame(xrows)
    wdf.to_csv(OUT_CSV, index=False); xdf.to_csv(OUT_CSV_X, index=False)

    # figure: (top) within-day potent_frac + angle R1 vs R2; (bottom) cross-day rotation by category
    fig, axes = plt.subplots(2, len(TARGETS), figsize=(6 * len(TARGETS), 8))
    colE = {"R1": "#7f7f7f", "R2": "#3498db"}
    for j, tgt in enumerate(TARGETS):
        w = wdf[wdf.target == tgt]
        ax = axes[0, j]
        for i, ep in enumerate(["R1", "R2"]):
            g = w[w.epoch == ep]
            ax.scatter(np.full(len(g), i) + rng.normal(0, 0.05, len(g)), g.potent_frac,
                       color=colE[ep], s=45, alpha=.8, edgecolors="white", zorder=3)
            ax.hlines(g.potent_frac.mean(), i - 0.25, i + 0.25, color=colE[ep], lw=2.5, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["R1", "R2"])
        ax.set_ylabel("potent fraction (var in read-out subspace)")
        ax.set_title(f"{tgt.replace('relative_','')} — is R2 closer to read-out space?", fontsize=10)
        ax.grid(alpha=.3, axis="y")

        x = xdf[xdf.target == tgt]
        axb = axes[1, j]
        order = ["R1->R1", "R1->R2", "R2->R1"]
        colC = {"R1->R1": "#7f7f7f", "R1->R2": "#e74c3c", "R2->R1": "#3498db"}
        for i, cat in enumerate(order):
            g = x[x.cat == cat]
            axb.scatter(np.full(len(g), i) + rng.normal(0, 0.05, len(g)), g.mean_angle_deg,
                        color=colC[cat], s=30, alpha=.6, edgecolors="white", zorder=3)
            if len(g):
                axb.hlines(g.mean_angle_deg.mean(), i - 0.25, i + 0.25, color=colC[cat], lw=2.5, zorder=4)
        axb.set_xticks(range(len(order))); axb.set_xticklabels(order)
        axb.set_ylabel("cross-day read-out rotation (principal angle, deg)")
        axb.set_title(f"{tgt.replace('relative_','')} — read-out remapping across days", fontsize=10)
        axb.grid(alpha=.3, axis="y")
    fig.suptitle("L3 — read-out geometry: neural-PC vs read-out alignment (top) and cross-day "
                 "read-out rotation (bottom)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nsaved {OUT_CSV}\nsaved {OUT_CSV_X}\nsaved {FIG}")


if __name__ == "__main__":
    main()
