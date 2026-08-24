"""Cross-phase (cross-time) decoding generalization — tests reassociation.

Bougou-style cross-time decoding adapted to our continuous reach. Each reach is
resampled to P phase bins (0 = reach start, 1 = peak). A linear (ridge) decoder
neural→velocity is trained at phase p and tested at phase p', giving a P×P
generalization matrix:
  diagonal  = within-phase decoding (how decodable velocity is at that phase)
  off-diag  = does a decoder trained at one phase still work at another?
              (sustained off-diagonal = temporally stable neural→movement map;
               narrow diagonal = the map reconfigures across the reach)

Four matrices:
  R1→R1, R2→R2  : within-epoch, 5-fold CV over trials, averaged over sessions
  R1→R2, R2→R1  : cross-epoch, CCA-aligned canonical space, averaged over pairs

Why this tests reassociation: reassociation = the neural→movement *mapping*
changed across the interference period while the repertoire was preserved. If so,
the cross-epoch matrices (R1→R2 / R2→R1) should degrade relative to the
within-epoch ones in a structured (phase-specific) way, even though the geometry
is preserved.

Config = LOCKED_CONFIG (bin=30, butter_o2, sigma=50ms, 0828 trial-41 excluded),
target = relative_velocity, K_PCS = 15, P = 8 phase bins.

Outputs:
  Results/workflows/manifold_geometry/cross_phase_matrices.npz
  Results/workflows/manifold_geometry/figures/fig_cross_phase.png
"""
from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

import decoder_utils as du
from manifold_align import pca_neural, trial_average_pc, cca_align, apply_alignment
from big_sweep_phase2_crossday import (
    SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS, K_PCS, N_PHASE_BINS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")

BIN_SIZE_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
TARGET = "relative_velocity"
P = 4                 # reach segments (early → late); reaches are short
N_FOLDS = 5
RIDGE_ALPHA = 1.0
SEED = 0

REPO_ROOT = _THIS.parents[1]
OUT_NPZ = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "cross_phase_matrices.npz"
FIG_DIR = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "figures"


def build_session(session, exclude=()):
    """Return per-trial phase-resampled (Yc_pc[n,P,K], X[n,P,3]) + traj for CCA."""
    bin_s = BIN_SIZE_MS / 1000.0
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_s
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, TARGET, bin_size=bin_s, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_SIZE_MS)
    Y_pc, _, _ = pca_neural(Y_sm, k=K_PCS)
    traj = trial_average_pc(Y_pc, meta, n_phase_bins=N_PHASE_BINS)
    return X, Y_pc, meta, traj


def corr_series(true, pred):
    """Mean within-series Pearson r across the 3 kinematic dims (over time bins)."""
    vals = []
    for d in range(true.shape[1]):
        a, b = true[:, d], pred[:, d]
        if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(vals)) if vals else np.nan


def seg_corr(model, X_te, Y_te, meta_te, S=P):
    """Per-reach-segment decoding corr.

    Decoder is already trained on the full reach. For each test trial we split
    its bins into S segments by within-trial phase and compute the within-trial
    (over-time) correlation between true and decoded velocity inside each
    segment — i.e. the M2 metric localized to early/.../late reach. Averaged
    over trials → length-S curve. This keeps the decodable within-reach velocity
    profile inside each segment (unlike fixing a single phase, which removes it).
    """
    seg_vals = [[] for _ in range(S)]
    for _, idx in meta_te.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < S + 2:
            continue
        pred = model.predict(Y_te[idx])
        true = X_te[idx]
        phase = np.linspace(0, 1, len(idx), endpoint=False)
        seg_id = np.minimum((phase * S).astype(int), S - 1)
        for s in range(S):
            m = seg_id == s
            if m.sum() >= 3:
                seg_vals[s].append(corr_series(true[m], pred[m]))
    return np.array([np.nanmean(v) if v else np.nan for v in seg_vals])


def fit_full_reach(X_tr, Y_tr):
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(Y_tr, X_tr)
    return model


def within_epoch_curve(sessions):
    """Average per-segment corr curve over sessions, 5-fold CV over trials."""
    curves = []
    for s in sessions:
        X, Y_pc, meta, _ = build_session(s, EXCLUDE_TRIALS.get(s, []))
        if meta["trial_number"].nunique() < N_FOLDS:
            continue
        fold_curves = []
        for tr_mask, te_mask in du.kfold_split_by_trial(meta, n_splits=N_FOLDS, random_seed=SEED):
            if tr_mask.sum() < 50 or te_mask.sum() < 20:
                continue
            model = fit_full_reach(X[tr_mask], Y_pc[tr_mask])
            fold_curves.append(seg_corr(model, X[te_mask], Y_pc[te_mask],
                                        meta[te_mask].reset_index(drop=True)))
        if fold_curves:
            curves.append(np.nanmean(fold_curves, axis=0))
    curves = np.array(curves)
    return np.nanmean(curves, 0), np.nanstd(curves, 0) / np.sqrt(len(curves)), len(curves)


def cross_epoch_curve(train_sessions, test_sessions):
    """Average per-segment corr curve over (train,test) pairs, CCA-aligned."""
    cache = {s: build_session(s, EXCLUDE_TRIALS.get(s, []))
             for s in set(train_sessions) | set(test_sessions)}
    curves = []
    for s_tr, s_te in product(train_sessions, test_sessions):
        X1, Y1, m1, traj1 = cache[s_tr]
        X2, Y2, m2, traj2 = cache[s_te]
        W1, W2, mu1, mu2 = cca_align(traj1, traj2)
        Y1c = apply_alignment(Y1, W1, mu1)
        Y2c = apply_alignment(Y2, W2, mu2)
        model = fit_full_reach(X1, Y1c)
        curves.append(seg_corr(model, X2, Y2c, m2))
    curves = np.array(curves)
    return np.nanmean(curves, 0), np.nanstd(curves, 0) / np.sqrt(len(curves)), len(curves)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("[R1→R1] within-epoch ...")
    r1r1, s_r1r1, n_r1 = within_epoch_curve(SESSIONS_R1)
    print("[R2→R2] within-epoch ...")
    r2r2, s_r2r2, n_r2 = within_epoch_curve(SESSIONS_R2)
    print("[R1→R2] cross-epoch ...")
    r1r2, s_r1r2, n_12 = cross_epoch_curve(SESSIONS_R1, SESSIONS_R2)
    print("[R2→R1] cross-epoch ...")
    r2r1, s_r2r1, n_21 = cross_epoch_curve(SESSIONS_R2, SESSIONS_R1)

    np.savez(OUT_NPZ, r1r1=r1r1, r2r2=r2r2, r1r2=r1r2, r2r1=r2r1,
             sem_r1r1=s_r1r1, sem_r2r2=s_r2r2, sem_r1r2=s_r1r2, sem_r2r1=s_r2r1,
             P=P, target=TARGET)
    print(f"\nsaved {OUT_NPZ}")
    for name, c in [("R1→R1", r1r1), ("R2→R2", r2r2), ("R1→R2", r1r2), ("R2→R1", r2r1)]:
        print(f"  {name}: mean corr over phases = {np.nanmean(c):.3f}   "
              f"(early={c[0]:.3f}, mid={c[P//2]:.3f}, late={c[-1]:.3f})")

    import matplotlib.pyplot as plt
    phases = np.linspace(0, 1, P)
    colors = {"R1→R1": "#7f8c8d", "R2→R2": "#34495e", "R1→R2": "#e74c3c", "R2→R1": "#3498db"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, c, sem, n in [("R1→R1", r1r1, s_r1r1, n_r1), ("R2→R2", r2r2, s_r2r2, n_r2),
                            ("R1→R2", r1r2, s_r1r2, n_12), ("R2→R1", r2r1, s_r2r1, n_21)]:
        ax.plot(phases, c, "-o", color=colors[name], lw=2, ms=4, label=f"{name} (n={n})")
        ax.fill_between(phases, c - sem, c + sem, color=colors[name], alpha=0.15)
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel("reach phase (0 = start → 1 = peak)")
    ax.set_ylabel("velocity decoding corr")
    ax.set_title("Cross-phase decoding: full-reach decoder evaluated per phase\n"
                 "(within-epoch CV vs cross-epoch CCA; gap = where the neural→velocity map fails)",
                 fontsize=11)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_cross_phase.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
