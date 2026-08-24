"""H5 + H6(step 1) — does R2's global-state variability (§5) explain why R1's decoder fails on R2 (§4)?

Two findings we are trying to bridge:
  §5  R2 has extra trial-to-trial NEURAL variability that is low-D / shared (a global-state /
      engagement signature), for cleaner behaviour.
  §4  R2 is a hard decode TARGET: an R1-trained decoder fails on R2 (R1->R2 drops), though R2 itself
      is decodable (R2->R2 high).
Bridge hypothesis (Qin's prior): the shared global-state signal is exactly what R1's read-out cannot
follow -> it IS the reason R2 is a hard target.

H5 (control — is the variability real, movement-independent?):  regress each day's PC activity on the
FULL kinematics (position + velocity), take the residual, and measure its trial-to-trial variability /
dimensionality. §5 only subtracted the mean reach; H5 removes the actual per-trial movement. If R2's
residual variability is still higher & lower-D, the global-state signal is genuinely movement-independent.

H6 step 1 (the bridge, with the falsifying control):  define the global-state axis G in R2 (top shared
dims of the movement-residual). Per R2 trial, global-state magnitude = displacement along G. Correlate it
with per-trial decode goodness under (a) an R1-trained decoder (R1->R2) and (b) an R2-trained decoder
(R2->R2, the CONTROL).
  bridge  : negative corr with R1->R2  AND  ~0 corr with R2->R2  (global state specifically breaks the
            R1 read-out; R2's own decoder was trained WITH this state so it tolerates it).
  just noise: both correlate (noisy trials decode worse for everyone) -> NOT the bridge.

Config: single-trial CCA, K_PCS=12, decode target = position (headline). 0828 trial-41 excluded. n(R2)=3.
Output: Results/manifold_geometry/global_state_bridge.csv (+ figure).  Reads NWB -> HatLab env.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from manifold_align import pca_neural, trial_average_pc
from dimension_sweep import align_full
from big_sweep_phase2_crossday import (
    SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS, N_PHASE_BINS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
    kalman_fit_predict, corr_1d,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 12
K_GLOBAL = 2                 # dims of the global-state subspace G
N_PHASE = N_PHASE_BINS
SEED = 0
REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "manifold_geometry" / "global_state_bridge.csv"
FIG = REPO / "Results" / "manifold_geometry" / "figures" / "fig_global_state_bridge.png"


def load(session, exclude=()):
    """Return X_pos (decode target), Kin (pos+vel, for regression), Y_pc (K), meta, trial_avg."""
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        Xp, Y, meta = du.build_decoder_dataset(
            nwb, reach, "relative_position", bin_size=BIN_MS / 1000.0, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    Xp, Y, meta = filter_trials(Xp, Y, meta, exclude)
    Ysm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS)
    Ypc = pca_neural(Ysm, k=K)[0]
    # velocity proxy = per-trial finite difference of position (same rows as Xp, so no build mismatch)
    Xv = np.zeros_like(Xp)
    for _, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) > 1:
            for d in range(Xp.shape[1]):
                Xv[idx, d] = np.gradient(Xp[idx, d])
    Kin = np.column_stack([Xp, Xv])
    return dict(Xp=Xp, Kin=Kin, Ypc=Ypc, meta=meta,
                tavg=trial_average_pc(Ypc, meta, n_phase_bins=N_PHASE))


def kin_residual(Ypc, Kin):
    """PC activity with the linear movement (pos+vel) regressed out."""
    Kc = Kin - Kin.mean(0, keepdims=True)
    Yc = Ypc - Ypc.mean(0, keepdims=True)
    W, *_ = np.linalg.lstsq(Kc, Yc, rcond=None)
    return Yc - Kc @ W                      # (T, K) movement-residual


def phase_stack(A, meta, n_phase=N_PHASE):
    tt = np.linspace(0, 1, n_phase)
    out, trials = [], []
    for tr, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 3:
            continue
        ts = np.linspace(0, 1, len(idx))
        out.append(np.column_stack([np.interp(tt, ts, A[idx, d]) for d in range(A.shape[1])]))
        trials.append(tr)
    return np.stack(out, 0), np.array(trials)       # (nT, P, K), (nT,)


def participation_ratio(C):
    ev = np.linalg.eigvalsh(C); ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def resid_stats(R, meta):
    """H5: trial-to-trial variability of the movement-residual (magnitude, PR, shared frac)."""
    st, _ = phase_stack(R, meta)                    # (nT, P, K)
    tt = st - st.mean(0, keepdims=True)             # trial-to-trial fluctuation
    flat = tt.reshape(-1, tt.shape[-1])
    C = np.cov(flat.T)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]; ev = ev[ev > 0]
    return dict(noise_mag=float((tt ** 2).mean()), noise_PR=participation_ratio(C),
                shared_frac=float(ev[:3].sum() / ev.sum()))


def global_axis(R, meta, k=K_GLOBAL):
    """G = top-k shared dims of the movement-residual trial-to-trial cloud (in PC space)."""
    st, _ = phase_stack(R, meta)
    tt = (st - st.mean(0, keepdims=True)).reshape(-1, st.shape[-1])
    C = np.cov(tt.T)
    w, V = np.linalg.eigh(C)
    return V[:, np.argsort(w)[::-1][:k]]             # (K, k)


def trial_global_magnitude(R, meta, G):
    """Per trial: displacement of the trial's mean residual along G (centred across trials)."""
    st, trials = phase_stack(R, meta)               # (nT, P, K)
    proj = st.mean(1) @ G                            # (nT, k) trial-mean projection onto G
    proj = proj - proj.mean(0, keepdims=True)
    mag = np.linalg.norm(proj, axis=1)              # (nT,)
    return dict(zip(trials, mag))


def per_trial_corr(X_te, pred, meta_te):
    out = {}
    for tr, idx in meta_te.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 4:
            continue
        cs = [corr_1d(X_te[idx, d], pred[idx, d]) for d in range(X_te.shape[1])]
        out[tr] = float(np.nanmean(cs))
    return out


def decode_per_trial(train_sessions, test_s, cache, rng):
    """Mean per-trial decode goodness on test_s trials, averaged over the train sessions.
    cache[s] is a 4-tuple (Xp, Ypc, meta, tavg) for align_full's tuple unpacking."""
    acc = {}
    for s1 in train_sessions:
        Ya, Yb = align_full("single_trial", K, cache[s1], cache[test_s], rng)
        if Ya is None:
            continue
        Xc, pred = kalman_fit_predict(cache[s1][0], Ya, cache[test_s][0], Yb, cache[test_s][2])
        for tr, c in per_trial_corr(Xc, pred, cache[test_s][2]).items():
            acc.setdefault(tr, []).append(c)
    return {tr: float(np.nanmean(v)) for tr, v in acc.items()}


# ----- H6 step 2: causal removal -----
def resid_basis(R):
    """Orthonormal basis of the movement-residual (movement-orthogonal) subspace, desc by shared var."""
    w, V = np.linalg.eigh(np.cov(R.T))
    order = np.argsort(w)[::-1]
    return V[:, order[w[order] > 1e-9]]        # (K, m)


def remove_subspace(Ypc, Q):
    """Project a subspace (columns of Q, orthonormal) out of the PC activity."""
    return Ypc - (Ypc @ Q) @ Q.T


def decode_mean_R1toR2(test_cache, cache, rng):
    """Mean R1->R2 decode corr on a (possibly cleaned) R2 test cache, over all R1 train sessions."""
    vals = []
    for s1 in SESSIONS_R1:
        Ya, Yb = align_full("single_trial", K, cache[s1], test_cache, rng)
        if Ya is None:
            continue
        Xc, pred = kalman_fit_predict(cache[s1][0], Ya, test_cache[0], Yb, test_cache[2])
        pt = per_trial_corr(Xc, pred, test_cache[2])
        if pt:
            vals.append(float(np.nanmean(list(pt.values()))))
    return float(np.nanmean(vals))


def main():
    FIG.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # align_full expects cache[s] indexable as (X, Ypc, meta, tavg); wrap dict -> tuple view
    raw = {s: load(s, EXCLUDE_TRIALS.get(s, [])) for s in SESSIONS_R1 + SESSIONS_R2}
    # 4-tuple (Xp, Ypc, meta, tavg) so align_full's `Xa,Ya,ma_,tja = ca` unpacks correctly
    cache = {s: (raw[s]["Xp"], raw[s]["Ypc"], raw[s]["meta"], raw[s]["tavg"]) for s in raw}
    print("loaded", len(raw), "sessions")

    # ---- H5: movement-residual variability, R1 vs R2 ----
    h5 = []
    for ep, sess in (("R1", SESSIONS_R1), ("R2", SESSIONS_R2)):
        for s in sess:
            R = kin_residual(raw[s]["Ypc"], raw[s]["Kin"])
            st = resid_stats(R, raw[s]["meta"]); st.update(epoch=ep, session=s[4:12])
            h5.append(st)
    h5 = pd.DataFrame(h5)
    print("\n=== H5: movement-residual (pos+vel regressed out) trial-to-trial variability ===")
    print(h5.groupby("epoch")[["noise_mag", "noise_PR", "shared_frac"]].mean().round(3).to_string())

    # ---- H6 step 1: correlation, per R2 trial ----
    rows = []
    for s2 in SESSIONS_R2:
        R = kin_residual(raw[s2]["Ypc"], raw[s2]["Kin"])
        G = global_axis(R, raw[s2]["meta"])
        gmag = trial_global_magnitude(R, raw[s2]["meta"], G)
        r1r2 = decode_per_trial(SESSIONS_R1, s2, cache, rng)                      # R1-trained
        r2r2 = decode_per_trial([o for o in SESSIONS_R2 if o != s2], s2, cache, rng)  # R2-trained (control)
        for tr in gmag:
            if tr in r1r2 and tr in r2r2:
                rows.append(dict(session=s2[4:12], trial=tr, gmag=gmag[tr],
                                 r1r2=r1r2[tr], r2r2=r2r2[tr]))
    d = pd.DataFrame(rows)
    d.to_csv(OUT_CSV, index=False)

    # per-session Spearman then pooled
    print("\n=== H6 step 1: corr(global-state magnitude, per-trial decode goodness) ===")
    print("  bridge predicts: NEGATIVE vs R1->R2, ~0 vs R2->R2")
    for s2 in d.session.unique():
        g = d[d.session == s2]
        rho1, p1 = spearmanr(g.gmag, g.r1r2)
        rho2, p2 = spearmanr(g.gmag, g.r2r2)
        print(f"  [{s2}] n={len(g):3d}   R1->R2 rho={rho1:+.3f} (p={p1:.3f})   "
              f"R2->R2 rho={rho2:+.3f} (p={p2:.3f})")
    # pooled (within-session ranks to avoid session offsets)
    dz = d.copy()
    for c in ["gmag", "r1r2", "r2r2"]:
        dz[c] = d.groupby("session")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    rho1, p1 = spearmanr(dz.gmag, dz.r1r2)
    rho2, p2 = spearmanr(dz.gmag, dz.r2r2)
    print(f"  POOLED (session-z) n={len(dz)}   R1->R2 rho={rho1:+.3f} (p={p1:.3g})   "
          f"R2->R2 rho={rho2:+.3f} (p={p2:.3g})")

    # ---- H6 step 2: causal removal (systematic version) ----
    print("\n=== H6 step 2: remove R2's shared global-state subspace, re-decode R1->R2 ===")
    print("  bridge predicts: remove-G lifts R1->R2 ABOVE remove-random and baseline")
    s2rows = []
    for s2 in SESSIONS_R2:
        Ypc, meta, Xp = raw[s2]["Ypc"], raw[s2]["meta"], raw[s2]["Xp"]
        R = kin_residual(Ypc, raw[s2]["Kin"])
        B = resid_basis(R)                       # movement-orthogonal basis
        G = B[:, :K_GLOBAL]                       # top-k shared (systematic + trial-varying)
        base = decode_mean_R1toR2(cache[s2], cache, rng)
        yG = remove_subspace(Ypc, G)
        gcache = (Xp, yG, meta, trial_average_pc(yG, meta, n_phase_bins=N_PHASE))
        removeG = decode_mean_R1toR2(gcache, cache, rng)
        rand = []
        for _ in range(10):
            Qr = np.linalg.qr(rng.standard_normal((B.shape[1], K_GLOBAL)))[0]
            yr = remove_subspace(Ypc, B @ Qr)
            rc = (Xp, yr, meta, trial_average_pc(yr, meta, n_phase_bins=N_PHASE))
            rand.append(decode_mean_R1toR2(rc, cache, rng))
        row = dict(session=s2[4:12], baseline=base, removeG=removeG,
                   removeRand_mean=float(np.mean(rand)), removeRand_sd=float(np.std(rand)))
        s2rows.append(row)
        print(f"  [{row['session']}] baseline={base:.3f}  remove-G={removeG:+.3f}  "
              f"remove-rand={row['removeRand_mean']:.3f}±{row['removeRand_sd']:.3f}   "
              f"ΔG={removeG-base:+.3f} vs Δrand={row['removeRand_mean']-base:+.3f}")
    s2df = pd.DataFrame(s2rows)
    s2df.to_csv(OUT_CSV.with_name("global_state_bridge_step2.csv"), index=False)
    dG = (s2df.removeG - s2df.baseline).mean()
    dR = (s2df.removeRand_mean - s2df.baseline).mean()
    print(f"  MEAN over R2 days: remove-G Δ={dG:+.3f}  remove-random Δ={dR:+.3f}  "
          f"-> {'BRIDGE (G lifts more)' if dG > dR + 0.01 else 'NO bridge (G ~ random)'}")

    # step-2 figure
    figs, axs = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(s2df))
    axs.bar(x - 0.25, s2df.baseline, 0.25, color="#95a5a6", label="baseline R1→R2")
    axs.bar(x, s2df.removeG, 0.25, color="#e74c3c", label="remove global-state G")
    axs.bar(x + 0.25, s2df.removeRand_mean, 0.25, yerr=s2df.removeRand_sd, capsize=3,
            color="#3498db", label="remove random (movement-⊥)")
    axs.set_xticks(x); axs.set_xticklabels(s2df.session)
    axs.set_ylabel("R1→R2 decode corr"); axs.set_title(
        "H6 step 2: does removing R2's global-state subspace rescue R1→R2?\n"
        f"mean ΔG={dG:+.3f} vs Δrandom={dR:+.3f} — "
        f"{'bridge' if dG > dR + 0.01 else 'no bridge (G behaves like random)'}", fontsize=10)
    axs.legend(fontsize=8); axs.grid(alpha=.3, axis="y")
    figs.tight_layout()
    figs.savefig(FIG.with_name("fig_global_state_bridge_step2.png"), dpi=150, bbox_inches="tight")
    print(f"  saved {FIG.with_name('fig_global_state_bridge_step2.png')}")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, (ep_key, lab) in enumerate([("noise_PR", "H5: residual noise PR (lower=shared)"),
                                       ("shared_frac", "H5: frac in top-3 (higher=shared)")]):
        for ep in ("R1", "R2"):
            v = h5[h5.epoch == ep][ep_key].values
            x = 0 if ep == "R1" else 1
            ax[i].scatter(np.full(len(v), x) + rng.normal(0, .04, len(v)), v,
                          s=45, alpha=.8, edgecolor="k", color="#7f7f7f" if ep == "R1" else "#e67e22")
            ax[i].hlines(v.mean(), x - .2, x + .2, lw=3, color="#7f7f7f" if ep == "R1" else "#e67e22")
        ax[i].set_xticks([0, 1]); ax[i].set_xticklabels(["R1", "R2"]); ax[i].set_title(lab, fontsize=10)
        ax[i].grid(alpha=.3, axis="y")
    ax[2].scatter(dz.gmag, dz.r1r2, s=14, alpha=.5, color="#e74c3c", label=f"R1→R2 (ρ={rho1:+.2f})")
    ax[2].scatter(dz.gmag, dz.r2r2, s=14, alpha=.5, color="#3498db", label=f"R2→R2 ctrl (ρ={rho2:+.2f})")
    ax[2].set_xlabel("global-state magnitude (session-z)")
    ax[2].set_ylabel("per-trial decode goodness (session-z)")
    ax[2].set_title("H6: does global state break the R1 read-out\nselectively? (bridge: red↓, blue≈0)", fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle("Global-state bridge (§5↔§4): is R2's shared global-state signal what R1's decoder can't follow?",
                 fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nsaved {OUT_CSV}\nsaved {FIG}")


if __name__ == "__main__":
    main()
