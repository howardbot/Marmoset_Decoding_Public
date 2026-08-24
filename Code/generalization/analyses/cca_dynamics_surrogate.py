"""Is the cross-day CCA alignment carried by *real shared dynamics*, or would CCA align
anything with our covariance structure?  (TME-style control, in the spirit of Gallego 2020
Extended Data Fig. 7 / Lee Miller's tensor-maximum-entropy surrogate.)

Our circular-shift null (cross_day_decoder.py) is a *decoding* null — it tests neural↔behaviour
coupling, not whether the CCA *alignment* recovers genuine shared neural dynamics. CCA maximises
correlation by construction, so the honest control is Miller's: surrogate data that PRESERVES
each day's covariance but DESTROYS the dynamics, run through the identical alignment.

Metric — **held-out** canonical correlation (NOT in-sample): fit CCA on one trial-half's
trial-averaged PC trajectory, score the canonical corr on the other half. (In-sample CC is
useless here — 30 phase-bins × 15 dims saturates to 1.0; held-out collapses to a noise floor
past ~dim 2, exactly as in fig_cca_score, so it is the discriminating metric.)

Surrogate — **per-day phase-bin permutation**: permute the order of the 30 phase bins of a day's
trajectory. This preserves the K×K covariance and per-dim marginals *exactly* (it is a row
permutation) but destroys the temporal dynamics. Independent permutations for the two days
destroy their shared dynamics; the SAME permutation is applied to a day's fit- and eval-halves
so within-day held-out scoring stays valid. Real and surrogate use the *identical* trial splits,
differing only by the permutation.

  - Real held-out CC1  >>  surrogate  -> alignment REQUIRES the real shared dynamics (good).
  - Real ≈ surrogate                  -> CCA is "too powerful", aligns covariance alone (bad).

Reference category R1-R1 (within-epoch) is the positive control.
Output: Results/workflows/manifold_geometry/cca_dynamics_surrogate.csv (+ figure).
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
from manifold_align import pca_neural, trial_average_pc, cca_align
from big_sweep_phase2_crossday import (
    SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS, N_PHASE_BINS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 12                # K_PCS = 12, matches the v2 report re-anchor (Gallego-2018 manifold dim)
N_SPLITS = 100        # random trial-half splits (= # surrogate draws)
SEED = 0

REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "cca_dynamics_surrogate.csv"
OUT_CSV_BYDIM = REPO / "Results" / "workflows" / "manifold_geometry" / "cca_dynamics_surrogate_bydim.csv"
FIG = REPO / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_cca_dynamics_surrogate.png"
FIG_BYDIM = REPO / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_cca_dynamics_surrogate_bydim.png"


def load_cache(session, exclude=()):
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, "relative_velocity", bin_size=BIN_MS / 1000.0, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    Ysm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS)
    Ypc = pca_neural(Ysm, k=K)[0]
    return {"Y_pc": Ypc, "meta": meta}

# generating trial average trajectory, split in half
def traj(cache, trials):
    mask = cache["meta"]["trial_number"].isin(trials).to_numpy()
    return trial_average_pc(cache["Y_pc"][mask], cache["meta"][mask].reset_index(drop=True),
                            n_phase_bins=N_PHASE_BINS)

# Pre generating 100 times
def precompute_splits(cache, rng, n=N_SPLITS):
    """Return n (fit_traj, eval_traj) pairs from random trial-half splits (each 30×K)."""
    t = np.array(sorted(cache["meta"]["trial_number"].unique()))
    out = []
    for _ in range(n):
        p = rng.permutation(t); h = len(p) // 2
        if h < 2:
            continue
        out.append((traj(cache, p[:h]), traj(cache, p[h:])))
    return out

# Calculating corr
def cc_dim(fit_a, fit_b, ev_a, ev_b):
    """Held-out canonical correlations: fit CCA on fit-halves, score on eval-halves."""
    W_a, W_b, m_a, m_b = cca_align(fit_a, fit_b)
    ca = (ev_a - m_a) @ W_a
    cb = (ev_b - m_b) @ W_b
    return np.array([np.corrcoef(ca[:, d], cb[:, d])[0, 1] for d in range(ca.shape[1])])


def main():
    FIG.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    caches = {s: load_cache(s, EXCLUDE_TRIALS.get(s, [])) for s in SESSIONS_R1 + SESSIONS_R2}
    splits = {s: precompute_splits(caches[s], rng) for s in caches}
    print("loaded + split", len(caches), "sessions")
    nph = N_PHASE_BINS

    cats = {
        "R1-R1": list(combinations(SESSIONS_R1, 2)),
        "R1-R2": list(product(SESSIONS_R1, SESSIONS_R2)),
        "R2-R2": list(combinations(SESSIONS_R2, 2)),
    }

    rows = []          # dim-1 summary (backward-compatible schema)
    bydim_rows = []    # per-dim held-out CC, all K canonical dims
    for cat, pairs in cats.items():
        for a, b in pairs:
            n = min(len(splits[a]), len(splits[b]))
            real = np.full((n, K), np.nan)   # per-split held-out CC across all dims
            surr = np.full((n, K), np.nan)
            for i in range(n):
                fa, ea = splits[a][i]
                fb, eb = splits[b][i]
                # real held-out CC across ALL canonical dims
                rc = cc_dim(fa, fb, ea, eb)
                real[i, :len(rc)] = rc
                # surrogate: independent phase-bin permutation per day (same within a day)
                pa = rng.permutation(nph); pb = rng.permutation(nph)
                sc = cc_dim(fa[pa], fb[pb], ea[pa], eb[pb])
                surr[i, :len(sc)] = sc
            real_by = np.nanmean(real, axis=0)          # (K,) mean over splits per dim
            surr_by = np.nanmean(surr, axis=0)
            surr_p95 = np.nanpercentile(surr, 95, axis=0)
            rows.append(dict(
                cat=cat, a=a[4:12], b=b[4:12], n=n,
                real_cc1=float(real_by[0]),
                surr_cc1_mean=float(surr_by[0]),
                surr_cc1_p95=float(surr_p95[0]),
                real_gt_surr_p95=bool(real_by[0] > surr_p95[0]),
            ))
            for d in range(K):
                bydim_rows.append(dict(
                    cat=cat, a=a[4:12], b=b[4:12], dim=d + 1, n=n,
                    real_cc=float(real_by[d]),
                    surr_cc_mean=float(surr_by[d]),
                    surr_cc_p95=float(surr_p95[d]),
                    real_gt_surr_p95=bool(real_by[d] > surr_p95[d]),
                ))
        sub = pd.DataFrame([r for r in rows if r["cat"] == cat])
        subd = pd.DataFrame([r for r in bydim_rows if r["cat"] == cat])
        print(f"\n[{cat}]  n_pairs={len(sub)}")
        print(f"  real      held-out CC1 = {sub.real_cc1.mean():.3f}")
        print(f"  surrogate held-out CC1 = {sub.surr_cc1_mean.mean():.3f}  (p95 {sub.surr_cc1_p95.mean():.3f})")
        print(f"  real > surrogate-p95 in {sub.real_gt_surr_p95.mean()*100:.0f}% of pairs (dim 1)")
        # per-dim: how far up the manifold does real stay above the surrogate p95?
        by = subd.groupby("dim").agg(real=("real_cc", "mean"),
                                     surr=("surr_cc_mean", "mean"),
                                     frac_gt=("real_gt_surr_p95", "mean")).reset_index()
        beat = by[by.frac_gt >= 0.5].dim.max()
        print(f"  real > surrogate-p95 (>=50% of pairs) up to dim {int(beat) if beat==beat else 0}")
        print("  per-dim real / surrogate: " +
              "  ".join(f"d{int(r.dim)}={r.real:.2f}/{r.surr:.2f}" for _, r in by.iterrows() if r.dim <= 6))

    df = pd.DataFrame(rows)
    df_bydim = pd.DataFrame(bydim_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    df_bydim.to_csv(OUT_CSV_BYDIM, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["R1-R1", "R1-R2", "R2-R2"]
    # R1-R1 in green (NOT grey) so grey is unambiguously the surrogate
    col = {"R1-R1": "#2ca02c", "R1-R2": "#e74c3c", "R2-R2": "#3498db"}
    for i, cat in enumerate(order):
        sub = df[df.cat == cat]
        ax.scatter(np.full(len(sub), i - 0.18) + rng.normal(0, 0.03, len(sub)), sub.real_cc1,
                   color=col[cat], s=40, alpha=.8, edgecolors="white", zorder=3,
                   label="real held-out CC1" if i == 0 else None)
        ax.scatter(np.full(len(sub), i + 0.18) + rng.normal(0, 0.03, len(sub)), sub.surr_cc1_mean,
                   facecolors="none", edgecolors="grey", s=34, alpha=.8, zorder=2,
                   label="surrogate (dynamics destroyed)" if i == 0 else None)
        ax.hlines(sub.real_cc1.mean(), i - 0.33, i - 0.03, color=col[cat], lw=2.5, zorder=4)
        ax.hlines(sub.surr_cc1_mean.mean(), i + 0.03, i + 0.33, color="grey", lw=2.5, zorder=4)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylabel("held-out canonical correlation (dim 1)")
    ax.axhline(0, color="k", lw=.6)
    ax.set_title("CCA alignment requires real dynamics — TME-style surrogate\n"
                 "real (R1-R1 green / R1-R2 red / R2-R2 blue) vs covariance-matched, "
                 "dynamics-destroyed surrogate (open grey)", fontsize=9.5)
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG, dpi=150, bbox_inches="tight")

    # --- per-dim figure: real vs surrogate held-out CC across ALL canonical dims ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    dims = np.arange(1, K + 1)
    for cat in order:
        g = df_bydim[df_bydim.cat == cat].groupby("dim").real_cc.mean().reindex(dims)
        ax2.plot(dims, g.values, "-o", color=col[cat], ms=5, lw=2, label=f"real · {cat}", zorder=3)
    # surrogate collapses to the same floor for every category -> plot one pooled grey band
    surr_by_dim = df_bydim.groupby("dim").surr_cc_mean.mean().reindex(dims)
    surr_p95_dim = df_bydim.groupby("dim").surr_cc_p95.mean().reindex(dims)
    ax2.plot(dims, surr_by_dim.values, "--", color="grey", lw=2,
             label="surrogate (dynamics destroyed)", zorder=2)
    ax2.fill_between(dims, surr_by_dim.values, surr_p95_dim.values, color="grey", alpha=.18,
                     label="surrogate 95th pctile", zorder=1)
    ax2.axhline(0, color="k", lw=.6)
    ax2.set_xticks(dims)
    ax2.set_xlabel("canonical dim"); ax2.set_ylabel("held-out canonical correlation")
    ax2.set_title("CCA alignment requires real dynamics — held-out CC across ALL dims\n"
                  "real (colored) stays above the dynamics-destroyed surrogate only where the "
                  "shared subspace lives (~first 3–4 dims); both hit the floor after", fontsize=9.5)
    ax2.legend(fontsize=8.5, ncol=2); ax2.grid(alpha=.3)
    fig2.tight_layout(); fig2.savefig(FIG_BYDIM, dpi=150, bbox_inches="tight")

    print(f"\nsaved {OUT_CSV}\nsaved {OUT_CSV_BYDIM}\nsaved {FIG}\nsaved {FIG_BYDIM}")


if __name__ == "__main__":
    main()
