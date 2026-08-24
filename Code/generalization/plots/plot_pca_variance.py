"""PCA cumulative variance: single-trial population vs trial-averaged reach signal.

Shows *why* a variance threshold is the wrong way to pick dimensionality: 80% of the
single-trial population variance needs ~46 PCs, but the trial-averaged reach *signal*
saturates by ~2-3 PCs. The ~44 extra dims are trial-to-trial noise.
Writes Results/manifold_geometry/figures/fig_pca_variance_cumulative.png
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))   # Code/
sys.path.insert(0, str(_THIS.parent))        # generalization/
import decoder_utils as du
from manifold_align import trial_average_pc
from big_sweep_phase2_crossday import (
    SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS, N_PHASE_BINS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials)
warnings.filterwarnings("ignore")
BIN_MS = 30; SK = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
NPC = 80
OUT = _THIS.parents[2] / "Results" / "manifold_geometry" / "figures" / "fig_pca_variance_cumulative.png"

def load(s, exclude=()):
    du.SESSION = s; du.PROCESSED_NWB = du.DATA_DIR / f"{s}_processed.nwb"; du.BIN_SIZE_SECONDS = BIN_MS/1000
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(nwb, reach, "relative_velocity", bin_size=BIN_MS/1000,
            unit_qualities=UNIT_QUALITIES, trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SK)
    finally: io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    return du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS/BIN_MS), meta

def cumvar(M):
    Mc = M - M.mean(0); lam = np.clip(np.linalg.eigvalsh(np.cov(Mc.T))[::-1], 0, None)
    return np.cumsum(lam) / lam.sum()

def stack(curves):
    out = np.full((len(curves), NPC), np.nan)
    for i, c in enumerate(curves): out[i, :min(NPC, len(c))] = c[:NPC]
    return out

import pandas as pd
CACHE = _THIS.parents[2] / "Results" / "manifold_geometry" / "_pca_variance_cache.npz"
if CACHE.exists():                                  # cached cumvar curves -> instant re-plots
    _z = np.load(CACHE); ST, SIG = _z["ST"], _z["SIG"]
else:
    st, sig = [], []
    for s in SESSIONS_R1 + SESSIONS_R2:
        Ysm, meta = load(s, EXCLUDE_TRIALS.get(s, []))
        st.append(cumvar(Ysm))
        sig.append(cumvar(trial_average_pc(Ysm, meta, n_phase_bins=N_PHASE_BINS)))
    ST, SIG = stack(st), stack(sig)
    CACHE.parent.mkdir(parents=True, exist_ok=True); np.savez(CACHE, ST=ST, SIG=SIG)
x = np.arange(1, NPC + 1)
mean_st = np.nanmean(ST, 0)
d80 = int(np.argmax(mean_st >= 0.8) + 1); d90 = int(np.argmax(mean_st >= 0.9) + 1)

fig, ax = plt.subplots(figsize=(9.5, 5.5))
# LEFT axis — cumulative variance explained
ax.plot(x, mean_st, color="#34495e", lw=2.5, label="cum. var · single-trial population")
ax.fill_between(x, np.nanmin(ST, 0), np.nanmax(ST, 0), color="#34495e", alpha=.12)
ax.plot(x, np.nanmean(SIG, 0), color="#e67e22", lw=2.5, label="cum. var · trial-avg signal")
for thr, ls in [(0.8, "--"), (0.9, ":")]:
    ax.axhline(thr, color="grey", ls=ls, lw=1)
ax.axvspan(1, 5, color="#2ecc71", alpha=.06)
ax.annotate(f"80% variance\n@ ~{d80} PCs", (d80, 0.8), (d80 + 3, 0.52), fontsize=9, color="#34495e",
            arrowprops=dict(arrowstyle="->", color="#34495e"))
ax.set_xlabel("# PCs (K_PCS)"); ax.set_ylabel("cumulative variance explained"); ax.set_ylim(0, 1.02)
ax.set_xlim(0.5, 45.5)   # match the decode sweep's x-range so both curves span the full width

# RIGHT axis — cross-day decode corr vs K_PCS (single-trial CCA, decode top-2 canonical)
dim = pd.read_csv(_THIS.parents[2] / "Results" / "manifold_geometry" / "dimension_sweep_long.csv")
dd = dim[(dim.trial_mode == "single_trial") & (dim.d == 2)]
ax2 = ax.twinx()
for pc, c in [("R1->R1", "#9aa0a6"), ("R2->R1", "#3498db"), ("R1->R2", "#e74c3c")]:
    g = dd[dd.pair_cat == pc].groupby("K_PCS").decode.mean()
    ax2.plot(g.index, g.values, "--o", color=c, ms=5, lw=1.8, label=f"decode {pc}")
ax2.set_ylabel("cross-day decode corr (top-2 canonical)")
ax.text(5.4, 0.06, "decode saturates ~2–5 dims", fontsize=9, color="#2e7d32")

h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8, framealpha=.9)
ax.set_title("PCA dimensionality trade-off — variance keeps climbing (left), decoding saturates early (right)\n"
             f"80% variance needs ~{d80} PCs, but decode is flat past ~2–5 → pick dims by decoding, not variance",
             fontsize=10)
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"single-trial dim80={d80} dim90={d90}; saved {OUT}")
