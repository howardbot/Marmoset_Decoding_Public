"""Is R2's neural->movement read-out TIGHTER / MORE RELIABLE / MORE GENERALISABLE
than R1's? (the live mechanism hypothesis after potent-inclusion was rejected)

The R1->R2 / R2->R1 asymmetry survives inside the shared read-out directions, so it
is not about *which* directions are used. The remaining specific hypothesis: R2 encodes
the reach with a tighter, more read-out-aligned mapping, so an R2-fit decoder is
well-determined and generalises, while an R1-fit decoder is under-determined and
generalises worse. Necessary precondition, tested here per session (own PCA space, K=12):

  1. within_decode : within-day held-out ridge decode corr (train trial-half -> test
                     trial-half). The day's OWN generalisability = read-out reliability.
  2. weight_stab   : read-out weight stability across bootstrap trial-resamples
                     (mean pairwise cosine of flattened W). Higher = better-determined.
  3. nb_cc         : held-out neural<->behaviour canonical correlation (leading dim).
                     Higher = code is more tightly aligned to the read-out.

Prediction (hypothesis): R2 >= R1 on all three. If not, the hypothesis fails cheaply.

Config: bin=30, butter_o2, sigma=50ms, K_PCS=12, ridge read-out, both targets.
Reads NWB -> HatLab env. Output: Results/workflows/manifold_geometry/readout_reliability.csv (+ fig).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from manifold_align import pca_neural
from big_sweep_phase2_crossday import (
    SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 12
L2 = 1e-2
N_SPLIT = 20        # within-day held-out splits
N_BOOT = 30         # bootstrap resamples for weight stability
SEED = 0
TARGETS = ["relative_position", "relative_velocity"]
REPO_ROOT = _THIS.parents[2]
OUT_CSV = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "readout_reliability.csv"
FIG = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_readout_reliability.png"


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
    return X, Ypc, meta


def ridge_fit(Y, X):
    return np.linalg.solve(Y.T @ Y + L2 * np.eye(Y.shape[1]), Y.T @ X)


def trial_index(meta):
    return [np.asarray(idx) for _, idx in meta.groupby("trial_number").indices.items()
            if len(idx) >= 4]


def corr_avg(Xtrue, pred, trials):
    cors = []
    for idx in trials:
        ck = [np.corrcoef(Xtrue[idx, d], pred[idx, d])[0, 1] for d in range(Xtrue.shape[1])]
        cors.append(np.nanmean(ck))
    return float(np.nanmean(cors)) if cors else np.nan


def within_decode(X, Y, trials, rng):
    out = []
    for _ in range(N_SPLIT):
        perm = rng.permutation(len(trials))
        h = len(trials) // 2
        tr = np.concatenate([trials[i] for i in perm[:h]])
        te_tr = [trials[i] for i in perm[h:]]
        te = np.concatenate(te_tr)
        W = ridge_fit(Y[tr], X[tr])
        out.append(corr_avg(X, Y @ W, te_tr))
    return float(np.nanmean(out))


def weight_stability(X, Y, trials, rng):
    Ws = []
    n = len(trials)
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)                       # bootstrap trials
        idx = np.concatenate([trials[i] for i in pick])
        Ws.append(ridge_fit(Y[idx], X[idx]).ravel())
    Ws = np.array(Ws)
    # mean pairwise cosine similarity of the weight vectors
    Wn = Ws / (np.linalg.norm(Ws, axis=1, keepdims=True) + 1e-12)
    C = Wn @ Wn.T
    iu = np.triu_indices(len(Ws), 1)
    return float(C[iu].mean())


def nb_canoncorr(X, Y, trials, rng):
    """Held-out leading neural<->behaviour canonical correlation."""
    out = []
    for _ in range(N_SPLIT):
        perm = rng.permutation(len(trials))
        h = len(trials) // 2
        tr = np.concatenate([trials[i] for i in perm[:h]])
        te = np.concatenate([trials[i] for i in perm[h:]])
        try:
            cca = CCA(n_components=1, max_iter=500)
            cca.fit(Y[tr], X[tr])
            a, b = cca.transform(Y[te], X[te])
            out.append(abs(np.corrcoef(a[:, 0], b[:, 0])[0, 1]))
        except Exception:
            pass
    return float(np.nanmean(out)) if out else np.nan


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for tgt in TARGETS:
        for epoch, sessions in (("R1", SESSIONS_R1), ("R2", SESSIONS_R2)):
            for s in sessions:
                X, Y, meta = load(s, tgt, EXCLUDE_TRIALS.get(s, []))
                trials = trial_index(meta)
                rows.append({
                    "target": tgt, "epoch": epoch, "session": s.replace("TSAL", "")[:8],
                    "n_trials": len(trials),
                    "within_decode": within_decode(X, Y, trials, rng),
                    "weight_stab": weight_stability(X, Y, trials, rng),
                    "nb_cc": nb_canoncorr(X, Y, trials, rng),
                })
                print(f"  {tgt[:12]:12s} {epoch} {rows[-1]['session']}: "
                      f"within={rows[-1]['within_decode']:.3f} "
                      f"stab={rows[-1]['weight_stab']:.3f} nb_cc={rows[-1]['nb_cc']:.3f} "
                      f"(n={rows[-1]['n_trials']})")
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}\n")
    # summary: R1 vs R2 per target
    for tgt in TARGETS:
        g = df[df.target == tgt]
        m = g.groupby("epoch")[["within_decode", "weight_stab", "nb_cc", "n_trials"]].mean()
        print(f"=== {tgt} : R1 vs R2 (mean) ===")
        print(m.round(3).to_string())
        print()
    make_figure(df, FIG)


def make_figure(df, out):
    metrics = [("within_decode", "within-day held-out decode\n(read-out generalisability)"),
               ("weight_stab", "read-out weight stability\n(bootstrap cosine)"),
               ("nb_cc", "neural↔behaviour canonical corr\n(coupling tightness)")]
    fig, axes = plt.subplots(len(TARGETS), 3, figsize=(13, 7))
    col = {"R1": "#7f8c8d", "R2": "#e67e22"}
    for r, tgt in enumerate(TARGETS):
        g = df[df.target == tgt]
        for c, (mkey, mlab) in enumerate(metrics):
            ax = axes[r, c]
            for ep in ("R1", "R2"):
                vals = g[g.epoch == ep][mkey].values
                x = 0 if ep == "R1" else 1
                ax.scatter(np.full(len(vals), x) + rng_jit(len(vals)), vals, color=col[ep], s=28, alpha=.8)
                ax.hlines(np.nanmean(vals), x - .2, x + .2, color=col[ep], lw=3)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["R1", "R2"])
            if r == 0:
                ax.set_title(mlab, fontsize=10)
            if c == 0:
                ax.set_ylabel(tgt, fontsize=10)
            ax.grid(alpha=.3, axis="y")
    fig.suptitle("Is R2's read-out tighter / more reliable than R1's? (premise of the mapping-reliability "
                 "hypothesis)\nprediction: R2 ≥ R1 on all three", fontsize=12, y=1.0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


_rng = np.random.default_rng(1)
def rng_jit(n):
    return (_rng.random(n) - .5) * 0.12


if __name__ == "__main__":
    main()
