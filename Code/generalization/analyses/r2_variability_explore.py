"""Raw exploration (theme E): R2 is more NEURALLY variable than R1 despite being
BEHAVIOURALLY less variable — is the extra variability shared (population/global-state,
e.g. attention/engagement) or private (single-unit noise)?

Per session, in the neural space (no CCA — this is about raw variability):
  * noise_mag : total trial-to-trial residual variance (each trial's phase-aligned
                trajectory minus the session mean reach), summed over units.
  * noise_PR  : participation ratio of the residual (noise) covariance = effective
                DIMENSIONALITY of the trial-to-trial variability. Low PR = the extra
                variability is low-D / shared / correlated across units (population,
                global-state flavour). High PR = spread across units (private noise).
  * shared_frac : fraction of residual variance in the top-3 residual PCs (shared).

Prediction (global-state account): R2 has higher noise_mag AND lower noise_PR /
higher shared_frac -> the excess is population-level, not private single-unit noise.

Exploratory, honest: n(R2)=3, report per-session values + epoch means, effect sizes.
Reads NWB -> HatLab env. Output: Results/workflows/manifold_geometry/r2_variability_explore.csv (+ fig).
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

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from big_sweep_phase2_crossday import (
    SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
N_PHASE = 30
REPO_ROOT = _THIS.parents[2]
OUT_CSV = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "r2_variability_explore.csv"
FIG = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_r2_variability_explore.png"


def load(session, exclude=()):
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, "relative_velocity", bin_size=BIN_MS / 1000.0,
            unit_qualities=UNIT_QUALITIES, trial_results=TRIAL_RESULTS,
            trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    Ysm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS)
    return Ysm, meta


def phase_stack(Y, meta, n_phase=N_PHASE):
    Y = np.asarray(Y, float)
    out = []
    tt = np.linspace(0, 1, n_phase)
    for _, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 3:
            continue
        ts = np.linspace(0, 1, len(idx))
        out.append(np.column_stack([np.interp(tt, ts, Y[idx, u]) for u in range(Y.shape[1])]))
    return np.stack(out, 0)          # (n_trials, n_phase, n_units)


def participation_ratio(cov):
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def run(session, exclude):
    Ysm, meta = load(session, exclude)
    st = phase_stack(Ysm, meta)                      # (T, P, U)
    mu = st.mean(0)                                  # mean reach (P, U)
    resid = (st - mu[None]).reshape(-1, st.shape[-1])   # (T*P, U) trial-to-trial residual cloud
    # z-score units so a few high-rate units don't dominate the magnitude/PR
    sd = resid.std(0) + 1e-9
    resid_z = resid / sd
    C = np.cov(resid_z.T)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    ev = ev[ev > 0]
    return {
        "n_units": st.shape[-1],
        "noise_mag": float((resid ** 2).mean()),      # raw magnitude (unnormalised)
        "noise_PR": participation_ratio(C),           # effective noise dimensionality
        "shared_frac": float(ev[:3].sum() / ev.sum()),  # top-3 residual PCs (shared)
    }


def main():
    rows = []
    for epoch, sessions in (("R1", SESSIONS_R1), ("R2", SESSIONS_R2)):
        for s in sessions:
            r = run(s, EXCLUDE_TRIALS.get(s, []))
            r.update(epoch=epoch, session=s.replace("TSAL", "")[:8])
            rows.append(r)
            print(f"  {epoch} {r['session']}: noise_mag={r['noise_mag']:.3f} "
                  f"noise_PR={r['noise_PR']:.1f} shared_frac(top3)={r['shared_frac']:.2f} "
                  f"(U={r['n_units']})")
    df = pd.DataFrame(rows)
    # merge behavioural + neural var from artifact_controls for the decoupling panel
    ac = pd.read_csv(REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "artifact_controls.csv")
    ac["session"] = ac["session"].astype(str)
    df = df.merge(ac[["session", "behav_var", "neural_var"]], on="session", how="left")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}\n")
    print("=== R1 vs R2 means ===")
    print(df.groupby("epoch")[["neural_var", "behav_var", "noise_mag", "noise_PR", "shared_frac"]]
          .mean().round(3).to_string())
    make_figure(df, FIG)


def make_figure(df, out):
    col = {"R1": "#7f8c8d", "R2": "#e67e22"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    # (A) neural vs behavioural variability decoupling
    for ep in ("R1", "R2"):
        g = df[df.epoch == ep]
        ax[0].scatter(g.behav_var, g.neural_var, color=col[ep], s=55, label=ep, edgecolor="k")
    ax[0].set_xlabel("behavioural trial-to-trial var"); ax[0].set_ylabel("neural trial-to-trial var")
    ax[0].set_title("(A) R2: more NEURAL var, less BEHAVIOURAL var\n(neural noise ↑ for cleaner behaviour)", fontsize=10)
    ax[0].legend(); ax[0].grid(alpha=.3)
    # (B) noise dimensionality (PR): lower = more shared/population
    # (C) shared fraction (top-3 residual PCs)
    for j, (key, lab) in enumerate([("noise_PR", "(B) noise dimensionality (PR)\nlower = shared/population"),
                                    ("shared_frac", "(C) frac noise in top-3 PCs\nhigher = shared/population")]):
        a = ax[j + 1]
        for ep in ("R1", "R2"):
            v = df[df.epoch == ep][key].values
            x = 0 if ep == "R1" else 1
            a.scatter(np.full(len(v), x) + (np.random.RandomState(0).rand(len(v)) - .5) * .12,
                      v, color=col[ep], s=45, alpha=.85, edgecolor="k")
            a.hlines(np.mean(v), x - .2, x + .2, color=col[ep], lw=3)
        a.set_xticks([0, 1]); a.set_xticklabels(["R1", "R2"]); a.set_title(lab, fontsize=10)
        a.grid(alpha=.3, axis="y")
    fig.suptitle("R2's excess neural variability is low-dimensional / shared — a population/global-state "
                 "(attention-engagement) signature, not private single-unit noise", fontsize=12, y=1.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
