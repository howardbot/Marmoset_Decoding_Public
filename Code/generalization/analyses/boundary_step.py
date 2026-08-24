"""Is there a STEP at the interference boundary, or smooth drift?
(best within-data proxy for the missing 'no-interference long-gap' control)

For every ordered session pair, in the CCA-aligned space (single-trial, K_PCS=15),
compute cross-day decode quality and alignment quality, vs CALENDAR-day gap. Fit
the within-R1 drift (quality ~ gap), extrapolate to the longer R1<->R2 gap, and
test whether the actual R1->R2 / R2->R1 values fall ON the drift line (=> ordinary
drift explains the asymmetry) or BELOW it (=> interference adds an extra drop).

  excess = predicted_by_within_R1_drift - actual    (positive => below trend => step)

Also reports within-R1 directional drift (earlier->later vs later->earlier).
Output: printed summary + CSV + figure.
NOTE: within-R1 gaps reach ~13 d; R1<->R2 gaps are ~15-29 d, so this EXTRAPOLATES
slightly beyond the fitted range. n(R2)=3.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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
K, D, SEED = 12, 12, 0         # v2 re-anchor: K_PCS 15->12, decode ALL canonical dims (was top-2)
TARGETS = ["relative_velocity", "relative_position"]
OUT = _THIS.parents[2] / "Results" / "workflows" / "manifold_geometry"
FIG = OUT / "figures"


def sdate(s):
    match = re.search(r"(20\d{6})", s)
    if match is None:
        raise ValueError(f"cannot parse session date: {s}")
    return datetime.strptime(match.group(1), "%Y%m%d")


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
    return X, pca_neural(du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS), k=K)[0], meta, None


def ridge_fit(Y, X, l2=1e-2):
    A = np.hstack([Y, np.ones((len(Y), 1))])
    return np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ X)


def ridge_pred(Y, W):
    return np.hstack([Y, np.ones((len(Y), 1))]) @ W


def corr_avg(t, p):
    return float(np.mean([np.corrcoef(t[:, j], p[:, j])[0, 1] for j in range(t.shape[1])]))


def epoch(s, sessions_r2):
    return "R2" if s in sessions_r2 else "R1"


def run_target(target, rng, sessions_r1, sessions_r2):
    sessions = sessions_r1 + sessions_r2
    cache = {
        s: load(s, target, EXCLUDE_TRIALS.get(s, [])) for s in sessions
    }
    rows = []
    for a, b in product(sessions, repeat=2):
        if a == b:
            continue
        Ya, Yb = align_full("single_trial", K, cache[a], cache[b], rng)
        if Ya is None:
            continue
        Xa, ma = cache[a][0], cache[a][2]
        Xb, mb = cache[b][0], cache[b][2]
        dec = corr_avg(Xb, ridge_pred(Yb[:, :D], ridge_fit(Ya[:, :D], Xa)))   # train a -> test b
        mua = trial_average_pc(Ya, ma, N_PHASE_BINS); mub = trial_average_pc(Yb, mb, N_PHASE_BINS)
        align = float(np.mean([np.corrcoef(mua[:, k], mub[:, k])[0, 1] for k in range(D)]))
        rows.append(dict(a=a[4:12], b=b[4:12],
                         ea=epoch(a, sessions_r2), eb=epoch(b, sessions_r2),
                         gap=(sdate(b) - sdate(a)).days,
                         pair=epoch(a, sessions_r2) + "->" + epoch(b, sessions_r2),
                         decode=dec, align=align))
    return pd.DataFrame(rows)


def analyse(df, col):
    w1 = df[df.pair == "R1->R1"]
    coef = np.polyfit(np.abs(w1.gap), w1[col], 1)            # quality ~ |gap|
    pred = lambda g: np.polyval(coef, np.abs(g))
    out = {}
    for p in ["R1->R2", "R2->R1"]:
        g = df[df.pair == p]
        excess = pred(g.gap.values) - g[col].values          # +ve => below drift trend
        try:
            pv = wilcoxon(excess, alternative="greater").pvalue
        except ValueError:
            pv = np.nan
        out[p] = (g[col].mean(), pred(g.gap.values).mean(), excess.mean(), pv)
    # within-R1 directional
    fwd = w1[w1.gap > 0][col].mean(); bwd = w1[w1.gap < 0][col].mean()
    return coef, out, fwd, bwd


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
    suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ci, tgt in enumerate(TARGETS):
        df = run_target(tgt, rng, sessions_r1, sessions_r2)
        df.insert(0, "animal", args.animal)
        data[tgt] = df
        df.to_csv(
            OUT / f"boundary_step_{tgt}{suffix}.csv", index=False
        )
        print(f"\n########## {tgt} ##########")
        for ri, col in enumerate(["decode", "align"]):
            coef, out, fwd, bwd = analyse(df, col)
            print(f"  [{col}] within-R1 drift slope = {coef[0]*7:+.3f}/week (intercept {coef[1]:.3f}); "
                  f"within-R1 dir: fwd {fwd:.3f} vs bwd {bwd:.3f}")
            for p, (act, prd, exc, pv) in out.items():
                # Verdict on EFFECT SIZE only. The Wilcoxon p is over day-pairs, which are
                # pseudo-replicated (the same 14 R1 / 3 R2 sessions recombine) at n(R2)=3,
                # so it is descriptive (flagged p*), NOT valid inference.
                verdict = "BELOW trend (step-like)" if exc > 0.02 else "on/above drift trend"
                print(f"      {p}: actual {act:.3f}  drift-predicts {prd:.3f}  excess {exc:+.3f}  "
                      f"p*={pv:.3f}  -> {verdict}")
            print(
                "      (* p over pseudo-replicated day-pairs, "
                f"n(R2)={len(sessions_r2)} — descriptive only, not inference)"
            )
            # plot
            ax = axes[ri, ci]
            cols = {"R1->R1": "#bbbbbb", "R1->R2": "#e74c3c", "R2->R1": "#3498db", "R2->R2": "#2ecc71"}
            for p, c in cols.items():
                g = df[df.pair == p]
                ax.scatter(np.abs(g.gap), g[col], s=28, alpha=.6, color=c, label=p, edgecolors="none")
            gg = np.linspace(0, df.gap.abs().max(), 50)
            ax.plot(gg, np.polyval(coef, gg), "k--", lw=1.5, label="within-R1 drift fit")
            ax.axvspan(13, df.gap.abs().max(), color="orange", alpha=.06)
            ax.set_xlabel("calendar-day gap"); ax.set_ylabel(col + " quality")
            ax.set_title(f"{tgt.split('_')[1]} — {col}", fontsize=11)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=.3)
    fig.suptitle(
        f"{args.animal} boundary-step test: does R1<->R2 fall on the "
        f"within-R1 drift line (drift) or below it (interference)?  "
        f"shaded = extrapolation zone   n(R2)={len(sessions_r2)}",
        y=1.0,
    )
    fig.tight_layout()
    figure_path = FIG / f"fig_boundary_step{suffix}.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {figure_path} + per-target CSVs")


if __name__ == "__main__":
    main()
