"""H0 control: is the R1->R2 / R2->R1 asymmetry just a decoder-conditioning (SNR) artifact?

Premise H0 challenges: the asymmetry might NOT be representational. R2 is the
*noisier* training epoch (trial-to-trial neural var 1.47 > R1's 1.16,
`artifact_controls.py`). Training a (Kalman) decoder on noisier inputs acts like
noise-injection regularization -> shrunk, more robust weights -> better cross-day
generalization. So "R2->R1 holds, R1->R2 drops" could be pure statistics: R2 trains
a more robust decoder, R1 overfits. No representational change required.

Test (increase-only per-dim variance control): inject independent Gaussian noise
into the *training* day's aligned canonical activity so each decode canonical dim is
raised to the OTHER day's std only when train std < test std. This is a diagonal
variance control, not a full covariance or temporal-noise match. Re-measure the
asymmetry.
  - H0 supported  : asymmetry collapses toward 0 (R1->R2 rises toward R2->R1) once
                    R1-training is noised up to R2's variance.
  - H0 rejected   : asymmetry survives variance-matching -> representational.

Also runs a noise-level sweep (alpha x per-dim std injected into R1-training). That
figure is a noise-flooding sensitivity sweep; the actual per-dim matched condition is
stored as ``cond == "var_matched"`` in the CSV.

Config: locked single-trial CCA, K_PCS=12, decode all d=12 dims, 0828 trial-41
excluded.
Targets: position (credible) + velocity. n=17 (R1=14, R2=3).
Output: Results/manifold_geometry/h0_snr_control.csv (+ figure).
"""

# Inject Gaussian noise into training activity for diagonal variance controls.
from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))   # Code/
sys.path.insert(0, str(_THIS.parent))        # generalization/

import decoder_utils as du
from manifold_align import pca_neural, trial_average_pc
from dimension_sweep import align_full
from big_sweep_phase2_crossday import (
    SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS, N_PHASE_BINS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
    kalman_fit_predict, m2_per_trial,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 12                         # v2 re-anchor (was 15)
D = 12                         # decode ALL canonical dims (was top-2)
SEED = 0
N_SEEDS = 15                   # I2: repeat noise injection over many seeds -> error bars
TARGETS = ["relative_position", "relative_velocity"]
# alpha = noise std injected into a TRAIN dim, as a multiple of that dim's own std.
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "manifold_geometry" / "h0_snr_control.csv"
FIG = REPO / "Results" / "manifold_geometry" / "figures" / "fig_h0_snr_control.png"


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


def decode(Xtr, Ytr, Xte, Yte, mte):
    Xc, pred = kalman_fit_predict(Xtr, Ytr[:, :D], Xte, Yte[:, :D], mte)
    return m2_per_trial(Xc, pred, mte)


def inject_to_match(Ytr, Yte, rng, dims=D):
    """Raise each of the first `dims` columns of TRAIN to the TEST day's per-dim std
    by adding independent Gaussian noise (variance-match training to the noisier day).
    Only adds noise where train std < test std; never removes variance."""
    Ytr = Ytr.copy()
    for d in range(dims):
        s_tr = Ytr[:, d].std()
        s_te = Yte[:, d].std()
        if s_te > s_tr:
            add = np.sqrt(max(s_te ** 2 - s_tr ** 2, 0.0))
            Ytr[:, d] = Ytr[:, d] + rng.normal(0, add, size=Ytr.shape[0])
    return Ytr


def inject_alpha(Ytr, alpha, rng, dims=D):
    """Add noise of std = alpha * (that dim's std) to the first `dims` train columns."""
    if alpha <= 0:
        return Ytr
    Ytr = Ytr.copy()
    for d in range(dims):
        Ytr[:, d] = Ytr[:, d] + rng.normal(0, alpha * Ytr[:, d].std(), size=Ytr.shape[0])
    return Ytr


def run_seed(cache, rng):
    """One seed: align each ordered pair ONCE (the single_trial CCA is the cost), then apply the
    noise conditions on top of the cached alignment."""
    pairs_fwd = list(product(SESSIONS_R1, SESSIONS_R2))   # R1->R2 (forward)
    pairs_rev = list(product(SESSIONS_R2, SESSIONS_R1))   # R2->R1 (reverse)
    aln = {(a, b): align_full("single_trial", K, cache[a], cache[b], rng)
           for a, b in pairs_fwd + pairs_rev}

    def dec(a, b, mode):
        Ya, Yb = aln[(a, b)]
        if Ya is None:
            return None
        if mode == "match":
            Ya = inject_to_match(Ya, Yb, rng)
        elif isinstance(mode, (int, float)) and mode > 0:
            Ya = inject_alpha(Ya, float(mode), rng)
        return decode(cache[a][0], Ya, cache[b][0], Yb, cache[b][2])

    def blk(pairs, mode):
        v = [x for x in (dec(a, b, mode) for a, b in pairs) if x is not None]
        return float(np.nanmean(v)) if v else float("nan")

    rows = []
    f0, r0 = blk(pairs_fwd, "baseline"), blk(pairs_rev, "baseline")
    fm, rm = blk(pairs_fwd, "match"), blk(pairs_rev, "match")
    rows.append(dict(cond="baseline", fwd=f0, rev=r0, asym=r0 - f0))
    rows.append(dict(cond="var_matched", fwd=fm, rev=rm, asym=rm - fm))
    for al in ALPHAS:
        f = blk(pairs_fwd, al)
        rows.append(dict(cond=f"alpha_fwd_{al}", fwd=f, rev=r0, asym=r0 - f))
    return rows


def main():
    FIG.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for tgt in TARGETS:
        print(f"\n########## {tgt} ##########  ({N_SEEDS} seeds)")
        cache = {s: load(s, tgt, EXCLUDE_TRIALS.get(s, [])) for s in SESSIONS_R1 + SESSIONS_R2}
        for seed in range(N_SEEDS):
            for r in run_seed(cache, np.random.default_rng(seed)):
                r.update(target=tgt, seed=seed)
                all_rows.append(r)
        print(f"  {N_SEEDS} seeds done")

    raw = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT_CSV.with_name("h0_snr_control_seeds.csv"), index=False)

    # aggregate across seeds -> mean, SD, 95% CI
    agg = raw.groupby(["target", "cond"]).agg(
        fwd=("fwd", "mean"), fwd_sd=("fwd", "std"),
        rev=("rev", "mean"), rev_sd=("rev", "std"),
        asym=("asym", "mean"), asym_sd=("asym", "std"), n=("asym", "size")).reset_index()
    agg["asym_ci"] = 1.96 * agg.asym_sd / np.sqrt(agg.n)
    agg.to_csv(OUT_CSV, index=False)

    for tgt in TARGETS:
        a = agg[agg.target == tgt]
        base = a[a.cond == "baseline"].iloc[0]
        match = a[a.cond == "var_matched"].iloc[0]
        ret = match.asym / base.asym * 100 if base.asym else float("nan")
        print(f"\n[{tgt}]")
        print(f"  baseline    : R1->R2={base.fwd:.3f}  R2->R1={base.rev:.3f}  asym={base.asym:+.3f} ± {base.asym_ci:.3f}")
        print(f"  var-matched : R1->R2={match.fwd:.3f}  R2->R1={match.rev:.3f}  asym={match.asym:+.3f} ± {match.asym_ci:.3f}")
        print(f"  -> asymmetry retained after variance-match: {ret:.0f}%  (CI excludes 0: {base.asym - base.asym_ci > 0})")

    # figure: alpha sweep per target with error bars (SD across seeds)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 4.5))
    for ax, tgt in zip(np.atleast_1d(axes), TARGETS):
        a = agg[(agg.target == tgt) & (agg.cond.str.startswith("alpha_fwd_"))].copy()
        a["al"] = [float(c.split("_")[-1]) for c in a.cond]
        a = a.sort_values("al")
        ax.errorbar(a.al, a.fwd, yerr=a.fwd_sd, fmt="-o", color="#e74c3c", capsize=3,
                    label="R1→R2 (train R1 + noise), ±SD")
        base = agg[(agg.target == tgt) & (agg.cond == "baseline")].iloc[0]
        ax.axhline(base.rev, color="#3498db", ls="--", label=f"R2→R1 baseline ({base.rev:.3f})")
        ax.axhline(base.fwd, color="#e74c3c", ls=":", alpha=.6, label=f"R1→R2 baseline ({base.fwd:.3f})")
        ax.set_title(f"{tgt}\nH0: noising R1-train should lift R1→R2 toward R2→R1", fontsize=10)
        ax.set_xlabel("injected noise α (× per-dim std)"); ax.set_ylabel("cross-day decode corr")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle(f"H0 SNR control — training-noise injection, {N_SEEDS} seeds (error bars = SD)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nsaved {OUT_CSV}\nsaved {FIG}")


if __name__ == "__main__":
    main()
