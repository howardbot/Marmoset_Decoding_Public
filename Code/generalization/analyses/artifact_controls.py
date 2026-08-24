"""Layer-1 artifact controls for the R1->R2 / R2->R1 asymmetry.

Before asking *why* the asymmetry happens, rule out the boring explanation:
"r2 is just cleaner / more stereotyped, so an r2-trained decoder generalizes
better." Compare r1 vs r2 on:
  1. behavioral trial-to-trial variability (velocity, phase-aligned)
  2. neural trial-to-trial variability (in PC space)
  3. reach duration + peak speed (did the movement itself change?)
  4. within-day decoding accuracy (is r2 simply easier to decode?)

If r2 is markedly less variable / easier to decode, the asymmetry could be a
data-quality artifact. If r1 and r2 are comparable, the asymmetry is more likely
a genuine representational effect.

Config: locked (bin=30, butter_o2, sigma=50ms, 0828 trial-41 excluded), velocity.
Output: Results/workflows/manifold_geometry/artifact_controls.csv (+ figure).
NOTE: r2 has only 3 sessions -> its epoch stats are n=3 (interpret with caution).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# The location of the script
_THIS = Path(__file__).resolve().parent
# Make sibling/generalization modules importable when this script is launched
# directly from the repo root.
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

# Shared dataset builder and trial utilities live one level above.
import decoder_utils as du
# PCA helper: raw/smoothed neural activity -> low-dimensional PC activity.
from manifold_align import pca_neural
# Reuse the exact Kalman + M2 metric semantics from the big within-day sweep.
from big_sweep_phase1_withinday import fit_kalman, m2_per_trial
# Reuse the canonical session list, cleaning choices, and fixed preprocessing
# constants from the current cross-day sweep pipeline.
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS, EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

# Keep the script output readable; NWB/sklearn can emit noisy benign warnings.
warnings.filterwarnings("ignore")

# This script follows the current figure/sweep locked config, not the older
# 20 ms cross_day_decoder.py config.
BIN_SIZE_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
TARGET = "relative_velocity"
# Number of phase bins used when putting variable-length reaches on a common
# 0..1 reach-phase axis for trial-to-trial variability.
P = 30
# Neural variability is measured after projecting each day to its own top PCs.
K_PCS = 12                     # v2 re-anchor (was 15)
N_FOLDS = 5
SEED = 0

REPO_ROOT = _THIS.parents[2]
OUT_CSV = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "artifact_controls.csv"
FIG_DIR = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "figures"


def load_session(session, exclude=()):
    """Load one session and return the locked-config X/Y/meta arrays.

    X is the behavioral target (velocity), Y_sm is causally-smoothed neural
    spike-count activity, and meta preserves trial identity for later grouping.
    """
    # decoder_utils expects bin size in seconds.
    bin_s = BIN_SIZE_MS / 1000.0
    # decoder_utils uses module-level globals to decide which NWB to open, so
    # every research script rebinding a session follows this pattern.
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_s
    # Open the processed NWB and find its reaching_segments interval table.
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        # Build per-bin behavioral velocity X, neural spike counts Y, and trial
        # metadata using exactly the same preprocessing as the figure pipeline.
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, TARGET, bin_size=bin_s, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        # Always close the NWB handle, even if dataset construction fails.
        io.close()
    # Drop documented bad trials, currently 0828 trial 41 only.
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    # Smooth neural counts causally within each trial; sigma is specified in ms,
    # so convert it into bins for the current bin size.
    Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_SIZE_MS)
    return X, Y_sm, meta


def resample(arr, idx, P=P):
    """Put one variable-length trial onto a fixed reach-phase grid."""
    # Original trial time is normalized to phase 0..1, independent of duration.
    # This is just constructing two index arrays
    ts = np.linspace(0, 1, len(idx)); tt = np.linspace(0, 1, P)
    # Interpolate every behavioral/neural dimension on the same phase grid.
    return np.column_stack([np.interp(tt, ts, arr[idx, d]) for d in range(arr.shape[1])])


def trial_to_trial_var(arr, meta):
    """Mean over (phase, dim) of the across-trial std of phase-aligned arr."""
    # Creating empty array for the result
    stk = []
    # Reconstruct each trial's sample indices from the stacked bin matrix.
    for _, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        # Very short trials cannot support a meaningful interpolation.
        if len(idx) < 3:
            continue
        # Store one (P phase bins x D dimensions) trajectory per trial.
        stk.append(resample(arr, idx))
    stk = np.stack(stk, 0)             # (n_trials, P, dim)
    # For each phase/dim, compute across-trial std; then average those stds into
    # one scalar "how stereotyped are trials in this session?" metric.
    return float(np.mean(np.std(stk, axis=0)))


def within_day_decode(X, Y_pc, meta):
    """Estimate whether a session is intrinsically easier to decode.

    This is not the cross-day asymmetry test. It is a same-day sanity check:
    if R2 had much higher within-day decoding, the R2-trained decoder might
    generalize better simply because R2 is cleaner/easier.
    """
    scores = []
    # Use whole-trial folds so adjacent bins from the same reach never split
    # across train and test.
    for tr, te in du.kfold_split_by_trial(meta, n_splits=N_FOLDS, random_seed=SEED):
        # Skip folds that become too small after filtering.
        if tr.sum() < 50 or te.sum() < 20:
            continue
        # fit_kalman expects test metadata indexed from zero.
        mte = meta[te].reset_index(drop=True)
        try:
            # Fit on train trials in PC space and predict held-out trials.
            Xc, pred = fit_kalman(X[tr], Y_pc[tr], X[te], Y_pc[te], mte)
            # M2 = per-trial Pearson r averaged over trials and dimensions.
            scores.append(m2_per_trial(Xc, pred, mte))
        except Exception:
            # A failed fold should not kill the whole artifact-control summary.
            pass
    return float(np.nanmean(scores)) if scores else np.nan


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
    output_csv = OUT_CSV.with_name(f"{OUT_CSV.stem}{suffix}.csv")
    # Output directory for the summary figure.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    # Each session becomes one row in the artifact-control table.
    for s in all_sessions:
        # Apply the documented outlier exclusion only for sessions that need it.
        X, Y_sm, meta = load_session(s, EXCLUDE_TRIALS.get(s, []))
        # Neural variability and within-day decoding use each session's own PC
        # space; no cross-day CCA is involved in this layer-1 control.
        Y_pc = pca_neural(Y_sm, k=K_PCS)[0]
        # Trial duration in bins; convert to milliseconds later.
        durs = [len(idx) for _, idx in meta.groupby("trial_number").indices.items()]
        # Empty array for each trial's peak speed
        speeds = []
        for _, idx in meta.groupby("trial_number").indices.items():
            # X is relative velocity, so ||X|| is speed; keep each trial's peak.
            speeds.append(np.max(np.linalg.norm(X[np.asarray(idx)], axis=1)))
        rows.append({
            # Keep a compact YYYYMMDD label in the CSV/printout.
            "session": s[4:12],
            "epoch": "R2" if s in sessions_r2 else "R1",
            "n_trials": meta["trial_number"].nunique(),
            # Behavioral stereotypy: lower means reaches are more similar.
            "behav_var": trial_to_trial_var(X, meta),
            # Neural stereotypy in PC space: lower means neural trajectories are
            # more similar across trials.
            "neural_var": trial_to_trial_var(Y_pc, meta),
            "reach_dur_ms": float(np.median(durs) * BIN_SIZE_MS),
            "peak_speed": float(np.median(speeds)),
            "within_day_decode": within_day_decode(X, Y_pc, meta),
        })
        print(f"  {rows[-1]['session']} done")

    # Persist the per-session metrics so later writeups do not rely on terminal logs.
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("\n" + df.to_string(index=False))

    # These are the scalar controls shown in the figure and epoch-mean table.
    cols = ["behav_var", "neural_var", "reach_dur_ms", "peak_speed", "within_day_decode"]
    print("\n=== epoch means ===")
    print(df.groupby("epoch")[cols].mean().round(4).to_string())

    # One small panel per control metric: dots are sessions, horizontal lines
    # are epoch means. This is intentionally descriptive, not a high-powered
    # inferential test, because R2 has only three sessions.
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4.5))
    for ax, c in zip(axes, cols):
        for ep, color in [("R1", "#7f8c8d"), ("R2", "#e74c3c")]:
            g = df[df.epoch == ep][c]
            ax.scatter([ep] * len(g), g, color=color, s=45, alpha=0.7, edgecolors="white")
            # Draw the epoch mean as a short horizontal bar.
            ax.hlines(g.mean(), -0.3 + (0 if ep == "R1" else 1), 0.3 + (0 if ep == "R1" else 1),
                      color=color, lw=2)
        ax.set_title(c, fontsize=10); ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"{args.animal} layer-1 artifact controls: R1 vs R2 "
        f"(is R2 just cleaner / different?)\n"
        f"n(R1)={len(sessions_r1)}, n(R2)={len(sessions_r2)} — descriptive",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / f"fig_artifact_controls{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}\nsaved {output_csv}")


if __name__ == "__main__":
    main()
