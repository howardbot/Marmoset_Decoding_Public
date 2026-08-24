"""Per-session kinematic variability diagnostics.

For each r1 + r2 session we build the locked decoder dataset (20 ms bins,
relative velocity) and compute three complementary variability measures:

  1. total_var   = Var(X) over all stacked bins, summed across dims.
                   Captures overall spread of velocity values the session ever sees.
  2. between_var = Var(per-trial mean X) summed across dims.
                   "How different are reaches from each other?" — the key knob
                   for a regression decoder, because the model learns to map
                   neural activity to *contrasts* between trials.
  3. within_var  = mean over trials of Var(X within trial), summed across dims.
                   "How dynamic is the velocity within a single reach?" —
                   captures within-reach modulation.
  4. trial_pair_r = mean pairwise Pearson correlation between trial-averaged
                    velocity trajectories (after phase resampling). Approaches
                    1.0 if every reach is a near-identical stereotyped motion.

If 0828 has Kalman fail not because of noise but because of stereotypy, we
expect: very high trial_pair_r AND very low between_var, relative to r1.

Output:
  Results/workflows/generalization/kinematic_variance.csv
  Results/workflows/generalization/kinematic_variance.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR))
from cross_day_decoder import (  # noqa: E402
    build_session_cache_entry,
    list_sessions,
    session_date,
    session_epoch,
    N_PHASE_BINS,
)

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
OUT_DIR = REPO_ROOT / "Results" / "workflows" / "generalization"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def per_trial_resample(X, meta, n_phase_bins=N_PHASE_BINS):
    """Resample each trial's X to a common phase grid -> (n_trials, n_phase_bins, 3)."""
    X = np.asarray(X, dtype=float)
    blocks = []
    for _, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 3:
            continue
        t_data = np.linspace(0.0, 1.0, len(idx))
        t_targ = np.linspace(0.0, 1.0, n_phase_bins)
        resampled = np.column_stack([
            np.interp(t_targ, t_data, X[idx, d]) for d in range(X.shape[1])
        ])
        blocks.append(resampled)
    return np.stack(blocks, axis=0) if blocks else np.zeros((0, n_phase_bins, X.shape[1]))


def trial_pair_similarity(trials_resampled):
    """Mean off-diagonal pairwise Pearson r between trial trajectories."""
    if trials_resampled.shape[0] < 2:
        return np.nan
    # flatten each trial's (n_phase_bins, 3) -> (n_phase_bins*3,)
    flat = trials_resampled.reshape(trials_resampled.shape[0], -1)
    # center each trial
    flat = flat - flat.mean(axis=1, keepdims=True)
    # pairwise correlation matrix
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = flat / norms
    corr = unit @ unit.T
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(corr[mask].mean())


def diagnose_one(session_tag):
    data = build_session_cache_entry(session_tag)
    X = data["X"]  # (T, 3) velocity
    meta = data["meta"]
    info = {
        "session": session_tag,
        "date": session_date(session_tag),
        "epoch": session_epoch(session_tag),
        "n_bins": X.shape[0],
        "n_trials": meta["trial_number"].nunique(),
    }
    # 1. total variance
    info["total_var"] = float(np.sum(np.var(X, axis=0)))
    # 2. between-trial variance (mean per trial)
    trial_means = []
    for _, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) > 0:
            trial_means.append(X[idx].mean(axis=0))
    trial_means = np.stack(trial_means, axis=0)
    info["between_var"] = float(np.sum(np.var(trial_means, axis=0)))
    # 3. within-trial mean variance
    within_vars = []
    for _, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) >= 2:
            within_vars.append(np.var(X[idx], axis=0))
    info["within_var"] = float(np.mean(np.sum(within_vars, axis=1)))
    # 4. trial-pair similarity after phase resampling
    trials_r = per_trial_resample(X, meta)
    info["trial_pair_r"] = trial_pair_similarity(trials_r)
    info["n_trials_resampled"] = trials_r.shape[0]
    return info


def main():
    sessions = list_sessions()
    rows = []
    for s in sessions:
        print(f"  computing {s}")
        try:
            rows.append(diagnose_one(s))
        except Exception as e:
            print(f"    SKIP: {type(e).__name__}: {e}")
            rows.append({
                "session": s, "date": session_date(s), "epoch": session_epoch(s),
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "kinematic_variance.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Print summary table
    print("\n=== Per-session kinematic variability ===")
    cols = ["date", "epoch", "n_trials", "total_var", "between_var",
            "within_var", "trial_pair_r"]
    print(df[cols].round(3).to_string(index=False))

    print("\n=== r1 baseline (n=13) vs r2 sessions ===")
    r1 = df[df.epoch == "r1"]
    r2 = df[df.epoch == "r2"]
    header = f"{'metric':<16} {'r1 median':>10} {'r1 IQR':>16}  {'0828':>9} {'0829':>9}"
    print(header)
    print("-" * len(header))
    for col in ["total_var", "between_var", "within_var", "trial_pair_r"]:
        if col not in r1.columns:
            continue
        vals = r1[col].dropna()
        if vals.empty:
            continue
        med = float(np.median(vals))
        lo = float(np.percentile(vals, 25))
        hi = float(np.percentile(vals, 75))
        def fmt(date):
            sub = r2[r2.date == date]
            if sub.empty or pd.isna(sub[col].iloc[0]):
                return "      --"
            return f"{sub[col].iloc[0]:>9.3f}"
        print(f"{col:<16} {med:>10.3f}  [{lo:>5.3f}, {hi:>5.3f}] "
              f"{fmt('20250828')} {fmt('20250829')}")

    # Plot: 4 panels for the 4 metrics, color by epoch
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    metrics = ["total_var", "between_var", "within_var", "trial_pair_r"]
    titles = ["Total velocity variance",
              "Between-trial variance\n(reach-to-reach diversity)",
              "Within-trial variance\n(reach dynamics)",
              "Trial-pair similarity\n(stereotypy: high = identical reaches)"]
    for ax, col, title in zip(axes, metrics, titles):
        plot_df = df.dropna(subset=[col]).sort_values("date")
        colors = ["tab:blue" if e == "r1" else "tab:orange" for e in plot_df["epoch"]]
        x = np.arange(len(plot_df))
        ax.bar(x, plot_df[col], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["date"], rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        # mark r1 median
        r1_med = np.median(df[df.epoch == "r1"][col].dropna())
        ax.axhline(r1_med, color="grey", linestyle="--", linewidth=1, label=f"r1 median = {r1_med:.2f}")
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Value")
    fig.suptitle("Per-session kinematic variability (r1 blue, r2 orange)")
    fig.tight_layout()
    out_png = OUT_DIR / "kinematic_variance.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
