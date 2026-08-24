"""Sweep the cross-day decoder pipeline on r1 across 2 bin sizes x 2 smoothing sigmas.

For each of 4 configurations we rebuild the per-session manifold cache and
recompute the full 13x13 r1 cross-day matrix in PC + CCA-aligned space.
We report both M1 (concat Pearson r) and M2 (per-trial Pearson r averaged)
on every pair and aggregate off-diagonal and diagonal means.

This is a parameter sensitivity study, not a re-locking of the standard
config — the locked configuration for downstream analyses remains
20 ms / sigma 50 ms as set in `cross_day_decoder.py`.

Outputs:
  Results/workflows/generalization/r1_config_sweep_long.csv     one row per (bin, sigma, train, test)
  Results/workflows/generalization/r1_config_sweep_summary.csv  one row per (bin, sigma)
  Results/workflows/generalization/r1_config_sweep_heatmap.png  2 metric rows x 4 config cols
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR))

import decoder_utils as du  # noqa: E402
from cross_day_decoder import (  # noqa: E402
    build_session_cache_entry, list_sessions, session_date, session_epoch,
    eval_off_diagonal, eval_diagonal, load_optimal_lag_per_session,
    K_PCS, N_PHASE_BINS, N_SPLITS_DIAGONAL,
)
import cross_day_decoder as cdd  # to override BIN_SIZE_MS / SMOOTH_SIGMA_MS  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
RES = REPO_ROOT / "Results" / "workflows" / "generalization"

BIN_SIZES = [10, 20, 30, 40, 50]
SIGMAS = [50, 100]


def run_one_config(bin_size_ms, sigma_ms):
    """Build cache + 13x13 r1 matrix for one (bin, sigma) config."""
    cdd.BIN_SIZE_MS = bin_size_ms
    cdd.BIN_SIZE_S = bin_size_ms / 1000.0
    cdd.SMOOTH_SIGMA_MS = sigma_ms

    sessions = [t for t in list_sessions() if session_epoch(t) == "r1"]
    t0 = time.time()
    cache = {}
    for tag in sessions:
        cache[session_date(tag)] = build_session_cache_entry(tag)
    print(f"  cache built in {time.time() - t0:.1f}s")

    lags = load_optimal_lag_per_session()
    default_lag = int(np.median(list(lags.values()))) if lags else 100

    rows = []
    dates = sorted(cache.keys())
    rng = np.random.default_rng(0)
    for i, d_tr in enumerate(dates):
        lag_ms = int(lags.get(d_tr, default_lag))
        lag_bins = lag_ms // bin_size_ms
        for j, d_te in enumerate(dates):
            try:
                if i == j:
                    out = eval_diagonal(cache[d_tr], lag_bins, rng=rng, with_null=False)
                else:
                    out = eval_off_diagonal(cache[d_tr], cache[d_te], lag_bins,
                                            rng=rng, with_null=False)
                rows.append({
                    "bin_size_ms": bin_size_ms,
                    "sigma_ms": sigma_ms,
                    "train_date": d_tr,
                    "test_date": d_te,
                    "lag_ms": lag_ms,
                    "M1_concat": out["corr_concat_mean"],
                    "M2_per_trial": out["corr_per_trial_mean"],
                })
            except Exception as e:
                print(f"    FAIL {d_tr}->{d_te}: {type(e).__name__}: {e}")
                rows.append({
                    "bin_size_ms": bin_size_ms, "sigma_ms": sigma_ms,
                    "train_date": d_tr, "test_date": d_te, "lag_ms": lag_ms,
                    "M1_concat": np.nan, "M2_per_trial": np.nan,
                })
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for bin_size in BIN_SIZES:
        for sigma in SIGMAS:
            print(f"\n=== bin={bin_size}ms, sigma={sigma}ms ===")
            t0 = time.time()
            df = run_one_config(bin_size, sigma)
            all_rows.append(df)
            print(f"  done in {(time.time() - t0)/60:.1f} min")

    long = pd.concat(all_rows, ignore_index=True)
    long.to_csv(RES / "r1_config_sweep_long.csv", index=False)
    print(f"\nSaved {RES / 'r1_config_sweep_long.csv'}")

    # ---- Summary per config ----
    summary_rows = []
    for (bs, sg), grp in long.groupby(["bin_size_ms", "sigma_ms"]):
        diag = grp[grp.train_date == grp.test_date]
        off = grp[grp.train_date != grp.test_date]
        summary_rows.append({
            "bin_size_ms": bs,
            "sigma_ms": sg,
            "within_M1_mean": float(diag["M1_concat"].mean()),
            "within_M2_mean": float(diag["M2_per_trial"].mean()),
            "off_M1_mean": float(off["M1_concat"].mean()),
            "off_M2_mean": float(off["M2_per_trial"].mean()),
            "drop_M1": float(diag["M1_concat"].mean() - off["M1_concat"].mean()),
            "drop_M2": float(diag["M2_per_trial"].mean() - off["M2_per_trial"].mean()),
            "n_pairs": len(off),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RES / "r1_config_sweep_summary.csv", index=False)
    print("\n=== Sweep summary ===")
    print(summary.round(3).to_string(index=False))

    # ---- Heatmap grid: 2 rows (metrics) x N cols (configs) ----
    configs = [(bs, sg) for bs in BIN_SIZES for sg in SIGMAS]
    ncols = len(configs)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 8.5))

    # Compute global color range for fair visual comparison within each metric
    all_M1 = long.pivot_table(index=["bin_size_ms", "sigma_ms", "train_date"],
                              columns="test_date", values="M1_concat").to_numpy()
    all_M2 = long.pivot_table(index=["bin_size_ms", "sigma_ms", "train_date"],
                              columns="test_date", values="M2_per_trial").to_numpy()
    vmax_M1 = float(np.nanmax(all_M1))
    vmax_M2 = float(np.nanmax(all_M2))

    for col, (bs, sg) in enumerate(configs):
        sub = long[(long.bin_size_ms == bs) & (long.sigma_ms == sg)]
        for row, (metric_col, vmax, label) in enumerate([
            ("M1_concat", vmax_M1, "M1 (Origin concat)"),
            ("M2_per_trial", vmax_M2, "M2 (per-trial)"),
        ]):
            mat = sub.pivot(index="train_date", columns="test_date", values=metric_col)
            ax = axes[row, col]
            im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
            ax.set_xticks(range(len(mat.columns)))
            ax.set_yticks(range(len(mat.index)))
            ax.set_xticklabels([str(c)[-4:] for c in mat.columns], rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels([str(c)[-4:] for c in mat.index], fontsize=7)
            for i in range(len(mat)):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="red", linewidth=1.0))
            off_mean = summary.loc[(summary.bin_size_ms == bs) & (summary.sigma_ms == sg),
                                   "off_M2_mean" if metric_col == "M2_per_trial" else "off_M1_mean"].iloc[0]
            ax.set_title(f"bin={bs}ms, σ={sg}ms\n{label}, off-diag mean = {off_mean:.3f}",
                         fontsize=9)
            if col == 0:
                ax.set_ylabel("Train day")
            if row == 1:
                ax.set_xlabel("Test day")
            plt.colorbar(im, ax=ax, label="mean velocity corr")

    fig.suptitle("r1 cross-day decoder transfer — 2 bin sizes × 2 smoothing σ × 2 evaluation metrics",
                 fontsize=12)
    fig.tight_layout()
    out = RES / "r1_config_sweep_heatmap.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
