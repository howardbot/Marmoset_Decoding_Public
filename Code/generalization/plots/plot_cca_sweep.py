"""Plot the CCA sweep: two figures sharing the same x-axis (# CCA dimensions).

  fig_cca_decode.png  -- cross-day DECODING corr vs # canonical dims, Kalman,
                         velocity. Four directed-pair curves (R1->R1, R1->R2,
                         R2->R1, R2->R2), each mean +/- 1 SEM across pairs.

  fig_cca_score.png   -- held-out CANONICAL CORRELATION vs canonical dimension.
                         CC is direction-symmetric, so direction is collapsed
                         into the mean: three unordered categories (R1-R1,
                         R1-R2, R2-R2), each mean +/- 1 SEM.

  fig_cca_score_cumulative.png -- cumulative view: for each K, the running MEAN
                         of held-out CC over the first K dims ("overall alignment
                         using the top K dims"). Starts ~0.9 at K=1 and decays as
                         noise-aligned dims dilute the average.

Together: the alignment figure shows only the first canonical dimension is
genuinely shared across days (CC ~0.9; the rest fall to the noise floor), which
explains why the decoding figure tops out and then declines past the first few
dimensions. The decoding asymmetry (R1->R2 below R2->R1) lives in the decode
figure but NOT in the symmetric alignment figure -- i.e. the asymmetry is a
readout phenomenon, not an alignment-quality one.

Reads Results/generalization/cca_sweep_long.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS))

from plotting_common import PAIR_COLORS, config_caption, ensure_fig_dir

REPO_ROOT = _THIS.parents[1]
CSV = REPO_ROOT / "Results" / "generalization" / "cca_sweep_long.csv"

DIRECTED = ["R1->R1", "R1->R2", "R2->R1", "R2->R2"]
UNORDERED_COLORS = {"R1-R1": "#7f8c8d", "R1-R2": "#e74c3c", "R2-R2": "#34495e"}
CC_FLOOR = 0.2


def _curve(ax, g, color, label):
    stats = g.groupby("n_cca")["corr"].agg(["mean", "sem"]).reset_index()
    d, m, s = stats["n_cca"], stats["mean"], stats["sem"]
    ax.plot(d, m, "-o", color=color, ms=4, lw=1.9, label=label)
    ax.fill_between(d, m - s, m + s, color=color, alpha=0.18)


def plot_decode(df, out_dir):
    sub = df[(df["metric"] == "decode") & (df["target_mode"] == "relative_velocity")]
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in DIRECTED:
        g = sub[sub["pair_category"] == cat]
        if g.empty:
            continue
        n = g.groupby(["train_session", "test_session"]).ngroups
        _curve(ax, g, PAIR_COLORS[cat], f"{cat}  (n={n})")
    ax.set_xlabel("# CCA dimensions kept (PCA fixed at 15)")
    ax.set_ylabel("cross-day decoding corr")
    ax.set_xticks(range(1, 16))
    ax.set_title("Cross-day decoding vs # CCA dimensions  ·  Kalman · velocity", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.suptitle(config_caption(), fontsize=9, y=0.98, color="gray")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "fig_cca_decode.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def plot_score(df, out_dir):
    sub = df[df["metric"] == "cca"].copy()
    # Collapse direction: map directed category to unordered (CC is symmetric).
    sub["uocat"] = sub["pair_category"].map({
        "R1->R1": "R1-R1", "R1->R2": "R1-R2", "R2->R1": "R1-R2", "R2->R2": "R2-R2",
    })
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in ["R1-R1", "R1-R2", "R2-R2"]:
        g = sub[sub["uocat"] == cat]
        if g.empty:
            continue
        n = g.groupby(["train_session", "test_session"]).ngroups
        _curve(ax, g, UNORDERED_COLORS[cat], f"{cat}  (n={n})")
    ax.axhline(CC_FLOOR, color="black", ls=":", lw=1, alpha=0.6)
    ax.text(15, CC_FLOOR + 0.01, f"CC={CC_FLOOR} (noise floor)", ha="right",
            fontsize=8, color="#555")
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel("canonical dimension index")
    ax.set_ylabel("held-out canonical correlation")
    ax.set_xticks(range(1, 16))
    ax.set_title("CCA alignment quality across days (held-out)", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.suptitle(f"{config_caption()}  ·  10 random trial-half splits, "
                 "fit CCA on one half / score CC on the other",
                 fontsize=9, y=0.98, color="gray")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "fig_cca_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def plot_score_cumulative(df, out_dir):
    """Cumulative view: for each K, the running MEAN of held-out CC over the
    first K canonical dimensions = 'overall alignment using the top K dims'."""
    sub = df[df["metric"] == "cca"].copy()
    sub["uocat"] = sub["pair_category"].map({
        "R1->R1": "R1-R1", "R1->R2": "R1-R2", "R2->R1": "R1-R2", "R2->R2": "R2-R2",
    })
    rows = []
    for (tr, te), g in sub.groupby(["train_session", "test_session"]):
        g = g.sort_values("n_cca")
        cc = g["corr"].to_numpy()
        cummean = np.cumsum(cc) / np.arange(1, len(cc) + 1)
        for k, v in zip(g["n_cca"].to_numpy(), cummean):
            rows.append({"uocat": g["uocat"].iloc[0], "train_session": tr,
                         "test_session": te, "K": int(k), "cummean": v})
    cum = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in ["R1-R1", "R1-R2", "R2-R2"]:
        g = cum[cum["uocat"] == cat]
        if g.empty:
            continue
        stats = g.groupby("K")["cummean"].agg(["mean", "sem"]).reset_index()
        n = g.groupby(["train_session", "test_session"]).ngroups
        ax.plot(stats["K"], stats["mean"], "-o", color=UNORDERED_COLORS[cat],
                ms=4, lw=1.9, label=f"{cat}  (n={n})")
        ax.fill_between(stats["K"], stats["mean"] - stats["sem"],
                        stats["mean"] + stats["sem"], color=UNORDERED_COLORS[cat], alpha=0.18)
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel("# canonical dimensions included (first K)")
    ax.set_ylabel("cumulative mean held-out CC")
    ax.set_xticks(range(1, 16))
    ax.set_title("Cumulative CCA alignment: overall corr using the first K dims", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.suptitle(f"{config_caption()}  ·  running mean of held-out CC over dims 1..K",
                 fontsize=9, y=0.98, color="gray")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "fig_cca_score_cumulative.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def main():
    out_dir = ensure_fig_dir()
    df = pd.read_csv(CSV)
    plot_decode(df, out_dir)
    plot_score(df, out_dir)
    plot_score_cumulative(df, out_dir)


if __name__ == "__main__":
    main()
