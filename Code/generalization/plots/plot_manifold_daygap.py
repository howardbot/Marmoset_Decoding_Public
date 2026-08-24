"""Day-gap control for manifold geometry asymmetry.

Asks: is the R1<->R2 alignment difference (R2-in-R1 0.835 vs R1-in-R2 0.762)
real, or is it explained by 'R1 spans 14 days, R2 only 2' sample-size confound?

For every ordered (train, test) session pair we have:
  - alignment(test in train basis)
  - outside variance fraction
  - calendar day gap

Plot alignment / outside vs day_gap, with the four directional pair categories
in 4 colors. Fit a regression to R1->R1 to get the natural within-epoch drift.
The headline question: do R1->R2 / R2->R1 points lie ON or OFF that drift line?

If R1->R2 / R2->R1 lie on the R1->R1 drift extrapolation -> no real asymmetry.
If they deviate (especially asymmetrically) -> real interference signature.

Reads Results/workflows/manifold_geometry/pairwise_metrics_long.csv.
Writes Results/workflows/manifold_geometry/figures/fig_manifold_daygap.png.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parents[1]
REPO_ROOT = _THIS.parents[1]
CSV = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "pairwise_metrics_long.csv"
FIG_DIR = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Directional pair colors matching plotting_common style.
PAIR_COLORS = {
    "r1->r1": "#7f8c8d",
    "r2->r2": "#34495e",
    "r1->r2": "#e74c3c",  # interference forward
    "r2->r1": "#3498db",  # interference reverse
}


def parse_date(d):
    return datetime.strptime(str(int(d)), "%Y%m%d")


def add_daygap(df):
    df = df.copy()
    df["train_dt"] = df["train_date"].apply(parse_date)
    df["test_dt"] = df["test_date"].apply(parse_date)
    df["day_gap"] = (df["test_dt"] - df["train_dt"]).abs().dt.days.astype(int)
    df["pair_cat"] = df["train_epoch"] + "->" + df["test_epoch"]
    return df


def panel(ax, df, ycol, title):
    # Drop self pairs (day_gap=0 from same session).
    sub = df[df["train_session"] != df["test_session"]].copy()

    # R1->R1 drift line (linear regression).
    r1 = sub[sub["pair_cat"] == "r1->r1"]
    x_r1 = r1["day_gap"].values.astype(float)
    y_r1 = r1[ycol].values.astype(float)
    good = np.isfinite(x_r1) & np.isfinite(y_r1)
    slope, intercept = np.polyfit(x_r1[good], y_r1[good], 1)
    resid = y_r1[good] - (slope * x_r1[good] + intercept)
    r1_sd = float(np.std(resid))
    xs = np.linspace(0, sub["day_gap"].max() + 1, 50)
    ax.plot(xs, slope * xs + intercept, color="#7f8c8d", linewidth=1.2, alpha=0.7,
            label=f"R1->R1 drift fit ({slope:+.4f}/day)", zorder=2)
    ax.fill_between(xs, slope * xs + intercept - r1_sd, slope * xs + intercept + r1_sd,
                    color="#7f8c8d", alpha=0.10, zorder=1, label="R1->R1 ±1 SD")

    # Scatter each category.
    for cat, color in PAIR_COLORS.items():
        g = sub[sub["pair_cat"] == cat]
        if g.empty:
            continue
        ax.scatter(g["day_gap"], g[ycol], color=color, s=30, alpha=0.7,
                   edgecolors="white", linewidth=0.5,
                   label=f"{cat.upper()} (n={len(g)})", zorder=3)

    ax.set_xlabel("day gap (calendar days between train & test)")
    ax.set_ylabel(ycol)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.95, loc="best")
    return slope, intercept, r1_sd


def deviation_table(df, ycol, slope, intercept, r1_sd):
    """How far do R1->R2 / R2->R1 sit from the R1->R1 drift line at their day gap?"""
    sub = df[df["train_session"] != df["test_session"]].copy()
    out = []
    for cat in ["r1->r2", "r2->r1"]:
        g = sub[sub["pair_cat"] == cat]
        if g.empty:
            continue
        predicted = slope * g["day_gap"] + intercept
        observed = g[ycol]
        residual = observed - predicted
        out.append({
            "pair_cat": cat.upper(),
            "n": len(g),
            "observed_mean": float(observed.mean()),
            "predicted_mean": float(predicted.mean()),
            "residual_mean": float(residual.mean()),
            "residual_in_sd": float(residual.mean() / r1_sd) if r1_sd > 0 else np.nan,
        })
    return pd.DataFrame(out)


def main():
    df = pd.read_csv(CSV)
    df = add_daygap(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    s_a, i_a, sd_a = panel(axes[0, 0], df, "align_test_in_train",
                            "Alignment (single-trial cov) vs day gap")
    s_o, i_o, sd_o = panel(axes[0, 1], df, "outside_test_vs_train",
                            "Outside-manifold variance vs day gap")
    s_at, i_at, sd_at = panel(axes[1, 0], df, "align_traj_test_in_train",
                               "Alignment (trial-averaged trajectory) vs day gap")
    s_p, i_p, sd_p = panel(axes[1, 1], df, "procrustes_disparity",
                            "Procrustes disparity vs day gap")

    fig.suptitle("Day-gap control: do R1<->R2 metrics deviate from within-R1 drift?",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_manifold_daygap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}\n")

    # Console summary: how many SDs do the R1<->R2 averages sit off the R1->R1 line?
    print("=== alignment(test in train basis) ===")
    print(deviation_table(df, "align_test_in_train", s_a, i_a, sd_a).round(4).to_string(index=False))
    print(f"  R1->R1 drift slope = {s_a:+.5f}/day, intercept = {i_a:.4f}, SD of residuals = {sd_a:.4f}\n")
    print("=== outside_test_vs_train ===")
    print(deviation_table(df, "outside_test_vs_train", s_o, i_o, sd_o).round(4).to_string(index=False))
    print(f"  R1->R1 drift slope = {s_o:+.5f}/day, intercept = {i_o:.4f}, SD of residuals = {sd_o:.4f}\n")
    print("=== alignment_traj ===")
    print(deviation_table(df, "align_traj_test_in_train", s_at, i_at, sd_at).round(4).to_string(index=False))
    print(f"  R1->R1 drift slope = {s_at:+.5f}/day, intercept = {i_at:.4f}, SD of residuals = {sd_at:.4f}\n")
    print("=== procrustes_disparity ===")
    print(deviation_table(df, "procrustes_disparity", s_p, i_p, sd_p).round(4).to_string(index=False))
    print(f"  R1->R1 drift slope = {s_p:+.5f}/day, intercept = {i_p:.4f}, SD of residuals = {sd_p:.4f}")


if __name__ == "__main__":
    main()
