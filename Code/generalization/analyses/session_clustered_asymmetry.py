"""Session-level summaries for the R1/R2 directional decode contrast.

The 42 ordered pairs reuse 14 R1 and 3 R2 sessions. This script pairs forward
and reverse values for the same R1/R2 session pair, reports one mean contrast
per R2 day, leave-one-R2-day-out sensitivity, and a hierarchical resampling
interval. The interval is descriptive for this one animal, not population-level
inference.

Output:
  Results/workflows/manifold_geometry/session_clustered_asymmetry_by_r2.csv
  Results/workflows/manifold_geometry/session_clustered_asymmetry_summary.csv
  Results/workflows/manifold_geometry/figures/fig_session_clustered_asymmetry.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
REPO = _THIS.parents[2]
IN_CSV = REPO / "Results" / "workflows" / "generalization" / "cca_sweep_long.csv"
OUT_DAY = REPO / "Results" / "workflows" / "manifold_geometry" / "session_clustered_asymmetry_by_r2.csv"
OUT_SUMMARY = REPO / "Results" / "workflows" / "manifold_geometry" / "session_clustered_asymmetry_summary.csv"
FIG = REPO / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_session_clustered_asymmetry.png"

N_CCA = 12
N_BOOT = 20_000
SEED = 20260713
PAIR_KEYS = ["train_session", "test_session"]


def canonical_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer trial-41-excluded rows, with include rows as pair fallback."""
    excluded = frame[frame.outlier_mode == "exclude"]
    included = frame[frame.outlier_mode == "include"]
    present = set(map(tuple, excluded[PAIR_KEYS].to_numpy()))
    fallback = included[~included[PAIR_KEYS].apply(tuple, axis=1).isin(present)]
    return pd.concat([excluded, fallback], ignore_index=True)


def pair_directions(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per R1/R2 pair with forward, reverse and paired contrast."""
    forward = frame[frame.pair_category == "R1->R2"][
        ["train_session", "test_session", "corr"]
    ].rename(columns={
        "train_session": "r1_session",
        "test_session": "r2_session",
        "corr": "forward_corr",
    })
    reverse = frame[frame.pair_category == "R2->R1"][
        ["train_session", "test_session", "corr"]
    ].rename(columns={
        "train_session": "r2_session",
        "test_session": "r1_session",
        "corr": "reverse_corr",
    })
    paired = forward.merge(reverse, on=["r1_session", "r2_session"], validate="one_to_one")
    paired["asymmetry"] = paired.reverse_corr - paired.forward_corr
    return paired


def hierarchical_bootstrap(
    paired: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> np.ndarray:
    """Resample R2 days, then paired R1 rows within each sampled R2 day."""
    groups = {key: group.asymmetry.to_numpy() for key, group in paired.groupby("r2_session")}
    r2_days = np.array(sorted(groups), dtype=object)
    if not len(r2_days):
        raise ValueError("No paired R2 sessions")
    boot = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        sampled_days = rng.choice(r2_days, size=len(r2_days), replace=True)
        day_means = []
        for day in sampled_days:
            values = groups[day]
            day_means.append(rng.choice(values, size=len(values), replace=True).mean())
        boot[iteration] = np.mean(day_means)
    return boot


def summarize_target(paired: pd.DataFrame, target: str, rng) -> tuple[pd.DataFrame, dict]:
    by_day = paired.groupby("r2_session").agg(
        n_r1_pairs=("asymmetry", "size"),
        forward_corr=("forward_corr", "mean"),
        reverse_corr=("reverse_corr", "mean"),
        asymmetry=("asymmetry", "mean"),
    ).reset_index()
    by_day.insert(0, "target", target)

    boot = hierarchical_bootstrap(paired, rng)
    leave_one_out = []
    for day in sorted(paired.r2_session.unique()):
        kept = paired[paired.r2_session != day]
        leave_one_out.append(kept.groupby("r2_session").asymmetry.mean().mean())
    summary = {
        "target": target,
        "n_r1_sessions": paired.r1_session.nunique(),
        "n_r2_sessions": paired.r2_session.nunique(),
        "cluster_mean_asymmetry": by_day.asymmetry.mean(),
        "hier_boot_lo": np.percentile(boot, 2.5),
        "hier_boot_hi": np.percentile(boot, 97.5),
        "leave_one_r2_min": np.min(leave_one_out),
        "leave_one_r2_max": np.max(leave_one_out),
        "interpretation": "descriptive one-animal sensitivity interval",
    }
    return by_day, summary


def main():
    FIG.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(IN_CSV)
    rng = np.random.default_rng(SEED)
    day_tables = []
    summaries = []
    paired_by_target = {}

    for target in ("relative_position", "relative_velocity"):
        subset = raw[
            (raw.metric == "decode")
            & (raw.target_mode == target)
            & (raw.n_cca == N_CCA)
        ]
        paired = pair_directions(canonical_pairs(subset))
        by_day, summary = summarize_target(paired, target, rng)
        paired_by_target[target] = paired
        day_tables.append(by_day)
        summaries.append(summary)
        print(f"\n{target}")
        print(by_day.round(3).to_string(index=False))
        print(pd.Series(summary).to_string())

    pd.concat(day_tables, ignore_index=True).to_csv(OUT_DAY, index=False)
    pd.DataFrame(summaries).to_csv(OUT_SUMMARY, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    for ax, target in zip(axes, ("relative_position", "relative_velocity")):
        by_day = day_tables[0 if target == "relative_position" else 1]
        x = np.arange(len(by_day))
        for index, row in by_day.iterrows():
            ax.plot([x[index] - 0.12, x[index] + 0.12],
                    [row.forward_corr, row.reverse_corr], color="#7f8c8d", lw=1.5)
        ax.scatter(x - 0.12, by_day.forward_corr, color="#e74c3c", s=55, label="R1->R2")
        ax.scatter(x + 0.12, by_day.reverse_corr, color="#3498db", s=55, label="R2->R1")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s)[4:12] for s in by_day.r2_session], rotation=25, ha="right")
        ax.set_title(target.replace("relative_", ""))
        ax.set_ylabel("mean decode corr across paired R1 sessions")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Directional contrast by R2 session (three descriptive clusters, one animal)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nsaved {OUT_DAY}\nsaved {OUT_SUMMARY}\nsaved {FIG}")


if __name__ == "__main__":
    main()
