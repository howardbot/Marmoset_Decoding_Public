"""R2-session summaries for nested and transductive CCA validation results."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from session_clustered_asymmetry import hierarchical_bootstrap

REPO = _THIS.parents[2]
IN_CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "nested_cca_validation.csv"
OUT_DAY = REPO / "Results" / "workflows" / "manifold_geometry" / "nested_cca_by_r2_session.csv"
OUT_SUMMARY = REPO / "Results" / "workflows" / "manifold_geometry" / "nested_cca_session_summary.csv"
N_BOOT = 20_000
SEED = 20260713


def pair_directions(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    forward = frame[frame.pair_category == "R1->R2"][
        ["train_session", "test_session", value]
    ].rename(columns={
        "train_session": "r1_session",
        "test_session": "r2_session",
        value: "forward_corr",
    })
    reverse = frame[frame.pair_category == "R2->R1"][
        ["train_session", "test_session", value]
    ].rename(columns={
        "train_session": "r2_session",
        "test_session": "r1_session",
        value: "reverse_corr",
    })
    paired = forward.merge(reverse, on=["r1_session", "r2_session"], validate="one_to_one")
    paired["asymmetry"] = paired.reverse_corr - paired.forward_corr
    return paired


def main():
    raw = pd.read_csv(IN_CSV)
    required = {"alignment_mode", "fold"}
    if not required.issubset(raw.columns):
        raise RuntimeError("nested_cca_validation.csv predates alignment-mode validation; rerun it")
    pair_means = raw.groupby([
        "target", "alignment_mode", "dims", "pair_category",
        "train_session", "test_session",
    ], as_index=False)[["nested_corr", "transductive_corr"]].mean()

    rng = np.random.default_rng(SEED)
    day_rows = []
    summary_rows = []
    grouping = ["target", "alignment_mode", "dims"]
    for keys, frame in pair_means.groupby(grouping):
        target, alignment_mode, dims = keys
        for method in ("nested_corr", "transductive_corr"):
            paired = pair_directions(frame, method)
            by_day = paired.groupby("r2_session").agg(
                n_r1_pairs=("asymmetry", "size"),
                forward_corr=("forward_corr", "mean"),
                reverse_corr=("reverse_corr", "mean"),
                asymmetry=("asymmetry", "mean"),
            ).reset_index()
            by_day.insert(0, "method", method.replace("_corr", ""))
            by_day.insert(0, "dims", dims)
            by_day.insert(0, "alignment_mode", alignment_mode)
            by_day.insert(0, "target", target)
            day_rows.extend(by_day.to_dict("records"))

            boot = hierarchical_bootstrap(paired, rng, n_boot=N_BOOT)
            leave_one_out = []
            for day in sorted(paired.r2_session.unique()):
                kept = paired[paired.r2_session != day]
                leave_one_out.append(kept.groupby("r2_session").asymmetry.mean().mean())
            summary_rows.append({
                "target": target,
                "alignment_mode": alignment_mode,
                "dims": dims,
                "method": method.replace("_corr", ""),
                "n_r2_sessions": paired.r2_session.nunique(),
                "cluster_mean_asymmetry": by_day.asymmetry.mean(),
                "hier_boot_lo": np.percentile(boot, 2.5),
                "hier_boot_hi": np.percentile(boot, 97.5),
                "leave_one_r2_min": np.min(leave_one_out),
                "leave_one_r2_max": np.max(leave_one_out),
                "interpretation": "descriptive one-animal sensitivity interval",
            })

    days = pd.DataFrame(day_rows)
    summary = pd.DataFrame(summary_rows)
    days.to_csv(OUT_DAY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.round(3).to_string(index=False))
    print(f"\nsaved {OUT_DAY}\nsaved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
