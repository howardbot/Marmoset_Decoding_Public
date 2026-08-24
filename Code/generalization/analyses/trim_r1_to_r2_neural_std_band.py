"""Trim only R1 trials until neural trajectory STD enters the R2-day band."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from big_sweep_phase2_crossday import ANIMAL_SESSIONS
from match_trial_pair_variability import pair_matrix
from trim_r2_to_r1_std_band import (
    date_label,
    r1_band,
    trajectory_std,
    trim_to_band,
)

REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "manifold_geometry"
PAIR_COLUMN = "neural_pair_msd"


def make_figure(
    reference: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    animal: str,
) -> None:
    lower = float(summary["r2_band_lower"].iloc[0])
    upper = float(summary["r2_band_upper"].iloc[0])
    mean = float(summary["r2_mean_std"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.axhspan(lower, upper, color="#f58518", alpha=0.15, label="R2 mean ± 1 across-day SD")
    ax.axhline(mean, color="#f58518", linestyle="--", linewidth=1.4)
    r1_x = np.arange(len(summary))
    ax.scatter(r1_x, summary["start_std"], color="#4c78a8", marker="x", s=65, label="R1 full")
    ax.scatter(r1_x, summary["selected_std"], color="#54a24b", marker="D", s=44, label="R1 selected")
    for x, row in zip(r1_x, summary.itertuples(index=False)):
        ax.plot([x, x], [row.start_std, row.selected_std], color="#777777", linewidth=1)
        ax.annotate(
            f"{row.n_remaining}/{row.n_start}",
            (x, row.selected_std),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    r2_x = len(summary) + 1 + np.arange(len(reference))
    ax.scatter(r2_x, reference["trajectory_std"], color="#f58518", s=48, label="R2 full reference")
    labels = [d[4:] for d in summary["r1_date"]] + [""] + [d[4:] for d in reference["date"]]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_ylabel("neural trajectory STD")
    ax.set_title("R1-only neural trimming into the fixed R2 day-level STD band")
    ax.legend(fontsize=9, loc="best")

    ax = axes[1]
    x = np.arange(len(summary))
    ax.bar(x, summary["n_start"], color="#d9d9d9", label="original R1 trials")
    ax.bar(x, summary["n_remaining"], color="#54a24b", label="retained R1 trials")
    for xi, row in zip(x, summary.itertuples(index=False)):
        ax.text(xi, row.n_remaining + 1, str(row.n_remaining), ha="center", fontsize=8)
    ax.set_xticks(x, [d[4:] for d in summary["r1_date"]], rotation=45, ha="right")
    ax.set_ylabel("trial count")
    ax.set_title("Trials remaining after reverse neural-STD matching")
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"{animal}: reverse one-sided neural matching (R2 fixed, delete only R1)\n"
        "above band: remove highest contributor; below band: remove lowest contributor",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal", choices=sorted(ANIMAL_SESSIONS), default="TS")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--band-sd", type=float, default=1.0)
    parser.add_argument("--min-trials", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.band_sd < 0:
        raise ValueError("--band-sd must be non-negative")
    r1_sessions, r2_sessions = ANIMAL_SESSIONS[args.animal]
    pair_path = args.pairs or OUT_DIR / f"trial_pair_variability_{args.animal}_pairs.csv"
    pairs = pd.read_csv(pair_path)

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for session in list(r1_sessions) + list(r2_sessions):
        day = pairs.loc[pairs["session"] == session]
        cache[session] = pair_matrix(day, PAIR_COLUMN)

    reference_rows = []
    reference_values = []
    for session in r2_sessions:
        trial_ids, matrix = cache[session]
        value = trajectory_std(matrix, np.arange(len(trial_ids)))
        reference_values.append(value)
        reference_rows.append(
            {
                "animal": args.animal,
                "r2_session": session,
                "date": date_label(session),
                "n_trials": len(trial_ids),
                "trajectory_std": value,
            }
        )
    band = r1_band(np.asarray(reference_values), args.band_sd)

    summary_rows = []
    path_rows = []
    trial_rows = []
    for session in r1_sessions:
        trial_ids, matrix = cache[session]
        result = trim_to_band(matrix, band["lower"], band["upper"], args.min_trials)
        selected_set = set(result["keep"].tolist())
        removal_map = {
            int(row["removed_index"]): row
            for row in result["removals"]
            if row["removal_step"] <= result["selected_step"]
        }
        summary_rows.append(
            {
                "animal": args.animal,
                "metric": "neural",
                "r1_session": session,
                "r1_date": date_label(session),
                "r2_n_days": len(r2_sessions),
                "r2_mean_std": band["mean"],
                "r2_across_day_sd": band["across_day_sd"],
                "r2_band_sd_multiplier": args.band_sd,
                "r2_band_lower": band["lower"],
                "r2_band_upper": band["upper"],
                "start_std": result["start_std"],
                "selected_std": result["selected_std"],
                "direction": result["direction"],
                "status": result["status"],
                "within_band": result["within_band"],
                "selected_step": result["selected_step"],
                "n_start": len(trial_ids),
                "n_remaining": len(result["keep"]),
                "n_removed": len(trial_ids) - len(result["keep"]),
            }
        )
        for row in result["path"]:
            path_rows.append(
                {
                    "animal": args.animal,
                    "metric": "neural",
                    "r1_session": session,
                    "r1_date": date_label(session),
                    **row,
                }
            )
        for index, trial_id in enumerate(trial_ids):
            removed = removal_map.get(index, {})
            trial_rows.append(
                {
                    "animal": args.animal,
                    "metric": "neural",
                    "r1_session": session,
                    "r1_date": date_label(session),
                    "trial": trial_id,
                    "selected": index in selected_set,
                    "removal_step": removed.get("removal_step", np.nan),
                    "contribution_at_removal": removed.get("contribution_at_removal", np.nan),
                }
            )

    reference_df = pd.DataFrame(reference_rows)
    summary_df = pd.DataFrame(summary_rows)
    path_df = pd.DataFrame(path_rows)
    trials_df = pd.DataFrame(trial_rows)
    stem = f"r1_to_r2_neural_std_band_{args.animal}"
    outputs = {
        "reference": OUT_DIR / f"{stem}_r2_daily.csv",
        "summary": OUT_DIR / f"{stem}_summary.csv",
        "path": OUT_DIR / f"{stem}_path.csv",
        "trials": OUT_DIR / f"{stem}_trials.csv",
        "figure": OUT_DIR / "figures" / f"fig_{stem}.png",
    }
    reference_df.to_csv(outputs["reference"], index=False)
    summary_df.to_csv(outputs["summary"], index=False)
    path_df.to_csv(outputs["path"], index=False)
    trials_df.to_csv(outputs["trials"], index=False)
    make_figure(reference_df, summary_df, outputs["figure"], args.animal)

    print(summary_df.to_string(index=False))
    for output in outputs.values():
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
