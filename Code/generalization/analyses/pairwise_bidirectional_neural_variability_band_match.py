"""Construct pair-specific variability-matched trial subsets.

The input table contains neural and position mean-squared distances (MSDs) for
every within-session pair of trials.  ``--metric`` selects which one defines
the matching band.  For each R1/R2 session pair, this script runs the matching
in both directions:

1. hold the complete R1 session fixed and trim trials from R2 until its mean
   pair-MSD is inside the R1 mean +/- SD band; and
2. hold the complete R2 session fixed and trim trials from R1 using the
   corresponding R2 band.

The script only selects and records trial subsets.  Cross-day decoding of the
selected subsets is performed by a separate script.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep Matplotlib/font caches in a writable temporary location on compute nodes.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# These analysis scripts are run directly rather than as an installed package;
# expose the shared generalization modules and neighboring helper scripts.
_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from big_sweep_phase2_crossday import ANIMAL_SESSIONS
from match_all_trial_pair_variability import DECODER_COMPAT_EXCLUDE_TRIALS
from match_trial_pair_variability import pair_matrix, subset_mean, subset_values
from trim_r2_to_r1_std_band import date_label, distance_to_band, trial_contributions

REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "manifold_geometry"
# The matching algorithm is identical for the two metrics; only the input
# pair-distance matrix changes.  Neural remains the default so existing
# commands reproduce the original analysis.
PAIR_COLUMNS = {
    "neural": "neural_pair_msd",
    "position": "position_pair_msd",
}
# Run both possible choices of fixed (anchor) day and trimmed day.
DIRECTIONS = ("trim_r2_to_r1", "trim_r1_to_r2")


def pair_distribution_stats(matrix: np.ndarray, keep: np.ndarray) -> dict:
    """Summarize unique trial-pair MSD values for the selected trial indices."""
    values = subset_values(matrix, keep)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "n_pairs": len(values),
    }


def trim_pair_mean_to_band(
    matrix: np.ndarray,
    lower: float,
    upper: float,
    min_trials: int = 3,
) -> dict:
    """Delete highest/lowest contributors until pair-MSD mean enters a band."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not (0.0 <= lower <= upper):
        raise ValueError("band must satisfy 0 <= lower <= upper")
    if not (3 <= min_trials <= len(matrix)):
        raise ValueError("min_trials must be between 3 and the starting count")

    keep = np.arange(len(matrix), dtype=int)
    start_mean = subset_mean(matrix, keep)
    if start_mean > upper:
        direction = "down"
    elif start_mean < lower:
        direction = "up"
    else:
        direction = "inside"
    start_row = {
        "step": 0,
        "n_remaining": len(keep),
        "pair_msd_mean": start_mean,
        "distance_to_band": distance_to_band(start_mean, lower, upper),
        "removed_index": np.nan,
        "removed_contribution": np.nan,
    }
    path = [start_row]
    states = [(keep.copy(), start_row)]
    removals: list[dict] = []
    status = "inside_initial" if direction == "inside" else "searching"

    while direction != "inside" and len(keep) > min_trials:
        contributions = trial_contributions(matrix, keep)
        remove_local = (
            int(np.argmax(contributions))
            if direction == "down"
            else int(np.argmin(contributions))
        )
        removed_index = int(keep[remove_local])
        removed_contribution = float(contributions[remove_local])
        keep = np.delete(keep, remove_local)
        value = subset_mean(matrix, keep)
        row = {
            "step": len(path),
            "n_remaining": len(keep),
            "pair_msd_mean": value,
            "distance_to_band": distance_to_band(value, lower, upper),
            "removed_index": removed_index,
            "removed_contribution": removed_contribution,
        }
        path.append(row)
        states.append((keep.copy(), row))
        removals.append(
            {
                "removed_index": removed_index,
                "removal_step": row["step"],
                "contribution_at_removal": removed_contribution,
            }
        )
        if row["distance_to_band"] == 0.0:
            status = "entered_band"
            break
        crossed = (direction == "down" and value < lower) or (
            direction == "up" and value > upper
        )
        if crossed:
            status = "crossed_without_entry"
            break

    if status == "searching":
        status = "no_entry_before_minimum"
    best_keep, best_row = min(
        states,
        key=lambda item: (item[1]["distance_to_band"], -item[1]["n_remaining"]),
    )
    return {
        "keep": best_keep,
        "start_mean": start_mean,
        "selected_mean": float(best_row["pair_msd_mean"]),
        "selected_step": int(best_row["step"]),
        "distance_to_band": float(best_row["distance_to_band"]),
        "direction": direction,
        "status": status,
        "within_band": best_row["distance_to_band"] == 0.0,
        "path": path,
        "removals": removals,
    }


def _heatmap(
    ax,
    table: pd.DataFrame,
    r1_dates: list[str],
    r2_dates: list[str],
    value: str,
    title: str,
    cmap: str,
    fmt: str,
    vmin=None,
    vmax=None,
):
    """Draw one annotated date-pair heatmap for the matching diagnostics."""
    pivot = table.pivot(index="r1_date", columns="r2_date", values=value)
    matrix = pivot.reindex(index=r1_dates, columns=r2_dates).to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(r2_dates)), [d[4:] for d in r2_dates])
    ax.set_yticks(range(len(r1_dates)), [d[4:] for d in r1_dates])
    ax.set_xlabel("R2 date")
    ax.set_ylabel("R1 date")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=7)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def make_figure(
    summary: pd.DataFrame,
    r1_sessions: tuple[str, ...],
    r2_sessions: tuple[str, ...],
    output: Path,
    animal: str,
    metric: str,
    band_sd: float,
) -> None:
    """Plot retained trial counts and anchor-relative matched means."""
    r1_dates = [date_label(session) for session in r1_sessions]
    r2_dates = [date_label(session) for session in r2_sessions]
    fig, axes = plt.subplots(2, 2, figsize=(13, max(10, 5 + 0.75 * len(r1_dates))))
    for row, direction in enumerate(DIRECTIONS):
        table = summary.loc[summary["match_direction"] == direction]
        trimmed = "R2" if direction == "trim_r2_to_r1" else "R1"
        anchor = "R1" if direction == "trim_r2_to_r1" else "R2"
        _heatmap(
            axes[row, 0], table, r1_dates, r2_dates,
            "n_trimmed_remaining", f"{trimmed} trials retained; {anchor} day fixed",
            "viridis", ".0f", vmin=0,
        )
        _heatmap(
            axes[row, 1], table, r1_dates, r2_dates,
            "selected_anchor_z", f"selected {trimmed} mean relative to {anchor}-day mean (SD units)",
            "coolwarm", ".2f", vmin=-1, vmax=1,
        )
    fig.suptitle(
        f"{animal}: pair-specific bidirectional {metric} variability-band matching\n"
        f"fixed day target = its raw trial-pair MSD mean ± {band_sd:g} SD",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    """Parse the animal, pairwise metric, band width, and minimum trial count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal", choices=sorted(ANIMAL_SESSIONS), default="TS")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--metric", choices=sorted(PAIR_COLUMNS), default="neural")
    parser.add_argument("--band-sd", type=float, default=1.0)
    parser.add_argument("--min-trials", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    """Run bidirectional band matching and save paths, trials, and summaries."""
    args = parse_args(argv)
    if args.band_sd < 0:
        raise ValueError("--band-sd must be non-negative")
    r1_sessions, r2_sessions = ANIMAL_SESSIONS[args.animal]
    pair_path = args.pairs or OUT_DIR / f"trial_pair_variability_{args.animal}_pairs.csv"
    pairs = pd.read_csv(pair_path)
    cache: dict[str, dict] = {}
    for session in list(r1_sessions) + list(r2_sessions):
        day = pairs.loc[pairs["session"] == session]
        incompatible = DECODER_COMPAT_EXCLUDE_TRIALS.get(session, ())
        if incompatible:
            day = day.loc[
                ~day["trial_i"].isin(incompatible)
                & ~day["trial_j"].isin(incompatible)
            ]
        trial_ids, matrix = pair_matrix(day, PAIR_COLUMNS[args.metric])
        stats = pair_distribution_stats(matrix, np.arange(len(trial_ids)))
        cache[session] = {"trial_ids": trial_ids, "matrix": matrix, **stats}

    summary_rows = []
    path_rows = []
    trial_rows = []
    pair_id = 0
    for r1_session in r1_sessions:
        for r2_session in r2_sessions:
            pair_id += 1
            r1, r2 = cache[r1_session], cache[r2_session]
            specifications = (
                ("trim_r2_to_r1", r1_session, r1, r2_session, r2),
                ("trim_r1_to_r2", r2_session, r2, r1_session, r1),
            )
            for direction, anchor_session, anchor, trimmed_session, trimmed in specifications:
                lower = max(0.0, anchor["mean"] - args.band_sd * anchor["sd"])
                upper = anchor["mean"] + args.band_sd * anchor["sd"]
                result = trim_pair_mean_to_band(
                    trimmed["matrix"], lower, upper, args.min_trials
                )
                selected_set = set(result["keep"].tolist())
                removal_map = {
                    int(row["removed_index"]): row
                    for row in result["removals"]
                    if row["removal_step"] <= result["selected_step"]
                }
                summary_rows.append(
                    {
                        "animal": args.animal,
                        "match_metric": args.metric,
                        "pair_id": pair_id,
                        "r1_session": r1_session,
                        "r1_date": date_label(r1_session),
                        "r2_session": r2_session,
                        "r2_date": date_label(r2_session),
                        "match_direction": direction,
                        "anchor_session": anchor_session,
                        "trimmed_session": trimmed_session,
                        "anchor_pair_mean": anchor["mean"],
                        "anchor_pair_sd": anchor["sd"],
                        "anchor_band_sd_multiplier": args.band_sd,
                        "anchor_band_lower": lower,
                        "anchor_band_upper": upper,
                        "trimmed_start_pair_mean": result["start_mean"],
                        "selected_pair_mean": result["selected_mean"],
                        "selected_anchor_z": (
                            (result["selected_mean"] - anchor["mean"]) / anchor["sd"]
                            if anchor["sd"] > 0
                            else 0.0
                        ),
                        "distance_to_anchor_band": result["distance_to_band"],
                        "deletion_direction": result["direction"],
                        "status": result["status"],
                        "within_anchor_band": result["within_band"],
                        "selected_step": result["selected_step"],
                        "n_anchor_full": len(anchor["trial_ids"]),
                        "n_trimmed_start": len(trimmed["trial_ids"]),
                        "n_trimmed_remaining": len(result["keep"]),
                        "n_trimmed_removed": len(trimmed["trial_ids"]) - len(result["keep"]),
                    }
                )
                for row in result["path"]:
                    path_rows.append(
                        {
                            "animal": args.animal,
                            "match_metric": args.metric,
                            "pair_id": pair_id,
                            "r1_date": date_label(r1_session),
                            "r2_date": date_label(r2_session),
                            "match_direction": direction,
                            "anchor_session": anchor_session,
                            "trimmed_session": trimmed_session,
                            "anchor_band_lower": lower,
                            "anchor_band_upper": upper,
                            **row,
                        }
                    )
                for index, trial_id in enumerate(trimmed["trial_ids"]):
                    removed = removal_map.get(index, {})
                    trial_rows.append(
                        {
                            "animal": args.animal,
                            "match_metric": args.metric,
                            "pair_id": pair_id,
                            "r1_date": date_label(r1_session),
                            "r2_date": date_label(r2_session),
                            "match_direction": direction,
                            "trimmed_session": trimmed_session,
                            "trial": trial_id,
                            "selected": index in selected_set,
                            "removal_step": removed.get("removal_step", np.nan),
                            "contribution_at_removal": removed.get("contribution_at_removal", np.nan),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    path_df = pd.DataFrame(path_rows)
    trials_df = pd.DataFrame(trial_rows)
    band_suffix = "" if np.isclose(args.band_sd, 1.0) else f"_sd{args.band_sd:g}"
    stem = (
        f"pairwise_bidirectional_{args.metric}_variability_band{band_suffix}_"
        f"match_{args.animal}"
    )
    outputs = {
        "summary": OUT_DIR / f"{stem}_summary.csv",
        "path": OUT_DIR / f"{stem}_path.csv",
        "trials": OUT_DIR / f"{stem}_trials.csv",
        "figure": OUT_DIR / "figures" / f"fig_{stem}.png",
    }
    summary_df.to_csv(outputs["summary"], index=False)
    path_df.to_csv(outputs["path"], index=False)
    trials_df.to_csv(outputs["trials"], index=False)
    make_figure(
        summary_df,
        r1_sessions,
        r2_sessions,
        outputs["figure"],
        args.animal,
        args.metric,
        args.band_sd,
    )

    print(f"=== Pair-specific {args.metric} pair-MSD mean +/- SD matching ===")
    for direction in DIRECTIONS:
        table = summary_df.loc[summary_df["match_direction"] == direction]
        print(
            f"{direction}: inside {int(table['within_anchor_band'].sum())}/{len(table)}, "
            f"unchanged {int((table['n_trimmed_removed'] == 0).sum())}/{len(table)}, "
            f"median retained={table['n_trimmed_remaining'].median():.0f}, "
            f"range={table['n_trimmed_remaining'].min()}-{table['n_trimmed_remaining'].max()}"
        )
    for output in outputs.values():
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
