"""Trim only R2 trials until trajectory STD enters the R1 across-day band."""
from __future__ import annotations

import argparse
import os
import re
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
from match_trial_pair_variability import pair_matrix, subset_mean

REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "workflows" / "manifold_geometry"
METRICS = {
    "neural": "neural_pair_msd",
    "position": "position_pair_msd",
}


def date_label(session: str) -> str:
    match = re.search(r"\d{8}", session)
    return match.group(0) if match else session


def trajectory_std(matrix: np.ndarray, keep: np.ndarray) -> float:
    """RMS across-feature trial STD, equivalent to sqrt(pairwise MSD / 2)."""
    return float(np.sqrt(max(subset_mean(matrix, keep), 0.0) / 2.0))


def trial_contributions(matrix: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Mean pairwise MSD from each retained trial to the retained set."""
    if len(keep) < 2:
        raise ValueError("at least two retained trials are required")
    sub = matrix[np.ix_(keep, keep)]
    return sub.sum(axis=1) / (len(keep) - 1)


def distance_to_band(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def trim_to_band(
    matrix: np.ndarray,
    lower: float,
    upper: float,
    min_trials: int = 3,
) -> dict:
    """Delete high- or low-contribution trials until STD enters a fixed band."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not (0.0 <= lower <= upper):
        raise ValueError("STD band must satisfy 0 <= lower <= upper")
    if not (3 <= min_trials <= len(matrix)):
        raise ValueError("min_trials must be between 3 and the starting count")

    keep = np.arange(len(matrix), dtype=int)
    start_std = trajectory_std(matrix, keep)
    if start_std > upper:
        direction = "down"
    elif start_std < lower:
        direction = "up"
    else:
        direction = "inside"

    path = [
        {
            "step": 0,
            "n_remaining": len(keep),
            "trajectory_std": start_std,
            "distance_to_band": distance_to_band(start_std, lower, upper),
            "removed_index": np.nan,
            "removed_contribution": np.nan,
        }
    ]
    removal_rows: list[dict] = []
    states = [(keep.copy(), path[-1])]
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
        value = trajectory_std(matrix, keep)
        row = {
            "step": len(path),
            "n_remaining": len(keep),
            "trajectory_std": value,
            "distance_to_band": distance_to_band(value, lower, upper),
            "removed_index": removed_index,
            "removed_contribution": removed_contribution,
        }
        path.append(row)
        states.append((keep.copy(), row))
        removal_rows.append(
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
        "start_std": start_std,
        "selected_std": float(best_row["trajectory_std"]),
        "selected_step": int(best_row["step"]),
        "direction": direction,
        "status": status,
        "within_band": best_row["distance_to_band"] == 0.0,
        "path": path,
        "removals": removal_rows,
    }


def trim_to_target(
    matrix: np.ndarray,
    target_std: float,
    min_trials: int = 3,
) -> dict:
    """Delete from one session until its STD first crosses an exact-day target."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if target_std < 0:
        raise ValueError("target_std must be non-negative")
    if not (3 <= min_trials <= len(matrix)):
        raise ValueError("min_trials must be between 3 and the starting count")

    keep = np.arange(len(matrix), dtype=int)
    start_std = trajectory_std(matrix, keep)
    if start_std > target_std:
        direction = "down"
    elif start_std < target_std:
        direction = "up"
    else:
        direction = "exact_initial"
    start_row = {
        "step": 0,
        "n_remaining": len(keep),
        "trajectory_std": start_std,
        "signed_target_difference": start_std - target_std,
        "absolute_target_difference": abs(start_std - target_std),
        "removed_index": np.nan,
        "removed_contribution": np.nan,
    }
    path = [start_row]
    states = [(keep.copy(), start_row)]
    removal_rows: list[dict] = []
    status = "exact_initial" if direction == "exact_initial" else "searching"

    while direction != "exact_initial" and len(keep) > min_trials:
        contributions = trial_contributions(matrix, keep)
        remove_local = (
            int(np.argmax(contributions))
            if direction == "down"
            else int(np.argmin(contributions))
        )
        removed_index = int(keep[remove_local])
        removed_contribution = float(contributions[remove_local])
        keep = np.delete(keep, remove_local)
        value = trajectory_std(matrix, keep)
        row = {
            "step": len(path),
            "n_remaining": len(keep),
            "trajectory_std": value,
            "signed_target_difference": value - target_std,
            "absolute_target_difference": abs(value - target_std),
            "removed_index": removed_index,
            "removed_contribution": removed_contribution,
        }
        path.append(row)
        states.append((keep.copy(), row))
        removal_rows.append(
            {
                "removed_index": removed_index,
                "removal_step": row["step"],
                "contribution_at_removal": removed_contribution,
            }
        )
        if value == target_std:
            status = "exact_target"
            break
        crossed = (direction == "down" and value < target_std) or (
            direction == "up" and value > target_std
        )
        if crossed:
            status = "crossed_target"
            break

    if status == "searching":
        status = "no_crossing_before_minimum"
    best_keep, best_row = min(
        states,
        key=lambda item: (
            item[1]["absolute_target_difference"],
            -item[1]["n_remaining"],
        ),
    )
    return {
        "keep": best_keep,
        "start_std": start_std,
        "selected_std": float(best_row["trajectory_std"]),
        "selected_step": int(best_row["step"]),
        "signed_target_difference": float(best_row["signed_target_difference"]),
        "absolute_target_difference": float(best_row["absolute_target_difference"]),
        "direction": direction,
        "status": status,
        "crossed_target": status in {"exact_initial", "exact_target", "crossed_target"},
        "path": path,
        "removals": removal_rows,
    }


def r1_band(values: np.ndarray, band_sd: float = 1.0) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("at least two R1 days are required")
    mean = float(np.mean(values))
    across_day_sd = float(np.std(values, ddof=1))
    return {
        "mean": mean,
        "across_day_sd": across_day_sd,
        "lower": max(0.0, mean - band_sd * across_day_sd),
        "upper": mean + band_sd * across_day_sd,
    }


def make_figure(
    r1_daily: pd.DataFrame,
    summary: pd.DataFrame,
    path: pd.DataFrame,
    output: Path,
    animal: str,
    band_sd: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.cm.tab10.colors
    for col, metric in enumerate(METRICS):
        r1 = r1_daily.loc[r1_daily["metric"] == metric].reset_index(drop=True)
        selected = summary.loc[summary["metric"] == metric].reset_index(drop=True)
        lower = float(selected["r1_band_lower"].iloc[0])
        upper = float(selected["r1_band_upper"].iloc[0])
        mean = float(selected["r1_mean_std"].iloc[0])

        ax = axes[0, col]
        ax.axhspan(lower, upper, color="#4c78a8", alpha=0.15)
        ax.axhline(mean, color="#4c78a8", linestyle="--", linewidth=1.4)
        r1_x = np.arange(len(r1))
        ax.scatter(r1_x, r1["trajectory_std"], color="#4c78a8", s=42, label="R1 full")
        r2_x = len(r1) + 1 + np.arange(len(selected))
        ax.scatter(r2_x, selected["start_std"], color="#f58518", marker="x", s=70, label="R2 full")
        ax.scatter(r2_x, selected["selected_std"], color="#54a24b", marker="D", s=46, label="R2 selected")
        for x, row in zip(r2_x, selected.itertuples(index=False)):
            ax.plot([x, x], [row.start_std, row.selected_std], color="#777777", linewidth=1)
            ax.annotate(
                f"{row.n_remaining}/{row.n_start}",
                (x, row.selected_std),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        labels = [d[4:] for d in r1["date"]] + [""] + [d[4:] for d in selected["r2_date"]]
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_ylabel("trajectory STD")
        ax.set_title(f"{metric}: R1 reference band and R2 trimming")
        ax.legend(fontsize=8, loc="best")

        ax = axes[1, col]
        ax.axhspan(lower, upper, color="#4c78a8", alpha=0.15, label=f"R1 mean ± {band_sd:g} SD")
        ax.axhline(mean, color="#4c78a8", linestyle="--", linewidth=1.2)
        for index, (session, group) in enumerate(path.loc[path["metric"] == metric].groupby("r2_session", sort=False)):
            row = selected.loc[selected["r2_session"] == session].iloc[0]
            color = colors[index % len(colors)]
            ax.plot(group["step"], group["trajectory_std"], marker="o", markersize=3, color=color, label=row["r2_date"][4:])
            chosen = group.loc[group["step"] == row["selected_step"]]
            ax.scatter(chosen["step"], chosen["trajectory_std"], color=color, marker="D", s=55, zorder=3)
            ax.annotate(
                f"N={int(row['n_remaining'])}",
                (float(chosen["step"].iloc[0]), float(chosen["trajectory_std"].iloc[0])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("R2 trials removed")
        ax.set_ylabel("R2 trajectory STD")
        ax.set_title(f"{metric}: one-sided deletion path")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"{animal}: R2-only trimming into the R1 day-level trajectory-STD band\n"
        "above band: remove highest contributor; below band: remove lowest contributor",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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

    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for session in list(r1_sessions) + list(r2_sessions):
        day = pairs.loc[pairs["session"] == session]
        for metric, column in METRICS.items():
            cache[(session, metric)] = pair_matrix(day, column)

    r1_rows = []
    summary_rows = []
    path_rows = []
    trial_rows = []
    for metric in METRICS:
        r1_values = []
        for session in r1_sessions:
            trial_ids, matrix = cache[(session, metric)]
            value = trajectory_std(matrix, np.arange(len(trial_ids)))
            r1_values.append(value)
            r1_rows.append(
                {
                    "animal": args.animal,
                    "metric": metric,
                    "session": session,
                    "date": date_label(session),
                    "n_trials": len(trial_ids),
                    "trajectory_std": value,
                }
            )
        band = r1_band(np.asarray(r1_values), args.band_sd)

        for session in r2_sessions:
            trial_ids, matrix = cache[(session, metric)]
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
                    "metric": metric,
                    "r2_session": session,
                    "r2_date": date_label(session),
                    "r1_n_days": len(r1_sessions),
                    "r1_mean_std": band["mean"],
                    "r1_across_day_sd": band["across_day_sd"],
                    "r1_band_sd_multiplier": args.band_sd,
                    "r1_band_lower": band["lower"],
                    "r1_band_upper": band["upper"],
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
                        "metric": metric,
                        "r2_session": session,
                        "r2_date": date_label(session),
                        **row,
                    }
                )
            for index, trial_id in enumerate(trial_ids):
                removed = removal_map.get(index, {})
                trial_rows.append(
                    {
                        "animal": args.animal,
                        "metric": metric,
                        "r2_session": session,
                        "r2_date": date_label(session),
                        "trial": trial_id,
                        "selected": index in selected_set,
                        "removal_step": removed.get("removal_step", np.nan),
                        "contribution_at_removal": removed.get("contribution_at_removal", np.nan),
                    }
                )

    r1_df = pd.DataFrame(r1_rows)
    summary_df = pd.DataFrame(summary_rows)
    path_df = pd.DataFrame(path_rows)
    trials_df = pd.DataFrame(trial_rows)
    stem = f"r2_to_r1_std_band_{args.animal}"
    outputs = {
        "r1": OUT_DIR / f"{stem}_r1_daily.csv",
        "summary": OUT_DIR / f"{stem}_summary.csv",
        "path": OUT_DIR / f"{stem}_path.csv",
        "trials": OUT_DIR / f"{stem}_trials.csv",
        "figure": OUT_DIR / "figures" / f"fig_{stem}.png",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r1_df.to_csv(outputs["r1"], index=False)
    summary_df.to_csv(outputs["summary"], index=False)
    path_df.to_csv(outputs["path"], index=False)
    trials_df.to_csv(outputs["trials"], index=False)
    make_figure(r1_df, summary_df, path_df, outputs["figure"], args.animal, args.band_sd)

    print(summary_df.to_string(index=False))
    for output in outputs.values():
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
