"""Run equal-N variability matching for every configured R1 x R2 day pair.

Three target-informed sensitivity analyses are run for every day pair:

``neural``
    Match the mean neural trial-pair MSD.  Position is an unselected side-effect
    diagnostic.
``position``
    Match the mean position trial-pair MSD.  Neural variability is an unselected
    side-effect diagnostic.
``joint``
    Greedily minimize the larger of the neural and position relative gaps and
    require both to be within the requested tolerance.

Every selected R1/R2 subset has the same number of trials.  The selected result
is the largest equal trial count on its deterministic greedy path that meets the
tolerance.  Results that cannot meet the constraint before ``min_trials`` are
retained for diagnosis but explicitly marked ``within_tolerance=False``.

This selection and its evaluation use the same data.  It is descriptive and
must not be treated as an independent confirmatory test.

Outputs
-------
Results/manifold_geometry/variability_match_all42_summary.csv
Results/manifold_geometry/variability_match_all42_trials.csv
Results/manifold_geometry/figures/fig_variability_match_all42_heatmaps.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from math import comb
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from big_sweep_phase2_crossday import ANIMAL_SESSIONS
from match_trial_pair_variability import (
    pair_matrix,
    subset_mean,
    subset_values,
    symmetric_relative_gap,
)

REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "manifold_geometry"
OUT_SUMMARY = OUT_DIR / "variability_match_all42_summary.csv"
OUT_TRIALS = OUT_DIR / "variability_match_all42_trials.csv"
OUT_FIGURE = OUT_DIR / "figures" / "fig_variability_match_all42_heatmaps.png"
MIN_TRIALS = 20
RELATIVE_TOLERANCE = 0.05
MATCH_MODES = ("neural", "position", "joint")
# relative_velocity drops one additional too-short trial after differencing.
# Excluding it here keeps the selected trial IDs valid for both locked decoder
# targets (position and velocity).
DECODER_COMPAT_EXCLUDE_TRIALS = {
    "TSAL20250810_0830_staticAndStaticFree001": (40,),
}


def date_label(session: str) -> str:
    """Extract the first eight-digit date token from a session name."""
    for token in session.split("_"):
        digits = "".join(character for character in token if character.isdigit())
        if len(digits) >= 8:
            return digits[:8]
    return session


def build_session_matrices(pairs: pd.DataFrame, session: str) -> dict:
    """Build aligned neural and position pair matrices for one session."""
    day = pairs.loc[pairs["session"] == session]
    incompatible = DECODER_COMPAT_EXCLUDE_TRIALS.get(session, ())
    if incompatible:
        day = day.loc[
            ~day["trial_i"].isin(incompatible)
            & ~day["trial_j"].isin(incompatible)
        ]
    if day.empty:
        raise ValueError(f"no pairwise data for session {session}")
    trial_ids, neural = pair_matrix(day, "neural_pair_msd")
    position_ids, position = pair_matrix(day, "position_pair_msd")
    if not np.array_equal(trial_ids, position_ids):
        raise RuntimeError(f"neural/position trial IDs differ for {session}")
    return {"trial_ids": trial_ids, "neural": neural, "position": position}


def choose_single_metric_match(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    min_trials: int,
    tolerance: float,
    fixed_trials: int | None = None,
) -> dict:
    """Adaptively match one metric using the same search freedom as joint mode."""
    keep1 = np.arange(len(matrix1), dtype=int)
    keep2 = np.arange(len(matrix2), dtype=int)
    minimum = min(min_trials, len(keep1), len(keep2))

    # Equalize counts by testing every possible deletion from the larger side.
    while len(keep1) > len(keep2):
        after = _means_after_each_removal(matrix1, keep1)
        other = subset_mean(matrix2, keep2)
        gaps = np.array([symmetric_relative_gap(value, other) for value in after])
        keep1 = np.delete(keep1, int(np.argmin(gaps)))
    while len(keep2) > len(keep1):
        after = _means_after_each_removal(matrix2, keep2)
        other = subset_mean(matrix1, keep1)
        gaps = np.array([symmetric_relative_gap(other, value) for value in after])
        keep2 = np.delete(keep2, int(np.argmin(gaps)))

    def record():
        """Snapshot the current equal-N subset and its primary-metric gap."""
        value1 = subset_mean(matrix1, keep1)
        value2 = subset_mean(matrix2, keep2)
        return {
            "n_trials": len(keep1),
            "keep_r1": keep1.copy(),
            "keep_r2": keep2.copy(),
            "primary_r1_mean": value1,
            "primary_r2_mean": value2,
            "primary_gap": symmetric_relative_gap(value1, value2),
        }

    candidates = [record()]
    while len(keep1) > minimum:
        values1 = _means_after_each_removal(matrix1, keep1)
        values2 = _means_after_each_removal(matrix2, keep2)
        gap_grid = _relative_gap_grid(values1, values2)
        remove1, remove2 = np.unravel_index(
            int(np.argmin(gap_grid)), gap_grid.shape
        )
        keep1 = np.delete(keep1, remove1)
        keep2 = np.delete(keep2, remove2)
        candidates.append(record())
    if fixed_trials is not None:
        selected = next(row for row in candidates if row["n_trials"] == fixed_trials)
        selected["within_tolerance"] = selected["primary_gap"] <= tolerance
    else:
        eligible = [row for row in candidates if row["primary_gap"] <= tolerance]
        if eligible:
            selected = eligible[0]
            selected["within_tolerance"] = True
        else:
            selected = min(
                candidates,
                key=lambda row: (row["primary_gap"], -row["n_trials"]),
            )
            selected["within_tolerance"] = False
    selected["direction_r1"] = "adaptive"
    selected["direction_r2"] = "adaptive"
    return selected


def _means_after_each_removal(matrix: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Vector of subset means obtained by removing each retained index once."""
    sub = matrix[np.ix_(keep, keep)]
    n_trials = len(keep)
    if n_trials <= 2:
        raise ValueError("at least three retained trials are required")
    pair_sum = float(np.sum(sub) / 2.0)
    row_sum = np.sum(sub, axis=1)
    return (pair_sum - row_sum) / comb(n_trials - 1, 2)


def _current_joint_values(
    neural1: np.ndarray,
    position1: np.ndarray,
    keep1: np.ndarray,
    neural2: np.ndarray,
    position2: np.ndarray,
    keep2: np.ndarray,
) -> dict:
    """Measure neural, position, and worst-case gaps for current subsets."""
    n1, n2 = subset_mean(neural1, keep1), subset_mean(neural2, keep2)
    p1, p2 = subset_mean(position1, keep1), subset_mean(position2, keep2)
    neural_gap = symmetric_relative_gap(n1, n2)
    position_gap = symmetric_relative_gap(p1, p2)
    return {
        "n_trials": min(len(keep1), len(keep2)),
        "keep_r1": keep1.copy(),
        "keep_r2": keep2.copy(),
        "neural_r1_mean": n1,
        "neural_r2_mean": n2,
        "position_r1_mean": p1,
        "position_r2_mean": p2,
        "neural_gap": neural_gap,
        "position_gap": position_gap,
        "joint_gap": max(neural_gap, position_gap),
    }


def _relative_gap_grid(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
    """Evaluate the symmetric relative gap for every removal-pair candidate."""
    a = np.abs(values1)[:, None]
    b = np.abs(values2)[None, :]
    denom = (a + b) / 2.0
    diff = np.abs(values1[:, None] - values2[None, :])
    return np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)


def choose_joint_match(
    neural1: np.ndarray,
    position1: np.ndarray,
    neural2: np.ndarray,
    position2: np.ndarray,
    min_trials: int,
    tolerance: float,
    fixed_trials: int | None = None,
) -> dict:
    """Greedily match neural and position means simultaneously at equal N."""
    keep1 = np.arange(len(neural1), dtype=int)
    keep2 = np.arange(len(neural2), dtype=int)
    minimum = min(min_trials, len(keep1), len(keep2))

    # Equalize the starting counts.  Remove from the larger set whichever trial
    # gives the smallest maximum neural/position gap to the unchanged side.
    while len(keep1) > len(keep2):
        neural_after = _means_after_each_removal(neural1, keep1)
        position_after = _means_after_each_removal(position1, keep1)
        neural_other = subset_mean(neural2, keep2)
        position_other = subset_mean(position2, keep2)
        loss = np.maximum(
            np.array([symmetric_relative_gap(x, neural_other) for x in neural_after]),
            np.array([symmetric_relative_gap(x, position_other) for x in position_after]),
        )
        keep1 = np.delete(keep1, int(np.argmin(loss)))
    while len(keep2) > len(keep1):
        neural_after = _means_after_each_removal(neural2, keep2)
        position_after = _means_after_each_removal(position2, keep2)
        neural_other = subset_mean(neural1, keep1)
        position_other = subset_mean(position1, keep1)
        loss = np.maximum(
            np.array([symmetric_relative_gap(neural_other, x) for x in neural_after]),
            np.array([symmetric_relative_gap(position_other, x) for x in position_after]),
        )
        keep2 = np.delete(keep2, int(np.argmin(loss)))

    candidates = [
        _current_joint_values(
            neural1, position1, keep1, neural2, position2, keep2
        )
    ]
    while len(keep1) > minimum:
        neural1_after = _means_after_each_removal(neural1, keep1)
        position1_after = _means_after_each_removal(position1, keep1)
        neural2_after = _means_after_each_removal(neural2, keep2)
        position2_after = _means_after_each_removal(position2, keep2)
        neural_gaps = _relative_gap_grid(neural1_after, neural2_after)
        position_gaps = _relative_gap_grid(position1_after, position2_after)
        joint_loss = np.maximum(neural_gaps, position_gaps)
        remove1, remove2 = np.unravel_index(
            int(np.argmin(joint_loss)), joint_loss.shape
        )
        keep1 = np.delete(keep1, remove1)
        keep2 = np.delete(keep2, remove2)
        candidates.append(
            _current_joint_values(
                neural1, position1, keep1, neural2, position2, keep2
            )
        )

    if fixed_trials is not None:
        selected = next(row for row in candidates if row["n_trials"] == fixed_trials)
        selected["within_tolerance"] = (
            selected["neural_gap"] <= tolerance
            and selected["position_gap"] <= tolerance
        )
    else:
        eligible = [
            row
            for row in candidates
            if row["neural_gap"] <= tolerance and row["position_gap"] <= tolerance
        ]
        if eligible:
            selected = eligible[0]
            selected["within_tolerance"] = True
        else:
            selected = min(
                candidates,
                key=lambda row: (row["joint_gap"], -row["n_trials"]),
            )
            selected["within_tolerance"] = False
    selected["primary_gap"] = selected["joint_gap"]
    selected["direction_r1"] = "joint"
    selected["direction_r2"] = "joint"
    return selected


def _trial_score(matrix: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Return full-length per-trial mean pair distances; NaN if not selected."""
    scores = np.full(len(matrix), np.nan)
    sub = matrix[np.ix_(keep, keep)]
    scores[keep] = sub.sum(axis=1) / (len(keep) - 1)
    return scores


def summarize_match(
    pair_id: int,
    mode: str,
    r1_session: str,
    r2_session: str,
    cache1: dict,
    cache2: dict,
    selected: dict,
    tolerance: float,
    min_trials: int,
) -> tuple[dict, list[dict]]:
    """Create one summary row and one selection row per original trial."""
    keep1, keep2 = selected["keep_r1"], selected["keep_r2"]
    n1_all = np.arange(len(cache1["trial_ids"]))
    n2_all = np.arange(len(cache2["trial_ids"]))
    before = {}
    after = {}
    for metric in ("neural", "position"):
        before[f"r1_{metric}"] = subset_values(cache1[metric], n1_all)
        before[f"r2_{metric}"] = subset_values(cache2[metric], n2_all)
        after[f"r1_{metric}"] = subset_values(cache1[metric], keep1)
        after[f"r2_{metric}"] = subset_values(cache2[metric], keep2)

    row = {
        "pair_id": pair_id,
        "match_mode": mode,
        "r1_session": r1_session,
        "r2_session": r2_session,
        "r1_date": date_label(r1_session),
        "r2_date": date_label(r2_session),
        "r1_n_before": len(n1_all),
        "r2_n_before": len(n2_all),
        "n_retained_each": len(keep1),
        "r1_retain_fraction": len(keep1) / len(n1_all),
        "r2_retain_fraction": len(keep2) / len(n2_all),
        "direction_r1": selected["direction_r1"],
        "direction_r2": selected["direction_r2"],
        "within_tolerance": selected["within_tolerance"],
        "requested_tolerance": tolerance,
        "minimum_trials": min_trials,
    }
    for metric in ("neural", "position"):
        r1_before = before[f"r1_{metric}"]
        r2_before = before[f"r2_{metric}"]
        r1_after = after[f"r1_{metric}"]
        r2_after = after[f"r2_{metric}"]
        row.update(
            {
                f"r1_{metric}_mean_before": float(np.mean(r1_before)),
                f"r2_{metric}_mean_before": float(np.mean(r2_before)),
                f"{metric}_gap_before": symmetric_relative_gap(
                    float(np.mean(r1_before)), float(np.mean(r2_before))
                ),
                f"r1_{metric}_mean_after": float(np.mean(r1_after)),
                f"r2_{metric}_mean_after": float(np.mean(r2_after)),
                f"{metric}_gap_after": symmetric_relative_gap(
                    float(np.mean(r1_after)), float(np.mean(r2_after))
                ),
                f"{metric}_wasserstein_before": float(
                    wasserstein_distance(r1_before, r2_before)
                ),
                f"{metric}_wasserstein_after": float(
                    wasserstein_distance(r1_after, r2_after)
                ),
            }
        )

    trial_rows = []
    for epoch, session, cache, keep in (
        ("R1", r1_session, cache1, keep1),
        ("R2", r2_session, cache2, keep2),
    ):
        selected_set = set(keep.tolist())
        full_keep = np.arange(len(cache["trial_ids"]))
        full_neural_score = _trial_score(cache["neural"], full_keep)
        full_position_score = _trial_score(cache["position"], full_keep)
        selected_neural_score = _trial_score(cache["neural"], keep)
        selected_position_score = _trial_score(cache["position"], keep)
        for index, trial in enumerate(cache["trial_ids"]):
            trial_rows.append(
                {
                    "pair_id": pair_id,
                    "match_mode": mode,
                    "session": session,
                    "date": date_label(session),
                    "epoch": epoch,
                    "trial": trial,
                    "selected": index in selected_set,
                    "full_set_neural_trial_var": full_neural_score[index],
                    "selected_set_neural_trial_var": selected_neural_score[index],
                    "full_set_position_trial_var": full_position_score[index],
                    "selected_set_position_trial_var": selected_position_score[index],
                }
            )
    return row, trial_rows


def _heatmap(
    ax,
    table: pd.DataFrame,
    r1_dates: list[str],
    r2_dates: list[str],
    value: str,
    title: str,
    cmap: str,
    vmin=None,
    vmax=None,
    percent: bool = False,
):
    """Draw one R1-by-R2 matching-diagnostic heatmap."""
    pivot = table.pivot(index="r1_date", columns="r2_date", values=value)
    matrix = pivot.reindex(index=r1_dates, columns=r2_dates).to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(r2_dates)), [d[4:] for d in r2_dates], fontsize=8)
    ax.set_yticks(range(len(r1_dates)), [d[4:] for d in r1_dates], fontsize=7)
    ax.set_xlabel("R2 date")
    ax.set_ylabel("R1 date")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value_ij = matrix[i, j]
            label = f"{100 * value_ij:.0f}%" if percent else f"{value_ij:.0f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.5)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def make_heatmaps(
    summary: pd.DataFrame,
    output: Path,
    r1_sessions: tuple[str, ...] | list[str],
    r2_sessions: tuple[str, ...] | list[str],
):
    """Show retention, neural gap, and position gap for every mode/day pair."""
    r1_dates = [date_label(session) for session in r1_sessions]
    r2_dates = [date_label(session) for session in r2_sessions]
    figure_height = max(10.0, 5.0 + 1.2 * len(r1_dates))
    fig, axes = plt.subplots(3, 3, figsize=(13, figure_height))
    for row, mode in enumerate(MATCH_MODES):
        table = summary.loc[summary["match_mode"] == mode]
        _heatmap(
            axes[row, 0],
            table,
            r1_dates,
            r2_dates,
            "n_retained_each",
            f"{mode}: trials retained per group",
            "viridis",
            vmin=0,
        )
        _heatmap(
            axes[row, 1],
            table,
            r1_dates,
            r2_dates,
            "neural_gap_after",
            f"{mode}: neural mean gap",
            "magma_r",
            vmin=0,
            vmax=max(0.25, float(table["neural_gap_after"].max())),
            percent=True,
        )
        _heatmap(
            axes[row, 2],
            table,
            r1_dates,
            r2_dates,
            "position_gap_after",
            f"{mode}: position mean gap",
            "magma_r",
            vmin=0,
            vmax=max(0.25, float(table["position_gap_after"].max())),
            percent=True,
        )
    fig.suptitle(
        f"All {len(r1_dates)} R1 × {len(r2_dates)} R2 "
        "equal-trial-count variability matches\n"
        "Cell text: retained N or symmetric relative mean gap",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    """Parse animal, variability input, tolerance, and trial-count settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animal",
        choices=sorted(ANIMAL_SESSIONS),
        default="TS",
        help="animal/session set to process (default: TS)",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="pairwise-variability CSV; defaults to the selected animal",
    )
    parser.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    parser.add_argument("--tolerance", type=float, default=RELATIVE_TOLERANCE)
    parser.add_argument(
        "--output-tag",
        default="all42",
        help="filename tag for keeping alternative min-N/tolerance runs separate",
    )
    parser.add_argument(
        "--fixed-trials",
        type=int,
        default=None,
        help="force this equal trial count in every cell instead of maximizing N",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Select matched subsets for every date pair and export diagnostics."""
    args = parse_args(argv)
    if args.min_trials < 3:
        raise ValueError("--min-trials must be at least 3")
    if args.fixed_trials is not None and args.fixed_trials < 3:
        raise ValueError("--fixed-trials must be at least 3")
    if not (0.0 <= args.tolerance < 1.0):
        raise ValueError("--tolerance must be between 0 and 1")
    started = time.perf_counter()
    pair_path = args.pairs or OUT_DIR / f"trial_pair_variability_{args.animal}_pairs.csv"
    pairs = pd.read_csv(pair_path)
    r1_sessions, r2_sessions = ANIMAL_SESSIONS[args.animal]
    sessions = list(r1_sessions) + list(r2_sessions)
    n_day_pairs = len(r1_sessions) * len(r2_sessions)
    cache = {session: build_session_matrices(pairs, session) for session in sessions}
    smallest_session = min(len(item["trial_ids"]) for item in cache.values())
    if args.fixed_trials is not None and args.fixed_trials > smallest_session:
        raise ValueError(
            f"--fixed-trials={args.fixed_trials} exceeds the smallest session "
            f"({smallest_session} trials)"
        )
    search_min = args.fixed_trials if args.fixed_trials is not None else args.min_trials

    summary_rows = []
    trial_rows = []
    pair_id = 0
    for r1_session in r1_sessions:
        for r2_session in r2_sessions:
            pair_id += 1
            cache1, cache2 = cache[r1_session], cache[r2_session]
            for mode in MATCH_MODES:
                if mode == "neural":
                    selected = choose_single_metric_match(
                        cache1["neural"],
                        cache2["neural"],
                        search_min,
                        args.tolerance,
                        fixed_trials=args.fixed_trials,
                    )
                elif mode == "position":
                    selected = choose_single_metric_match(
                        cache1["position"],
                        cache2["position"],
                        search_min,
                        args.tolerance,
                        fixed_trials=args.fixed_trials,
                    )
                else:
                    selected = choose_joint_match(
                        cache1["neural"],
                        cache1["position"],
                        cache2["neural"],
                        cache2["position"],
                        search_min,
                        args.tolerance,
                        fixed_trials=args.fixed_trials,
                    )
                summary, rows = summarize_match(
                    pair_id,
                    mode,
                    r1_session,
                    r2_session,
                    cache1,
                    cache2,
                    selected,
                    args.tolerance,
                    search_min,
                )
                summary_rows.append(summary)
                trial_rows.extend(rows)
            print(
                f"finished {pair_id:02d}/{n_day_pairs}: {date_label(r1_session)} × "
                f"{date_label(r2_session)}",
                flush=True,
            )

    summary_df = pd.DataFrame(summary_rows)
    trials_df = pd.DataFrame(trial_rows)
    output_stem = f"variability_match_{args.output_tag}"
    summary_out = OUT_DIR / f"{output_stem}_summary.csv"
    trials_out = OUT_DIR / f"{output_stem}_trials.csv"
    figure_out = OUT_DIR / "figures" / f"fig_{output_stem}_heatmaps.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)
    trials_df.to_csv(trials_out, index=False)
    make_heatmaps(summary_df, figure_out, r1_sessions, r2_sessions)
    elapsed = time.perf_counter() - started

    print(f"\n=== Match-mode summary across {n_day_pairs} day pairs ===")
    for mode in MATCH_MODES:
        table = summary_df.loc[summary_df["match_mode"] == mode]
        print(
            f"{mode:>8}: success {int(table['within_tolerance'].sum())}/{n_day_pairs}, "
            f"median retained={table['n_retained_each'].median():.0f}, "
            f"range={table['n_retained_each'].min()}-{table['n_retained_each'].max()}, "
            f"median neural gap={100 * table['neural_gap_after'].median():.1f}%, "
            f"median position gap={100 * table['position_gap_after'].median():.1f}%"
        )
    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Saved {summary_out}")
    print(f"Saved {trials_out}")
    print(f"Saved {figure_out}")


if __name__ == "__main__":
    main()
