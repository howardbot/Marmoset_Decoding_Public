"""Match R1/R2 neural trial-pair variability while retaining equal trial counts.

The default analysis pairs the final three R1 days with the three R2 days in
chronological order.  This avoids treating the 14 R1 sessions as if they were
exchangeable trials against only three R2 sessions.

For every paired day and every possible equal retained count, two deterministic
greedy paths are constructed from the precomputed trial-pair matrices:

* R1: remove the trial with the smallest mean distance to the retained set,
  which raises R1 variability as efficiently as possible at that step.
* R2: remove the trial with the largest mean distance to the retained set,
  which lowers R2 variability as efficiently as possible at that step.

The largest retained count whose symmetric R1/R2 neural-mean gap is within the
requested tolerance is selected.  Position never drives selection; its before
and after values are reported as a kinematic side-effect check.

This is a descriptive, target-informed sensitivity analysis.  Selection and
evaluation use the same neural variability values and therefore should not be
presented as an independent confirmatory test.

Outputs
-------
Results/manifold_geometry/variability_match_last3_summary.csv
Results/manifold_geometry/variability_match_last3_trials.csv
Results/manifold_geometry/figures/fig_variability_match_last3.png
"""
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
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from big_sweep_phase2_crossday import SESSIONS_R1, SESSIONS_R2

REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "manifold_geometry"
PAIR_CSV = OUT_DIR / "trial_pair_variability_TS_pairs.csv"
OUT_SUMMARY = OUT_DIR / "variability_match_last3_summary.csv"
OUT_TRIALS = OUT_DIR / "variability_match_last3_trials.csv"
OUT_FIGURE = OUT_DIR / "figures" / "fig_variability_match_last3.png"
MIN_TRIALS = 20
RELATIVE_TOLERANCE = 0.05


def pair_matrix(day_pairs: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a symmetric trial-pair matrix from one session's long table."""
    if metric not in day_pairs:
        raise KeyError(f"missing metric column: {metric}")
    trial_ids = np.unique(
        np.concatenate(
            [day_pairs["trial_i"].to_numpy(), day_pairs["trial_j"].to_numpy()]
        )
    )
    index = {trial: i for i, trial in enumerate(trial_ids)}
    matrix = np.zeros((len(trial_ids), len(trial_ids)), dtype=float)
    seen = np.zeros_like(matrix, dtype=bool)
    for trial_i, trial_j, value in day_pairs[
        ["trial_i", "trial_j", metric]
    ].itertuples(index=False, name=None):
        i, j = index[trial_i], index[trial_j]
        matrix[i, j] = matrix[j, i] = float(value)
        seen[i, j] = seen[j, i] = True
    expected = ~np.eye(len(trial_ids), dtype=bool)
    if not np.all(seen[expected]):
        raise ValueError("pair table is incomplete for this session")
    return trial_ids, matrix


def subset_values(matrix: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Return the unique upper-triangle pair values for a retained subset."""
    keep = np.asarray(keep, dtype=int)
    sub = matrix[np.ix_(keep, keep)]
    return sub[np.triu_indices(len(keep), k=1)]


def subset_mean(matrix: np.ndarray, keep: np.ndarray) -> float:
    values = subset_values(matrix, keep)
    return float(np.mean(values))


def symmetric_relative_gap(value1: float, value2: float) -> float:
    """Absolute difference divided by the two-value mean."""
    denom = (abs(value1) + abs(value2)) / 2.0
    return float(abs(value2 - value1) / denom) if denom > 0 else 0.0


def greedy_path(
    matrix: np.ndarray,
    min_trials: int,
    direction: str,
) -> dict[int, np.ndarray]:
    """Return retained indices at every N along a variance-raising/lowering path."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if direction not in {"raise", "lower"}:
        raise ValueError("direction must be 'raise' or 'lower'")
    if min_trials < 2 or min_trials > len(matrix):
        raise ValueError("min_trials must be between 2 and the matrix size")

    keep = np.arange(len(matrix), dtype=int)
    path = {len(keep): keep.copy()}
    while len(keep) > min_trials:
        sub = matrix[np.ix_(keep, keep)]
        row_mean = sub.sum(axis=1) / (len(keep) - 1)
        if direction == "raise":
            remove_local = int(np.argmin(row_mean))
        else:
            remove_local = int(np.argmax(row_mean))
        keep = np.delete(keep, remove_local)
        path[len(keep)] = keep.copy()
    return path


def choose_largest_match(
    r1_matrix: np.ndarray,
    r2_matrix: np.ndarray,
    min_trials: int = MIN_TRIALS,
    tolerance: float = RELATIVE_TOLERANCE,
) -> dict:
    """Choose the largest equal-N R1/R2 subsets meeting the mean-gap tolerance."""
    max_equal_n = min(len(r1_matrix), len(r2_matrix))
    minimum = min(min_trials, max_equal_n)
    r1_path = greedy_path(r1_matrix, minimum, direction="raise")
    r2_path = greedy_path(r2_matrix, minimum, direction="lower")

    candidates = []
    for n_trials in range(max_equal_n, minimum - 1, -1):
        keep1, keep2 = r1_path[n_trials], r2_path[n_trials]
        value1 = subset_mean(r1_matrix, keep1)
        value2 = subset_mean(r2_matrix, keep2)
        candidates.append(
            {
                "n_trials": n_trials,
                "keep_r1": keep1,
                "keep_r2": keep2,
                "r1_mean": value1,
                "r2_mean": value2,
                "relative_gap": symmetric_relative_gap(value1, value2),
            }
        )

    eligible = [row for row in candidates if row["relative_gap"] <= tolerance]
    if eligible:
        selected = eligible[0]  # candidates are ordered from largest N down
        selected["within_tolerance"] = True
    else:
        selected = min(candidates, key=lambda row: (row["relative_gap"], -row["n_trials"]))
        selected["within_tolerance"] = False
    return selected


def _trial_rows(
    session: str,
    epoch: str,
    pair_id: int,
    trial_ids: np.ndarray,
    neural_matrix: np.ndarray,
    keep: np.ndarray,
) -> list[dict]:
    keep_set = set(np.asarray(keep, dtype=int))
    full_score = neural_matrix.sum(axis=1) / (len(neural_matrix) - 1)
    selected_score = np.full(len(trial_ids), np.nan)
    selected_sub = neural_matrix[np.ix_(keep, keep)]
    selected_score[keep] = selected_sub.sum(axis=1) / (len(keep) - 1)
    return [
        {
            "pair_id": pair_id,
            "session": session,
            "date": session.replace("TSAL", "")[:8],
            "epoch": epoch,
            "trial": trial,
            "selected": i in keep_set,
            "full_set_trial_var": full_score[i],
            "selected_set_trial_var": selected_score[i],
        }
        for i, trial in enumerate(trial_ids)
    ]


def match_one_pair(
    pairs: pd.DataFrame,
    r1_session: str,
    r2_session: str,
    pair_id: int,
    min_trials: int,
    tolerance: float,
) -> tuple[dict, list[dict], dict]:
    """Match one chronological R1/R2 day pair and return summary plus plot data."""
    g1 = pairs.loc[pairs["session"] == r1_session]
    g2 = pairs.loc[pairs["session"] == r2_session]
    if g1.empty or g2.empty:
        raise ValueError(f"missing pair data for {r1_session} or {r2_session}")

    ids1, neural1 = pair_matrix(g1, "neural_pair_msd")
    ids2, neural2 = pair_matrix(g2, "neural_pair_msd")
    pos_ids1, position1 = pair_matrix(g1, "position_pair_msd")
    pos_ids2, position2 = pair_matrix(g2, "position_pair_msd")
    if not np.array_equal(ids1, pos_ids1) or not np.array_equal(ids2, pos_ids2):
        raise RuntimeError("neural and position trial IDs do not align")

    match = choose_largest_match(neural1, neural2, min_trials, tolerance)
    keep1, keep2 = match["keep_r1"], match["keep_r2"]
    neural1_before = subset_values(neural1, np.arange(len(ids1)))
    neural2_before = subset_values(neural2, np.arange(len(ids2)))
    neural1_after = subset_values(neural1, keep1)
    neural2_after = subset_values(neural2, keep2)
    position1_before = subset_values(position1, np.arange(len(ids1)))
    position2_before = subset_values(position2, np.arange(len(ids2)))
    position1_after = subset_values(position1, keep1)
    position2_after = subset_values(position2, keep2)

    summary = {
        "pair_id": pair_id,
        "r1_session": r1_session,
        "r2_session": r2_session,
        "r1_date": r1_session.replace("TSAL", "")[:8],
        "r2_date": r2_session.replace("TSAL", "")[:8],
        "r1_n_before": len(ids1),
        "r2_n_before": len(ids2),
        "n_retained_each": match["n_trials"],
        "r1_retain_fraction": match["n_trials"] / len(ids1),
        "r2_retain_fraction": match["n_trials"] / len(ids2),
        "r1_neural_mean_before": float(np.mean(neural1_before)),
        "r2_neural_mean_before": float(np.mean(neural2_before)),
        "r1_neural_mean_after": float(np.mean(neural1_after)),
        "r2_neural_mean_after": float(np.mean(neural2_after)),
        "neural_relative_gap_after": match["relative_gap"],
        "neural_wasserstein_before": float(
            wasserstein_distance(neural1_before, neural2_before)
        ),
        "neural_wasserstein_after": float(
            wasserstein_distance(neural1_after, neural2_after)
        ),
        "r1_position_mean_before": float(np.mean(position1_before)),
        "r2_position_mean_before": float(np.mean(position2_before)),
        "r1_position_mean_after": float(np.mean(position1_after)),
        "r2_position_mean_after": float(np.mean(position2_after)),
        "position_relative_gap_after": symmetric_relative_gap(
            float(np.mean(position1_after)), float(np.mean(position2_after))
        ),
        "within_tolerance": match["within_tolerance"],
        "requested_tolerance": tolerance,
        "minimum_trials": min_trials,
    }
    trial_rows = _trial_rows(
        r1_session, "R1", pair_id, ids1, neural1, keep1
    ) + _trial_rows(r2_session, "R2", pair_id, ids2, neural2, keep2)
    plot_data = {
        "summary": summary,
        "neural_before": (neural1_before, neural2_before),
        "neural_after": (neural1_after, neural2_after),
        "position_after": (position1_after, position2_after),
    }
    return summary, trial_rows, plot_data


def _density(ax, values: np.ndarray, color: str, label: str):
    values = np.asarray(values, dtype=float)
    lo, hi = np.quantile(values, [0.005, 0.995])
    margin = max((hi - lo) * 0.15, 1e-6)
    grid = np.linspace(max(0.0, lo - margin), hi + margin, 300)
    if np.std(values) > 1e-12:
        density = gaussian_kde(values)(grid)
        ax.plot(grid, density, color=color, linewidth=1.6, label=label)
    ax.axvline(np.mean(values), color=color, linestyle="--", linewidth=1.0)


def make_figure(plot_rows: list[dict], output: Path):
    """Plot neural distributions before/after matching and position after matching."""
    fig, axes = plt.subplots(len(plot_rows), 3, figsize=(14, 3.6 * len(plot_rows)), squeeze=False)
    colors = {"R1": "#4C78A8", "R2": "#F58518"}
    for row, data in enumerate(plot_rows):
        summary = data["summary"]
        panels = (
            ("neural_before", "Neural before matching"),
            ("neural_after", "Neural after matching"),
            ("position_after", "Position after neural matching"),
        )
        for col, (key, title) in enumerate(panels):
            ax = axes[row, col]
            values1, values2 = data[key]
            _density(ax, values1, colors["R1"], "R1")
            _density(ax, values2, colors["R2"], "R2")
            ax.set_title(title, fontsize=10)
            ax.set_yticks([])
            ax.grid(alpha=0.2, axis="x")
            ax.spines[["top", "right", "left"]].set_visible(False)
            if row == 0 and col == 0:
                ax.legend(frameon=False)
            if col == 0:
                ax.set_ylabel(
                    f"R1 {summary['r1_date']}  vs  R2 {summary['r2_date']}\n"
                    f"retain {summary['n_retained_each']} each",
                    fontsize=9,
                )
            ax.set_xlabel("pairwise mean squared difference", fontsize=9)
        axes[row, 1].text(
            0.98,
            0.95,
            f"mean gap = {100 * summary['neural_relative_gap_after']:.1f}%",
            transform=axes[row, 1].transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    fig.suptitle(
        "Equal-trial-count neural variability matching\n"
        "dashed lines show distribution means; position is not used for selection",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=PAIR_CSV)
    parser.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    parser.add_argument("--tolerance", type=float, default=RELATIVE_TOLERANCE)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.min_trials < 2:
        raise ValueError("--min-trials must be at least 2")
    if not (0.0 <= args.tolerance < 1.0):
        raise ValueError("--tolerance must be between 0 and 1")
    pairs = pd.read_csv(args.pairs)

    r1_sessions = SESSIONS_R1[-3:]
    r2_sessions = SESSIONS_R2
    summaries = []
    trial_rows = []
    plot_rows = []
    for pair_id, (r1_session, r2_session) in enumerate(
        zip(r1_sessions, r2_sessions), start=1
    ):
        summary, rows, plot_data = match_one_pair(
            pairs,
            r1_session,
            r2_session,
            pair_id,
            args.min_trials,
            args.tolerance,
        )
        summaries.append(summary)
        trial_rows.extend(rows)
        plot_rows.append(plot_data)
        print(
            f"pair {pair_id}: retain {summary['n_retained_each']} each; "
            f"neural {summary['r1_neural_mean_after']:.6f} vs "
            f"{summary['r2_neural_mean_after']:.6f} "
            f"(gap={100 * summary['neural_relative_gap_after']:.2f}%)"
        )

    summary_df = pd.DataFrame(summaries)
    trials_df = pd.DataFrame(trial_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    trials_df.to_csv(OUT_TRIALS, index=False)
    make_figure(plot_rows, OUT_FIGURE)
    print("\nEqual-day average after matching:")
    print(
        f"  R1={summary_df['r1_neural_mean_after'].mean():.6f}, "
        f"R2={summary_df['r2_neural_mean_after'].mean():.6f}, "
        f"gap={100 * symmetric_relative_gap(summary_df['r1_neural_mean_after'].mean(), summary_df['r2_neural_mean_after'].mean()):.2f}%"
    )
    print(f"Saved {OUT_SUMMARY}")
    print(f"Saved {OUT_TRIALS}")
    print(f"Saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
