"""Pair-specific bidirectional neural-STD matching for every R1 x R2 day pair."""
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
from trim_r2_to_r1_std_band import date_label, trajectory_std, trim_to_target

REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "manifold_geometry"
PAIR_COLUMN = "neural_pair_msd"
DIRECTIONS = ("trim_r2_to_r1", "trim_r1_to_r2")


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
    fmt: str = ".0f",
):
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
) -> None:
    r1_dates = [date_label(session) for session in r1_sessions]
    r2_dates = [date_label(session) for session in r2_sessions]
    residual_limit = max(0.001, float(summary["absolute_target_difference"].max()))
    fig, axes = plt.subplots(2, 2, figsize=(13, max(10, 5 + 0.75 * len(r1_dates))))
    for row, direction in enumerate(DIRECTIONS):
        table = summary.loc[summary["match_direction"] == direction]
        trimmed = "R2" if direction == "trim_r2_to_r1" else "R1"
        target = "R1" if direction == "trim_r2_to_r1" else "R2"
        _heatmap(
            axes[row, 0],
            table,
            r1_dates,
            r2_dates,
            "n_trimmed_remaining",
            f"{trimmed} trials retained; {target} fixed",
            "viridis",
            vmin=0,
        )
        _heatmap(
            axes[row, 1],
            table,
            r1_dates,
            r2_dates,
            "signed_target_difference",
            f"selected {trimmed} STD − exact {target}-day STD",
            "coolwarm",
            vmin=-residual_limit,
            vmax=residual_limit,
            fmt=".3f",
        )
    fig.suptitle(
        f"{animal}: pair-specific bidirectional neural trajectory-STD matching\n"
        "one day fixed; delete highest/lowest contributor from the other until first target crossing",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal", choices=sorted(ANIMAL_SESSIONS), default="TS")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--min-trials", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    r1_sessions, r2_sessions = ANIMAL_SESSIONS[args.animal]
    pair_path = args.pairs or OUT_DIR / f"trial_pair_variability_{args.animal}_pairs.csv"
    pairs = pd.read_csv(pair_path)
    cache: dict[str, dict] = {}
    for session in list(r1_sessions) + list(r2_sessions):
        trial_ids, matrix = pair_matrix(
            pairs.loc[pairs["session"] == session],
            PAIR_COLUMN,
        )
        cache[session] = {
            "trial_ids": trial_ids,
            "matrix": matrix,
            "full_std": trajectory_std(matrix, np.arange(len(trial_ids))),
        }

    summary_rows = []
    path_rows = []
    trial_rows = []
    pair_id = 0
    for r1_session in r1_sessions:
        for r2_session in r2_sessions:
            pair_id += 1
            r1 = cache[r1_session]
            r2 = cache[r2_session]
            specifications = (
                ("trim_r2_to_r1", r1_session, r1, r2_session, r2),
                ("trim_r1_to_r2", r2_session, r2, r1_session, r1),
            )
            for direction, anchor_session, anchor, trimmed_session, trimmed in specifications:
                result = trim_to_target(
                    trimmed["matrix"],
                    target_std=anchor["full_std"],
                    min_trials=args.min_trials,
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
                        "pair_id": pair_id,
                        "r1_session": r1_session,
                        "r1_date": date_label(r1_session),
                        "r2_session": r2_session,
                        "r2_date": date_label(r2_session),
                        "match_direction": direction,
                        "anchor_session": anchor_session,
                        "trimmed_session": trimmed_session,
                        "target_std": anchor["full_std"],
                        "trimmed_start_std": result["start_std"],
                        "selected_std": result["selected_std"],
                        "signed_target_difference": result["signed_target_difference"],
                        "absolute_target_difference": result["absolute_target_difference"],
                        "deletion_direction": result["direction"],
                        "status": result["status"],
                        "crossed_target": result["crossed_target"],
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
                            "pair_id": pair_id,
                            "r1_date": date_label(r1_session),
                            "r2_date": date_label(r2_session),
                            "match_direction": direction,
                            "anchor_session": anchor_session,
                            "trimmed_session": trimmed_session,
                            "target_std": anchor["full_std"],
                            **row,
                        }
                    )
                for index, trial_id in enumerate(trimmed["trial_ids"]):
                    removed = removal_map.get(index, {})
                    trial_rows.append(
                        {
                            "animal": args.animal,
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
    stem = f"pairwise_bidirectional_neural_std_match_{args.animal}"
    outputs = {
        "summary": OUT_DIR / f"{stem}_summary.csv",
        "path": OUT_DIR / f"{stem}_path.csv",
        "trials": OUT_DIR / f"{stem}_trials.csv",
        "figure": OUT_DIR / "figures" / f"fig_{stem}.png",
    }
    summary_df.to_csv(outputs["summary"], index=False)
    path_df.to_csv(outputs["path"], index=False)
    trials_df.to_csv(outputs["trials"], index=False)
    make_figure(summary_df, r1_sessions, r2_sessions, outputs["figure"], args.animal)

    print("=== Pair-specific neural STD match ===")
    for direction in DIRECTIONS:
        table = summary_df.loc[summary_df["match_direction"] == direction]
        print(
            f"{direction}: crossed {int(table['crossed_target'].sum())}/{len(table)}, "
            f"median retained={table['n_trimmed_remaining'].median():.0f}, "
            f"range={table['n_trimmed_remaining'].min()}-{table['n_trimmed_remaining'].max()}, "
            f"median |STD residual|={table['absolute_target_difference'].median():.6f}"
        )
    for output in outputs.values():
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
