"""Decode pair-specific variability mean +/- SD matched trial subsets."""
from __future__ import annotations

import argparse
import os
import sys
import time
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
from decode_variability_matched_crossday import (
    baseline_matrices,
    decode_direction,
    load_raw_session,
    subset_cache,
)
from trim_r2_to_r1_std_band import date_label

REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "workflows" / "manifold_geometry"
TARGET_MODES = ("relative_position", "relative_velocity")
MATCH_DIRECTIONS = ("trim_r2_to_r1", "trim_r1_to_r2")
MATCH_METRICS = ("neural", "position")


def selected_trials_for_cell(
    selections: pd.DataFrame,
    pair_id: int,
    match_direction: str,
    session: str,
) -> np.ndarray:
    """Return the selected trial IDs for one pair, trimming direction, and day.

    A missing selection is treated as an error because silently falling back to
    all trials would make the reported matched decoder incomparable across
    cells.
    """
    rows = selections.loc[
        (selections["pair_id"] == pair_id)
        & (selections["match_direction"] == match_direction)
        & (selections["trimmed_session"] == session)
        & selections["selected"]
    ]
    if rows.empty:
        raise ValueError(
            f"no selected trials for pair={pair_id}, direction={match_direction}, "
            f"session={session}"
        )
    return rows["trial"].to_numpy(dtype=int)


def summarize_asymmetry(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize forward/reverse scores and signed gaps by target and match arm.

    The signed directional gap is defined as ``R2->R1 - R1->R2``.  Positive
    values therefore indicate better transfer when the decoder is trained on
    R2 and tested on R1.
    """
    rows = []
    for (target_mode, match_direction), table in results.groupby(
        ["target_mode", "match_direction"], sort=False
    ):
        forward = table.loc[table["decoder_direction"] == "R1->R2"].set_index("pair_id")
        reverse = table.loc[table["decoder_direction"] == "R2->R1"].set_index("pair_id")
        common = forward.index.intersection(reverse.index)
        baseline_cell_gap = (
            reverse.loc[common, "baseline_corr"] - forward.loc[common, "baseline_corr"]
        )
        matched_cell_gap = (
            reverse.loc[common, "matched_corr"] - forward.loc[common, "matched_corr"]
        )
        rows.append(
            {
                "target_mode": target_mode,
                "match_direction": match_direction,
                "n_cells": len(common),
                "baseline_r1_to_r2_mean": forward.loc[common, "baseline_corr"].mean(),
                "baseline_r2_to_r1_mean": reverse.loc[common, "baseline_corr"].mean(),
                "baseline_signed_gap": baseline_cell_gap.mean(),
                "matched_r1_to_r2_mean": forward.loc[common, "matched_corr"].mean(),
                "matched_r2_to_r1_mean": reverse.loc[common, "matched_corr"].mean(),
                "matched_signed_gap": matched_cell_gap.mean(),
                "gap_change": matched_cell_gap.mean() - baseline_cell_gap.mean(),
                "mean_absolute_cell_gap": matched_cell_gap.abs().mean(),
            }
        )
    return pd.DataFrame(rows)


def run_target(
    target_mode: str,
    selection_summary: pd.DataFrame,
    selections: pd.DataFrame,
    r1_sessions: tuple[str, ...],
    r2_sessions: tuple[str, ...],
) -> pd.DataFrame:
    """Refit and score both transfer directions for every R1-by-R2 pair.

    Only the day designated as ``trimmed_session`` is subsetted.  The anchor
    day remains complete, matching the selection procedure used to construct
    the variability band.
    """
    sessions = list(r1_sessions) + list(r2_sessions)
    print(f"\nLoading {len(sessions)} sessions for {target_mode} ...", flush=True)
    raw = {session: load_raw_session(session, target_mode) for session in sessions}
    full_cache = {
        session: subset_cache(
            values,
            values["meta"]["trial_number"].drop_duplicates().to_numpy(),
        )
        for session, values in raw.items()
    }
    baseline = baseline_matrices(target_mode)
    summary_index = selection_summary.set_index(["pair_id", "match_direction"])
    rows = []
    pair_id = 0
    total = len(r1_sessions) * len(r2_sessions)
    for r1_session in r1_sessions:
        for r2_session in r2_sessions:
            pair_id += 1
            for match_direction in MATCH_DIRECTIONS:
                match_info = summary_index.loc[(pair_id, match_direction)]
                if match_direction == "trim_r2_to_r1":
                    r1_cache = full_cache[r1_session]
                    r2_trials = selected_trials_for_cell(
                        selections, pair_id, match_direction, r2_session
                    )
                    r2_cache = subset_cache(raw[r2_session], r2_trials)
                else:
                    r1_trials = selected_trials_for_cell(
                        selections, pair_id, match_direction, r1_session
                    )
                    r1_cache = subset_cache(raw[r1_session], r1_trials)
                    r2_cache = full_cache[r2_session]

                directional = (
                    ("R1->R2", r1_session, r2_session, r1_cache, r2_cache),
                    ("R2->R1", r2_session, r1_session, r2_cache, r1_cache),
                )
                for decoder_direction, train_session, test_session, train_cache, test_cache in directional:
                    score = decode_direction(train_cache, test_cache)
                    baseline_score = (
                        float(baseline.at[train_session, test_session])
                        if baseline is not None
                        else decode_direction(
                            full_cache[train_session], full_cache[test_session]
                        )
                    )
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "target_mode": target_mode,
                            "match_direction": match_direction,
                            "decoder_direction": decoder_direction,
                            "r1_session": r1_session,
                            "r2_session": r2_session,
                            "r1_date": date_label(r1_session),
                            "r2_date": date_label(r2_session),
                            "train_session": train_session,
                            "test_session": test_session,
                            "n_trials_r1": r1_cache["n_trials"],
                            "n_trials_r2": r2_cache["n_trials"],
                            "n_trials_train": train_cache["n_trials"],
                            "n_trials_test": test_cache["n_trials"],
                            "anchor_pair_mean": float(match_info["anchor_pair_mean"]),
                            "anchor_pair_sd": float(match_info["anchor_pair_sd"]),
                            "selected_pair_mean": float(match_info["selected_pair_mean"]),
                            "selected_anchor_z": float(match_info["selected_anchor_z"]),
                            "baseline_corr": baseline_score,
                            "matched_corr": score,
                            "delta_corr": score - baseline_score,
                        }
                    )
            print(
                f"decoded {pair_id:02d}/{total}: {date_label(r1_session)[4:]} × "
                f"{date_label(r2_session)[4:]}",
                flush=True,
            )
    return pd.DataFrame(rows)


def _matrix(
    rows: pd.DataFrame,
    value: str,
    r1_dates: list[str],
    r2_dates: list[str],
) -> np.ndarray:
    """Pivot long-form cell results into an ordered R1-by-R2 matrix."""
    pivot = rows.pivot(index="r1_date", columns="r2_date", values=value)
    return pivot.reindex(index=r1_dates, columns=r2_dates).to_numpy(dtype=float)


def _plot(ax, values, title, r1_dates, r2_dates, cmap, vmin, vmax):
    """Draw one annotated heatmap panel using a shared color range."""
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(r2_dates)), [d[4:] for d in r2_dates], fontsize=8)
    ax.set_yticks(range(len(r1_dates)), [d[4:] for d in r1_dates], fontsize=7)
    ax.set_xlabel("R2 date")
    ax.set_ylabel("R1 date")
    span = vmax - vmin
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > vmin + 0.58 * span else "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="corr")


def make_figure(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    target_mode: str,
    output: Path,
    r1_sessions: tuple[str, ...],
    r2_sessions: tuple[str, ...],
    match_metric: str,
    band_sd: float,
) -> None:
    """Create the six-panel matched-decoding summary for one target variable."""
    target = results.loc[results["target_mode"] == target_mode]
    r1_dates = [date_label(session) for session in r1_sessions]
    r2_dates = [date_label(session) for session in r2_sessions]
    corr_values = target["matched_corr"].to_numpy(dtype=float)
    corr_min = min(0.0, float(np.nanmin(corr_values)))
    corr_max = float(np.nanquantile(corr_values, 0.99))
    cell_gaps = []
    for match_direction in MATCH_DIRECTIONS:
        table = target.loc[target["match_direction"] == match_direction]
        fwd = table.loc[table["decoder_direction"] == "R1->R2"]
        rev = table.loc[table["decoder_direction"] == "R2->R1"]
        cell_gaps.append(
            _matrix(rev, "matched_corr", r1_dates, r2_dates)
            - _matrix(fwd, "matched_corr", r1_dates, r2_dates)
        )
    gap_limit = max(0.05, float(np.nanmax(np.abs(cell_gaps))))

    fig, axes = plt.subplots(2, 3, figsize=(13, 15.5))
    for row, match_direction in enumerate(MATCH_DIRECTIONS):
        table = target.loc[target["match_direction"] == match_direction]
        fwd = table.loc[table["decoder_direction"] == "R1->R2"]
        rev = table.loc[table["decoder_direction"] == "R2->R1"]
        trim_label = "R2 trimmed; R1 fixed" if row == 0 else "R1 trimmed; R2 fixed"
        _plot(
            axes[row, 0], _matrix(fwd, "matched_corr", r1_dates, r2_dates),
            f"{trim_label}\nR1→R2", r1_dates, r2_dates, "viridis", corr_min, corr_max,
        )
        _plot(
            axes[row, 1], _matrix(rev, "matched_corr", r1_dates, r2_dates),
            f"{trim_label}\nR2→R1", r1_dates, r2_dates, "viridis", corr_min, corr_max,
        )
        gap = _matrix(rev, "matched_corr", r1_dates, r2_dates) - _matrix(fwd, "matched_corr", r1_dates, r2_dates)
        _plot(
            axes[row, 2], gap,
            f"cellwise asymmetry\nR2→R1 − R1→R2", r1_dates, r2_dates,
            "RdBu_r", -gap_limit, gap_limit,
        )
    target_summary = summary.loc[summary["target_mode"] == target_mode]
    labels = []
    for row in target_summary.itertuples(index=False):
        short = "trim R2" if row.match_direction == "trim_r2_to_r1" else "trim R1"
        labels.append(
            f"{short}: original gap={row.baseline_signed_gap:.3f}, "
            f"matched gap={row.matched_signed_gap:.3f}"
        )
    short_target = target_mode.replace("relative_", "")
    fig.suptitle(
        f"Pair-specific {match_metric} variability mean ± {band_sd:g} SD "
        "matched decoding\n"
        f"target={short_target}; " + "; ".join(labels),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    """Parse command-line paths, matching settings, and decoder targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal", choices=("TS",), default="TS")
    parser.add_argument("--match-metric", choices=MATCH_METRICS, default="neural")
    parser.add_argument("--band-sd", type=float, default=1.0)
    parser.add_argument("--selection-summary", type=Path, default=None)
    parser.add_argument("--selection-trials", type=Path, default=None)
    parser.add_argument("--targets", nargs="+", choices=TARGET_MODES, default=list(TARGET_MODES))
    return parser.parse_args(argv)


def main(argv=None):
    """Validate selections, run matched decoding, and save tables and figures."""
    args = parse_args(argv)
    if args.band_sd < 0:
        raise ValueError("--band-sd must be non-negative")
    started = time.perf_counter()
    r1_sessions, r2_sessions = ANIMAL_SESSIONS[args.animal]
    band_suffix = "" if np.isclose(args.band_sd, 1.0) else f"_sd{args.band_sd:g}"
    selection_stem = (
        f"pairwise_bidirectional_{args.match_metric}_variability_band{band_suffix}_"
        f"match_{args.animal}"
    )
    selection_summary_path = (
        args.selection_summary or OUT_DIR / f"{selection_stem}_summary.csv"
    )
    selection_trials_path = (
        args.selection_trials or OUT_DIR / f"{selection_stem}_trials.csv"
    )
    selection_summary = pd.read_csv(selection_summary_path)
    selections = pd.read_csv(selection_trials_path)
    if "anchor_band_sd_multiplier" in selection_summary:
        observed = selection_summary["anchor_band_sd_multiplier"].to_numpy(dtype=float)
        if not np.allclose(observed, args.band_sd):
            raise ValueError(
                "selection summary band does not match --band-sd: "
                f"observed {np.unique(observed)}, requested {args.band_sd:g}"
            )
    expected = len(r1_sessions) * len(r2_sessions) * len(MATCH_DIRECTIONS)
    if len(selection_summary) != expected:
        raise ValueError(f"expected {expected} selection rows, found {len(selection_summary)}")
    if not selection_summary["within_anchor_band"].all():
        raise ValueError("all decoded selections must be inside their exact anchor-day band")

    all_rows = []
    for target_mode in args.targets:
        rows = run_target(
            target_mode, selection_summary, selections, r1_sessions, r2_sessions
        )
        all_rows.append(rows)
        checkpoint = OUT_DIR / (
            f"pairwise_{args.match_metric}_variability_band{band_suffix}_"
            f"decoding_{args.animal}_"
            f"{target_mode.replace('relative_', '')}.csv"
        )
        rows.to_csv(checkpoint, index=False)
        print(f"Saved {checkpoint}")
    results = pd.concat(all_rows, ignore_index=True)
    summary = summarize_asymmetry(results)
    output_stem = (
        f"pairwise_{args.match_metric}_variability_band{band_suffix}_"
        f"decoding_{args.animal}"
    )
    output = OUT_DIR / f"{output_stem}.csv"
    summary_output = OUT_DIR / f"{output_stem}_summary.csv"
    results.to_csv(output, index=False)
    summary.to_csv(summary_output, index=False)
    for target_mode in args.targets:
        figure = OUT_DIR / "figures" / (
            f"fig_pairwise_{args.match_metric}_variability_band{band_suffix}_"
            f"decoding_{args.animal}_"
            f"{target_mode.replace('relative_', '')}.png"
        )
        make_figure(
            results,
            summary,
            target_mode,
            figure,
            r1_sessions,
            r2_sessions,
            args.match_metric,
            args.band_sd,
        )
        print(f"Saved {figure}")
    print("\n=== Asymmetry summary (signed gap = R2->R1 - R1->R2) ===")
    print(summary.round(4).to_string(index=False))
    print(f"Elapsed: {time.perf_counter() - started:.1f} seconds")
    print(f"Saved {output}")
    print(f"Saved {summary_output}")


if __name__ == "__main__":
    main()
