"""Re-run the 14 x 3 bidirectional cross-day decoder after variability matching.

Each R1/R2 heatmap cell uses the cell-specific, equal-N trial subsets exported
by ``match_all_trial_pair_variability.py``.  PCA is refit independently on each
selected session subset, CCA is fit on the selected trial-average trajectories,
and the locked lag-0 Kalman decoder is evaluated in both directions.

The default selection is the high-retention control: exactly 40 trials from
each side, jointly optimized for neural and position trial-pair variability.
``--match-mode`` can instead select the neural-only or position-only trial
lists exported by the same fixed-N matching run.
Cells that do not reach both requested variability-gap tolerances remain in the
heatmap but are marked with a red border and ``!``; no low-N substitution is
silently made.

Outputs
-------
Results/workflows/manifold_geometry/variability_matched_crossday_fixed40.csv
Results/workflows/manifold_geometry/figures/
    fig_variability_matched_crossday_fixed40_velocity.png
    fig_variability_matched_crossday_fixed40_position.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from big_sweep_phase2_crossday import (
    EXCLUDE_TRIALS,
    K_PCS,
    N_PHASE_BINS,
    SESSIONS_R1,
    SESSIONS_R2,
    SMOOTH_SIGMA_MS,
    TRIAL_RESULTS,
    UNIT_QUALITIES,
    filter_trials,
    kalman_fit_predict,
    m2_per_trial,
)
from manifold_align import apply_alignment, cca_align, pca_neural, trial_average_pc
from plotting_common import SWEEP_CSV, filter_locked, load_sweep, pivot_matrix

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "workflows" / "manifold_geometry"
SELECTION_SUMMARY = OUT_DIR / "variability_match_all42_fixed40_tol10_summary.csv"
SELECTION_TRIALS = OUT_DIR / "variability_match_all42_fixed40_tol10_trials.csv"
OUT_CSV = OUT_DIR / "variability_matched_crossday_fixed40.csv"
FIG_DIR = OUT_DIR / "figures"
BIN_MS = 30
SMOOTHER_KW = {
    "smoother": "butter",
    "smooth_cutoff_hz": 6.0,
    "smooth_order": 2,
}
TARGET_MODES = ("relative_velocity", "relative_position")


def date_label(session: str) -> str:
    """Convert an internal session identifier to an eight-digit date label."""
    return session.replace("TSAL", "")[:8]


def load_raw_session(session: str, target_mode: str) -> dict:
    """Load one locked-config session before subset-specific PCA fitting."""
    bin_s = BIN_MS / 1000.0
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_s
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, neural, meta = du.build_decoder_dataset(
            nwb,
            reach,
            target_mode,
            bin_size=bin_s,
            unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS,
            trial_window="start_to_peak",
            **SMOOTHER_KW,
        )
    finally:
        io.close()
    X, neural, meta = filter_trials(
        X, neural, meta, EXCLUDE_TRIALS.get(session, ())
    )
    neural_smoothed = du.smooth_neural_causal(
        neural,
        meta,
        sigma_bins=SMOOTH_SIGMA_MS / BIN_MS,
    )
    return {"X": X, "neural": neural_smoothed, "meta": meta}


def subset_cache(raw: dict, selected_trials) -> dict:
    """Subset whole trials and refit the day-specific 12-D PCA manifold."""
    requested = set(int(value) for value in selected_trials)
    mask = raw["meta"]["trial_number"].isin(requested).to_numpy()
    X = raw["X"][mask]
    neural = raw["neural"][mask]
    meta = raw["meta"][mask].reset_index(drop=True)
    present = set(int(value) for value in meta["trial_number"].unique())
    missing = requested - present
    if missing:
        raise ValueError(f"selected trials absent from decoder dataset: {sorted(missing)}")
    if len(present) != len(requested):
        raise RuntimeError("selected trial count changed during decoder subsetting")
    neural_pc, loadings, mean = pca_neural(neural, k=K_PCS)
    trajectory = trial_average_pc(
        neural_pc,
        meta,
        n_phase_bins=N_PHASE_BINS,
    )
    return {
        "X": X,
        "Y_pc": neural_pc,
        "meta": meta,
        "PCA_V": loadings,
        "PCA_mean": mean,
        "traj": trajectory,
        "n_trials": len(present),
    }


def decode_direction(train_cache: dict, test_cache: dict) -> float:
    """Locked single-trial CCA + lag-0 Kalman transfer score."""
    W_train, W_test, mean_train, mean_test = cca_align(
        train_cache["traj"], test_cache["traj"]
    )
    neural_train = apply_alignment(train_cache["Y_pc"], W_train, mean_train)
    neural_test = apply_alignment(test_cache["Y_pc"], W_test, mean_test)
    X_test_centered, prediction = kalman_fit_predict(
        train_cache["X"],
        neural_train,
        test_cache["X"],
        neural_test,
        test_cache["meta"],
    )
    return m2_per_trial(X_test_centered, prediction, test_cache["meta"])


def selected_trials_for_pair(
    selections: pd.DataFrame,
    pair_id: int,
    session: str,
    match_mode: str = "joint",
) -> np.ndarray:
    """Return selected trial numbers for one pair/session/matching mode."""
    rows = selections.loc[
        (selections["pair_id"] == pair_id)
        & (selections["match_mode"] == match_mode)
        & (selections["session"] == session)
        & selections["selected"]
    ]
    if rows.empty:
        raise ValueError(f"no selected trials for pair={pair_id}, session={session}")
    return rows["trial"].to_numpy(dtype=int)


def baseline_matrices(target_mode: str) -> pd.DataFrame | None:
    """Return the saved full-trial matrix, or ``None`` when it is unavailable.

    The Phase-2 sweep is a convenient cache, not a scientific dependency of
    this analysis.  A fresh clone may contain the NWBs and matching selections
    without the gitignored sweep CSV.  In that case ``run_target`` reconstructs
    the same locked-config cross-epoch baselines from the already loaded full
    trial sets.
    """
    if not SWEEP_CSV.exists():
        return None
    sweep = load_sweep()
    locked = filter_locked(sweep, target_mode=target_mode)
    return pivot_matrix(locked)


def run_target(
    target_mode: str,
    selection_summary: pd.DataFrame,
    selections: pd.DataFrame,
    match_mode: str = "joint",
) -> pd.DataFrame:
    """Evaluate all 42 session pairs in both transfer directions."""
    print(f"\nLoading 17 sessions for {target_mode} ...", flush=True)
    raw = {
        session: load_raw_session(session, target_mode)
        for session in list(SESSIONS_R1) + list(SESSIONS_R2)
    }
    baseline = baseline_matrices(target_mode)
    full_cache = None
    if baseline is None:
        print(
            f"Saved Phase-2 sweep not found at {SWEEP_CSV}; "
            "recomputing locked full-trial cross-epoch baselines.",
            flush=True,
        )
        full_cache = {
            session: subset_cache(
                values,
                values["meta"]["trial_number"].drop_duplicates().to_numpy(),
            )
            for session, values in raw.items()
        }
    rows = []
    match_summary = selection_summary.loc[
        selection_summary["match_mode"] == match_mode
    ].set_index("pair_id")
    pair_id = 0
    for r1_session in SESSIONS_R1:
        for r2_session in SESSIONS_R2:
            pair_id += 1
            trials1 = selected_trials_for_pair(
                selections, pair_id, r1_session, match_mode=match_mode
            )
            trials2 = selected_trials_for_pair(
                selections, pair_id, r2_session, match_mode=match_mode
            )
            cache1 = subset_cache(raw[r1_session], trials1)
            cache2 = subset_cache(raw[r2_session], trials2)
            if cache1["n_trials"] != cache2["n_trials"]:
                raise RuntimeError("R1/R2 selected decoder trial counts differ")
            match_info = match_summary.loc[pair_id]
            directional = (
                ("R1->R2", r1_session, r2_session, cache1, cache2),
                ("R2->R1", r2_session, r1_session, cache2, cache1),
            )
            for direction, train_session, test_session, train_cache, test_cache in directional:
                score = decode_direction(train_cache, test_cache)
                if baseline is None:
                    baseline_score = decode_direction(
                        full_cache[train_session], full_cache[test_session]
                    )
                else:
                    baseline_score = float(baseline.at[train_session, test_session])
                rows.append(
                    {
                        "pair_id": pair_id,
                        "match_mode": match_mode,
                        "target_mode": target_mode,
                        "direction": direction,
                        "r1_session": r1_session,
                        "r2_session": r2_session,
                        "r1_date": date_label(r1_session),
                        "r2_date": date_label(r2_session),
                        "train_session": train_session,
                        "test_session": test_session,
                        "n_trials_train": train_cache["n_trials"],
                        "n_trials_test": test_cache["n_trials"],
                        "neural_gap": float(match_info["neural_gap_after"]),
                        "position_gap": float(match_info["position_gap_after"]),
                        "within_tolerance": bool(match_info["within_tolerance"]),
                        "matched_corr": score,
                        "baseline_corr": baseline_score,
                        "delta_corr": score - baseline_score,
                    }
                )
            print(
                f"decoded {pair_id:02d}/42 {date_label(r1_session)} × "
                f"{date_label(r2_session)} ({cache1['n_trials']} each)",
                flush=True,
            )
    return pd.DataFrame(rows)


def _matrix(rows: pd.DataFrame, value: str) -> np.ndarray:
    """Pivot a cell-level column into the canonical 14-by-3 date order."""
    r1_dates = [date_label(session) for session in SESSIONS_R1]
    r2_dates = [date_label(session) for session in SESSIONS_R2]
    pivot = rows.pivot(index="r1_date", columns="r2_date", values=value)
    return pivot.reindex(index=r1_dates, columns=r2_dates).to_numpy(dtype=float)


def _plot_heatmap(
    ax,
    values: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    failed: np.ndarray,
):
    """Draw an annotated heatmap and flag cells that missed the tolerance."""
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(
        range(len(SESSIONS_R2)),
        [date_label(session)[4:] for session in SESSIONS_R2],
        fontsize=8,
    )
    ax.set_yticks(
        range(len(SESSIONS_R1)),
        [date_label(session)[4:] for session in SESSIONS_R1],
        fontsize=7,
    )
    ax.set_xlabel("R2 date")
    ax.set_ylabel("R1 date")
    span = vmax - vmin
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if values[row, col] > vmin + 0.58 * span else "black"
            suffix = "!" if failed[row, col] else ""
            ax.text(
                col,
                row,
                f"{values[row, col]:.2f}{suffix}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )
            if failed[row, col]:
                ax.add_patch(
                    Rectangle(
                        (col - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#d62728",
                        linewidth=1.2,
                    )
                )
    return image


def make_figure(
    rows: pd.DataFrame,
    target_mode: str,
    output: Path,
    match_mode: str = "joint",
):
    """Compare original and matched forward/reverse 14x3 heatmaps."""
    target_rows = rows.loc[rows["target_mode"] == target_mode]
    forward = target_rows.loc[target_rows["direction"] == "R1->R2"]
    reverse = target_rows.loc[target_rows["direction"] == "R2->R1"]
    failed = _matrix(forward, "within_tolerance") < 0.5
    panels = (
        (forward, "baseline_corr", "R1→R2 original"),
        (forward, "matched_corr", "R1→R2 variance-balanced"),
        (forward, "delta_corr", "R1→R2 matched − original"),
        (reverse, "baseline_corr", "R2→R1 original"),
        (reverse, "matched_corr", "R2→R1 variance-balanced"),
        (reverse, "delta_corr", "R2→R1 matched − original"),
    )
    baseline_and_matched = np.concatenate(
        [_matrix(table, value) for table, value, _ in panels if value != "delta_corr"]
    )
    corr_min = min(0.0, float(np.nanmin(baseline_and_matched)))
    corr_max = float(np.nanquantile(baseline_and_matched, 0.99))
    delta_values = np.concatenate(
        [_matrix(table, value).ravel() for table, value, _ in panels if value == "delta_corr"]
    )
    delta_limit = max(0.05, float(np.nanmax(np.abs(delta_values))))

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 15.5))
    for ax, (table, value, title) in zip(axes.ravel(), panels):
        values = _matrix(table, value)
        if value == "delta_corr":
            image = _plot_heatmap(
                ax, values, title, "RdBu_r", -delta_limit, delta_limit, failed
            )
        else:
            image = _plot_heatmap(
                ax, values, title, "viridis", corr_min, corr_max, failed
            )
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="corr")
    success = int(forward["within_tolerance"].sum())
    short_target = target_mode.replace("relative_", "")
    match_label = {
        "joint": "joint neural+position",
        "neural": "neural-only",
        "position": "position-only",
    }[match_mode]
    threshold_label = (
        "both variability mean gaps" if match_mode == "joint"
        else f"{match_mode} variability mean gap"
    )
    fig.suptitle(
        f"Cross-day Kalman decoding after {match_label} variability balancing\n"
        f"target={short_target}, fixed 40 trials/session/cell, 30 ms, PCA-12, CCA, lag 0\n"
        f"{success}/42 cells have {threshold_label} ≤10%; "
        "red/! cells are best fixed-N subsets but exceed 10%",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    """Parse selection paths, requested targets, and matching mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-summary", type=Path, default=SELECTION_SUMMARY)
    parser.add_argument("--selection-trials", type=Path, default=SELECTION_TRIALS)
    parser.add_argument("--output", type=Path, default=OUT_CSV)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGET_MODES,
        default=list(TARGET_MODES),
    )
    parser.add_argument(
        "--match-mode",
        choices=("joint", "neural", "position"),
        default="joint",
        help="which fixed-40 trial-selection list to decode",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the fixed-N matched decoder and write target-specific outputs."""
    args = parse_args(argv)
    started = time.perf_counter()
    selection_summary = pd.read_csv(args.selection_summary)
    selections = pd.read_csv(args.selection_trials)
    matched = selection_summary.loc[
        selection_summary["match_mode"] == args.match_mode
    ]
    if len(matched) != 42:
        raise ValueError(
            f"selection summary must contain 42 {args.match_mode} day-pair rows"
        )
    retained = matched["n_retained_each"].unique()
    if len(retained) != 1:
        raise ValueError(f"decoder heatmap requires fixed N; found {retained.tolist()}")

    all_rows = []
    for target_mode in args.targets:
        target_rows = run_target(
            target_mode,
            selection_summary,
            selections,
            match_mode=args.match_mode,
        )
        all_rows.append(target_rows)
        suffix = target_mode.replace("relative_", "")
        target_checkpoint = args.output.with_name(
            f"{args.output.stem}_{suffix}{args.output.suffix}"
        )
        target_rows.to_csv(target_checkpoint, index=False)
        mode_tag = "" if args.match_mode == "joint" else f"_{args.match_mode}_only"
        figure = FIG_DIR / (
            f"fig_variability_matched_crossday_fixed40{mode_tag}_{suffix}.png"
        )
        make_figure(
            target_rows,
            target_mode,
            figure,
            match_mode=args.match_mode,
        )
        print(f"Saved {target_checkpoint}")
        print(f"Saved {figure}")
    results = pd.concat(all_rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print("\n=== Decoder summary ===")
    summary = results.groupby(["target_mode", "direction"])[
        ["baseline_corr", "matched_corr", "delta_corr"]
    ].mean()
    print(summary.round(4).to_string())
    print(f"Elapsed: {time.perf_counter() - started:.1f} seconds")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
