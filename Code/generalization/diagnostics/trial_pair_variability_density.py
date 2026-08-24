
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS,
    EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS,
    TRIAL_RESULTS,
    UNIT_QUALITIES,
    filter_trials,
)

warnings.filterwarnings("ignore")

# Bin the original signals at 30 ms before trial-phase normalization.
BIN_MS = 30
# Represent every start-to-peak trial on 30 equally spaced phase points.
N_PHASE = 30
# Smooth the kinematic signal before phase resampling.
SMOOTHER_KW = {
    "smoother": "butter",
    "smooth_cutoff_hz": 6.0,
    "smooth_order": 2,
}
REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "workflows" / "manifold_geometry"


def session_date(session: str) -> str:
    """Return the first YYYYMMDD token in a session tag."""
    match = re.search(r"\d{8}", session)
    return match.group(0) if match else session


def phase_resample_joint(
    position: np.ndarray,
    neural: np.ndarray,
    meta: pd.DataFrame,
    n_phase: int = N_PHASE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample aligned position and neural data on a common trial phase grid."""
    position = np.asarray(position, dtype=float)
    neural = np.asarray(neural, dtype=float)
    if position.ndim != 2 or neural.ndim != 2:
        raise ValueError("position and neural must both be two-dimensional")
    if len(position) != len(neural) or len(position) != len(meta):
        raise ValueError("position, neural, and meta must have matching rows")
    if n_phase < 2:
        raise ValueError("n_phase must be at least 2")

    # Phase 0 is movement start and phase 1 is peak movement. Using one target
    # grid lets trials with different durations be compared point by point.
    target_phase = np.linspace(0.0, 1.0, n_phase)
    trial_ids: list[object] = []
    position_trials: list[np.ndarray] = []
    neural_trials: list[np.ndarray] = []
    for trial, idx in meta.groupby("trial_number", sort=False).indices.items():
        idx = np.asarray(idx)
        if len(idx) < 3:
            continue

        # The input is sampled in equal-width time bins. Normalizing sample
        # indices to [0, 1] therefore gives an equally spaced source grid.
        source_phase = np.linspace(0.0, 1.0, len(idx))

        # For every target phase, np.interp takes a distance-weighted average
        # of the two neighboring source points. Coordinates are independent.
        position_trials.append(
            np.column_stack(
                [
                    np.interp(target_phase, source_phase, position[idx, d])
                    for d in range(position.shape[1])
                ]
            )
        )

        # Neural activity is smoothed first, then every unit is interpolated
        # independently on exactly the same target-phase grid as position.
        neural_trials.append(
            np.column_stack(
                [
                    np.interp(target_phase, source_phase, neural[idx, d])
                    for d in range(neural.shape[1])
                ]
            )
        )
        trial_ids.append(trial)

    if len(trial_ids) < 2:
        raise RuntimeError("fewer than two usable trials after phase resampling")
    return (
        np.asarray(trial_ids, dtype=object),
        np.stack(position_trials, axis=0),
        np.stack(neural_trials, axis=0),
    )


def pairwise_mean_squared_difference(stack: np.ndarray) -> np.ndarray:
    """Return one mean-squared trajectory difference per unique trial pair."""
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("stack must have shape (trial, phase, feature)")
    if stack.shape[0] < 2:
        raise ValueError("at least two trials are required")
    # Flatten phase and feature axes. Squared Euclidean distance sums squared
    # differences; division by vector length turns it into a mean squared
    # trajectory difference for each unique trial pair.
    flat = stack.reshape(stack.shape[0], -1)
    return pdist(flat, metric="sqeuclidean") / flat.shape[1]


def load_day(session: str, exclude: tuple[int, ...] | list[int], bin_ms: int):
    """Load aligned shoulder-centred position and smoothed neural spike counts."""
    bin_s = bin_ms / 1000.0
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_s
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        position, neural, meta = du.build_decoder_dataset(
            nwb,
            reach,
            "relative_position",
            bin_size=bin_s,
            unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS,
            trial_window="start_to_peak",
            **SMOOTHER_KW,
        )
    finally:
        io.close()
    # Remove known invalid trials before constructing any trial pairs.
    position, neural, meta = filter_trials(position, neural, meta, exclude)

    # Smooth neural activity within trials before treating it as a continuous
    # trajectory during phase interpolation.
    neural_smoothed = du.smooth_neural_causal(
        neural,
        meta,
        sigma_bins=SMOOTH_SIGMA_MS / bin_ms,
    )
    return position, neural_smoothed, meta


def compute_day(
    session: str,
    epoch: str,
    exclude: tuple[int, ...] | list[int] = (),
    bin_ms: int = BIN_MS,
    n_phase: int = N_PHASE,
) -> tuple[pd.DataFrame, dict]:
    """Compute all neural and position pair values for one session."""
    position, neural, meta = load_day(session, exclude, bin_ms)
    trial_ids, position_trials, neural_trials = phase_resample_joint(
        position, neural, meta, n_phase=n_phase
    )
    # With n trials, pdist returns n(n-1)/2 unique upper-triangle pairs.
    neural_values = pairwise_mean_squared_difference(neural_trials)
    position_values = pairwise_mean_squared_difference(position_trials)
    pair_i, pair_j = np.triu_indices(len(trial_ids), k=1)
    if not (len(pair_i) == len(neural_values) == len(position_values)):
        raise RuntimeError("internal pair-order mismatch")

    date = session_date(session)
    pairs = pd.DataFrame(
        {
            "animal": session[:2],
            "session": session,
            "date": date,
            "epoch": epoch.upper(),
            "trial_i": trial_ids[pair_i],
            "trial_j": trial_ids[pair_j],
            "neural_pair_msd": neural_values,
            "position_pair_msd": position_values,
        }
    )
    summary: dict[str, float | int | str] = {
        "animal": session[:2],
        "session": session,
        "date": date,
        "epoch": epoch.upper(),
        "n_trials": len(trial_ids),
        "n_pairs": len(pairs),
        "n_units": neural_trials.shape[-1],
        "bin_ms": bin_ms,
        "n_phase": n_phase,
        "excluded_trials": ",".join(map(str, exclude)),
    }
    for prefix, values in (
        ("neural", neural_values),
        ("position", position_values),
    ):
        summary[f"{prefix}_mean"] = float(np.mean(values))
        summary[f"{prefix}_sd"] = float(np.std(values, ddof=1))
        summary[f"{prefix}_q05"] = float(np.quantile(values, 0.05))
        summary[f"{prefix}_median"] = float(np.median(values))
        summary[f"{prefix}_q95"] = float(np.quantile(values, 0.95))
    return pairs, summary


def _finite_log10(values: np.ndarray) -> np.ndarray:
    """Log-transform positive finite values for a comparable density axis."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        raise ValueError("density values contain no positive finite observations")
    # Log10 is only for display; saved pairwise MSD values stay untransformed.
    return np.log10(values)


def _draw_density(ax, log_values: np.ndarray, grid: np.ndarray, color: str):
    """Draw a KDE, with a stable fallback for nearly constant values."""
    if len(log_values) >= 2 and np.std(log_values) > 1e-12:
        density = gaussian_kde(log_values)(grid)
        ax.plot(grid, density, color=color, linewidth=1.4)
        ax.fill_between(grid, 0.0, density, color=color, alpha=0.25)
        ax.set_ylim(0.0, density.max() * 1.12)
    else:
        ax.axvline(log_values[0], color=color, linewidth=1.4)
    ax.axvline(np.median(log_values), color="black", linestyle="--", linewidth=0.8)


def make_density_figure(pairs: pd.DataFrame, daily: pd.DataFrame, out: Path):
    """Plot one neural and one position density for every session/day."""
    order = daily.sort_values(["date", "epoch"])["session"].tolist()
    metric_specs = (
        ("neural_pair_msd", "Neural trial-pair variability"),
        ("position_pair_msd", "Position trial-pair variability"),
    )
    pooled = {
        metric: _finite_log10(pairs[metric].to_numpy())
        for metric, _ in metric_specs
    }
    grids = {}
    for metric, _ in metric_specs:
        values = pooled[metric]
        lo, hi = np.quantile(values, [0.001, 0.999])
        margin = max(0.05 * (hi - lo), 0.02)
        grids[metric] = np.linspace(lo - margin, hi + margin, 400)

    fig, axes = plt.subplots(
        len(order),
        2,
        figsize=(12.5, max(15.0, 1.35 * len(order))),
        sharex="col",
        squeeze=False,
    )
    colors = {"R1": "#4C78A8", "R2": "#F58518"}
    for row, session in enumerate(order):
        day_pairs = pairs.loc[pairs["session"] == session]
        day_info = daily.loc[daily["session"] == session].iloc[0]
        epoch = str(day_info["epoch"])
        color = colors.get(epoch, "#777777")
        for col, (metric, title) in enumerate(metric_specs):
            ax = axes[row, col]
            values = _finite_log10(day_pairs[metric].to_numpy())
            _draw_density(ax, values, grids[metric], color)
            ax.set_xlim(grids[metric][0], grids[metric][-1])
            ax.set_yticks([])
            ax.grid(alpha=0.18, axis="x")
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=12, weight="bold")
            if col == 0:
                ax.set_ylabel(
                    f"{day_info['date']}  {epoch}\n"
                    f"{int(day_info['n_trials'])} trials",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    labelpad=8,
                )
            if row < len(order) - 1:
                ax.tick_params(labelbottom=False)

    axes[-1, 0].set_xlabel(r"$\log_{10}$ mean squared neural difference")
    axes[-1, 1].set_xlabel(r"$\log_{10}$ mean squared position difference")
    fig.suptitle(
        "Within-day trial-to-trial variability distributions\n"
        "one value per unique trial pair; dashed line = daily median",
        fontsize=14,
        y=0.998,
    )
    fig.text(
        0.99,
        0.003,
        "R1 = blue, R2 = orange. Pair values share trials and are descriptive, not independent samples.",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0.12, 0.02, 1.0, 0.985), h_pad=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animal",
        choices=sorted(ANIMAL_SESSIONS),
        default="TS",
        help="animal/session set to process (default: TS)",
    )
    parser.add_argument("--bin-ms", type=int, default=BIN_MS)
    parser.add_argument("--n-phase", type=int, default=N_PHASE)
    parser.add_argument(
        "--include-known-outliers",
        action="store_true",
        help="do not apply the project's EXCLUDE_TRIALS mapping",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.bin_ms <= 0:
        raise ValueError("--bin-ms must be positive")
    if args.n_phase < 2:
        raise ValueError("--n-phase must be at least 2")

    r1_sessions, r2_sessions = ANIMAL_SESSIONS[args.animal]
    all_pairs = []
    daily_rows = []
    for epoch, sessions in (("R1", r1_sessions), ("R2", r2_sessions)):
        for session in sessions:
            exclude = () if args.include_known_outliers else EXCLUDE_TRIALS.get(session, ())
            print(f"Computing {epoch} {session_date(session)} ...", flush=True)
            pairs, summary = compute_day(
                session,
                epoch,
                exclude=exclude,
                bin_ms=args.bin_ms,
                n_phase=args.n_phase,
            )
            all_pairs.append(pairs)
            daily_rows.append(summary)
            print(
                f"  {summary['n_trials']} trials, {summary['n_pairs']} pairs, "
                f"neural median={summary['neural_median']:.6g}, "
                f"position median={summary['position_median']:.6g}",
                flush=True,
            )

    pairs_df = pd.concat(all_pairs, ignore_index=True)
    daily_df = pd.DataFrame(daily_rows).sort_values(["date", "epoch"])
    stem = args.animal.upper()
    pairs_out = OUT_DIR / f"trial_pair_variability_{stem}_pairs.csv"
    daily_out = OUT_DIR / f"trial_pair_variability_{stem}_daily.csv"
    figure_out = OUT_DIR / "figures" / f"fig_trial_pair_variability_density_{stem}.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(pairs_out, index=False)
    daily_df.to_csv(daily_out, index=False)
    make_density_figure(pairs_df, daily_df, figure_out)
    print(f"Saved {pairs_out}")
    print(f"Saved {daily_out}")
    print(f"Saved {figure_out}")


if __name__ == "__main__":
    main()
