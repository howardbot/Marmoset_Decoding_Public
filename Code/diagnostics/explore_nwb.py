
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend; we save figures, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import ndx_pose  # noqa: F401  registers pose extension so NWB can read it

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from decoder_utils import reach_marker_names, reach_side_for_session

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
FIG_DIR = REPO_ROOT / "Results" / "archive" / "legacy" / "diagnostics" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SESSION = "TSAL20250812_0830_staticAndStaticFree001"
PROCESSED_NWB = DATA_DIR / f"{SESSION}_processed.nwb"
ACQUISITION_NWB = DATA_DIR / f"{SESSION}_acquisition.nwb"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


_POSE_MISMATCH_WARNED: set[str] = set()


def _pose_data_and_timestamps(series, label: str | None = None):
    """Return pose data/timestamps with matching first-axis lengths."""
    data = series.data[:]
    ts = series.timestamps[:]
    if data.shape[0] != ts.shape[0]:
        n = min(data.shape[0], ts.shape[0])
        prefix = f"{label}: " if label else ""
        warn_key = label or str(id(series))
        if warn_key not in _POSE_MISMATCH_WARNED:
            print(f"  WARNING: {prefix}data/timestamps length mismatch "
                  f"({data.shape[0]} vs {ts.shape[0]}); using first {n} samples")
            _POSE_MISMATCH_WARNED.add(warn_key)
        data = data[:n]
        ts = ts[:n]
    return data, ts


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def summarize_intervals(nwb_prc) -> dict:
    """List every interval table and its length."""
    print_header("Intervals")
    intervals = dict(nwb_prc.intervals)
    for name, tbl in intervals.items():
        print(f"  {name:40s} n = {len(tbl)}")
    return intervals


def pick_reach_interval(intervals: dict):
    """Return (name, table) of the trial-period interval table.

    The NWB guide uses 'reaching_segments_static'; we match by prefix in
    case the suffix differs across sessions.
    """
    candidates = [n for n in intervals if "reaching_segments" in n]
    if not candidates:
        raise RuntimeError(
            f"No reaching_segments_* interval found. Available: {list(intervals)}"
        )
    name = candidates[0]
    if len(candidates) > 1:
        print(f"\nMultiple reach interval tables found, picking '{name}': {candidates}")
    else:
        print(f"\nUsing reach intervals: '{name}'  (n = {len(intervals[name])})")
    return name, intervals[name]


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------
def summarize_trials(reach_tbl) -> None:
    """Print trial count, duration stats, and ITI stats."""
    print_header("Trials")
    starts = np.asarray(reach_tbl.start_time[:])
    stops = np.asarray(reach_tbl.stop_time[:])
    durations = stops - starts
    itis = starts[1:] - stops[:-1]

    print(f"  N trials                 = {len(starts)}")
    print(f"  duration (s)   mean      = {durations.mean():.3f}")
    print(f"                 median    = {np.median(durations):.3f}")
    print(f"                 min / max = {durations.min():.3f} / {durations.max():.3f}")
    if itis.size:
        print(f"  inter-trial (s) median   = {np.median(itis):.3f}")
    df = reach_tbl.to_dataframe()
    print(f"  columns                  = {list(df.columns)}")


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
def summarize_units(nwb_prc):
    """Print unit count and firing-rate stats. Return (units_df, fr_hz)."""
    print_header("Units")
    units = nwb_prc.units.to_dataframe()
    print(f"  N units      = {len(units)}")
    print(f"  unit columns = {list(units.columns)}")

    # Per-unit firing rate, estimated over the unit's own spike-time span.
    # This is a quick first look — not yet trial-restricted.
    fr = np.full(len(units), np.nan)
    for i, spikes in enumerate(units.spike_times):
        spikes = np.asarray(spikes)
        if spikes.size >= 2:
            fr[i] = spikes.size / (spikes.max() - spikes.min())

    print(f"  firing rate (Hz)  median = {np.nanmedian(fr):.2f}")
    print(f"                    range  = {np.nanmin(fr):.2f}  -  {np.nanmax(fr):.2f}")
    return units, fr
def inspect_qc_columns(units, reach_tbl) -> None:
    """Print distinct values / stats of the columns we'll use for QC.

    Categorical columns (quality, channel_group, result) -> value_counts.
    Numeric columns (amp) -> describe.
    ID columns (original_cluster_id) -> nunique.

    Output guides our thresholds for filter_units() and filter_trials()
    later — e.g. which `quality` value means 'good', which `result`
    value means 'success'.
    """
    print_header("QC columns — distinct values")
    trials_df = reach_tbl.to_dataframe()

    # ---- categorical: which values exist, and how many of each? ----
    categorical = {
        "quality":       units,
        "channel_group": units,
        "result":        trials_df,
    }
    for col, df in categorical.items():
        print(f"\n[{col}]")
        if col not in df.columns:
            print("  (column not present)")
            continue
        print(df[col].value_counts(dropna=False).to_string())

    # ---- numeric: distribution stats ----
    print(f"\n[amp]")
    print(units["amp"].describe().to_string())

    # ---- id: just count distinct ----
    print(f"\n[original_cluster_id]")
    print(f"  n unique = {units['original_cluster_id'].nunique()}  "
          f"(of {len(units)} units)")

def plot_firing_rates(fr: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(fr[~np.isnan(fr)], bins=30, color="steelblue", edgecolor="k")
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Units")
    ax.set_title("Firing rate distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Array layout (PMd / M1 split lives on the x or y axis of the array)
# ---------------------------------------------------------------------------
def plot_array_layout(nwb_prc, units, save_path: Path) -> None:
    """Scatter electrode (x, y). Color electrodes that host >=1 sorted unit."""
    elecs = nwb_prc.electrodes.to_dataframe()
    xs = elecs["x"].to_numpy()
    ys = elecs["y"].to_numpy()
    has_unit = np.isin(
        elecs["electrode_label"].to_numpy(),
        units["electrode_label"].to_numpy(),
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs[~has_unit], ys[~has_unit], s=40, c="lightgray",
               edgecolors="k", label="no unit")
    ax.scatter(xs[has_unit], ys[has_unit], s=40, c="crimson",
               edgecolors="k", label=">=1 unit")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Array layout - electrodes with sorted units")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def plot_wrist_trajectories(
    nwb_prc,
    reach_tbl,
    save_path: Path,
) -> None:
    """Overlay wrist trajectory for every trial.

    The dominant wrist is selected from the session's animal code, matching the
    decoder path (TS is right-handed; TY is left-handed).
    """
    df = reach_tbl.to_dataframe()
    fig, ax = plt.subplots(figsize=(5, 5))
    plotted, marker_used = 0, None
    expected_marker = reach_marker_names(SESSION)[0]

    for _, row in df.iterrows():
        mod = nwb_prc.processing[row.kinematics_module]
        pose = mod.data_interfaces[row.video_event].pose_estimation_series
        if expected_marker not in pose:
            continue
        marker = expected_marker
        marker_used = marker
        pos, ts = _pose_data_and_timestamps(
            pose[marker], f"{row.video_event}/{marker}"
        )
        mask = (ts >= row.start_time) & (ts <= row.stop_time)
        if mask.sum() < 2:
            continue
        ax.plot(pos[mask, 0], pos[mask, 1], lw=0.6, alpha=0.5)
        plotted += 1

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Wrist trajectories ({marker_used}, n = {plotted} trials)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Acquisition file (raw electrical / LFP) — just confirm we can open it
# ---------------------------------------------------------------------------
def peek_acquisition() -> None:
    print_header("Acquisition file (raw electrical)")
    if not ACQUISITION_NWB.exists():
        print(f"  not found: {ACQUISITION_NWB}")
        return
    with NWBHDF5IO(ACQUISITION_NWB, mode="r") as io:
        nwb_acq = io.read()
        es = nwb_acq.acquisition["ElectricalSeries"]
        # Lazy access — does NOT load 3.7 GB into RAM.
        print(f"  ElectricalSeries.data.shape = {es.data.shape}")
        print(f"  sampling rate (Hz)          = {es.rate}")
        print(f"  starting_time (s)           = {es.starting_time}")

def inspect_kinematics(nwb_prc, reach_tbl) -> None:
    """Inspect the kinematics module structure and wrist trajectory.

    Reports:
      - which processing modules hold kinematics
      - what body-part markers are tracked
      - data dimensionality (2D vs 3D)
      - effective sampling rate
      - NaN fraction in the wrist marker (DLC drops low-confidence frames)
      - sample trial: shape of x/y vs time
    """
    print_header("Kinematics")

    df = reach_tbl.to_dataframe()

    # ---- which processing modules / video events appear ----
    mod_names = df["kinematics_module"].unique()
    vid_events = df["video_event"].unique()
    print(f"  kinematics modules        = {list(mod_names)}")
    print(f"  video_events in trials    = {list(vid_events)}")

    # ---- pick first trial's pose object to introspect markers ----
    row0 = df.iloc[0]
    mod = nwb_prc.processing[row0.kinematics_module]
    pose = mod.data_interfaces[row0.video_event].pose_estimation_series
    markers = list(pose.keys())
    print(f"  tracked markers (n={len(markers)}) = {markers}")

    # ---- inspect the session's dominant wrist marker ----
    marker_name = reach_marker_names(SESSION)[0]
    if marker_name not in pose:
        marker_name = markers[0]
    wrist = pose[marker_name]
    data, ts = _pose_data_and_timestamps(
        wrist, f"{row0.video_event}/{marker_name}"
    )
    print(f"\n  marker '{marker_name}' summary "
          f"(video_event '{row0.video_event}', not trial-restricted):")
    print(f"    data shape            = {data.shape}    "
          f"(T frames x D dims; D=2 -> 2D pixel, D=3 -> 3D mm)")
    print(f"    timestamps shape      = {ts.shape}")
    print(f"    duration              = {ts[-1] - ts[0]:.2f} s")
    print(f"    effective sample rate = {len(ts) / (ts[-1] - ts[0]):.2f} Hz")
    print(f"    NaN fraction          = {np.isnan(data).any(axis=1).mean():.3%}")
    print(f"    x range (units?)      = [{np.nanmin(data[:, 0]):.2f}, "
          f"{np.nanmax(data[:, 0]):.2f}]")
    print(f"    y range               = [{np.nanmin(data[:, 1]):.2f}, "
          f"{np.nanmax(data[:, 1]):.2f}]")

    # ---- per-trial: how many samples land in each trial, NaN rate inside trials ----
    n_per_trial, nan_in_trial = [], []
    for _, row in df.iterrows():
        mod = nwb_prc.processing[row.kinematics_module]
        pose = mod.data_interfaces[row.video_event].pose_estimation_series
        if marker_name not in pose:
            n_per_trial.append(0)
            continue
        data, ts = _pose_data_and_timestamps(
            pose[marker_name], f"{row.video_event}/{marker_name}"
        )
        m = (ts >= row.start_time) & (ts <= row.stop_time)
        n_per_trial.append(m.sum())
        if m.sum() > 0:
            nan_in_trial.append(np.isnan(data[m]).any(axis=1).mean())
    n_per_trial = np.array(n_per_trial)
    nan_in_trial = np.array(nan_in_trial)
    print(f"\n  per-trial samples       median = {int(np.median(n_per_trial))},  "
          f"min/max = {n_per_trial.min()}/{n_per_trial.max()}")
    if len(nan_in_trial):
        print(f"  per-trial NaN fraction  median = {np.median(nan_in_trial):.3%},  "
              f"max = {nan_in_trial.max():.3%}")
    else:
        print("  per-trial NaN fraction  no kinematic samples found in trials")


# ---------------------------------------------------------------------------
# Handedness check
# ---------------------------------------------------------------------------
def _trial_path_length(
    pose, marker: str, t0: float, t1: float, label: str | None = None
) -> float:
    """Total 3D path length of a marker during [t0, t1]. NaN-safe."""
    if marker not in pose:
        return np.nan
    data, ts = _pose_data_and_timestamps(pose[marker], label or marker)
    m = (ts >= t0) & (ts <= t1)
    pos = data[m]
    if pos.shape[0] < 2:
        return np.nan
    # frame-to-frame displacement, skip rows with any NaN
    diffs = np.diff(pos, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return float(np.nansum(seg))


def inspect_handedness(nwb_prc, reach_tbl) -> None:
    """Check which arm is dominant by comparing l-wrist vs r-wrist path length.

    For each trial: total 3D displacement of l-wrist and r-wrist during the
    trial window. The dominant arm should move more on most trials and
    almost certainly on success (S) trials. WA ('wrong arm') trials are
    the key sanity check: they should favor the non-dominant wrist if our
    interpretation of WA is correct.

    Reports:
      - overall: how often each wrist 'wins' (greater path length)
      - per-result breakdown (S / F / WA / N): mean path length each wrist,
        and 'wins' counts.
    """
    print_header("Handedness — l-wrist vs r-wrist path length per trial")

    df = reach_tbl.to_dataframe().copy()
    # peak_extension_times is stored as a string per trial (NWB serialization
    # quirk). Coerce to float once up front so we can use it as a numeric
    # endpoint of the reach-extension window [start_time, peak_time].
    df["peak_t"] = pd.to_numeric(df["peak_extension_times"], errors="coerce")

    l_path = np.full(len(df), np.nan)
    r_path = np.full(len(df), np.nan)

    for i, row in enumerate(df.itertuples(index=False)):
        if not np.isfinite(row.peak_t):
            continue  # missing peak -> leave NaN
        mod = nwb_prc.processing[row.kinematics_module]
        pose = mod.data_interfaces[row.video_event].pose_estimation_series
        l_path[i] = _trial_path_length(
            pose, "l-wrist", row.start_time, row.peak_t,
            f"{row.video_event}/l-wrist",
        )
        r_path[i] = _trial_path_length(
            pose, "r-wrist", row.start_time, row.peak_t,
            f"{row.video_event}/r-wrist",
        )

    df["l_path_mm"] = l_path
    df["r_path_mm"] = r_path
    # Strict inequality both ways; ties and NaNs end up in neither bucket
    # (instead of silently being credited to r-wrist).
    valid = ~(np.isnan(l_path) | np.isnan(r_path))
    df["l_wins"] = valid & (l_path > r_path)
    df["r_wins"] = valid & (r_path > l_path)

    # ---- overall ----
    n_valid = int((~np.isnan(l_path) & ~np.isnan(r_path)).sum())
    n_ties = n_valid - int(df["l_wins"].sum() + df["r_wins"].sum())
    print(f"  trials with both wrists tracked = {n_valid} / {len(df)}")
    print(f"  l-wrist wins (more movement)    = {int(df['l_wins'].sum())} trials")
    print(f"  r-wrist wins                    = {int(df['r_wins'].sum())} trials")
    if n_ties:
        print(f"  ties / invalid                  = {n_ties} trials")
    print(f"  median l-wrist path (mm) = {np.nanmedian(l_path):.2f}")
    print(f"  median r-wrist path (mm) = {np.nanmedian(r_path):.2f}")

    # ---- per-result breakdown ----
    print("\n  per-result breakdown:")
    print(f"  {'result':<8} {'n':>4} {'l_mean':>8} {'r_mean':>8}  {'l_wins':>7} / n")
    for res, sub in df.groupby("result"):
        n = len(sub)
        l_mean = sub["l_path_mm"].mean()
        r_mean = sub["r_path_mm"].mean()
        wins = int(sub["l_wins"].sum())
        print(f"  {res:<8} {n:>4} {l_mean:>8.2f} {r_mean:>8.2f}  {wins:>7} / {n}")

    # ---- explicit WA inspection ----
    wa = df[df["result"] == "WA"]
    if len(wa):
        dominant_side = reach_side_for_session(SESSION)
        wrong_side = "l" if dominant_side == "r" else "r"
        print(
            f"\n  WA trials (expect {wrong_side}-wrist > {dominant_side}-wrist "
            "if 'wrong arm' = non-dominant):"
        )
        for _, row in wa.iterrows():
            wrong_path = row[f"{wrong_side}_path_mm"]
            dominant_path = row[f"{dominant_side}_path_mm"]
            tag = (
                f"{wrong_side} > {dominant_side} ✓"
                if wrong_path > dominant_path
                else f"{dominant_side} > {wrong_side} ✗"
            )
            print(f"    t={row.start_time:7.2f}s  "
                  f"l={row.l_path_mm:6.2f}  r={row.r_path_mm:6.2f}   {tag}")

    # ---- explicit S inspection (Success trials) ----
    successes = df[df["result"] == "S"]
    if len(successes):
        print("\n  Success (S) trials - determining 'good' wrist per trial:")
        for _, row in successes.iterrows():
            # The 'good' wrist is the one with the longer path length
            if row.l_path_mm > row.r_path_mm:
                good_wrist = "l-wrist"
            else:
                good_wrist = "r-wrist"

            print(f"    Trial start t={row.start_time:7.2f}s | "
                  f"l_path={row.l_path_mm:6.2f} | r_path={row.r_path_mm:6.2f} | "
                  f"Good wrist: {good_wrist}")


def check_grab_timing_simple(reach_tbl) -> None:
    """Verify that trial stop_time aligns with the reach peak extension.

    Expectation: peak_extension_times typically slightly precedes stop_time
    (arm extends -> peak -> retracts -> trial ends). A negative median
    diff is normal. A positive diff would be suspicious (peak after trial
    ended -> mis-segmentation).

    Note: peak_extension_times may be an array per trial (multiple
    sub-reaches inside one segment). When it is, we use the last peak as
    the 'final grasp' moment.
    """
    print_header("Timing Check — peak_extension vs stop_time")

    df = reach_tbl.to_dataframe()
    col = "peak_extension_times"
    if col not in df.columns:
        print(f"  Column '{col}' not found. Available: {list(df.columns)}")
        return

    # ---- detect storage format ----
    # NWB stores some "times" columns as strings (single value per trial,
    # serialized as a float-as-string). Other columns might genuinely be
    # arrays. Detect which we have.
    sample = df[col].iloc[0]
    is_string = isinstance(sample, (str, bytes))
    is_array = hasattr(sample, "__len__") and not is_string
    if is_string:
        kind = "string per trial -> parsing to float"
    elif is_array:
        kind = "array per trial -> using last peak"
    else:
        kind = "scalar"
    print(f"  '{col}' value type = {type(sample).__name__}, {kind}")
    if is_array:
        n_peaks = df[col].apply(len)
        print(f"  peaks per trial: median = {int(n_peaks.median())}, "
              f"min/max = {n_peaks.min()}/{n_peaks.max()}")

    # ---- raw value diagnostic: explicit type/shape/repr for first 3 trials ----
    print("\n  Raw value diagnostic for first 3 trials:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        peak_raw = row[col]
        idx_raw = row["peak_extension_idxs"] if "peak_extension_idxs" in df.columns else None
        print(f"  --- trial {i} (start={row.start_time:.3f}, stop={row.stop_time:.3f}) ---")
        print(f"    {col}:")
        print(f"      type   = {type(peak_raw).__name__}")
        if hasattr(peak_raw, "dtype"):
            print(f"      dtype  = {peak_raw.dtype}")
        if hasattr(peak_raw, "shape"):
            print(f"      shape  = {peak_raw.shape}")
        elif hasattr(peak_raw, "__len__"):
            print(f"      len    = {len(peak_raw)}")
        print(f"      repr   = {peak_raw!r}")
        if idx_raw is not None:
            print(f"    peak_extension_idxs:")
            print(f"      type   = {type(idx_raw).__name__}, repr = {idx_raw!r}")

    # ---- restrict to success trials ----
    successes = df[df["result"] == "S"].copy()
    if successes.empty:
        print("  No success (S) trials found.")
        return

    # ---- pick scalar peak per trial, then diff against stop_time ----
    if is_array:
        peak = successes[col].apply(lambda x: x[-1]).astype(float)
    elif is_string:
        # column is a string-encoded float (NWB serialization quirk)
        peak = pd.to_numeric(successes[col], errors="coerce")
    else:
        peak = successes[col].astype(float)
    successes["peak"] = peak.values
    successes["time_diff"] = successes["peak"] - successes["stop_time"]

    # ---- preview ----
    print("\n  Preview of first 5 S-trials:")
    for _, row in successes.head(5).iterrows():
        print(f"    start={row.start_time:7.2f}s  "
              f"stop={row.stop_time:7.2f}s  "
              f"peak={row.peak:7.2f}s  "
              f"diff={row.time_diff:+.3f}s")

    # ---- summary ----
    med = successes["time_diff"].median()
    iqr_lo, iqr_hi = successes["time_diff"].quantile([0.25, 0.75])
    # ~5 video frames at 150 Hz ≈ 33 ms — treat anything within that as
    # frame-level alignment.
    frame_tol = 5 / 150.0
    print(f"\n  Median (peak - stop) = {med:+.3f} s   "
          f"IQR = [{iqr_lo:+.3f}, {iqr_hi:+.3f}]")

    if abs(med) <= frame_tol:
        print(f"  -> stop_time effectively equals peak extension "
              f"(within {frame_tol*1000:.0f} ms = ~5 video frames).")
    elif med < 0:
        print("  -> Peak extension occurs BEFORE stop_time (expected: "
              "arm extends, peaks, retracts, then trial ends).")
    else:
        print("  -> Peak extension occurs AFTER stop_time — suspicious, "
              "may indicate mis-segmentation.")
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Repo:    {REPO_ROOT}")
    print(f"Session: {SESSION}")
    if not PROCESSED_NWB.exists():
        raise FileNotFoundError(PROCESSED_NWB)

    with NWBHDF5IO(PROCESSED_NWB, mode="r") as io_prc:
        nwb_prc = io_prc.read()

        # ---- session metadata ----
        print_header("Session metadata")
        print(f"  identifier    = {nwb_prc.identifier}")
        print(f"  session_start = {nwb_prc.session_start_time}")
        print(f"  subject       = {nwb_prc.subject}")

        # ---- structure discovery ----
        intervals = summarize_intervals(nwb_prc)
        _, reach_tbl = pick_reach_interval(intervals)

        # ---- per-table summaries ----
        summarize_trials(reach_tbl)
        units, fr = summarize_units(nwb_prc)
        inspect_qc_columns(units, reach_tbl)
        inspect_kinematics(nwb_prc, reach_tbl)
        inspect_handedness(nwb_prc, reach_tbl)
        check_grab_timing_simple(reach_tbl)

        # ---- QC plots ----
        print_header("Plots")
        plot_firing_rates(fr, FIG_DIR / "firing_rate_dist.png")
        plot_array_layout(nwb_prc, units, FIG_DIR / "array_layout.png")
        plot_wrist_trajectories(nwb_prc, reach_tbl,
                                FIG_DIR / "wrist_trajectories.png")

    peek_acquisition()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
