from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import ndx_pose
from scipy.signal import savgol_filter, butter, sosfiltfilt


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data"
SESSION = "TSAL20250813_0830_staticAndStaticFree001"
PROCESSED_NWB = DATA_DIR / f"{SESSION}_processed.nwb"


TARGET_MODES = [
    "relative_position",
    "relative_velocity",
    "relative_position_velocity_acceleration",
]

UNIT_QUALITY_SETS = [
    ("good",),
    ("mua",),
    ("good", "mua"),
]

LAG_BINS = [-4, -3, -2, -1, 0, 1, 2, 3, 4] # not for Kalman

BIN_SIZE_SECONDS = 0.01


REACH_SIDE_BY_ANIMAL = {
    "TS": "r",
    "TY": "l",
}


def reach_side_for_session(session_tag):
    """Return the reaching-arm side for a session identifier.

    Session identifiers begin with the two-letter animal code. Tria (TS) is
    right-handed, whereas Tony (TY) is left-handed. Unknown animals fail
    explicitly so a new subject cannot silently be decoded from the wrong arm.
    """
    animal = str(session_tag).strip().upper()[:2]
    try:
        return REACH_SIDE_BY_ANIMAL[animal]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reaching side for session {session_tag!r}; "
            f"known animal codes are {sorted(REACH_SIDE_BY_ANIMAL)}"
        ) from exc


def reach_marker_names(session_tag):
    """Return the wrist and shoulder marker names for a session."""
    side = reach_side_for_session(session_tag)
    return f"{side}-wrist", f"{side}-shoulder"


def quality_label(qualities):
    """Convert a tuple of unit-quality labels into a stable display string."""
    return "+".join(qualities)


def state_names(target_mode):
    """Return the ordered state names produced by a given decoding target mode."""
    if target_mode == "relative_position":
        return ["rel_x", "rel_y", "rel_z"]
    if target_mode == "relative_velocity":
        return ["vel_x", "vel_y", "vel_z"]
    if target_mode == "relative_position_velocity_acceleration":
        return [
            "rel_x", "rel_y", "rel_z",
            "vel_x", "vel_y", "vel_z",
            "acc_x", "acc_y", "acc_z",
        ]
    raise ValueError(f"Unknown target_mode: {target_mode}")


def r2_score_1d(y_true, y_pred):
    """Compute scalar R^2 for one output dimension."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def r2_table(y_true, y_pred, names, prefix):
    """Map each output dimension to an R^2 entry using a shared name prefix."""
    return {f"{prefix}_{name}": r2_score_1d(y_true[:, i], y_pred[:, i])
            for i, name in enumerate(names)}


def print_top_results(df, n=8):
    print("\nTop settings")
    print("=" * 78)
    print(df.head(n).to_string(index=False))


def print_best_summary(rows):
    print("\n" + "=" * 78)
    print("Best summary")
    print("=" * 78)
    print(pd.DataFrame(rows).to_string(index=False))


def load_nwb_and_reach():
    """Open the processed NWB file and return the reach interval table used downstream.

    The project assumes one processed NWB per session and one interval table whose
    name contains ``reaching_segments``. That table defines the trial windows used
    later for extracting both kinematics and neural activity.
    """
    # Open the processed NWB once and find the interval table that stores reach segments.
    if not PROCESSED_NWB.exists():
        raise FileNotFoundError(PROCESSED_NWB)
    io = NWBHDF5IO(PROCESSED_NWB, mode="r")
    nwb_prc = io.read()
    intervals = dict(nwb_prc.intervals)
    candidates = [name for name in intervals if "reaching_segments" in name]
    if not candidates:
        io.close()
        raise RuntimeError(f"No reaching_segments interval found: {list(intervals)}")
    name = candidates[0]
    print(f"Repo:    {REPO_ROOT}")
    print(f"Session: {SESSION}")
    print(f"NWB:     {PROCESSED_NWB}")
    print(f"Using reach interval: {name}  n={len(intervals[name])}")
    return io, nwb_prc, intervals[name]


def get_unit_spike_times(nwb_prc, qualities=("good", "mua")):
    """Extract spike times for the requested unit classes.

    Returns a Python list where each element is a 1D numpy array of spike times for
    one unit. The order of this list defines the column order used later in the
    binned spike-count matrix.
    """
    # Keep only the requested unit classes and return one spike-time array per unit.
    units = nwb_prc.units.to_dataframe()
    selected = units[units["quality"].isin(qualities)].copy()
    spike_times = [np.asarray(spikes, dtype=float) for spikes in selected["spike_times"]]
    print(f"Using units {qualities}: {len(selected)} / {len(units)}")
    return spike_times


_POSE_MISMATCH_WARNED = set()


def load_pose_series(series, session_tag=""):
    """Read a pose-estimation series' data and timestamps, trimming to a common length.

    Several processed NWB files in this project have more pose data rows than
    timestamps (anipose/DLC writes one row per camera frame but a few trailing
    timestamps are missing). NumPy fancy-indexing then fails when a boolean mask
    derived from timestamps is applied to data. We trim both arrays to
    ``min(len(data), len(timestamps))`` and warn once per (session, series) pair.
    """
    data = np.asarray(series.data[:])
    ts = np.asarray(series.timestamps[:])
    nd, nt = data.shape[0], ts.shape[0]
    if nd != nt:
        key = (session_tag, getattr(series, "name", id(series)))
        if key not in _POSE_MISMATCH_WARNED:
            _POSE_MISMATCH_WARNED.add(key)
            print(
                f"[load_pose_series] {session_tag} {getattr(series, 'name', '?')}: "
                f"data={nd} timestamps={nt} -> trimming both to {min(nd, nt)} "
                f"(diff={nd - nt})"
            )
        n = min(nd, nt)
        data, ts = data[:n], ts[:n]
    return data, ts


def interpolate_marker_to_bins(marker_pos, marker_ts, bin_centers):
    """Linearly interpolate marker positions onto decoder bin centers.

    Parameters
    ----------
    marker_pos
        Raw marker positions with shape ``(time, 3)``.
    marker_ts
        Absolute timestamps for each raw marker sample.
    bin_centers
        Target timestamps where the decoder wants one position estimate per bin.

    Returns
    -------
    np.ndarray
        Array with shape ``(len(bin_centers), 3)`` containing interpolated x/y/z
        positions. Times outside the observed marker range are left as NaN rather
        than extrapolated.
    """
    # Resample marker positions onto decoder bin centers so kinematics and spikes share a time base.
    marker_pos = np.asarray(marker_pos, dtype=float)
    marker_ts = np.asarray(marker_ts, dtype=float)
    binned = np.full((len(bin_centers), 3), np.nan)
    for dim in range(3):
        # Interpolate each spatial coordinate independently.
        good = np.isfinite(marker_pos[:, dim]) & np.isfinite(marker_ts)
        if good.sum() < 2:
            continue
        binned[:, dim] = np.interp(
            bin_centers,
            marker_ts[good],
            marker_pos[good, dim],
            # Do not extrapolate beyond the observed marker time range.
            left=np.nan,
            right=np.nan,
        )
    return binned


def smooth_relative_trajectory(rel_xyz, window_length=7, polyorder=2):
    """Smooth the relative wrist-minus-shoulder trajectory coordinate by coordinate.

    Internal gaps are filled temporarily so Savitzky-Golay filtering can run on a
    contiguous segment, but leading and trailing missing regions are preserved as NaN.
    This keeps the decoder from inventing movement before the first valid sample or
    after the last valid sample.
    """
    # Smooth each coordinate independently after filling internal gaps by interpolation.
    # Leading/trailing NaNs are preserved so we do not fabricate motion outside valid tracking.
    rel_xyz = np.asarray(rel_xyz, dtype=float)
    if window_length is None or window_length < 3:
        return rel_xyz
    if window_length % 2 == 0:
        window_length += 1

    out = np.full_like(rel_xyz, np.nan, dtype=float)
    sample_idx = np.arange(rel_xyz.shape[0])
    for dim in range(rel_xyz.shape[1]):
        values = rel_xyz[:, dim]
        good = np.isfinite(values)
        if good.sum() < max(polyorder + 2, 3):
            continue
        # Fill only the interior missing points so the smoothing kernel has a continuous segment.
        filled = np.interp(sample_idx, sample_idx[good], values[good], left=np.nan, right=np.nan)
        valid = np.isfinite(filled)
        if valid.sum() < window_length:
            continue
        # Restrict smoothing to the maximal contiguous valid block.
        first = int(np.argmax(valid))
        last = int(len(valid) - np.argmax(valid[::-1]))
        segment = filled[first:last]
        local_window = min(window_length, len(segment))
        if local_window % 2 == 0:
            local_window -= 1
        if local_window <= polyorder:
            continue
        out[first:last, dim] = savgol_filter(
            segment,
            window_length=local_window,
            polyorder=polyorder,
            mode="interp",
        )
    return out


def smooth_relative_trajectory_butter(rel_xyz, fs, cutoff_hz=6.0, order=2):
    """Zero-phase Butterworth low-pass smoothing of the relative trajectory.

    Mirrors the NaN handling of ``smooth_relative_trajectory``:
      * leading and trailing NaNs are preserved (no fabricated motion outside
        the valid tracking window),
      * interior gaps are linearly interpolated so the filter sees a
        contiguous segment,
      * filtering is restricted to the maximal contiguous valid block per
        coordinate.

    Implementation notes:
      * ``butter(..., output='sos') + sosfiltfilt`` is used for numerical
        stability and zero phase distortion. The effective order is ``2*order``
        because filtfilt applies the filter twice (Winter convention:
        ``order=2`` yields a "4th-order zero-lag" filter).
      * If a segment is too short for the requested order's padlen, we
        fall back to ``order=2``; if still too short, we leave the segment
        unfiltered.

    Parameters
    ----------
    rel_xyz : array-like, shape (T, D)
        Relative wrist-minus-shoulder samples on the decoder bin grid.
    fs : float
        Sampling rate of ``rel_xyz`` in Hz (i.e. ``1 / bin_size``).
    cutoff_hz : float
        Low-pass cutoff in Hz. Must be < fs / 2.
    order : int
        Butterworth order passed to ``butter`` (filtfilt doubles the effective
        order).
    """
    rel_xyz = np.asarray(rel_xyz, dtype=float)
    if cutoff_hz is None or cutoff_hz <= 0:
        return rel_xyz
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} must be < Nyquist={nyq} (fs={fs})"
        )

    def _design(o):
        return butter(o, cutoff_hz, btype="low", fs=fs, output="sos")

    sos_primary = _design(order)
    # sosfiltfilt default padlen for SOS form is 3 * (2 * n_sections + 1) - 1;
    # we use the same conservative estimate as ``filtfilt`` so the fallback
    # branches before scipy raises.
    def _padlen(o):
        return 3 * (2 * o + 1)

    out = np.full_like(rel_xyz, np.nan, dtype=float)
    sample_idx = np.arange(rel_xyz.shape[0])
    for dim in range(rel_xyz.shape[1]):
        values = rel_xyz[:, dim]
        good = np.isfinite(values)
        if good.sum() < 4:
            continue
        # Fill only interior gaps so the filter sees a continuous segment.
        filled = np.interp(sample_idx, sample_idx[good], values[good],
                           left=np.nan, right=np.nan)
        valid = np.isfinite(filled)
        if valid.sum() < 4:
            continue
        # Restrict to the maximal contiguous valid block.
        first = int(np.argmax(valid))
        last = int(len(valid) - np.argmax(valid[::-1]))
        segment = filled[first:last]
        n = len(segment)
        if n > _padlen(order):
            sos = sos_primary
        elif order > 2 and n > _padlen(2):
            # Auto-degrade to order=2 when the requested order is too steep
            # for this trial length.
            sos = _design(2)
        else:
            # Too short to filter safely: keep raw (interpolated) segment.
            out[first:last, dim] = segment
            continue
        out[first:last, dim] = sosfiltfilt(sos, segment)
    return out


def build_relative_state(rel_xyz, dt, target_mode):
    """Convert relative position samples into the decoder target representation.

    Returns both the state matrix and the number of leading bins that were lost when
    taking temporal derivatives. That trim count is later applied to spike-count bins
    so the neural and kinematic targets remain aligned in time.
    """
    rel_xyz = np.asarray(rel_xyz, dtype=float)
    if target_mode == "relative_position":
        return rel_xyz, 0

    # np.diff shortens the sequence by one sample, so downstream spike bins need the same trim.
    vel_xyz = np.diff(rel_xyz, axis=0) / dt
    if target_mode == "relative_velocity":
        return vel_xyz, 1

    if target_mode == "relative_position_velocity_acceleration":
        # Acceleration loses one more sample; align position/velocity to the surviving acceleration bins.
        acc_xyz = np.diff(vel_xyz, axis=0) / dt
        state = np.column_stack([rel_xyz[2:], vel_xyz[1:], acc_xyz])
        return state, 2

    raise ValueError(f"Unknown target_mode: {target_mode}")


def count_spikes_in_bins(spike_times_list, bin_edges):
    """Count spikes for every unit inside a shared set of bin edges.

    Each output column corresponds to one unit from ``spike_times_list`` and each row
    corresponds to one half-open time bin defined by ``bin_edges``.
    """
    # Bin each unit's spike times with the same edges used for the kinematic trajectory.
    counts = np.zeros((len(bin_edges) - 1, len(spike_times_list)), dtype=float)
    for unit_idx, spikes in enumerate(spike_times_list):
        counts[:, unit_idx] = np.histogram(spikes, bins=bin_edges)[0]
    return counts

def binary_spikes_counts(counts):
    return ((np.asarray(counts))  > 0).astype(float)


def causal_gaussian_kernel(sigma_bins, truncate=3.0):
    """Return a one-sided (causal) Gaussian kernel normalized to sum to 1.

    kernel[0] is the weight on the current bin, kernel[k] is the weight on the
    bin ``k`` steps in the past. Using only past samples keeps the smoothed
    neural signal honest for prequential / online decoding.
    """
    # make SST in bin count
    sigma_bins = float(sigma_bins)
    # if less than zero do nothing
    if sigma_bins <= 0:
        return np.array([1.0])
    # How long we are looking back, don't be zero
    length = max(1, int(np.ceil(truncate * sigma_bins)))
    # get the bin index for this bin
    t = np.arange(length + 1, dtype=float)
    # Calculate the weight
    k = np.exp(-0.5 * (t / sigma_bins) ** 2)
    # Standard k, sum is 1
    k /= k.sum()
    return k


def smooth_neural_causal(Y, meta, sigma_bins, truncate=3.0):
    """Causal Gaussian-smooth a stacked spike-count matrix per trial.

    The decoder dataset is one long matrix of bins concatenated across trials.
    Convolving across the whole matrix would leak the tail of one trial into
    the head of the next, so we smooth each trial's slice independently.
    """
    # Make Y as an array, which is the neural activity
    Y = np.asarray(Y, dtype=float)
    # if no sigma then do nothing
    if sigma_bins is None or sigma_bins <= 0:
        return Y
    # Get the kernel
    kernel = causal_gaussian_kernel(sigma_bins, truncate=truncate)
    # Make an identical size array to store result
    out = np.empty_like(Y)
    # Separate them by trials
    for _, idx in meta.groupby("trial_number").indices.items():
        # Take the trial idx in Y
        idx = np.asarray(idx)
        # Take the trial neural activity
        block = Y[idx]
        # convolve each unit's column with the causal kernel; truncate to original length
        smoothed = np.empty_like(block)
        for u in range(block.shape[1]):
            # Do the convolution
            full = np.convolve(block[:, u], kernel, mode="full")
            # Get the part we need
            smoothed[:, u] = full[: len(idx)]
        # Put back
        out[idx] = smoothed
    return out


def parse_peak_time(row):
    """Normalize the peak-extension timestamp field from the reach table.

    The NWB field may appear as a scalar, a length-1 container, or a longer sequence.
    When multiple peaks are present, the downstream code consistently uses the last
    peak as the end of the reach window.
    """
    # peak_extension_times may be stored as a scalar or a sequence; use the last peak when multiple exist.
    peak = row.get("peak_extension_times", np.nan)
    if isinstance(peak, (str, bytes)):
        text = peak.decode() if isinstance(peak, bytes) else peak
        return float(text) if text.strip() else np.nan
    if hasattr(peak, "__len__"):
        if len(peak) == 0:
            return np.nan
        peak = peak[-1]
    try:
        return float(peak)
    except (TypeError, ValueError):
        return np.nan


def build_decoder_dataset(
    nwb_prc,
    reach_tbl,
    target_mode,
    bin_size=0.01,
    trial_results=("S", "F"),
    unit_qualities=("good", "mua"),
    smooth_window=7,
    smooth_polyorder=2,
    smoother="savgol",
    smooth_cutoff_hz=6.0,
    smooth_order=2,
    trial_window="start_to_peak",
    binary=False,
    reach_side=None,
):
    """Build aligned decoder inputs, targets, and sample-level metadata.

    Processing steps for each trial:
    1. Select the reach window.
    2. Interpolate wrist and shoulder positions onto decoder bin centers.
    3. Form the relative trajectory and smooth it.
    4. Convert that trajectory into the requested decoder target state.
    5. Bin spikes with the same time grid.
    6. Remove invalid samples and append per-bin metadata.

    The returned arrays are stacked across trials, while ``meta`` preserves the
    trial identity needed for trial-wise splitting, lagging, and history features.
    """
    # Build one sample per time bin:
    # X = behavioral state to decode, Y = population spike counts, meta = per-sample trial bookkeeping.
    # Taking out the certain quality neural spikes
    spike_times = get_unit_spike_times(nwb_prc, unit_qualities)
    if reach_side is None:
        wrist_name, shoulder_name = reach_marker_names(SESSION)
    else:
        reach_side = str(reach_side).strip().lower()
        if reach_side not in {"l", "r"}:
            raise ValueError("reach_side must be 'l', 'r', or None")
        wrist_name = f"{reach_side}-wrist"
        shoulder_name = f"{reach_side}-shoulder"
    print(f"Using reach markers: {wrist_name}, {shoulder_name}")
    # Make the reach table to be a dataframe
    trials = reach_tbl.to_dataframe().copy()
    # Keep the qualified trials
    trials = trials[trials["result"].isin(trial_results)].copy()
    print(f"Using {len(trials)} trials with result in {trial_results}")
    # Create X Y meta Lists
    X_trials, Y_trials, rows = [], [], []
    for trial_number, row in trials.iterrows():
        # Each trial points to a pose-estimation container inside the NWB processing module.
        pose = nwb_prc.processing[row.kinematics_module].data_interfaces[
            row.video_event
        ].pose_estimation_series
        if wrist_name not in pose or shoulder_name not in pose:
            continue
        # Take the starttime for this trial
        start = float(row.start_time)
        # The project supports two reach windows: start-to-peak or start-to-stop.
        if trial_window == "start_to_peak":
            stop = parse_peak_time(row)
        elif trial_window == "start_to_stop":
            stop = float(row.stop_time)
        else:
            raise ValueError(f"Unknown trial_window: {trial_window}")

        if not np.isfinite(stop):
            continue
        # If too short then skip
        if stop <= start + 2 * bin_size:
            continue

        # Bin edges define spike counts; bin centers define interpolated marker positions.
        bin_edges = np.arange(start, stop + bin_size, bin_size)
        if len(bin_edges) < 4:
            continue
        # Calculating each bin's center, everything but the last element
        bin_centers = bin_edges[:-1] + bin_size / 2

        wrist = pose[wrist_name]
        shoulder = pose[shoulder_name]
        # Pose data and timestamps can have mismatched lengths in some sessions;
        # load_pose_series trims to the shorter length and warns once per series.
        wrist_data, wrist_ts = load_pose_series(wrist, session_tag=SESSION)
        shoulder_data, shoulder_ts = load_pose_series(shoulder, session_tag=SESSION)
        # First express both markers on the same regular time grid.
        wrist_xyz = interpolate_marker_to_bins(wrist_data, wrist_ts, bin_centers)
        shoulder_xyz = interpolate_marker_to_bins(shoulder_data, shoulder_ts, bin_centers)
        # The decoder uses shoulder-centered wrist kinematics rather than absolute wrist position.
        rel_raw = wrist_xyz - shoulder_xyz
        if smoother == "savgol":
            rel_xyz = smooth_relative_trajectory(
                rel_raw,
                window_length=smooth_window,
                polyorder=smooth_polyorder,
            )
        elif smoother == "butter":
            rel_xyz = smooth_relative_trajectory_butter(
                rel_raw,
                fs=1.0 / bin_size,
                cutoff_hz=smooth_cutoff_hz,
                order=smooth_order,
            )
        else:
            raise ValueError(
                f"Unknown smoother={smoother!r}; expected 'savgol' or 'butter'."
            )

        # trim_start keeps neural bins aligned with velocity/acceleration after temporal differencing.
        X_trial, trim_start = build_relative_state(rel_xyz, bin_size, target_mode)
        Y_trial = count_spikes_in_bins(spike_times, bin_edges)[trim_start:]
        if binary:
            Y_trial = binary_spikes_counts(Y_trial)

        # Drop any time bins where either kinematics or neural features are invalid.
        good = np.isfinite(X_trial).all(axis=1) & np.isfinite(Y_trial).all(axis=1)
        X_trial = X_trial[good]
        Y_trial = Y_trial[good]
        # Skip very short fragments because later train/test splitting and lagging become unstable.
        if len(X_trial) < 10:
            continue
        # Adding to the list
        X_trials.append(X_trial)
        Y_trials.append(Y_trial)
        # For each bin
        for local_bin in range(len(X_trial)):
            # Store enough bookkeeping to reconstruct trial structure after stacking.
            rows.append({
                "trial_number": trial_number,
                "result": row.result,
                "local_bin": local_bin,
                "start_time": start,
                "end_time": stop,
                "trial_window": trial_window,
            })

    if not X_trials:
        raise RuntimeError("No usable trials found.")
    # Together in row
    X = np.vstack(X_trials)
    Y = np.vstack(Y_trials)
    meta = pd.DataFrame(rows)
    print(f"Built dataset: X={X.shape}, Y={Y.shape}, trials={meta['trial_number'].nunique()}")
    return X, Y, meta


def split_by_trial(X, Y, meta, test_fraction=0.2, random_seed=0):
    """Split stacked samples into train/test partitions using whole trials.

    This avoids leakage where neighboring bins from the same reach would otherwise
    appear in both the training and test sets.
    """
    # Split on whole trials so the decoder never sees part of a test reach during training.
    rng = np.random.default_rng(random_seed)
    trials = meta["trial_number"].unique().copy()
    rng.shuffle(trials)
    n_test = max(1, int(round(len(trials) * test_fraction)))
    test_trials = set(trials[:n_test])
    is_test = meta["trial_number"].isin(test_trials).to_numpy()
    return (
        X[~is_test],
        Y[~is_test],
        X[is_test],
        Y[is_test],
        meta[~is_test].reset_index(drop=True),
        meta[is_test].reset_index(drop=True),
    )


def apply_lag(X, Y, meta, lag_bins, verbose=True, max_examples=3):
    # X is binnum x dim Y is binnum x units meta is the trial info for each sample
    # Creating the arrays for after processing
    X_rows, Y_rows, meta_rows = [], [], []
    trial_summaries = []
    # For printing
    mapping_examples = []

    if verbose:
        lag_ms = int(round(lag_bins * BIN_SIZE_SECONDS * 1000))
        print(
            f"[apply_lag] lag_bins={lag_bins} lag_ms={lag_ms} | "
            f"input X={X.shape}, Y={Y.shape}, rows={len(meta)}, "
            f"trials={meta['trial_number'].nunique()}"
        )

    # For each trial
    for trial_number, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        # getting data based on index
        X_trial = X[idx]
        Y_trial = Y[idx]
        # reindex the metatable
        meta_trial = meta.iloc[idx].reset_index(drop=True)
        kept = 0 # How many samples left after lag
        first_mapping = None # The first pair
        last_mapping = None # The last pair
        # With in each Trial
        for target_t in range(len(idx)):
            # target_t indexes the kinematic sample; neural_t is the matched neural bin after lagging.
            neural_t = target_t - lag_bins
            # Check if neural_t is out of range
            if neural_t < 0 or neural_t >= len(idx):
                continue
            kept += 1
            if first_mapping is None:
                first_mapping = (target_t, neural_t)
            last_mapping = (target_t, neural_t)
            # Take the time spot's metadata as dictionary
            meta_row = meta_trial.iloc[target_t].to_dict()
            # Save how many bins used
            meta_row["lag_bins"] = lag_bins
            meta_row["lag_ms"] = int(round(lag_bins * BIN_SIZE_SECONDS * 1000))
            # Save the kinematic data
            X_rows.append(X_trial[target_t])
            # Save the neural data
            Y_rows.append(Y_trial[neural_t])
            # Save the metadata
            meta_rows.append(meta_row)
        # Having at least one sample then, for each trial we print this
        if kept:
            # Printing trial_num, length, how many sample left
            trial_summaries.append((trial_number, len(idx), kept))
            if len(mapping_examples) < max_examples:
                mapping_examples.append((trial_number, len(idx), first_mapping, last_mapping))
    # no samples
    if not X_rows:
        raise RuntimeError(f"No samples left after lag_bins={lag_bins}.")
    # List to arrays
    X_out = np.asarray(X_rows, dtype=float)
    Y_out = np.asarray(Y_rows, dtype=float)
    # To dataframe
    meta_out = pd.DataFrame(meta_rows)
    #
    if verbose:
        original_rows = len(meta)
        kept_rows = len(meta_out)
        dropped_rows = original_rows - kept_rows
        kept_trials = len(trial_summaries)
        print(
            f"[apply_lag] output X={X_out.shape}, Y={Y_out.shape}, rows={kept_rows}, "
            f"trials={kept_trials} | dropped_rows={dropped_rows}"
        )
        print(f"[apply_lag] expected drop per trial ~= {abs(lag_bins)} bin(s)")
        for trial_number, trial_len, first_mapping, last_mapping in mapping_examples:
            first_target, first_neural = first_mapping
            last_target, last_neural = last_mapping
            print(
                f"[apply_lag] example trial={trial_number} len={trial_len}: "
                f"first target->{first_target} uses neural->{first_neural}; "
                f"last target->{last_target} uses neural->{last_neural}"
            )

    return X_out, Y_out, meta_out


def make_history_features(X, Y, meta, history_bins, lag_bins=0):
    """Expand each neural sample into a short history window of consecutive bins.

    For each valid target time, the feature vector is
    ``[Y[t], Y[t-1], ..., Y[t-history_bins]]`` after applying any requested lag.
    This is useful for linear decoders that model short temporal context explicitly.
    """
    # Concatenate the current and previous neural bins into one feature vector per target time.
    X_rows, F_rows, meta_rows = [], [], []
    n_units = Y.shape[1]
    for trial_number, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) <= history_bins + abs(lag_bins):
            continue
        X_trial = X[idx]
        Y_trial = Y[idx]
        meta_trial = meta.iloc[idx].reset_index(drop=True)

        for target_t in range(len(idx)):
            neural_t = target_t - lag_bins
            if neural_t - history_bins < 0 or neural_t >= len(idx):
                continue
            # Concatenate newest-to-oldest neural bins so each sample carries its recent history.
            feat = [Y_trial[neural_t - lag] for lag in range(history_bins + 1)]
            F_rows.append(np.concatenate(feat))
            X_rows.append(X_trial[target_t])
            meta_row = meta_trial.iloc[target_t].to_dict()
            meta_row["lag_bins"] = lag_bins
            meta_row["lag_ms"] = int(round(lag_bins * BIN_SIZE_SECONDS * 1000))
            meta_rows.append(meta_row)
    return (
        np.asarray(F_rows, dtype=float).reshape(-1, n_units * (history_bins + 1)),
        np.asarray(X_rows, dtype=float),
        pd.DataFrame(meta_rows),
    )


def standardize_train_test(F_train, F_test):
    """Z-score features using only training-set statistics."""
    # Standardize using training statistics only, then reuse them on the test set.
    mean = F_train.mean(axis=0)
    std = F_train.std(axis=0)
    # Keep constant features finite instead of dividing by zero.
    std[std == 0] = 1.0
    return (F_train - mean) / std, (F_test - mean) / std


def fit_ridge(F_train, X_train, alpha):
    """Fit a multivariate ridge regressor in closed form."""
    # Closed-form ridge solution with an unpenalized intercept term.
    F_aug = np.column_stack([np.ones(len(F_train)), F_train])
    reg = np.eye(F_aug.shape[1]) * alpha
    # Leave the intercept unregularized.
    reg[0, 0] = 0.0
    return np.linalg.solve(F_aug.T @ F_aug + reg, F_aug.T @ X_train)


def predict_linear(F, coef):
    """Apply a linear model whose coefficient matrix expects an explicit intercept column."""
    F_aug = np.column_stack([np.ones(len(F)), F])
    return F_aug @ coef


def summarize_train_test(X_train, P_train, X_test, P_test, names):
    """Collect per-dimension and mean train/test R^2 metrics into one summary row."""
    # Report per-dimension and mean train/test R^2 values in one flat summary row.
    row = {}
    row.update(r2_table(X_train, P_train, names, "train"))
    row.update(r2_table(X_test, P_test, names, "test"))
    row["mean_train_r2"] = np.nanmean([row[f"train_{n}"] for n in names])
    row["mean_test_r2"] = np.nanmean([row[f"test_{n}"] for n in names])
    return row

def kfold_split_by_trial(meta, n_splits=5, random_seed=0):
    rng = np.random.RandomState(random_seed)
    trials = meta["trial_number"].unique().copy()
    rng.shuffle(trials)
    # splitting the dataset by trial
    folds = np.array_split(trials, n_splits)
    for k in range(n_splits):
        # Pick the test set
        test_trials = set(folds[k])
        # Making boolean marks to the trials
        is_test = meta["trial_number"].isin(test_trials).to_numpy()
        # Don't give it all immediately
        yield ~is_test, is_test
