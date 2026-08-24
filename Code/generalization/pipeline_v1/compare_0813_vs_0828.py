"""Text-only diagnostic: last r1 day (20250813) vs first r2 day (20250828).

Prints raw NWB inventory, locked-config decoder dataset statistics, per-trial
velocity-outlier ranking for 0828, and a leave-out check showing how much the
flagged trials drive the session-level velocity explosion.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO
import ndx_pose

_THIS_DIR = Path(__file__).resolve().parents[1]
_CODE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_THIS_DIR))

from cross_day_decoder import build_session_cache_entry

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
DATA_DIR = REPO_ROOT / "Data"

S1 = "TSAL20250813_0830_staticAndStaticFree001"
S2 = "TSAL20250828_0830_interferenceAndInterferenceFree001"
LABEL1 = "0813 (r1 last)"
LABEL2 = "0828 (r2 first)"
# Decoder-speed outliers above this value are too large to be plausible
# marmoset wrist velocities in this processed coordinate system.
OUTLIER_SPEED_THRESHOLD = 500.0
# Raw 3D pose coordinates for normal reaches are on the order of 1-20 here.
# Values above 100 are intentionally loose: they should catch only catastrophic
# marker reconstruction failures, not normal large reaches.
RAW_POSITION_ABS_THRESHOLD = 100.0
WRIST_SHOULDER_DISTANCE_THRESHOLD = 100.0


def basic_inventory(session_tag):
    """Read raw counts from the NWB before any decoder pipeline runs."""
    path = DATA_DIR / f"{session_tag}_processed.nwb"
    info = {"session": session_tag}
    with NWBHDF5IO(path, mode="r") as io:
        nwb = io.read()
        # The reach interval table tells us how many trials exist before the
        # decoder filters out non-S/F trials or very short/invalid fragments.
        reach_tbl_name = [n for n in nwb.intervals if "reaching_segments" in n][0]
        df = nwb.intervals[reach_tbl_name].to_dataframe()
        info["interval_table"] = reach_tbl_name
        info["n_trials_total"] = len(df)
        if "result" in df.columns:
            info["n_S"] = int((df["result"] == "S").sum())
            info["n_F"] = int((df["result"] == "F").sum())
            info["n_N"] = int((df["result"] == "N").sum())
            info["success_rate"] = info["n_S"] / max(info["n_trials_total"], 1)

        # Unit counts and average firing rate help rule out an obvious neural
        # data failure as the reason 0828 decodes poorly.
        udf = nwb.units.to_dataframe()
        info["n_units_total"] = len(udf)
        info["n_good"] = int((udf["quality"] == "good").sum()) if "quality" in udf.columns else None
        info["n_mua"] = int((udf["quality"] == "mua").sum()) if "quality" in udf.columns else None
        rates = [len(np.asarray(s)) / max(float(df["stop_time"].max() - df["start_time"].min()), 1e-9)
                 for s in udf["spike_times"]]
        info["unit_rate_mean_hz"] = float(np.mean(rates))

        # Pose data and timestamps can have mismatched lengths in these NWBs.
        # This inventory records the raw mismatch before decoder_utils trims to
        # the shared valid length.
        row0 = df.iloc[0]
        pose_mod = nwb.processing[row0["kinematics_module"]]
        pose = pose_mod.data_interfaces[row0["video_event"]].pose_estimation_series
        marker_info = []
        for marker_name in ("r-wrist", "r-shoulder"):
            if marker_name in pose:
                series = pose[marker_name]
                n_data = int(series.data.shape[0])
                n_ts = int(series.timestamps.shape[0])
                marker_info.append({
                    "marker": marker_name,
                    "n_pose_data": n_data,
                    "n_timestamps": n_ts,
                    "extra_rows": n_data - n_ts,
                })
        info["markers"] = marker_info
    return info


def parse_peak_time(row):
    """Return the scalar peak-extension time used as the start-to-peak window end."""
    peak = row.get("peak_extension_times", np.nan)
    if isinstance(peak, (str, bytes)):
        return float(peak)
    if hasattr(peak, "__len__"):
        if len(peak) == 0:
            return np.nan
        return float(peak[-1])
    return float(peak)


def kinematic_stats(X, meta):
    """Session-level velocity variance and speed-tail summaries."""
    n_trials = meta["trial_number"].nunique()
    # X is the decoder target matrix: one row per bin and three columns for
    # relative wrist-shoulder velocity. Sum variance across dimensions so each
    # session has one overall velocity-spread number.
    total_var = float(np.sum(np.var(X, axis=0)))
    # Trial means capture reach-to-reach differences. If this is huge, one or
    # more trials likely occupy a very different kinematic scale.
    trial_means = np.stack([
        X[np.asarray(idx)].mean(axis=0)
        for _, idx in meta.groupby("trial_number").indices.items()
    ])
    between_var = float(np.sum(np.var(trial_means, axis=0)))
    # Within-trial variance asks whether velocity is exploding inside reaches,
    # rather than only trial averages being separated.
    within_vars = [
        np.sum(np.var(X[np.asarray(idx)], axis=0))
        for _, idx in meta.groupby("trial_number").indices.items()
        if len(idx) >= 2
    ]
    within_var = float(np.mean(within_vars))
    # Speed-tail summaries separate "typical movement scale" from rare bad bins.
    # A normal median with a huge p99/max points to localized pose outliers.
    abs_v = np.linalg.norm(X, axis=1)
    return {
        "n_trials_in_dataset": int(n_trials),
        "n_bins_total": int(X.shape[0]),
        "total_var": total_var,
        "between_trial_var": between_var,
        "within_trial_var": within_var,
        "median_speed": float(np.median(abs_v)),
        "p99_speed": float(np.percentile(abs_v, 99)),
        "max_speed": float(np.max(abs_v)),
    }


def per_trial_velocity_stats(X, meta):
    """Return one velocity-tail summary per trial, sorted worst-first."""
    rows = []
    # The decoder dataset is stacked across trials, so use meta to recover
    # which rows of X came from each reach.
    for trial_num, idx_arr in meta.groupby("trial_number").indices.items():
        idx_arr = np.asarray(idx_arr)
        if len(idx_arr) == 0:
            continue
        # X has velocity components [vx, vy, vz]. The Euclidean norm is the
        # scalar speed used to find pose-derived velocity explosions.
        speed = np.linalg.norm(X[idx_arr], axis=1)
        rows.append({
            "trial": trial_num,
            "n_bins": len(idx_arr),
            "max_speed": float(np.max(speed)),
            "p95_speed": float(np.percentile(speed, 95)),
            "mean_speed": float(np.mean(speed)),
            "total_var": float(np.sum(np.var(X[idx_arr], axis=0))),
            "n_over_500": int(np.sum(speed > 500)),
            "n_over_1000": int(np.sum(speed > 1000)),
        })
    # Worst-first ordering makes a single bad trial obvious in the printed table.
    return sorted(rows, key=lambda r: r["max_speed"], reverse=True)


def print_trial_ranking(rows, label, n=12):
    print(f"\n=== {label}: worst trials by max speed ===")
    print(
        f"{'trial':>7} {'n_bins':>7} {'max_speed':>11} {'p95_speed':>11} "
        f"{'mean_speed':>11} {'total_var':>13} {'>500':>6} {'>1000':>7}"
    )
    print("-" * 82)
    for row in rows[:n]:
        print(
            f"{row['trial']:>7} {row['n_bins']:>7} "
            f"{row['max_speed']:>11.3f} {row['p95_speed']:>11.3f} "
            f"{row['mean_speed']:>11.3f} {row['total_var']:>13.3f} "
            f"{row['n_over_500']:>6} {row['n_over_1000']:>7}"
        )


def print_leave_out_check(X, meta, flagged_trials, label):
    if not flagged_trials:
        print(f"\n=== {label}: leave-out check ===")
        print(f"No trials exceeded max-speed threshold {OUTLIER_SPEED_THRESHOLD:g}.")
        return

    # Recompute the same session-level statistics after removing flagged trials.
    # If the huge variance disappears, the session problem is localized rather
    # than a day-wide kinematic scale shift.
    keep = ~meta["trial_number"].isin(flagged_trials).to_numpy()
    before = kinematic_stats(X, meta)
    after = kinematic_stats(X[keep], meta.loc[keep])
    print(f"\n=== {label}: leave-out flagged trials ===")
    print(f"Flagged trials: {flagged_trials}  (max_speed > {OUTLIER_SPEED_THRESHOLD:g})")
    print(f"{'metric':<22} {'all trials':>14} {'drop flagged':>14} {'after / before':>15}")
    print("-" * 70)
    for key in ["n_trials_in_dataset", "n_bins_total", "total_var", "median_speed", "p99_speed", "max_speed"]:
        v0, v1 = before[key], after[key]
        ratio = v1 / v0 if v0 not in (0, 0.0) else float("inf")
        print(f"{key:<22} {v0:>14.3f} {v1:>14.3f} {ratio:>15.4f}")


def safe_nan_stat(fn, values):
    """Run a nan-aware statistic, returning NaN if all values are non-finite."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return np.nan
    return float(fn(values))


def contiguous_true_runs(mask):
    """Convert a per-frame boolean mask into inclusive contiguous index runs."""
    runs = []
    in_run = False
    for i, value in enumerate(mask):
        # A False->True transition starts a new bad segment.
        if value and not in_run:
            start = i
            in_run = True
        # A True->False transition closes the current segment. The final frame
        # needs special handling because there may be no following False.
        if in_run and ((not value) or i == len(mask) - 1):
            stop = i - 1 if not value else i
            runs.append((start, stop))
            in_run = False
    return runs


def print_raw_pose_diagnosis(session_tag, trial_num):
    """Trace a flagged decoder-velocity trial back to raw marker positions."""
    path = DATA_DIR / f"{session_tag}_processed.nwb"
    with NWBHDF5IO(path, mode="r") as io:
        nwb = io.read()
        # The reach interval table gives the trial window and points to the
        # pose container used for that exact video event.
        reach_tbl_name = [n for n in nwb.intervals if "reaching_segments" in n][0]
        trials = nwb.intervals[reach_tbl_name].to_dataframe()
        row = trials.loc[trial_num]
        start = float(row.start_time)
        stop = parse_peak_time(row)
        pose = nwb.processing[row.kinematics_module].data_interfaces[
            row.video_event
        ].pose_estimation_series

        print(f"\n=== Raw pose diagnosis: trial {trial_num} ===")
        print(f"result={row.result}  window={start:.6f}..{stop:.6f}  duration={stop - start:.3f}s")
        print(f"video_event={row.video_event}")
        print(
            f"{'marker':>12} {'frames':>7} {'dt_med':>8} {'step_max':>10} "
            f"{'inst_max':>10} {'x_range':>23} {'y_range':>23} {'z_range':>23} "
            f"{'conf_nan':>9} {'conf_med':>9} {'conf_max':>9}"
        )
        print("-" * 148)

        marker_cache = {}
        for marker_name in ("r-wrist", "r-shoulder"):
            series = pose[marker_name]
            # PoseEstimationSeries stores raw 3D coordinates, timestamps, and
            # Anipose reprojection error. Some sessions have extra data rows, so
            # use the shared valid length before applying a time-window mask.
            data = np.asarray(series.data[:])
            ts = np.asarray(series.timestamps[:])
            conf = np.asarray(series.confidence[:])
            n = min(len(data), len(ts), len(conf))
            data, ts, conf = data[:n], ts[:n], conf[:n]
            # Keep only frames inside the reach window. This checks the raw
            # pose that produced the decoder target, before binning/differencing.
            mask = (ts >= start) & (ts <= stop)
            win_data, win_ts, win_conf = data[mask], ts[mask], conf[mask]
            marker_cache[marker_name] = (win_data, win_ts)

            # Frame-to-frame position jumps divided by frame dt are the raw
            # instantaneous speeds. If these are huge for wrist but not shoulder,
            # the velocity explosion is already present in the pose data.
            steps = np.linalg.norm(np.diff(win_data, axis=0), axis=1)
            dts = np.diff(win_ts)
            inst = steps / np.maximum(dts, 1e-12)
            print(
                f"{marker_name:>12} {len(win_ts):>7} {np.nanmedian(dts):>8.4f} "
                f"{safe_nan_stat(np.nanmax, steps):>10.3f} "
                f"{safe_nan_stat(np.nanmax, inst):>10.1f} "
                f"[{np.nanmin(win_data[:, 0]):>8.1f},{np.nanmax(win_data[:, 0]):>8.1f}] "
                f"[{np.nanmin(win_data[:, 1]):>8.1f},{np.nanmax(win_data[:, 1]):>8.1f}] "
                f"[{np.nanmin(win_data[:, 2]):>8.1f},{np.nanmax(win_data[:, 2]):>8.1f}] "
                f"{np.mean(~np.isfinite(win_conf)):>9.3f} "
                f"{safe_nan_stat(np.nanmedian, win_conf):>9.3f} "
                f"{safe_nan_stat(np.nanmax, win_conf):>9.3f}"
            )

        # Compare wrist to shoulder on the same frames. A real reach should keep
        # both coordinates in the same physical scale; extreme wrist norm or
        # wrist-shoulder distance marks raw 3D tracking/triangulation failure.
        wrist_data, wrist_ts = marker_cache["r-wrist"]
        shoulder_data, _ = marker_cache["r-shoulder"]
        n = min(len(wrist_data), len(shoulder_data), len(wrist_ts))
        wrist_data, shoulder_data, wrist_ts = wrist_data[:n], shoulder_data[:n], wrist_ts[:n]
        wrist_norm = np.linalg.norm(wrist_data, axis=1)
        wrist_shoulder_dist = np.linalg.norm(wrist_data - shoulder_data, axis=1)
        bad = (
            (wrist_norm > RAW_POSITION_ABS_THRESHOLD)
            | (wrist_shoulder_dist > WRIST_SHOULDER_DISTANCE_THRESHOLD)
        )
        print(
            f"\nBad raw wrist frames: {int(bad.sum())}/{len(bad)} "
            f"({np.mean(bad):.3f}); thresholds: |wrist|>{RAW_POSITION_ABS_THRESHOLD:g} "
            f"or wrist-shoulder distance>{WRIST_SHOULDER_DISTANCE_THRESHOLD:g}"
        )
        print(f"{'start_rel_s':>11} {'end_rel_s':>9} {'frames':>7} {'wrist_x0':>10} {'wrist_x1':>10} {'max_dist':>10}")
        print("-" * 66)
        for run_start, run_stop in contiguous_true_runs(bad)[:10]:
            # Convert bad-frame indices back into seconds relative to trial
            # start, which is easier to inspect against videos or pose traces.
            max_dist = float(np.nanmax(wrist_shoulder_dist[run_start:run_stop + 1]))
            print(
                f"{wrist_ts[run_start] - start:>11.4f} "
                f"{wrist_ts[run_stop] - start:>9.4f} "
                f"{run_stop - run_start + 1:>7} "
                f"{wrist_data[run_start, 0]:>10.3f} "
                f"{wrist_data[run_stop, 0]:>10.3f} "
                f"{max_dist:>10.3f}"
            )


def print_basic(info, label):
    print(f"\n=== {label}: {info['session']} ===")
    print(f"  interval table:     {info['interval_table']}")
    print(f"  n_trials_total:     {info['n_trials_total']}  "
          f"(S={info.get('n_S')}, F={info.get('n_F')}, N={info.get('n_N')}, "
          f"success_rate={info.get('success_rate', float('nan')):.2f})")
    print(f"  n_units:            {info['n_units_total']}  "
          f"(good={info['n_good']}, mua={info['n_mua']})")
    print(f"  unit firing rate:   {info['unit_rate_mean_hz']:.2f} Hz (mean)")
    print(f"  pose data vs timestamps (per marker, first video_event):")
    for m in info["markers"]:
        print(f"    {m['marker']:>12}: data={m['n_pose_data']}  ts={m['n_timestamps']}  "
              f"extra={m['extra_rows']:+d}")


def main():
    # First inspect raw NWB bookkeeping. This checks trial counts, unit counts,
    # and pose/timestamp mismatches before the decoder pipeline changes anything.
    info1 = basic_inventory(S1)
    info2 = basic_inventory(S2)
    print_basic(info1, LABEL1)
    print_basic(info2, LABEL2)

    print("\n--- Building decoder dataset (locked config: 20 ms bin, σ=50 ms) ---")
    # Reuse the locked generalization preprocessing so this diagnostic is tied
    # to the exact X/Y matrices used by the cross-day decoder.
    e1 = build_session_cache_entry(S1)
    e2 = build_session_cache_entry(S2)

    # Compare whole-session velocity distributions first. This tells us whether
    # 0828 is globally different or only has an extreme tail.
    k1 = kinematic_stats(e1["X"], e1["meta"])
    k2 = kinematic_stats(e2["X"], e2["meta"])

    print(f"\n=== Velocity statistics (relative wrist−shoulder, 3D) ===")
    print(f"{'metric':<22} {LABEL1:>18} {LABEL2:>18}    ratio (0828 / 0813)")
    print("-" * 88)
    for key in ["n_trials_in_dataset", "n_bins_total",
                "total_var", "between_trial_var", "within_trial_var",
                "median_speed", "p99_speed", "max_speed"]:
        v1, v2 = k1[key], k2[key]
        ratio = v2 / v1 if v1 not in (0, 0.0) else float("inf")
        print(f"{key:<22} {v1:>18.3f} {v2:>18.3f}    {ratio:>6.1f}×")

    # Drill from session-level tail explosion to the responsible trial(s).
    trial_rows = per_trial_velocity_stats(e2["X"], e2["meta"])
    print_trial_ranking(trial_rows, LABEL2)
    # Use a deliberately high absolute threshold so only catastrophic velocity
    # bins are flagged for raw-pose follow-up.
    flagged = [
        int(row["trial"])
        for row in trial_rows
        if row["max_speed"] > OUTLIER_SPEED_THRESHOLD
    ]
    print_leave_out_check(e2["X"], e2["meta"], flagged, LABEL2)
    # Finally, trace each flagged decoder trial back to the raw wrist/shoulder
    # coordinates to decide whether the issue is pose, timestamps, or biology.
    for trial_num in flagged:
        print_raw_pose_diagnosis(S2, trial_num)


if __name__ == "__main__":
    main()
