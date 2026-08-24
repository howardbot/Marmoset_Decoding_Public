"""Per-session / per-trial diagnostic for pose timestamp/data length mismatch.

For every processed NWB in Data/, report:
  - per pose_estimation_series: data rows vs timestamps rows (the global mismatch)
  - per reach trial (using the reaching_segments interval table): how many
    *trimmed-off* frames would have fallen inside the trial window, i.e. how
    much usable kinematics we lose to this mismatch.

The trimmed-off frames are the last ``data_rows - timestamps_rows`` rows of
``series.data``. They have no timestamps, so we cannot know exactly when they
occurred. We estimate by assuming the trimmed frames would have continued at the
same median dt past the last known timestamp and ask whether that estimated
window overlaps each trial.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import ndx_pose  # noqa: F401  register pose extension

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
REPORT_DIR = REPO_ROOT / "Results"
REPORT_DIR.mkdir(exist_ok=True)


def iter_pose_series(nwb):
    """Yield (kinematics_module_name, video_event_name, marker_name, series)."""
    for mod_name, mod in nwb.processing.items():
        for di_name, di in mod.data_interfaces.items():
            if not hasattr(di, "pose_estimation_series"):
                continue
            for mk, series in di.pose_estimation_series.items():
                yield mod_name, di_name, mk, series


def estimate_extrapolated_window(timestamps, n_extra):
    """Estimate (t_start, t_end) the trimmed frames would have occupied."""
    if n_extra <= 0 or len(timestamps) < 2:
        return None
    dt = float(np.median(np.diff(timestamps)))
    last = float(timestamps[-1])
    return last + dt, last + dt * n_extra


def diagnose_file(nwb_path):
    """Return (series_rows_df, trial_rows_df) for one session."""
    session = nwb_path.name.replace("_processed.nwb", "")
    date = session.split("_")[0].replace("TSAL", "")
    series_rows = []
    trial_rows = []

    with NWBHDF5IO(nwb_path, mode="r") as io:
        nwb = io.read()

        # Per-series global stats.
        series_meta = {}  # (di_name, mk) -> dict
        for mod_name, di_name, mk, series in iter_pose_series(nwb):
            ts = np.asarray(series.timestamps[:])
            nd = int(series.data.shape[0])
            nt = int(ts.shape[0])
            extra = nd - nt
            extrap = estimate_extrapolated_window(ts, extra)
            info = {
                "date": date,
                "module": mod_name,
                "video_event": di_name,
                "marker": mk,
                "data_rows": nd,
                "timestamps_rows": nt,
                "extra_rows": extra,
                "ts_first": float(ts[0]) if nt else np.nan,
                "ts_last": float(ts[-1]) if nt else np.nan,
                "extrap_start": extrap[0] if extrap else np.nan,
                "extrap_end": extrap[1] if extrap else np.nan,
            }
            series_rows.append(info)
            series_meta[(di_name, mk)] = info

        # Per-trial stats using the reaching_segments table(s).
        for tbl_name, tbl in nwb.intervals.items():
            if "reaching_segments" not in tbl_name:
                continue
            df = tbl.to_dataframe()
            for trial_idx, row in df.iterrows():
                start = float(row["start_time"])
                stop = float(row["stop_time"])
                video_event = row.get("video_event", None)
                # For each marker in this trial's pose container.
                for (di_name, mk), info in series_meta.items():
                    if di_name != video_event:
                        continue
                    overlap_extra = 0
                    if info["extra_rows"] > 0 and np.isfinite(info["extrap_start"]):
                        # Number of estimated extrapolated samples that would
                        # have fallen inside the trial window.
                        es, ee = info["extrap_start"], info["extrap_end"]
                        ov_lo = max(es, start)
                        ov_hi = min(ee, stop)
                        if ov_hi > ov_lo:
                            # rough count: extra_rows * fraction of extrap range overlapping
                            extrap_span = max(ee - es, 1e-9)
                            overlap_extra = int(
                                round(info["extra_rows"] * (ov_hi - ov_lo) / extrap_span)
                            )
                    trial_rows.append({
                        "date": date,
                        "interval_table": tbl_name,
                        "trial_index": trial_idx,
                        "result": row.get("result", None),
                        "video_event": video_event,
                        "marker": mk,
                        "start_time": start,
                        "stop_time": stop,
                        "trial_duration": stop - start,
                        "extra_rows_for_series": info["extra_rows"],
                        "estimated_lost_frames_in_trial": overlap_extra,
                    })

    return pd.DataFrame(series_rows), pd.DataFrame(trial_rows)


def main():
    files = sorted(DATA_DIR.glob("*_processed.nwb"))
    all_series, all_trials = [], []
    print(f"Scanning {len(files)} processed NWB files in {DATA_DIR}\n")
    for f in files:
        sdf, tdf = diagnose_file(f)
        all_series.append(sdf)
        all_trials.append(tdf)
        date = f.name.split("_")[0].replace("TSAL", "")
        n_series = len(sdf)
        n_bad = int((sdf["extra_rows"] != 0).sum())
        max_extra = int(sdf["extra_rows"].max()) if n_series else 0
        # how many trials have any estimated lost frames for any marker
        n_trials_affected = (
            tdf.groupby("trial_index")["estimated_lost_frames_in_trial"].max().gt(0).sum()
            if not tdf.empty
            else 0
        )
        n_trials_total = tdf["trial_index"].nunique() if not tdf.empty else 0
        print(
            f"{date}: series_mismatch={n_bad}/{n_series} (max +{max_extra} rows) | "
            f"trials_with_est_loss={n_trials_affected}/{n_trials_total}"
        )

    series_all = pd.concat(all_series, ignore_index=True)
    trials_all = pd.concat(all_trials, ignore_index=True)

    series_out = REPORT_DIR / "pose_mismatch_per_series.csv"
    trials_out = REPORT_DIR / "pose_mismatch_per_trial.csv"
    series_all.to_csv(series_out, index=False)
    trials_all.to_csv(trials_out, index=False)

    print(f"\nWrote {series_out}")
    print(f"Wrote {trials_out}")

    # Headline summary.
    print("\n=== Headline ===")
    summary = (
        series_all.assign(mismatch=(series_all["extra_rows"] != 0).astype(int))
        .groupby("date")
        .agg(
            n_series=("marker", "count"),
            n_mismatch=("mismatch", "sum"),
            max_extra=("extra_rows", "max"),
            median_extra_when_bad=(
                "extra_rows",
                lambda x: float(np.median(x[x != 0])) if (x != 0).any() else 0.0,
            ),
        )
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
