"""Test whether the pose data/timestamp mismatch is tail-loss vs interleaved.

Logic:
  - The reach interval table stores ``peak_extension_times``: the recorded
    moment of maximum reach for each trial, measured independently of the pose
    arrays we are trimming.
  - For each trial we read the trimmed wrist + shoulder series, compute the
    wrist-minus-shoulder distance, and find the time of its maximum inside the
    trial window. Call that the *observed* peak time.
  - Observed - recorded peak time should be small and noise-like if the missing
    timestamps were at the tail (trimming preserves alignment).
  - If timestamps were dropped mid-stream, this delta will grow systematically
    with trial index (later trials accumulate more missing frames before them).

Output:
  - Per-session: median |delta|, trend slope of delta vs trial_index, fraction
    of trials with |delta| > 50 ms.
  - CSV with all per-trial deltas to Results/workflows/data_quality/peak_alignment_deltas.csv.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import ndx_pose  # noqa

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
RESULTS = REPO_ROOT / "Results" / "workflows" / "data_quality"
RESULTS.mkdir(exist_ok=True)


def parse_peak(row):
    p = row.get("peak_extension_times", np.nan)
    try:
        if isinstance(p, (str, bytes)):
            s = p.decode() if isinstance(p, bytes) else p
            return float(s) if s.strip() else np.nan
        if hasattr(p, "__len__"):
            if len(p) == 0:
                return np.nan
            return float(p[-1])
        return float(p)
    except (ValueError, TypeError):
        return np.nan


def load_trimmed(series):
    data = np.asarray(series.data[:])
    ts = np.asarray(series.timestamps[:])
    n = min(data.shape[0], ts.shape[0])
    return data[:n], ts[:n], data.shape[0] - ts.shape[0]


def trial_peak_delta(nwb, row):
    """Return (observed_peak_t, recorded_peak_t, extra_rows) or None."""
    try:
        pose = nwb.processing[row["kinematics_module"]].data_interfaces[
            row["video_event"]
        ].pose_estimation_series
    except KeyError:
        return None
    if "r-wrist" not in pose or "r-shoulder" not in pose:
        return None
    recorded = parse_peak(row)
    if not np.isfinite(recorded):
        return None

    w_data, w_ts, w_extra = load_trimmed(pose["r-wrist"])
    s_data, s_ts, _ = load_trimmed(pose["r-shoulder"])

    start = float(row["start_time"])
    stop = float(row["stop_time"])

    w_mask = (w_ts >= start) & (w_ts <= stop)
    s_mask = (s_ts >= start) & (s_ts <= stop)
    if w_mask.sum() < 5 or s_mask.sum() < 5:
        return None

    w_t = w_ts[w_mask]
    s_t = s_ts[s_mask]
    # Put shoulder on wrist's clock by linear interpolation per axis.
    s_on_w = np.column_stack([
        np.interp(w_t, s_t, s_data[s_mask, d]) for d in range(3)
    ])
    rel = w_data[w_mask] - s_on_w
    dist = np.linalg.norm(rel, axis=1)
    if not np.isfinite(dist).any():
        return None
    observed = float(w_t[np.nanargmax(dist)])
    return observed, recorded, w_extra


def diagnose_session(nwb_path):
    session = nwb_path.name.replace("_processed.nwb", "")
    date = session.split("_")[0].replace("TSAL", "")
    rows = []
    with NWBHDF5IO(nwb_path, mode="r") as io:
        nwb = io.read()
        for tbl_name, tbl in nwb.intervals.items():
            if "reaching_segments" not in tbl_name:
                continue
            df = tbl.to_dataframe()
            for trial_idx, row in df.iterrows():
                r = trial_peak_delta(nwb, row)
                if r is None:
                    continue
                obs, rec, extra = r
                rows.append({
                    "date": date,
                    "trial_index": int(trial_idx),
                    "trial_start": float(row["start_time"]),
                    "recorded_peak_t": rec,
                    "observed_peak_t": obs,
                    "delta_sec": obs - rec,
                    "wrist_extra_rows": extra,
                })
    return pd.DataFrame(rows)


def main():
    files = sorted(DATA_DIR.glob("*_processed.nwb"))
    all_dfs = []
    print(f"{'date':>8}  {'n':>4}  {'median|d|ms':>11}  {'p95|d|ms':>8}  "
          f"{'frac>50ms':>9}  {'slope ms/trial':>14}  {'extra_rows':>10}")
    print("-" * 80)
    for f in files:
        df = diagnose_session(f)
        if df.empty:
            continue
        all_dfs.append(df)
        d = df["delta_sec"].to_numpy() * 1000  # ms
        ad = np.abs(d)
        # slope: ms drift per trial — if mid-loss, expect a non-zero slope
        if len(df) >= 3:
            slope = np.polyfit(df["trial_index"].to_numpy(), d, 1)[0]
        else:
            slope = np.nan
        print(
            f"{df['date'].iloc[0]:>8}  {len(df):>4}  "
            f"{np.median(ad):>11.1f}  {np.percentile(ad, 95):>8.1f}  "
            f"{(ad > 50).mean():>9.2%}  {slope:>14.3f}  "
            f"{df['wrist_extra_rows'].max():>10d}"
        )

    out = pd.concat(all_dfs, ignore_index=True)
    csv = RESULTS / "peak_alignment_deltas.csv"
    out.to_csv(csv, index=False)
    print(f"\nWrote {csv}")
    print("\nInterpretation:")
    print("  - If slope ~ 0 and median|d| small (< one frame ~16ms): tail-loss likely.")
    print("  - If slope grows with trial index: timestamps were dropped mid-stream.")
    print("  - 20250813 has zero extra_rows; use it as a control upper bound on noise.")


if __name__ == "__main__":
    main()
