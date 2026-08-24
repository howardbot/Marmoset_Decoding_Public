"""Diagnose the three r2 (post-interference) sessions vs the r1 baseline.

For each session we collect:
  - trial counts and result breakdown (S / F / N) from the reach interval table
  - trial duration statistics (start -> peak)
  - per-unit firing rate stats
  - pose data/timestamp mismatch sizes
  - 3D reach-trajectory length and peak displacement statistics

The r2 sessions are compared against the r1 distribution to flag outliers
(in particular 20250828, whose within-day decoder failed: mean vel corr = 0.023).

Output:
  Results/generalization/r2_diagnostics.csv  -- one row per session
  Results/generalization/r2_diagnostics.txt  -- printed report
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import ndx_pose  # noqa

_THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS_DIR.parent))
import decoder_utils as du  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
DATA_DIR = REPO_ROOT / "Data"
OUT_DIR = REPO_ROOT / "Results" / "generalization"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def session_date(tag):
    return tag.split("_")[0].replace("TSAL", "")


def session_epoch(tag):
    d = int(session_date(tag))
    if 20250731 <= d <= 20250813:
        return "r1"
    if 20250828 <= d <= 20250830:
        return "r2"
    return "?"


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


def reach_distance(pose, row):
    """Wrist - shoulder displacement at reach start and at peak; return their distance."""
    try:
        wrist = pose["r-wrist"]
        shoulder = pose["r-shoulder"]
    except KeyError:
        return np.nan, np.nan, np.nan
    w_d = np.asarray(wrist.data[:])
    w_t = np.asarray(wrist.timestamps[:])
    n = min(len(w_d), len(w_t))
    w_d, w_t = w_d[:n], w_t[:n]
    s_d = np.asarray(shoulder.data[:])
    s_t = np.asarray(shoulder.timestamps[:])
    m = min(len(s_d), len(s_t))
    s_d, s_t = s_d[:m], s_t[:m]
    start = float(row["start_time"])
    stop = float(row["stop_time"])
    peak = parse_peak(row)
    if not np.isfinite(peak):
        return np.nan, np.nan, np.nan
    # nearest neighbors in time
    def nearest(times, t):
        return int(np.argmin(np.abs(times - t)))
    try:
        i_s = nearest(w_t, start)
        i_p = nearest(w_t, peak)
        sj_s = nearest(s_t, start)
        sj_p = nearest(s_t, peak)
        rel_start = w_d[i_s] - s_d[sj_s]
        rel_peak = w_d[i_p] - s_d[sj_p]
        # displacement length, peak distance from origin, reach span
        peak_dist = float(np.linalg.norm(rel_peak))
        start_dist = float(np.linalg.norm(rel_start))
        span = float(np.linalg.norm(rel_peak - rel_start))
        return start_dist, peak_dist, span
    except Exception:
        return np.nan, np.nan, np.nan


def diagnose(session_tag):
    path = DATA_DIR / f"{session_tag}_processed.nwb"
    info = {
        "session": session_tag,
        "date": session_date(session_tag),
        "epoch": session_epoch(session_tag),
        "has_reach_table": False,
    }
    with NWBHDF5IO(path, mode="r") as io:
        nwb = io.read()
        # Reach table?
        reach_names = [n for n in nwb.intervals if "reaching_segments" in n]
        info["interval_tables"] = ",".join(nwb.intervals.keys())
        if not reach_names:
            return info
        info["has_reach_table"] = True
        df = nwb.intervals[reach_names[0]].to_dataframe()
        info["n_trials_total"] = len(df)
        if "result" in df.columns:
            for r in ("S", "F", "N"):
                info[f"n_result_{r}"] = int((df["result"] == r).sum())
        # Trial durations (start_to_peak window we actually use in decoder)
        durations = []
        spans = []
        starts = []
        peaks = []
        for _, row in df.iterrows():
            if row.get("result", None) not in ("S", "F"):
                continue
            t0 = float(row["start_time"])
            pk = parse_peak(row)
            if not np.isfinite(pk) or pk <= t0:
                continue
            durations.append(pk - t0)
            # kinematics if available
            try:
                pose = nwb.processing[row["kinematics_module"]].data_interfaces[
                    row["video_event"]
                ].pose_estimation_series
                s, p, sp = reach_distance(pose, row)
                starts.append(s)
                peaks.append(p)
                spans.append(sp)
            except Exception:
                pass
        if durations:
            info["n_trials_usable"] = len(durations)
            info["dur_median_s"] = float(np.median(durations))
            info["dur_iqr_s"] = float(np.percentile(durations, 75) - np.percentile(durations, 25))
            info["dur_min_s"] = float(np.min(durations))
            info["dur_max_s"] = float(np.max(durations))
        if spans:
            spans_arr = np.array([s for s in spans if np.isfinite(s)])
            peaks_arr = np.array([p for p in peaks if np.isfinite(p)])
            starts_arr = np.array([s for s in starts if np.isfinite(s)])
            if spans_arr.size:
                info["reach_span_median"] = float(np.median(spans_arr))
                info["reach_span_iqr"] = float(np.percentile(spans_arr, 75) - np.percentile(spans_arr, 25))
            if peaks_arr.size:
                info["peak_dist_median"] = float(np.median(peaks_arr))
            if starts_arr.size:
                info["start_dist_median"] = float(np.median(starts_arr))

        # Unit / neural stats
        if nwb.units is not None:
            udf = nwb.units.to_dataframe()
            info["n_units_total"] = len(udf)
            if "quality" in udf.columns:
                info["n_good"] = int((udf["quality"] == "good").sum())
                info["n_mua"] = int((udf["quality"] == "mua").sum())
            spike_counts = [len(np.asarray(s)) for s in udf["spike_times"]]
            session_dur = (
                float(df["stop_time"].max() - df["start_time"].min())
                if len(df) else np.nan
            )
            rates = [n / session_dur if session_dur > 0 else np.nan for n in spike_counts]
            info["session_duration_s"] = session_dur
            info["unit_rate_median_hz"] = float(np.median(rates))
            info["unit_rate_mean_hz"] = float(np.mean(rates))
            info["unit_rate_silent_count"] = int(np.sum(np.array(rates) < 0.1))

        # Pose mismatch (just one example marker)
        try:
            row0 = df.iloc[0]
            pose = nwb.processing[row0["kinematics_module"]].data_interfaces[
                row0["video_event"]
            ].pose_estimation_series
            if "r-wrist" in pose:
                wd = pose["r-wrist"].data.shape[0]
                wt = pose["r-wrist"].timestamps.shape[0]
                info["pose_data_minus_ts_example"] = wd - wt
        except Exception:
            info["pose_data_minus_ts_example"] = np.nan

    return info


def main():
    sessions = sorted(p.name.replace("_processed.nwb", "")
                      for p in DATA_DIR.glob("*_processed.nwb")
                      if session_epoch(p.name.replace("_processed.nwb", "")) in ("r1", "r2"))
    rows = []
    for s in sessions:
        print(f"Diagnosing {s} ...")
        try:
            rows.append(diagnose(s))
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            rows.append({"session": s, "epoch": session_epoch(s),
                         "date": session_date(s), "error": str(e)})

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "r2_diagnostics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Build a comparison report: r1 stats vs each r2 session
    numeric_cols = [
        "n_trials_total", "n_result_S", "n_result_F", "n_result_N",
        "n_trials_usable", "dur_median_s", "dur_iqr_s",
        "reach_span_median", "peak_dist_median", "start_dist_median",
        "n_units_total", "n_good", "n_mua",
        "session_duration_s", "unit_rate_median_hz", "unit_rate_mean_hz",
        "unit_rate_silent_count", "pose_data_minus_ts_example",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    r1 = df[df.epoch == "r1"]
    r2 = df[df.epoch == "r2"]

    print("\n=== Comparison: r1 baseline (n=13) vs r2 each session ===")
    header = (
        f"{'metric':<28} {'r1 median':>12} {'r1 IQR':>14}  "
        f"{'0828':>9} {'0829':>9} {'0830':>9}"
    )
    print(header)
    print("-" * len(header))
    for col in numeric_cols:
        r1_vals = r1[col].dropna()
        if r1_vals.empty:
            continue
        med = float(np.median(r1_vals))
        iqr_lo = float(np.percentile(r1_vals, 25))
        iqr_hi = float(np.percentile(r1_vals, 75))

        def fmt_r2(date):
            sub = r2[r2.date == date]
            if sub.empty or col not in sub.columns:
                return "--"
            v = sub[col].iloc[0]
            if pd.isna(v):
                return "--"
            return f"{v:>9.2f}" if isinstance(v, float) else f"{v:>9.0f}"

        print(
            f"{col:<28} {med:>12.2f}  [{iqr_lo:>5.2f},{iqr_hi:>5.2f}] "
            f"{fmt_r2('20250828')} {fmt_r2('20250829')} {fmt_r2('20250830')}"
        )

    print("\n=== r2 interval table contents ===")
    for date in ("20250828", "20250829", "20250830"):
        sub = r2[r2.date == date]
        if not sub.empty:
            print(f"  {date}: {sub.iloc[0].get('interval_tables', '?')}")


if __name__ == "__main__":
    main()
