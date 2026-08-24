from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # Code/ for local imports
from decoder_utils import (
    count_spikes_in_bins,
    get_unit_spike_times,
    interpolate_marker_to_bins,
    load_nwb_and_reach,
    parse_peak_time,
    smooth_relative_trajectory,
)


RESULTS_DIR = Path(__file__).resolve().parents[2] / "Results"


def build_parser():
    """Define CLI arguments for selecting a trial, a unit, and the binning setup."""
    parser = argparse.ArgumentParser(
        description="Plot one sampled reach trial with raw and interpolated wrist/shoulder positions plus one unit's binned spikes."
    )
    parser.add_argument("--bin-size", type=float, default=0.01, help="Bin size in seconds. Default: 0.01")
    parser.add_argument("--trial-window", choices=("start_to_peak", "start_to_stop"), default="start_to_peak")
    parser.add_argument("--trial-number", type=int, default=1, help="Trial number to visualize. Default: 51")
    parser.add_argument("--unit-index", type=int, default=1, help="Unit index from the filtered unit list. Default: 117")
    parser.add_argument("--qualities", nargs="+", default=("good", "mua"), help="Unit quality labels to keep.")
    return parser


def get_selected_trial(nwb_prc, reach_tbl, args):
    """Return the user-selected trial after validating that it is usable."""
    trials = reach_tbl.to_dataframe().copy()
    if args.trial_number not in trials.index:
        raise ValueError(f"Trial {args.trial_number} is not present in the reach table.")

    row = trials.loc[args.trial_number]
    pose = nwb_prc.processing[row.kinematics_module].data_interfaces[row.video_event].pose_estimation_series
    if "r-wrist" not in pose or "r-shoulder" not in pose:
        raise ValueError(f"Trial {args.trial_number} does not contain both r-wrist and r-shoulder markers.")

    start = float(row.start_time)
    if args.trial_window == "start_to_peak":
        stop = parse_peak_time(row)
    else:
        stop = float(row.stop_time)
    # just in case it's before that
    if not np.isfinite(stop) or stop <= start + 2 * args.bin_size:
        raise ValueError(f"Trial {args.trial_number} is too short for bin_size={args.bin_size}.")

    return row, start, stop


def slice_marker_series(series, start, stop):
    """Extract raw marker samples whose timestamps fall inside one trial window."""
    pos = np.asarray(series.data[:], dtype=float)
    ts = np.asarray(series.timestamps[:], dtype=float)
    mask = (ts >= start) & (ts <= stop)
    return ts[mask], pos[mask]


def print_trial_summary(trial_number, unit_index, start, stop, bin_size):
    print("\n" + "=" * 80)
    print("Sampled trial summary")
    print("=" * 80)
    print(f"trial_number: {trial_number}")
    print(f"unit_index:   {unit_index}")
    print(f"window:       {start:.6f} -> {stop:.6f} s")
    print(f"bin_size:     {bin_size * 1000:.1f} ms")


def plot_trial(trial_number, unit_index, bin_size, wrist_ts, wrist_xyz, shoulder_ts, shoulder_xyz, plot_data, save_path):
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axes[0].scatter(wrist_ts, wrist_xyz[:, 1], s=18, color="tab:blue", alpha=0.65, label="raw wrist_y")
    axes[0].scatter(shoulder_ts, shoulder_xyz[:, 1], s=18, color="tab:orange", alpha=0.65, label="raw shoulder_y")
    axes[0].set_title("Raw samples")
    axes[0].set_ylabel("x position")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].eventplot(plot_data["spike_times_s"], colors="0.25", lineoffsets=0.5, linelengths=0.8)
    axes[1].set_title("Raw spike times")
    axes[1].set_yticks([])
    axes[1].grid(alpha=0.2)

    axes[2].plot(plot_data["bin_center_s"], plot_data["wrist_x"], color="tab:blue", lw=2, label="interp wrist_y")
    axes[2].plot(plot_data["bin_center_s"], plot_data["shoulder_x"], color="tab:orange", lw=2, label="interp shoulder_y")
    axes[2].bar(
        plot_data["bin_center_s"],
        plot_data["spike_count"],
        width=bin_size * 0.9,
        color="0.85",
        alpha=0.8,
        label="spike count",
    )
    axes[2].set_title("Interpolated to bin centers")
    axes[2].set_ylabel("x / count")
    axes[2].grid(alpha=0.2)
    axes[2].legend(loc="best", fontsize=9)

    axes[3].plot(plot_data["bin_center_s"], plot_data["relative_x_smooth"], color="tab:green", lw=2)
    axes[3].set_title("Smoothed relative_x (utils)")
    axes[3].set_ylabel("relative x")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(alpha=0.2)

    fig.suptitle(
        f"Trial {trial_number} | unit {unit_index} | raw -> interpolated -> smoothed | bin={bin_size * 1000:.1f} ms"
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {save_path}")


def main():
    """Load one trial, align marker and spike data, and save the comparison figure."""
    args = build_parser().parse_args()

    io, nwb_prc, reach_tbl = load_nwb_and_reach()
    try:
        spike_times = get_unit_spike_times(nwb_prc, tuple(args.qualities))
        if args.unit_index < 0 or args.unit_index >= len(spike_times):
            raise ValueError(f"unit_index must be in [0, {len(spike_times) - 1}]")

        row, start, stop = get_selected_trial(nwb_prc, reach_tbl, args)

        # The reach table stores where to find the per-trial pose-estimation data.
        pose = nwb_prc.processing[row.kinematics_module].data_interfaces[row.video_event].pose_estimation_series
        wrist = pose["r-wrist"]
        shoulder = pose["r-shoulder"]

        wrist_ts, wrist_xyz = slice_marker_series(wrist, start, stop)
        shoulder_ts, shoulder_xyz = slice_marker_series(shoulder, start, stop)

        # Use the same regular bin grid as the decoder utilities so the visualization matches
        # the data preparation logic.
        bin_edges = np.arange(start, stop + args.bin_size, args.bin_size)
        if len(bin_edges) < 2:
            raise RuntimeError("Bin edges are empty for the selected trial.")
        bin_centers = bin_edges[:-1] + args.bin_size / 2

        # Interpolate raw marker samples onto bin centers, then reproduce the smoothed
        # shoulder-centered trajectory used by decoder_utils.
        wrist_interp = interpolate_marker_to_bins(wrist_xyz, wrist_ts, bin_centers)
        shoulder_interp = interpolate_marker_to_bins(shoulder_xyz, shoulder_ts, bin_centers)
        relative_interp = wrist_interp - shoulder_interp
        relative_smooth = smooth_relative_trajectory(relative_interp)[:, 0]

        # Keep raw spike timestamps for the raster-style panel and binned counts for the
        # decoder-aligned panel.
        unit_spikes = np.asarray(spike_times[args.unit_index], dtype=float)
        spike_mask = (unit_spikes >= start) & (unit_spikes <= stop)
        spike_count = count_spikes_in_bins([unit_spikes], bin_edges)[:, 0]

        # Store only the minimal 1D quantities needed by the plotting function.
        plot_data = {
            "bin_center_s": bin_centers,
            "wrist_x": wrist_interp[:, 0],
            "shoulder_x": shoulder_interp[:, 0],
            "spike_count": spike_count,
            "spike_times_s": unit_spikes[spike_mask],
            "relative_x_smooth": relative_smooth,
        }

        print_trial_summary(args.trial_number, args.unit_index, start, stop, args.bin_size)

        save_path = RESULTS_DIR / (
            f"trial_alignment_trial_{args.trial_number}_unit_{args.unit_index}_{int(round(args.bin_size * 1000))}ms.png"
        )
        plot_trial(
            args.trial_number,
            args.unit_index,
            args.bin_size,
            wrist_ts,
            wrist_xyz,
            shoulder_ts,
            shoulder_xyz,
            plot_data,
            save_path,
        )
    finally:
        io.close()


if __name__ == "__main__":
    main()
