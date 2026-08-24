from __future__ import annotations

from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO
import ndx_pose  # noqa: F401  register pose extension


REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION = "TSAL20250812_0830_staticAndStaticFree001"
PROCESSED_NWB = REPO_ROOT / "Data" / f"{SESSION}_processed.nwb"


def toy_example() -> None:
    print("\n=== Toy example: boolean mask must match data length ===")

    data = np.array(
        [
            [10, 100],
            [11, 101],
            [12, 102],
            [13, 103],
            [14, 104],
        ]
    )
    timestamps = np.array([0.0, 0.1, 0.2, 0.3])
    mask = timestamps >= 0.15

    print(f"data has {len(data)} rows:")
    print(data)
    print(f"timestamps has {len(timestamps)} values:")
    print(timestamps)
    print(f"mask made from timestamps has {len(mask)} values:")
    print(mask)

    print("\nTrying data[mask] means:")
    for i, keep in enumerate(mask):
        print(f"  timestamp row {i}: keep={keep}")
    print("  data row 4: has no matching mask value")

    try:
        print(data[mask])
    except IndexError as exc:
        print(f"\nNumPy error: {exc}")

    n = min(len(data), len(timestamps))
    print(f"\nMinimal fix for inspection: trim both to common length n={n}")
    print(data[:n][mask])


def nwb_example() -> None:
    print("\n=== Real NWB example ===")
    print(f"file: {PROCESSED_NWB}")

    with NWBHDF5IO(PROCESSED_NWB, mode="r") as io:
        nwb = io.read()
        reach_tbl = nwb.intervals["reaching_segments_static"]
        row0 = reach_tbl.to_dataframe().iloc[0]

        mod = nwb.processing[row0.kinematics_module]
        pose = mod.data_interfaces[row0.video_event].pose_estimation_series
        wrist = pose["l-wrist"]

        data = wrist.data[:]
        timestamps = wrist.timestamps[:]

        print(f"video_event: {row0.video_event}")
        print(f"marker: l-wrist")
        print(f"data shape:       {data.shape}")
        print(f"timestamps shape: {timestamps.shape}")
        print(f"extra data rows without timestamps: {data.shape[0] - timestamps.shape[0]}")

        mask = (timestamps >= row0.start_time) & (timestamps <= row0.stop_time)
        print(f"\ntrial window: {row0.start_time:.3f} to {row0.stop_time:.3f} sec")
        print(f"mask length: {len(mask)}")
        print(f"data rows:   {data.shape[0]}")
        print(f"samples in this trial according to timestamps: {mask.sum()}")

        print("\nOriginal failing operation: data[mask]")
        try:
            _ = data[mask]
        except IndexError as exc:
            print(f"NumPy error: {exc}")

        n = min(data.shape[0], timestamps.shape[0])
        aligned_data = data[:n]
        aligned_timestamps = timestamps[:n]
        aligned_mask = (
            (aligned_timestamps >= row0.start_time)
            & (aligned_timestamps <= row0.stop_time)
        )
        trial_data = aligned_data[aligned_mask]

        print("\nAfter trimming data and timestamps to the same length:")
        print(f"aligned data shape:       {aligned_data.shape}")
        print(f"aligned timestamps shape: {aligned_timestamps.shape}")
        print(f"trial_data shape:         {trial_data.shape}")
        print("first 3 trial samples:")
        print(trial_data[:3])


def main() -> None:
    toy_example()
    nwb_example()


if __name__ == "__main__":
    main()
