# Data Contract

Processed NWB inputs are not distributed in this public repository. To run
the analyses, place authorized NWB files under `Data/` using the relative
paths listed in [session_manifest.csv](session_manifest.csv).

| Experiment | Animal | R1 | R2 |
|---|---|---:|---:|
| Interference | TS | 14 | 3 |
| Interference replication | TY | 11 | 3 |
| Forget control | TS | 3 | 3 |

## Known limitations

- TS forget R1 2026-06-09: forget event 2 has unusable pose in the current
  processed file, leaving approximately 70 trials unavailable. Reprocessing
  is pending.
- TS forget R1 2026-06-10: the file is complete, but only 31 S/F reaching
  trials are decoder-usable. This defines the current common equal-N limit.
- `forgetFree` video timestamps alone are not decoder-ready. Trial segments
  and usable wrist/shoulder kinematics are also required.

NWB, HDF5, NumPy, pickle, and parquet files are excluded by `.gitignore`.
