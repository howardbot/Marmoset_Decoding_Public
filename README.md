# Marmoset Cross-Day Neural Decoding

Research code and curated results for cross-day neural decoding analyses in
marmosets. The repository covers the primary TS interference experiment, the
TY interference replication, and the TS no-interference forget control.

Start with [PROJECT_INDEX.md](PROJECT_INDEX.md), which maps the active code,
data contract, current Markdown report, and curated result files.

## Repository contents

- `Code/`: decoder implementations, maintained analysis workflows, plots,
  diagnostics, and hypothesis tests.
- `Data/`: a data dictionary and session manifest only. Processed NWB files
  are intentionally not distributed in this repository.
- `Reports/current/`: the current integrated scientific report in Markdown.
- `Results/current/`: curated outputs for the latest analyses.
- `Results/generalization/` and `Results/manifold_geometry/`: only the
  additional tables and figures cited directly by the current report.

## Quick navigation

List the maintained analysis entry points:

```bash
python Code/generalization/run_analysis.py list
```

Validate Markdown links:

```bash
python Code/generalization/run_analysis.py check-links
```

The primary report is
[Reports/current/directional_gap_report.md](Reports/current/directional_gap_report.md).

## Data availability

The analyses expect processed NWB inputs at the relative paths recorded in
[Data/session_manifest.csv](Data/session_manifest.csv). Raw and processed
recordings are excluded because they are large research datasets with
separate access requirements. See [Data/README.md](Data/README.md) for the
expected organization and known limitations.

## Reproducibility status

The active scripts and curated numerical outputs are included, but the source
NWB files and a locked software-environment specification are not. Numerical
results should therefore be treated as the documented output of the current
analysis snapshot, not as a turnkey data release.
