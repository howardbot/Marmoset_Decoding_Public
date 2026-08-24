# Marmoset Cross-Day Decoding

This repository contains the maintained analysis code, compact public result
snapshot, data contract, and Markdown report for the TS/TY marmoset
cross-session decoding project.

Start with [PROJECT_INDEX.md](PROJECT_INDEX.md), which maps the active code,
data contract, current report, and curated result files.

## Repository contents

- `Code/`: decoder implementations, maintained analysis workflows, plots,
  diagnostics, hypothesis tests, and result-publication tools.
- `Data/`: a data dictionary and session manifest only. Processed NWB files
  are intentionally not distributed in this repository.
- `Reports/current/`: the current integrated scientific report in Markdown.
- `Results/current/`: the tracked, report-facing tables and figures.
- `Results/workflows/`: complete local analysis outputs and intermediates;
  excluded from Git.
- `Results/archive/`: frozen historical outputs; excluded from Git.

## Quick navigation

List maintained analysis and maintenance commands:

```bash
python Code/generalization/run_analysis.py list
```

After rerunning analyses, publish and verify the current result snapshot:

```bash
python Code/generalization/run_analysis.py publish-results
python Code/generalization/run_analysis.py publish-check
python Code/generalization/run_analysis.py path-check
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
analysis snapshot, not as a turnkey data release. Each published result has a
workflow source mapping and SHA-256 digest under `Results/current/`.
