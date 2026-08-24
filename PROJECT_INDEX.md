# Project Index

This is the canonical entry point for the public analysis snapshot.

## Experimental grids

| Dataset | Animal | R1 dates | R2 dates | Grid |
|---|---|---:|---:|---:|
| Original interference | TS | 14 | 3 | 14 × 3 |
| Interference replication | TY | 11 | 3 | 11 × 3 |
| No-interference forget control | TS | 3 | 3 | 3 × 3 |

Session identifiers and known data issues are recorded in
[Data/session_manifest.csv](Data/session_manifest.csv). Canonical session
grids and path definitions live in
[Code/generalization/project_config.py](Code/generalization/project_config.py).

## Current report and results

- [Directional-gap report](Reports/current/directional_gap_report.md)
- [Curated result index](Results/current/README.md)
- [Curated result manifest](Results/current/result_manifest.csv)

## Active code

- [Analysis runner](Code/generalization/run_analysis.py)
- [Core pipeline map](Code/generalization/README.md)
- [Analysis index](Code/generalization/analyses/README.md)
- [Hypothesis-test index](Code/generalization/hypothesis_function_tests/README.md)
- [Script manifest](Code/generalization/script_manifest.csv)
- [Methods documentation](Code/generalization/docs/methods/)

## Integrity checks

```bash
python Code/generalization/run_analysis.py list
python Code/generalization/run_analysis.py check-links
```
