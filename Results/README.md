# Results

The result tree has three explicit lifecycle levels.

## `current/`: report-facing snapshot

`current/` contains the compact, tracked tables and figures cited by the
current scientific report.  These are the only result artifacts intended for
the public repository.

- `source_map.csv` records the workflow source for every promoted artifact.
- `result_manifest.csv` records type, size, SHA-256, and source provenance.
- Tables and figures are grouped by experiment and analysis family.

Do not copy files into this tree by hand.  Refresh and validate it with:

```bash
python Code/generalization/run_analysis.py publish-results
python Code/generalization/run_analysis.py publish-check
python Code/generalization/run_analysis.py path-check
```

## `workflows/`: complete local analysis outputs

`workflows/` contains full analysis products, including long tables,
parameter sweeps, mechanism analyses, diagnostics, and the sources promoted
into `current/`.  Analysis scripts write here by default.  The directory is
local-only and excluded from Git because it is roughly 863 MB.

Main families are:

- `workflows/generalization/`
- `workflows/manifold_geometry/`
- `workflows/decoder_benchmarks/`
- `workflows/data_quality/`

## `archive/`: frozen historical outputs

`archive/legacy/` preserves superseded pipelines, generated QA bundles, and
historical backups.  It is local-only, excluded from Git, and must not be used
as an input by maintained analyses.

New result names use lowercase snake_case.  Avoid ambiguous labels such as
`final` and `latest`; lifecycle is represented by the directory level.
