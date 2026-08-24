# Generalization Analysis Code

Use [run_analysis.py](run_analysis.py) for maintained entry commands and
[project_config.py](project_config.py) for session grids and repository paths.

## Directory roles

- `analyses/`: hypothesis, mechanism, control, time-resolved, and plotting
  analyses.
- `diagnostics/`: input-data and model diagnostics.
- `plots/`: established sweep and publication plotting scripts.
- `hypothesis_function_tests/`: unit and regression tests.
- `pipeline_v1/`: frozen earlier pipeline implementation.
- `sweeps/`: parameter sweeps.
- `tools/`: manifests, link checks, and current-result publication.
- `docs/`: stable methods documentation.

## Result lifecycle

Maintained analyses write complete outputs below `Results/workflows/`.
Report-facing artifacts are promoted according to
`Results/current/source_map.csv`; reports reference only `Results/current/`.
Historical pipelines and outputs live below `Results/archive/`.

## Locked analysis metadata

The current position report uses 30 ms neural bins, 12 PCs, trial-average
CCA, lag-0 Kalman decoding, and start-to-peak reaching data. These values are
recorded in `project_config.LOCKED_DECODER`; specialized older figures may
retain explicitly documented alternatives.

## Maintained jobs

```bash
python Code/generalization/run_analysis.py list
python Code/generalization/run_analysis.py forget-equal-n
python Code/generalization/run_analysis.py ty-locked-matrix
python Code/generalization/run_analysis.py ty-significance
python Code/generalization/run_analysis.py publish-results
python Code/generalization/run_analysis.py publish-check
python Code/generalization/run_analysis.py path-check
python Code/generalization/run_analysis.py check-links
```

Internal worklogs, archived indexes, full local results, and superseded
presentations are not included in the public snapshot.
