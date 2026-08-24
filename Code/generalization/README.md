# Generalization Analysis Code

Use [run_analysis.py](run_analysis.py) for maintained entry commands and
[project_config.py](project_config.py) for all session grids and repository
paths.

## Directory roles

- `analyses/`: hypothesis, mechanism, control, time-resolved, and plotting
  analyses. This replaces the ambiguous former name `why/`.
- `diagnostics/`: input-data and model diagnostics.
- `plots/`: established sweep/publication plotting scripts.
- `hypothesis_function_tests/`: unit and regression tests.
- `pipeline_v1/`: frozen earlier pipeline implementation.
- `sweeps/`: parameter sweeps.
- `tools/`: project manifest and link-integrity utilities.
- `docs/`: stable methods documentation included in this public snapshot.

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
```

Internal worklogs, archived indexes, and superseded presentations are not
included in the public snapshot.
