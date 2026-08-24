# Current Results

This directory is the canonical, report-facing result snapshot.

## Analysis families

- `interference/neural_variability_matching/`: neural-variability matching,
  matched decoding, and directional-significance outputs.
- `interference/kinematic_variability_matching/`: 1-, 2-, and 3-SD movement
  variability matching controls.
- `interference/trial_count_control/`: random fixed-40 position and velocity
  controls.
- `interference/variability_gap_association/`: neural/movement variability
  differences versus directional gap.
- `forget_control/equal_n_3x3/`: complete TS forget-control fixed-31 and
  dropout-clean fixed-31 results.
- `comparisons/interference_vs_forget/`: direct condition contrast,
  paired-direction plot, and within-reach timing comparison.
- `cross_animal/ty_locked_position/`: TY 11-by-3 locked position replication.

## Provenance

[`source_map.csv`](source_map.csv) maps local workflow artifacts to this
snapshot.  [`result_manifest.csv`](result_manifest.csv) adds a SHA-256 digest
and source path for every current artifact.

Refresh and verify the snapshot after rerunning an analysis:

```bash
python Code/generalization/run_analysis.py publish-results
python Code/generalization/run_analysis.py publish-check
python Code/generalization/run_analysis.py path-check
python Code/generalization/run_analysis.py check-links
```
