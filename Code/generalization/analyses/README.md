# Analysis Script Index

This directory contains the mechanism and control analyses formerly stored
under the ambiguous name `why/`. Existing import relationships were preserved
during the rename.

## Current headline workflows

- `forget_control_equal_n_crossday.py`: complete TS forget-control 3 × 3
  fixed-31 and dropout-clean fixed-31 analysis.
- `position_asymmetry_significance.py`: variability-matched TS significance
  and locked TS/TY directional-gap summaries.
- `plot_ty_paired_directional_significance.py`: TY 11 × 3 paired-direction
  figure and date-aware statistics.
- `random_fixed40_crossday_control.py`: TS interference equal-trial-count
  control.
- `pairwise_bidirectional_neural_variability_band_match.py` and
  `decode_pairwise_neural_variability_band.py`: pair-specific neural
  variability matching and decoder refit.
- `match_all_trial_pair_variability.py` and
  `decode_variability_matched_crossday.py`: equal-N neural/kinematic matching.

## Time-resolved analyses

- `locked_position_time_resolved.py`
- `forget_control_position_time_resolved.py`
- `forget_control_position_cumulative_m2.py`
- `interference_position_time_resolved_random_fixed40.py`
- `interference_position_cumulative_m2_random_fixed40.py`
- `plot_interference_forget_instantaneous_and_cumulative.py`

## Mechanism families

- Variance/noise/support: `h0_*`, `structured_noise_null.py`,
  `coverage_controls*.py`, `r2_variance_*`, `remove_to_match_*`.
- CCA/manifold: `cca_*`, `nested_cca_*`, `manifold_geometry.py`,
  `subspace_inclusion.py`.
- Readout/private space: `private_*`, `readout_*`, `shared_*`.
- Kalman decomposition: `kalman_*`, `h_observation_*`,
  `h_intercept_centering_control.py`.
- Cross-animal/common-target: `decoder_consensus_crossanimal.py`,
  `common_target_source_test.py`, and their `summarize_*` companions.

Scripts with a matching `*_summary.py` or `plot_*.py` are post-processing
stages; they should not independently redefine sessions or paths.
