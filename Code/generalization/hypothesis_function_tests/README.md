# Hypothesis Function Tests

Small, focused tests for the hypothesis-verification scripts in
`Code/generalization/analyses/`.

Start with pure helper functions and saved-result checks, then add expensive
NWB integration tests only when needed.

Run the H0 tests from the repository root with:

```bash
python -m unittest Code.generalization.hypothesis_function_tests.test_h0_snr_control
```

Run the H0 real-data noise visualization with:

```bash
python Code/generalization/hypothesis_function_tests/h0_real_data_noise_demo.py
```

That script loads one real R1/R2 pair, injects noise into the aligned training
latents to reach a target standard deviation, and saves before/after plots.
