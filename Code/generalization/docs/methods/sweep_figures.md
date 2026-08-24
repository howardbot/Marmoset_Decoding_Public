# Cross-day sweep + figures — pipeline & reproduction

Milestone: a full parameter sweep of the cross-day decoder, plus the figure set
that reports the R1↔R2 (retrograde-interference) result for one marmoset (TSAL).

Two stages:
1. **Sweep** — `big_sweep_phase1_withinday.py`, `big_sweep_phase2_crossday.py`
   → write long-form CSVs of decoding scores over a parameter grid.
2. **Figures** — `plotting_common.py` + `plot_fig*.py` / `plot_supp_*.py`
   → read the cross-day CSV at one locked config and render the figures.

> **Outputs are git-ignored.** `Results/` (CSVs and PNGs) is excluded by
> `.gitignore`. Only the *code* and this doc are committed. Re-run the sweep,
> then the plot scripts, to regenerate everything (see Reproduction below).

---

## Read this first — smoothing convention

Hand trajectories are low-pass filtered with a **Butterworth filter at 6 Hz**
applied via **`scipy.signal.sosfiltfilt` (zero-phase, forward + backward)**.
`filtfilt` runs the filter twice, so **the effective order is doubled**: the
sweep label `butter_o2` = a 2nd-order Butterworth applied twice = an
**effective 4th-order zero-lag** filter (the standard "4th-order zero-lag
Butterworth" of the biomechanics literature). Zero-phase means the smoothed
trajectory is not shifted in time — important for aligning it with neural
activity. Implemented in `decoder_utils.smooth_relative_trajectory_butter`;
the original Savitzky-Golay smoother is kept as the `savgol` option.

---

## Locked configuration (every figure uses this)

Defined once in `plotting_common.LOCKED_CONFIG`:

| knob | value | why |
|---|---|---|
| `bin_size_ms` | 30 | near-best cross-day Kalman; matches Gallego 2020 |
| `smoother` | `butter_o2` | 6 Hz, effective 4th-order zero-lag |
| `decoder` | `kalman` | best decoder; state-space prior = soft regularization |
| `lag_ms` | 0 | strongest position signal; cleanest interpretation |
| `target_mode` | `relative_velocity` | primary; `relative_position` shown in parallel |
| `outlier_mode` | `exclude` | drop 0828 trial 41 (tracking failure, see below) |

The metric on every figure is **`corr`** = per-trial Pearson correlation
between true and decoded movement, averaged over trials (the sweep stores it as
`M2_mean`; figures label it `corr`).

**Keep this in sync with the sweep.** If the sweep grid drops a value
`LOCKED_CONFIG` points at, every figure breaks. There is a reminder note at the
top of `big_sweep_phase2_crossday.py`.

---

## Stage 1 — sweep

### Grid (both phases)
- `bin_size_ms` ∈ {10, 20, 30, 40, 50}
- `smoother` ∈ {savgol, butter_o2, butter_o4}
- `target_mode` ∈ {relative_velocity, relative_position}
- `lag_ms` = bin-integer multiples of [0, 150]
- `decoder` = Kalman (KordingLab) + Wiener (KordingLab); Wiener `history_ms` ∈ {50, 100}
- neural smoothing σ = 50 ms (fixed); units = good+mua; trials = S+F (fixed)
- `outlier_mode` = include / exclude 0828 trial 41

### Scripts
| script | scope | output CSV |
|---|---|---|
| `big_sweep_phase1_withinday.py` | within-day, 5-fold CV per session | `Results/generalization/big_sweep_withinday_long.csv` |
| `big_sweep_phase2_crossday.py` | 15×15 cross-day matrix (diagonal = 5-fold CV; off-diagonal = CCA-aligned) | `Results/generalization/big_sweep_crossday_long.csv` |

Both run on `N_WORKERS = 3` processes and **checkpoint per cell** — re-running
skips any cell already in the CSV, so they are safe to Ctrl-C and resume.

The figures only need Phase 2. Phase 1 is the parameter-selection study that
justified `LOCKED_CONFIG`.

### 0828 trial 41
A pose-tracking failure (wrist coordinates blow up to ~1500× normal scale for
~700 ms). Diagnosed in `compare_0813_vs_0828.py`. Hardcoded as
`EXCLUDE_TRIALS = {0828: [41]}` in both sweep scripts. Including it drags every
0828-involving cell to ≈0; excluding it restores them. No Butterworth /
Savgol / median filter removes it — it is a wrong reconstruction, not noise — so
the trial is dropped. `outlier_mode` keeps both versions for the comparison.

---

## Stage 2 — figures

`plotting_common.py` holds `LOCKED_CONFIG`, session order, the R1/R2 boundary,
colors, and helpers (`filter_locked`, `pivot_matrix`, `forward_reverse_pairs`,
`day_gap`, …). Every plot script imports from it.

| script | figure | shows |
|---|---|---|
| `plot_fig1_crossday_matrix.py` | F1 | 15×15 transfer matrix (Kalman vel + pos), red-framed diagonal, R1/R2 split |
| `plot_fig2_r1r2_asymmetry.py` | F2 | deviation-from-baseline bars: forward (R1→R2) drops, reverse (R2→R1) holds |
| `plot_fig3_time_drift.py` | F3 | corr vs calendar day-gap; forward<reverse at matched gaps (time controlled) |
| `plot_fig4_learning_trend.py` | F4 | within-day corr vs session, OLS fit + R²/p (Sami's "d") |
| `plot_fig5_within_vs_cross.py` | F5 | per-session within-day vs mean cross-day (generalization drop) |
| `plot_supp_decoder_consistency.py` | S2 | Wiener heatmaps + Kalman-vs-Wiener scatter (decoder-independent) |
| `plot_supp_target_consistency.py` | S3 | velocity-vs-position scatter, colored by R1/R2 direction |
| `plot_supp_trial41.py` | — | include / exclude / Δ matrix for 0828 trial 41 |

All write PNGs to `Results/generalization/figures/` (git-ignored).

### Key findings (plain language)
- **Transfer works but drops ~40%** off the training day (F1, F5).
- **R1→R2 is the directional signal** (F2): a decoder built before the
  interference task struggles to read activity after it, but a decoder built
  after reads the earlier activity fine. Symmetric "the periods differ" would
  drop both — it doesn't.
- **Not just time** (F3): forward and reverse use the *same* session pairs, so
  identical day-gaps — yet reverse stays higher. (Clean for position; velocity
  drift is noisy so we lean on the matched-pair argument, not the extrapolation.)
- **Learning trend** (F4): within-day decoding rises across R1, significant for
  position (R²≈0.58, p≈0.003), weak/non-significant for velocity (R²≈0.19,
  p≈0.14).
- **Robustness**: Wiener reproduces Kalman (S2, ρ≈0.92); the asymmetry shows in
  both velocity and position (S3).

---

## Coverage vs Sami's Figure-3 manuscript

| Sami | idea | our figure | status |
|---|---|---|---|
| a | method schematic | — | not done (illustration) |
| b | example aligned dimensions | — | not done (illustration) |
| c | generalization heatmap | F1 | done |
| d | within-day regression, significant | F4 | done (OLS + R²/p; position significant) |
| e | "bright spot in later learning" | (visible in F1) | not a standalone figure |
| f | corr by day-gap | F3 | done |
| g | interference drop | F2 | done |
| h | isolate interference from time/forgetting | F3 | done |
| "both monkeys" | replicate c–g on a 2nd animal | — | only one marmoset's data so far |

---

## Reproduction

```bash
# from repo root, with the HatLab env
PY=/opt/anaconda3/envs/HatLab/bin/python

# 1. sweep (Phase 2 is what the figures need; ~hours, checkpointed)
$PY Code/generalization/big_sweep_phase2_crossday.py
# (optional) Phase 1 within-day parameter study
$PY Code/generalization/big_sweep_phase1_withinday.py

# 2. figures (seconds each; read the cross-day CSV at LOCKED_CONFIG)
for f in fig1_crossday_matrix fig2_r1r2_asymmetry fig3_time_drift \
         fig4_learning_trend fig5_within_vs_cross \
         supp_decoder_consistency supp_target_consistency supp_trial41; do
  $PY Code/generalization/plot_${f}.py
done
```

Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, and the KordingLab
`Neural_Decoding` package (Kalman + Wiener decoders).
