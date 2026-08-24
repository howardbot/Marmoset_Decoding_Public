"""Tests for the cross-fitted residual-variance removal dose response."""
from __future__ import annotations

import inspect
import unittest

import numpy as np
import pandas as pd

from Code.generalization.analyses import r2_variance_removal_dose_response_crossfit as dose


def make_meta(n_trials: int, bins_per_trial: int) -> pd.DataFrame:
    return pd.DataFrame({
        "trial_number": np.repeat(np.arange(n_trials), bins_per_trial),
    })


class TestVarianceRemovalDoseResponse(unittest.TestCase):
    def test_trial_folds_keep_whole_trials_and_cover_each_trial_once(self):
        meta = make_meta(20, 5)
        folds = dose.trial_folds(meta, 5, 31)
        self.assertEqual(len(folds), 5)
        coverage = np.zeros(len(meta), dtype=int)
        for evaluation in folds:
            coverage += evaluation.astype(int)
            for _, indices in meta.groupby("trial_number").indices.items():
                values = evaluation[np.asarray(indices)]
                self.assertTrue(values.all() or (~values).all())
        np.testing.assert_array_equal(coverage, np.ones(len(meta), dtype=int))

    def test_match_scaler_uses_calibration_only(self):
        meta = make_meta(6, 5)
        time = np.linspace(0, 1, len(meta))[:, None]
        kin1 = np.column_stack([time, time**2, time**3])
        kin2 = kin1 + 0.1
        calibration = meta["trial_number"].isin([0, 1, 2, 3]).to_numpy()
        evaluation = ~calibration
        scaler_a = dose.fit_match_scaler(
            kin1, meta, calibration, kin2, meta, calibration
        )
        changed = kin2.copy()
        changed[evaluation] = 1e6
        scaler_b = dose.fit_match_scaler(
            kin1, meta, calibration, changed, meta, calibration
        )
        np.testing.assert_allclose(scaler_a.mean, scaler_b.mean)
        np.testing.assert_allclose(scaler_a.scale, scaler_b.scale)

    def test_cutoff_is_learned_on_calibration_not_forced_on_evaluation(self):
        calibration = np.arange(10.0)
        evaluation = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        cal_keep, eval_keep, cutoff = dose.calibration_cutoff_selection(
            calibration, evaluation, 0.5
        )
        np.testing.assert_array_equal(cal_keep, np.arange(5))
        np.testing.assert_array_equal(eval_keep, np.asarray([0, 1, 2]))
        self.assertEqual(cutoff, 4.0)

        cal_all, eval_all, all_cutoff = dose.calibration_cutoff_selection(
            calibration, evaluation, 1.0
        )
        np.testing.assert_array_equal(cal_all, np.arange(10))
        np.testing.assert_array_equal(eval_all, np.arange(6))
        self.assertTrue(np.isinf(all_cutoff))

    def test_neural_gate_score_has_no_kinematic_argument(self):
        parameters = inspect.signature(dose.trial_change_scores).parameters
        self.assertNotIn("kinematics", parameters)
        self.assertNotIn("movement", parameters)
        meta = make_meta(2, 4)
        activity = np.arange(24.0).reshape(8, 3)
        quiet = activity.copy()
        quiet[meta.trial_number.to_numpy() == 1] -= 2.0
        scores = dose.trial_change_scores(activity, quiet, meta, [0, 1])
        self.assertAlmostEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[1], 4.0)

    def test_gate_fit_ignores_evaluation_residuals(self):
        rng = np.random.default_rng(4)
        activity1 = rng.normal(size=(40, 4))
        activity2 = rng.normal(size=(40, 4))
        signal1 = rng.normal(size=(40, 4))
        signal2 = rng.normal(size=(40, 4))
        residual1 = rng.normal(scale=0.5, size=(40, 4))
        residual2 = rng.normal(scale=1.2, size=(40, 4))
        calibration = np.zeros(40, dtype=bool)
        calibration[:24] = True

        quiet_a, diagnostics_a = dose.fit_neural_only_gate(
            activity1,
            activity2,
            signal1,
            signal2,
            residual1,
            residual2,
            calibration,
            calibration,
            "r2",
        )
        changed_signal2 = signal2.copy()
        changed_residual2 = residual2.copy()
        changed_signal2[~calibration] = 1e8
        changed_residual2[~calibration] = -1e8
        quiet_b, diagnostics_b = dose.fit_neural_only_gate(
            activity1,
            activity2,
            signal1,
            changed_signal2,
            residual1,
            changed_residual2,
            calibration,
            calibration,
            "r2",
        )
        np.testing.assert_allclose(quiet_a, quiet_b)
        self.assertEqual(diagnostics_a["gate_fit_id"], diagnostics_b["gate_fit_id"])

    def test_paired_subsets_never_create_orphans(self):
        pairs = dose.TrialPairs(
            r1_trials=np.asarray([10, 11, 12, 13]),
            r2_trials=np.asarray([20, 21, 22, 23]),
            distance=np.asarray([0.1, 0.2, 0.3, 0.4]),
        )
        subset = pairs.subset(np.asarray([0, 2]))
        np.testing.assert_array_equal(subset.r1_trials, [10, 12])
        np.testing.assert_array_equal(subset.r2_trials, [20, 22])
        self.assertEqual(len(subset.r1_trials), len(subset.r2_trials))

    def test_random_subsets_are_exact_count_and_reproducible(self):
        first = dose.random_pair_indices(20, 7, np.random.default_rng(12))
        second = dose.random_pair_indices(20, 7, np.random.default_rng(12))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(len(np.unique(first)), 7)

    def test_factorial_routing_uses_requested_train_and_evaluation_pairs(self):
        def pair(label: int) -> dose.TrialPairs:
            return dose.TrialPairs(
                r1_trials=np.asarray([label]),
                r2_trials=np.asarray([label + 100]),
                distance=np.asarray([float(label)]),
            )

        variance_calibration = pair(1)
        variance_evaluation = pair(2)
        random_calibration = pair(3)
        random_evaluation = pair(4)
        cells = dose.factorial_pair_sets(
            variance_calibration,
            variance_evaluation,
            random_calibration,
            random_evaluation,
        )
        self.assertIs(cells["random_both"][0], random_calibration)
        self.assertIs(cells["random_both"][1], random_evaluation)
        self.assertIs(cells["variance_train_only"][0], variance_calibration)
        self.assertIs(cells["variance_train_only"][1], random_evaluation)
        self.assertIs(cells["variance_eval_only"][0], random_calibration)
        self.assertIs(cells["variance_eval_only"][1], variance_evaluation)
        self.assertIs(cells["variance_both"][0], variance_calibration)
        self.assertIs(cells["variance_both"][1], variance_evaluation)

    def test_fold_effect_uses_equal_n_random_mean(self):
        base = {
            "r1": "a",
            "r2": "b",
            "r1_session": "r1",
            "r2_session": "r2",
            "selector": "neural_only",
            "retention_fraction": 0.5,
            "repeat": 0,
            "fold": 0,
            "valid": True,
            "forward_corr": 0.4,
            "reverse_corr": 0.6,
            "gap_corr": 0.2,
            "gap_fisher_z": 0.3,
            "own_r1_corr": 0.7,
            "own_r2_corr": 0.7,
            "calibration_residual_variance_ratio": 1.0,
            "evaluation_residual_variance_ratio": 1.0,
            "calibration_kinematic_distance": 1.0,
            "evaluation_kinematic_distance": 1.0,
            "calibration_fraction_actual": 0.5,
            "evaluation_fraction_actual": 0.5,
        }
        rows = []
        selected = dict(base, condition="variance_both", random_rep=-1)
        selected["gap_fisher_z"] = 0.1
        rows.append(selected)
        for random_rep, random_gap in enumerate([0.2, 0.4]):
            random = dict(
                base,
                condition="random_both",
                random_rep=random_rep,
                gap_fisher_z=random_gap,
            )
            train = dict(
                base,
                condition="variance_train_only",
                random_rep=random_rep,
                gap_fisher_z=random_gap - 0.05,
            )
            evaluation = dict(
                base,
                condition="variance_eval_only",
                random_rep=random_rep,
                gap_fisher_z=random_gap - 0.08,
            )
            rows.extend([random, train, evaluation])
        effect = dose.fold_effects(pd.DataFrame(rows)).iloc[0]
        self.assertAlmostEqual(effect.random_gap_fisher_z, 0.3)
        self.assertAlmostEqual(effect.selected_gap_fisher_z, 0.1)
        self.assertAlmostEqual(effect.delta_gap_fisher_z, -0.2)
        self.assertAlmostEqual(effect.delta_train_only_gap_fisher_z, -0.05)
        self.assertAlmostEqual(effect.delta_eval_only_gap_fisher_z, -0.08)
        self.assertAlmostEqual(effect.interaction_gap_fisher_z, -0.07)

    def test_three_same_sign_r2_days_have_minimum_two_sided_p_point_25(self):
        self.assertAlmostEqual(dose.exact_sign_flip_p([0.1, 0.2, 0.3]), 0.25)

    def test_validity_coverage_counts_folds_not_random_condition_rows(self):
        identifiers = {
            "r1": "a",
            "r2": "b",
            "r1_session": "r1",
            "r2_session": "r2",
            "selector": "neural_only",
            "retention_fraction": 0.4,
            "repeat": 0,
            "high_variance_side": "r2",
        }
        rows = [
            dict(identifiers, fold=0, condition="variance_both", valid=True),
            dict(identifiers, fold=0, condition="random_both", valid=True),
            dict(identifiers, fold=0, condition="random_both", valid=True),
            dict(
                identifiers,
                fold=1,
                condition="invalid_too_few_pairs",
                valid=False,
            ),
        ]
        pairs, days, overall = dose.validity_coverage(pd.DataFrame(rows))
        for frame in (pairs, days, overall):
            self.assertEqual(int(frame.iloc[0].n_fold_instances_total), 2)
            self.assertEqual(int(frame.iloc[0].n_fold_instances_valid), 1)
            self.assertAlmostEqual(float(frame.iloc[0].valid_fold_fraction), 0.5)
        self.assertEqual(int(overall.iloc[0].n_session_pairs_with_valid), 1)
        self.assertEqual(int(overall.iloc[0].n_r2_days_with_valid), 1)


if __name__ == "__main__":
    unittest.main()
