"""Synthetic edge-case tests; no participant data or historical pipelines."""
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pares_ad import fixed_budget_yield, case_report, committee_features, Cohort, crossfit_scores


class ReleaseTests(unittest.TestCase):
    def test_zero_review_mass_stays_undefined(self):
        self.assertTrue(math.isnan(fixed_budget_yield([.8, .2], [1, 0], .1)))
        self.assertTrue(math.isnan(fixed_budget_yield([.8, .2], [1, 0], 0)))

    def test_fractional_boundary_matches_cohort_engine(self):
        n = 30
        scores = np.arange(n, dtype=float)[::-1]
        errors = np.array([1, 0, 1] + [0] * (n - 3))
        frame = pd.DataFrame({'participant_key': [str(i) for i in range(n)],
                              'row_key': [f'{i:03d}' for i in range(n)], 'AD_label': np.zeros(n),
                              'model_probability_AD': np.where(errors, .9, .1),
                              'review_weight': np.ones(n), 'configuration_id': ['one'] * n})
        cohort = Cohort('synthetic', frame, ['configuration_id'])
        cohort.set_scores({'score': scores})
        selected, _ = cohort.evaluate(['score'], np.ones((1, n), dtype=int), np.arange(n), floor=False)
        for i, budget in enumerate([.05, .1, .2]):
            self.assertAlmostEqual(fixed_budget_yield(scores, errors, budget, floor=False),
                                   selected[0, i, 0, 1] / selected[0, i, 0, 0])
        self.assertAlmostEqual(fixed_budget_yield(scores, errors, .05, floor=False), 2 / 3)

    def test_budget_validation_and_ties(self):
        self.assertEqual(fixed_budget_yield([.5, .5], [1, 0], .5), 1)
        self.assertEqual(fixed_budget_yield([.5, .5], [1, 0], 1), .5)
        for scores, errors, budget in [([], [], .1), ([1], [0, 1], .1), ([float('nan')], [0], .1), ([1], [2], .1), ([1], [1], -1)]:
            with self.assertRaises(ValueError):
                fixed_budget_yield(scores, errors, budget)

    def test_report_requires_four_contributions(self):
        with self.assertRaises(ValueError):
            case_report('synthetic', .8, 0, [1, 2, 3], [1, 1, 1, 1], .1, 1)

    def test_report_extreme_finite_logit(self):
        text = case_report('synthetic', .8, -1000, [0] * 4, [1] * 4, .1, 1)
        self.assertIn('0.0% -> CN', text)
        with self.assertRaises(ValueError):
            case_report('synthetic', 1.2, 0, [0] * 4, [1] * 4, .1, 1)

    def test_disagreement_formula_is_not_entropy(self):
        frame = pd.DataFrame({'participant_key': ['a'] * 4, 'model_probability_AD': [.9, .1, .1, .1]})
        sd, disagreement = committee_features(frame)
        np.testing.assert_allclose(disagreement, .75)
        np.testing.assert_allclose(sd, np.std([.9, .1, .1, .1]))

    def test_crossfit_keeps_participants_together_and_weight_choice(self):
        n = 100
        frame = pd.DataFrame({'participant_key': np.repeat(np.arange(n), 2),
                              'model_probability_AD': np.repeat(np.linspace(.1, .9, n), 2),
                              'pathology_probability_AD': np.repeat(np.linspace(.9, .1, n), 2),
                              'AD_label': np.repeat(np.arange(n) % 2, 2),
                              'review_weight': np.repeat(np.linspace(.5, 1.5, n), 2)})
        for weight_col in [None, 'review_weight']:
            calls = []
            def fit(X, y, weights):
                calls.append(weights.copy())
                return set(map(tuple, X))
            def predict(training_rows, X):
                self.assertTrue(training_rows.isdisjoint(map(tuple, X)))
                return np.full(len(X), .5)
            with patch('pares_ad.scores.fit_head', side_effect=fit), patch('pares_ad.scores.predict_head', side_effect=predict):
                result = crossfit_scores(frame, weight_col=weight_col)
            self.assertEqual(len(calls), 20)
            self.assertTrue(all(np.isfinite(v).all() for v in result.values()))
            self.assertTrue(all(np.isclose(w.mean(), 1) for w in calls))
            if weight_col is None:
                self.assertTrue(all(np.allclose(w, 1) for w in calls))
            else:
                self.assertTrue(all(w.max() > w.min() for w in calls))


    def test_crossfit_default_uses_configuration_weights(self):
        n = 60
        frame = pd.DataFrame({'participant_key': np.repeat(np.arange(n), 2),
                              'model_probability_AD': np.repeat(np.linspace(.1, .9, n), 2),
                              'pathology_probability_AD': np.repeat(np.linspace(.9, .1, n), 2),
                              'AD_label': np.repeat(np.arange(n) % 2, 2),
                              'review_weight': np.repeat(np.linspace(.5, 1.5, n), 2)})
        default = crossfit_scores(frame)
        explicit = crossfit_scores(frame, weight_col='review_weight')
        for k in default:
            np.testing.assert_allclose(default[k], explicit[k])

if __name__ == '__main__':
    unittest.main()
