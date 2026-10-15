import unittest

import numpy as np
import pandas as pd

from research_v2.src.analysis.build_ad_mri_explanation_system import (
    apply_acquisition_validation_gate,
    assert_private_identifiers_absent,
    conformal_absolute_quantile,
    clustered_state_risk_ratio,
    fit_monotone_pathology_model,
    fit_two_stage_behavior,
    reliability_state,
    selective_curve,
)


class ADMRIExplanationSystemTests(unittest.TestCase):
    def test_monotone_pathology_model_has_nonnegative_coefficients(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(200, 4))
        probability = 1 / (1 + np.exp(-(features[:, 0] + 0.5 * features[:, 2])))
        labels = (rng.random(200) < probability).astype(int)
        intercept, coefficients = fit_monotone_pathology_model(features, labels)
        self.assertTrue(np.isfinite(intercept))
        self.assertTrue(np.all(coefficients >= 0))
        contributions = features * coefficients
        self.assertTrue(
            np.allclose(intercept + contributions.sum(axis=1), intercept + features @ coefficients)
        )

    def test_two_stage_behavior_separates_acquisition_residual(self):
        rng = np.random.default_rng(11)
        rows = []
        for index in range(300):
            phase = "ADNI1" if index % 2 == 0 else "ADNI3"
            manufacturer = "Siemens" if index % 3 else "GE"
            field = "1.5T" if index % 2 == 0 else "3.0T"
            concepts = rng.normal(size=4)
            acquisition = 0.8 if manufacturer == "GE" else -0.2
            rows.append(
                {
                    **{f"C00{j + 1}_severity": concepts[j] for j in range(4)},
                    "phase": phase,
                    "manufacturer": manufacturer,
                    "field_strength": field,
                    "model_logit": concepts[0] + acquisition + rng.normal(scale=0.05),
                }
            )
        frame = pd.DataFrame(rows)
        result = fit_two_stage_behavior(frame.iloc[:240], frame.iloc[240:])
        residual = (
            frame.iloc[240:]["model_logit"].to_numpy()
            - np.asarray(result["full_eval"])
        )
        self.assertLess(float(np.mean(np.abs(residual))), 0.15)
        self.assertGreater(
            float(np.std(np.asarray(result["acquisition_eval"]))), 0.1
        )

    def test_conformal_quantile_uses_finite_sample_order_statistic(self):
        residuals = np.arange(1, 11, dtype=float)
        self.assertEqual(conformal_absolute_quantile(residuals, 0.90), 10.0)

    def test_reliability_state_priority(self):
        state, _ = reliability_state(False, 2.0, 0.1, True)
        self.assertEqual(state, "UNEXPLAINED_ABSTAIN")
        state, dominant = reliability_state(True, 0.5, 0.2, True)
        self.assertTrue(dominant)
        self.assertEqual(state, "ACQUISITION_SENSITIVE")
        state, _ = reliability_state(True, 0.1, 0.5, True)
        self.assertEqual(state, "PATHOLOGY_CONFLICT")
        state, _ = reliability_state(True, 0.1, 0.5, False)
        self.assertEqual(state, "PATHOLOGY_ALIGNED")

    def test_selective_curve_rewards_correct_error_ranking(self):
        errors = np.array([0, 0, 0, 1, 1])
        good_risk = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        bad_risk = good_risk[::-1]
        good = selective_curve(errors, good_risk)
        bad = selective_curve(errors, bad_risk)
        self.assertLess(good["aurc_50_100"], bad["aurc_50_100"])

    def test_output_identifier_guard(self):
        assert_private_identifiers_absent(pd.DataFrame({"case_key": ["CASE0001"]}))
        with self.assertRaises(ValueError):
            assert_private_identifiers_absent(pd.DataFrame({"RID": ["1"]}))

    def test_unvalidated_acquisition_component_is_converted_to_abstention(self):
        frame = pd.DataFrame(
            {
                "family": ["F"] * 4,
                "configuration_id": ["C"] * 4,
                "model_logit": [0.0, 1.0, 2.0, 3.0],
                "disease_behavior_logit": [0.0, 1.0, 2.0, 3.0],
                "reconstructed_model_logit": [3.0, 2.0, 1.0, 0.0],
                "reliability_state": ["ACQUISITION_SENSITIVE"] * 4,
            }
        )
        output = apply_acquisition_validation_gate(frame)
        self.assertFalse(output["acquisition_component_model_validated"].any())
        self.assertTrue(
            output["reliability_state"]
            .eq("ACQUISITION_UNVALIDATED_ABSTAIN")
            .all()
        )

    def test_clustered_state_risk_ratio_resamples_cases_not_rows(self):
        frame = pd.DataFrame(
            {
                "case_key": ["A", "A", "B", "B", "C", "C"],
                "reliability_state": [
                    "PATHOLOGY_CONFLICT",
                    "PATHOLOGY_ALIGNED",
                    "PATHOLOGY_CONFLICT",
                    "PATHOLOGY_ALIGNED",
                    "PATHOLOGY_CONFLICT",
                    "PATHOLOGY_ALIGNED",
                ],
                "model_error": [1, 0, 1, 0, 0, 1],
            }
        )
        result = clustered_state_risk_ratio(frame, "PATHOLOGY_CONFLICT")
        self.assertEqual(result["state_n"], 3)
        self.assertEqual(result["reference_n"], 3)
        self.assertAlmostEqual(result["risk_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
