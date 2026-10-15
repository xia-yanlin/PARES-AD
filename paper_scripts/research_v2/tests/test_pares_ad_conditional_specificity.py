import unittest

import numpy as np
import pandas as pd

from research_v2.src.analysis.validate_pares_ad_conditional_specificity import (
    build_adjusted_design,
    build_control_features,
    state_risk,
)


class ParesAdConditionalSpecificityTests(unittest.TestCase):
    def test_control_features_use_frozen_disease_orientations(self):
        train = pd.DataFrame(
            {
                "EstimatedTotalIntraCranialVol": [1_200_000, 1_300_000, 1_400_000, 1_500_000],
                "Left_Caudate": [3000, 3100, 3200, 3300],
                "Right_Caudate": [3050, 3150, 3250, 3350],
                "Left_Pallidum": [1600, 1650, 1700, 1750],
                "Right_Pallidum": [1620, 1670, 1720, 1770],
                "lh_pericalcarine_thickness": [1.5, 1.6, 1.7, 1.8],
                "rh_pericalcarine_thickness": [1.6, 1.7, 1.8, 1.9],
                "lh_precentral_thickness": [2.0, 2.1, 2.2, 2.3],
                "rh_precentral_thickness": [2.1, 2.2, 2.3, 2.4],
            }
        )
        evaluation = train.iloc[[1]].copy()
        train_features, evaluation_features = build_control_features(train, evaluation)
        self.assertEqual(train_features.shape, (4, 4))
        self.assertEqual(evaluation_features.shape, (1, 4))
        self.assertTrue(np.isfinite(train_features).all())
        self.assertLess(train_features[-1, 2], train_features[0, 2])

    def test_state_risk_counts_cluster_units_without_ids_in_result(self):
        frame = pd.DataFrame(
            {
                "_RID": ["1", "1", "2", "3"],
                "model_error": [1, 1, 0, 0],
                "pathology_conflict": [True, True, False, False],
            }
        )
        result = state_risk(frame, "pathology_conflict")
        self.assertEqual(result["conflict_unique_participants"], 1)
        self.assertEqual(result["error_risk_ratio"], float("inf"))
        self.assertNotIn("_RID", result)

    def test_adjusted_design_retains_primary_conflict(self):
        cases = pd.DataFrame(
            {
                "pathology_conflict": [0, 1, 0, 1, 0, 1],
                "confidence_error_risk": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "model_predicted_class": [0, 0, 1, 1, 0, 1],
                "PHC_Age_T1": [60, 61, 62, 63, 64, 65],
                "education": [12, 13, 14, 15, 16, 17],
                "education_missing": [0, 0, 0, 0, 0, 0],
                "sex_missing": [0, 0, 0, 0, 0, 0],
                "sex": ["F", "M", "F", "M", "F", "M"],
                "phase": ["ADNI1", "ADNI1", "ADNI3", "ADNI3", "ADNI4", "ADNI4"],
                "manufacturer": ["GE", "GE", "Siemens", "Siemens", "Philips", "Philips"],
                "field_strength": ["1.5T", "1.5T", "3.0T", "3.0T", "3.0T", "3.0T"],
                "family": ["a", "a", "a", "b", "b", "b"],
                "configuration_id": ["x", "x", "x", "y", "y", "y"],
                "outer_domain": ["d1", "d1", "d2", "d2", "d3", "d3"],
                "model_error": [0, 1, 0, 1, 0, 1],
            }
        )
        design, outcomes, names = build_adjusted_design(cases)
        self.assertIn("pathology_conflict", names)
        self.assertEqual(design.shape[0], len(outcomes))


if __name__ == "__main__":
    unittest.main()
