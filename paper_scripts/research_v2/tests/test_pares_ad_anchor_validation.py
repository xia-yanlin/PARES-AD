import unittest

import numpy as np
import pandas as pd

from research_v2.src.analysis.run_pares_ad_anchor_validation import (
    crossfit_continuous,
    nearest_anchor,
)


class ParesAdAnchorValidationTests(unittest.TestCase):
    def test_nearest_anchor_prefers_prior_on_equal_gap(self):
        confirmation = pd.DataFrame(
            {
                "_RID": ["1"],
                "_scan_date": pd.to_datetime(["2025-01-10"]),
                "AD_label": [1],
                "outer_domain": ["d"],
                "PHC_Age_T1": [70.0],
                "phase": ["ADNI3"],
                "manufacturer": ["Siemens"],
                "field_strength": ["3.0T"],
            }
        )
        anchors = pd.DataFrame(
            {
                "RID": [1, 1],
                "DATE": ["2025-01-09", "2025-01-11"],
                "VALUE": [1.1, 1.9],
            }
        )
        linked = nearest_anchor(
            confirmation,
            anchors,
            date_column="DATE",
            value_column="VALUE",
            window_days=365,
            extra_mask=pd.Series([True, True]),
        )
        self.assertEqual(float(linked.loc[0, "_anchor_value"]), 1.1)

    def test_crossfit_continuous_returns_complete_predictions(self):
        count = 50
        score = np.linspace(-2, 2, count)
        frame = pd.DataFrame(
            {
                "PHC_Age_T1": np.linspace(60, 80, count),
                "education": np.linspace(10, 20, count),
                "education_missing": np.zeros(count),
                "sex_missing": np.zeros(count),
                "absolute_gap_days": np.arange(count) + 1,
                "AD_label": np.tile([0, 1], count // 2),
                "sex": np.tile(["F", "M"], count // 2),
                "phase": np.tile(["ADNI1", "ADNI3"], count // 2),
                "manufacturer": np.tile(["GE", "Siemens"], count // 2),
                "field_strength": np.tile(["1.5T", "3.0T"], count // 2),
                "pathology_logit": score,
                "control_logit": np.cos(np.arange(count)),
            }
        )
        outcome = 0.8 * score + np.sin(np.arange(count)) * 0.1
        residual, predictions, delta_r2 = crossfit_continuous(
            frame, outcome, ("pathology_logit", "control_logit")
        )
        self.assertTrue(np.isfinite(residual).all())
        self.assertTrue(all(np.isfinite(values).all() for values in predictions.values()))
        self.assertGreater(delta_r2["pathology_logit"], delta_r2["control_logit"])


if __name__ == "__main__":
    unittest.main()
