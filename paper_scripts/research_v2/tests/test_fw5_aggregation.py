import unittest

import numpy as np
import pandas as pd

from research_v2.src.execution.aggregate_fw5_outer import aggregate_seed_predictions, association_row, e3_profiles, rule_predictions


class FW5AggregationTests(unittest.TestCase):
    def test_seed_predictions_are_averaged_before_metrics(self):
        rows = []
        for seed, probability in zip([7, 42, 123, 202, 314], [0.1, 0.2, 0.3, 0.4, 0.5]):
            rows.append({"outer_domain": "O", "family": "F", "configuration_id": "C", "RID": "1", "AD_label": 0,
                         "performance_only_selected": True, "audit_guided_selected": False, "seed": seed,
                         "raw_probability_AD": probability, "calibrated_probability_AD": probability})
        output = aggregate_seed_predictions(pd.DataFrame(rows))
        self.assertEqual(len(output), 1)
        self.assertAlmostEqual(output.iloc[0]["mean_calibrated_probability_AD"], 0.3)

    def test_association_fails_closed_below_ten_rows(self):
        frame = pd.DataFrame({"x": np.arange(9), "y": np.arange(9)})
        row = association_row("E", "component", frame, "x", "y")
        self.assertEqual(row["status"], "not_estimable")
        self.assertTrue(np.isnan(row["standardized_coefficient"]))

    def test_rule_predictions_require_identical_subject_label_keys(self):
        frame = pd.DataFrame([
            {"outer_domain": "O", "family": "F", "configuration_id": "P", "RID": "1", "AD_label": 0, "performance_only_selected": True, "audit_guided_selected": False},
            {"outer_domain": "O", "family": "F", "configuration_id": "A", "RID": "2", "AD_label": 1, "performance_only_selected": False, "audit_guided_selected": True},
        ])
        with self.assertRaises(RuntimeError):
            rule_predictions(frame)

    def test_e3_validated_use_keeps_interpretable_domain_count_without_duplicate_columns(self):
        inner = pd.DataFrame([{
            "outer_domain": "O", "family": "F", "configuration_id": "C", "model_kind": "neural",
            "macro_mean_calibrated_auroc": 0.8,
        }])
        readability = pd.DataFrame([{
            "outer_domain": "O", "family": "F", "configuration_id": "C", "representation_layer": "penultimate",
            "probe_family": "linear", "concept_id": "C001", "profile_mean_primary_selectivity": 0.1,
            "profile_variance_primary_selectivity": 0.01, "estimable_unit_count": 6,
        }])
        fw4 = pd.DataFrame([{
            "outer_domain": "O", "family": "F", "configuration_id": "C", "condition_id": "M_C001",
            "mean_excess_auroc_drop_interpretable_domains": 0.02,
            "between_inner_domain_excess_variance_valid_domains": 0.001,
            "valid_inner_domain_count": 6, "interpretable_inner_domain_count": 4,
        }])
        output = e3_profiles(inner, readability, fw4)
        self.assertTrue(output.columns.is_unique)
        usable = output[output["profile_source"].eq("FW4_validated_use")]
        self.assertEqual(int(usable.iloc[0]["valid_inner_domain_count"]), 4)


if __name__ == "__main__":
    unittest.main()
