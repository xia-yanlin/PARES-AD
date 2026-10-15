import unittest

from research_v2.src.execution.aggregate_fw2_inner import (
    one_standard_error_set,
    select_performance_only,
    standard_error,
)


class FW2InnerAggregationTests(unittest.TestCase):
    def test_standard_error_uses_sample_sd_across_domains(self):
        self.assertAlmostEqual(standard_error([0.7, 0.8, 0.9]), 0.1 / (3 ** 0.5))

    def test_one_se_threshold_is_inclusive_and_deterministic(self):
        result = one_standard_error_set(
            {
                "config_b": [0.70, 0.80, 0.90],
                "config_a": [0.80, 0.80, 0.80],
                "config_c": [0.60, 0.60, 0.60],
            }
        )
        self.assertEqual(result["best_configuration_id"], "config_a")
        self.assertEqual(result["eligible_configuration_ids"], ["config_a", "config_b"])

    def test_performance_selection_uses_frozen_tie_break_order(self):
        self.assertEqual(
            select_performance_only(
                ["z", "a", "b"],
                {"z": 0.2, "a": 0.1, "b": 0.1},
                {"z": 50, "a": 20, "b": 10},
            ),
            "b",
        )
        self.assertEqual(
            select_performance_only(
                ["z", "a"],
                {"z": 0.1, "a": 0.1},
                {"z": 10, "a": 10},
            ),
            "a",
        )

    def test_one_se_requires_common_domain_coverage(self):
        with self.assertRaises(ValueError):
            one_standard_error_set({"a": [0.7, 0.8], "b": [0.9]})


if __name__ == "__main__":
    unittest.main()
