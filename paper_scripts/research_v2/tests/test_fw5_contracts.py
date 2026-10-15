import unittest

import pandas as pd

from research_v2.src.execution.build_fw5_manifest import build_manifest_rows
from research_v2.src.execution.fw5_common import validate_manifest_row


class FW5ContractTests(unittest.TestCase):
    def test_manifest_expands_frozen_eligible_units_without_outer_values(self):
        eligibility = {
            "outer_loop_eligibility": [
                {"outer_domain": f"O{outer}", "eligible_configuration_ids": [f"C{outer}_{unit}" for unit in range(8 + (outer < 2))]}
                for outer in range(7)
            ]
        }
        selection_rows = []
        for entry in eligibility["outer_loop_eligibility"]:
            for index, configuration_id in enumerate(entry["eligible_configuration_ids"]):
                selection_rows.append({"outer_domain": entry["outer_domain"], "family": "flat_mlp", "configuration_id": configuration_id,
                                       "performance_only_selected": index == 0, "audit_guided_selected": index == 1})
        rows = build_manifest_rows(eligibility, pd.DataFrame(selection_rows), [7, 42, 123, 202, 314])
        self.assertEqual(len(rows), 290)
        self.assertEqual(sum(row["performance_only_selected"] for row in rows), 35)
        self.assertNotIn("AD_label", rows[0])

    def test_outer_access_requires_explicit_post_freeze_value(self):
        row = {
            "fw5_job_id": "FW5J000001", "outer_domain": "O", "family": "flat_mlp", "configuration_id": "C", "seed": "7",
            "performance_only_selected": "true", "audit_guided_selected": "false", "outer_test_access": "forbidden", "output_scope": "private_outer_prediction",
        }
        with self.assertRaises(ValueError):
            validate_manifest_row(row)
        row["outer_test_access"] = "permitted_after_freeze"
        self.assertIsNone(validate_manifest_row(row))


if __name__ == "__main__":
    unittest.main()
