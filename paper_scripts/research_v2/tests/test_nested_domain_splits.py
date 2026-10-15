from __future__ import annotations

import unittest

import pandas as pd

from research_v2.src.data.build_nested_domain_splits import (
    build_nested_membership,
    build_split_registry,
    validate_membership,
)


class NestedDomainSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eligible = [f"D{index}" for index in range(1, 8)]
        self.stress = ["S1", "S2"]
        rows = []
        rid = 1
        for domain in self.eligible + self.stress:
            for label in [0, 1]:
                rows.append({"RID": str(rid), "label": label, "acquisition_domain": domain})
                rid += 1
        self.assignments = pd.DataFrame(rows)

    def test_nested_membership_passes_all_invariants(self) -> None:
        membership = build_nested_membership(self.assignments, self.eligible, self.stress)
        checks = validate_membership(membership, self.eligible, self.stress)
        self.assertTrue(all(checks.values()))
        self.assertEqual(len(membership), len(self.assignments) * 7)

    def test_registry_contains_42_outer_inner_pairs(self) -> None:
        membership = build_nested_membership(self.assignments, self.eligible, self.stress)
        registry = build_split_registry(membership, self.eligible, self.stress)
        self.assertEqual(len(registry), 42)
        self.assertTrue(all(row["stress_only_in_inner_train_n"] == 4 for row in registry))

    def test_outer_rows_cannot_enter_inner_training(self) -> None:
        membership = build_nested_membership(self.assignments, self.eligible, self.stress)
        outer = membership[membership["outer_domain"].eq("D1")]
        outer_test = outer[outer["outer_role"].eq("test")]
        for inner_domain in self.eligible[1:]:
            self.assertTrue((outer_test[f"inner_role__{inner_domain}"] == "excluded_outer").all())

    def test_stress_rows_are_never_test_or_validation(self) -> None:
        membership = build_nested_membership(self.assignments, self.eligible, self.stress)
        stress = membership[membership["acquisition_domain"].isin(self.stress)]
        self.assertTrue((stress["outer_role"] == "train_pool").all())
        for domain in self.eligible:
            values = stress[f"inner_role__{domain}"]
            active = values[values.ne("not_applicable")]
            self.assertTrue((active == "training").all())


if __name__ == "__main__":
    unittest.main()
