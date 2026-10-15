from __future__ import annotations

import unittest

import numpy as np

from research_v2.src.data.roi_mapping import ConceptDefinition, aggregate_concept, train_only_standardize


class RoiMappingTests(unittest.TestCase):
    def test_sum_mapping(self) -> None:
        definition = ConceptDefinition("hippocampus", ("lh_hip", "rh_hip"), "sum", "mm3", "lower")
        result = aggregate_concept(
            {"lh_hip": [2.0, 3.0], "rh_hip": [4.0, 5.0]}, definition
        )
        np.testing.assert_allclose(result, [6.0, 8.0])

    def test_missing_registered_roi_fails(self) -> None:
        definition = ConceptDefinition("hippocampus", ("lh_hip", "rh_hip"), "sum", "mm3", "lower")
        with self.assertRaisesRegex(KeyError, "rh_hip"):
            aggregate_concept({"lh_hip": [2.0]}, definition)

    def test_standardization_uses_training_statistics(self) -> None:
        result = train_only_standardize([0.0, 2.0], [1.0, 3.0])
        np.testing.assert_allclose(result, [0.0, np.sqrt(2.0)])


if __name__ == "__main__":
    unittest.main()
