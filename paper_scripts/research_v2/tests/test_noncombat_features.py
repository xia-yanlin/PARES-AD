from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_v2.src.data.noncombat_features import (
    FrozenROITransformer,
    build_noncombat_mapping,
    build_structural_covariance_adjacency,
)


class NonCombatFeatureTests(unittest.TestCase):
    def synthetic_mapping(self) -> list[dict]:
        return [
            {"index": 0, "name": "lh_a", "type": "cortical", "thickness_col": "lh_a_thickness", "volume_col": "lh_a_volume"},
            {"index": 1, "name": "lh_b", "type": "subcortical", "thickness_col": None, "volume_col": "Left_b"},
        ]

    def synthetic_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lh_a_thickness": [1.0, 3.0, 100.0],
                "lh_a_volume": [100.0, 130.0, 10000.0],
                "Left_b": [80.0, 100.0, 9000.0],
                "EstimatedTotalIntraCranialVol": [1000.0, 1200.0, 8000.0],
            }
        )

    def test_transformer_fits_training_rows_only(self) -> None:
        frame = self.synthetic_frame()
        transformer = FrozenROITransformer.fit(frame, self.synthetic_mapping(), [0, 1])
        transformed = transformer.transform(frame)
        self.assertAlmostEqual(float(transformed[[0, 1], 0, 0].mean()), 0.0, places=6)
        self.assertGreater(float(abs(transformed[2, 0, 0])), 10.0)

        changed = frame.copy()
        changed.loc[2, "lh_a_thickness"] = 100000.0
        second = FrozenROITransformer.fit(changed, self.synthetic_mapping(), [0, 1])
        np.testing.assert_allclose(transformer.thickness_mean, second.thickness_mean)

    def test_graph_uses_only_supplied_training_rows(self) -> None:
        rng = np.random.default_rng(1)
        features = rng.normal(size=(6, 4, 3)).astype(np.float32)
        first = build_structural_covariance_adjacency(features, [0, 1, 2, 3, 4], top_k=1)
        features[5] = 9999.0
        second = build_structural_covariance_adjacency(features, [0, 1, 2, 3, 4], top_k=1)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, first.T)

    def test_mapping_requires_exact_frozen_node_count(self) -> None:
        node = {"index": 0, "name": "a", "type": "subcortical", "thickness_col": None, "volume_col": "Left_a_combat"}
        with self.assertRaisesRegex(ValueError, "94"):
            build_noncombat_mapping([node], ["Left_a"])


if __name__ == "__main__":
    unittest.main()
