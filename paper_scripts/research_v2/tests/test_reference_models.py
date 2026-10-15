from __future__ import annotations

import unittest

import numpy as np

from research_v2.src.models.baselines import build_reference_model


class ReferenceModelTests(unittest.TestCase):
    def test_all_reference_families_fit_synthetic_data(self) -> None:
        rng = np.random.default_rng(42)
        features = rng.normal(size=(40, 8))
        labels = np.array([0, 1] * 20)
        configurations = {
            "logistic_regression": {"C": 1.0},
            "rbf_svm": {"C": 1.0},
            "hist_gradient_boosting": {"max_leaf_nodes": 15, "learning_rate": 0.1, "max_iter": 10},
        }
        for family, configuration in configurations.items():
            with self.subTest(family=family):
                model = build_reference_model(family, configuration, seed=7)
                model.fit(features, labels)
                self.assertEqual(len(model.predict(features)), len(labels))

    def test_unknown_reference_family_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown reference family"):
            build_reference_model("random_forest", {}, seed=7)


if __name__ == "__main__":
    unittest.main()
