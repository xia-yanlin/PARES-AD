from __future__ import annotations

import unittest

import numpy as np

from research_v2.src.statistics.metrics import binary_metrics, fit_platt, sigmoid


class MetricTests(unittest.TestCase):
    def test_sigmoid_is_stable_for_extreme_scores(self) -> None:
        probabilities = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities, [0.0, 0.5, 1.0], atol=1e-12)

    def test_platt_outputs_valid_probabilities(self) -> None:
        scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        calibrator = fit_platt(scores, labels)
        probabilities = calibrator.predict(scores)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertGreater(probabilities[-1], probabilities[0])

    def test_binary_metrics_return_registered_fields(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = binary_metrics(labels, probabilities)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertIn("calibration_slope", metrics)
        self.assertIn("ece_10", metrics)

    def test_metrics_require_both_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "both outcome classes"):
            binary_metrics(np.zeros(5), np.full(5, 0.1))


if __name__ == "__main__":
    unittest.main()
