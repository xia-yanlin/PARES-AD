from __future__ import annotations

import unittest

import numpy as np

from research_v2.src.execution.run_fw2_job import (
    deterministic_fit_calibration_indices,
    job_output_directory,
)


class FW2RunnerTests(unittest.TestCase):
    def test_formal_output_directory_is_job_scoped(self) -> None:
        directory = job_output_directory({"job_id": "FW2J99999"}, mode="formal", max_epochs=120)
        self.assertEqual(directory.name, "FW2J99999")

    def test_fit_calibration_split_is_deterministic_and_disjoint(self) -> None:
        labels = np.array([0] * 80 + [1] * 20)
        first_fit, first_calibration = deterministic_fit_calibration_indices(
            labels, seed=7, calibration_fraction=0.15
        )
        second_fit, second_calibration = deterministic_fit_calibration_indices(
            labels, seed=7, calibration_fraction=0.15
        )
        np.testing.assert_array_equal(first_fit, second_fit)
        np.testing.assert_array_equal(first_calibration, second_calibration)
        self.assertFalse(set(first_fit) & set(first_calibration))
        self.assertEqual(len(first_fit) + len(first_calibration), len(labels))
        self.assertEqual(set(labels[first_calibration]), {0, 1})

    def test_split_requires_both_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "both classes"):
            deterministic_fit_calibration_indices(
                np.zeros(20), seed=7, calibration_fraction=0.15
            )


if __name__ == "__main__":
    unittest.main()
