from __future__ import annotations

import unittest

from research_v2.src.execution.build_fw2_job_manifest import build_jobs


class FW2JobManifestTests(unittest.TestCase):
    def synthetic_registry(self) -> dict:
        return {
            "common_training": {"seeds": [1, 2, 3, 4, 5]},
            "neural_families": [
                {"family": f"n{family}", "configurations": [{"id": f"c{config}"} for config in range(3)]}
                for family in range(4)
            ],
            "strong_references": [
                {"family": f"r{family}", "configurations": [{"id": f"c{config}"} for config in range(3)]}
                for family in range(3)
            ],
        }

    def test_one_pair_has_expected_job_count(self) -> None:
        jobs = build_jobs(
            self.synthetic_registry(),
            [{"outer_domain": "D1", "inner_validation_domain": "D2"}],
        )
        self.assertEqual(len(jobs), 69)
        self.assertEqual(sum(job["model_kind"] == "neural" for job in jobs), 60)
        self.assertEqual(sum(job["model_kind"] == "reference" for job in jobs), 9)
        self.assertTrue(all(job["outer_test_access"] == "forbidden" for job in jobs))

    def test_job_ids_are_stable_and_unique(self) -> None:
        pairs = [
            {"outer_domain": "D1", "inner_validation_domain": "D2"},
            {"outer_domain": "D1", "inner_validation_domain": "D3"},
        ]
        first = build_jobs(self.synthetic_registry(), pairs)
        second = build_jobs(self.synthetic_registry(), pairs)
        self.assertEqual(first, second)
        self.assertEqual(len({job["job_id"] for job in first}), len(first))

    def test_outer_and_inner_cannot_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "must differ"):
            build_jobs(
                self.synthetic_registry(),
                [{"outer_domain": "D1", "inner_validation_domain": "D1"}],
            )


if __name__ == "__main__":
    unittest.main()
