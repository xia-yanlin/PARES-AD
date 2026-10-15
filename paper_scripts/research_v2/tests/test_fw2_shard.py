from __future__ import annotations

import unittest

from research_v2.src.execution.run_fw2_shard import select_manifest_jobs


class FW2ShardTests(unittest.TestCase):
    def test_selection_is_bounded_and_ordered(self) -> None:
        rows = [{"job_id": f"J{index}"} for index in range(10)]
        selected = select_manifest_jobs(rows, 3, 4)
        self.assertEqual([row["job_id"] for row in selected], ["J3", "J4", "J5", "J6"])

    def test_selection_truncates_at_manifest_end(self) -> None:
        rows = [{"job_id": f"J{index}"} for index in range(5)]
        self.assertEqual(len(select_manifest_jobs(rows, 3, 10)), 2)

    def test_invalid_selection_fails(self) -> None:
        with self.assertRaises(ValueError):
            select_manifest_jobs([], -1, 1)
        with self.assertRaises(ValueError):
            select_manifest_jobs([], 0, 0)


if __name__ == "__main__":
    unittest.main()
