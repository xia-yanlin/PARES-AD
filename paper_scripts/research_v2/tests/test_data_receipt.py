from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from research_v2.src.data.validate_data_receipt import validate_receipt


class DataReceiptTests(unittest.TestCase):
    def write_manifest(self, root: Path, relative: str, expected_hash: str, expected_bytes: int) -> Path:
        path = root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file_id",
                    "relative_path",
                    "cohort",
                    "data_class",
                    "expected_bytes",
                    "expected_sha256",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "file_id": "opaque-1",
                    "relative_path": relative,
                    "cohort": "SYNTHETIC",
                    "data_class": "test",
                    "expected_bytes": expected_bytes,
                    "expected_sha256": expected_hash,
                }
            )
        return path

    def test_valid_synthetic_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inbound = root / "inbound"
            inbound.mkdir()
            payload = b"synthetic-only\n"
            (inbound / "sample.bin").write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            manifest = self.write_manifest(root, "sample.bin", digest, len(payload))
            report = validate_receipt(inbound, manifest)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["passed_count"], 1)

    def test_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inbound = root / "inbound"
            inbound.mkdir()
            (inbound / "sample.bin").write_bytes(b"synthetic-only\n")
            manifest = self.write_manifest(root, "sample.bin", "0" * 64, 15)
            report = validate_receipt(inbound, manifest)
            self.assertEqual(report["status"], "FAIL")

    def test_path_traversal_fails_without_opening_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inbound = root / "inbound"
            inbound.mkdir()
            manifest = self.write_manifest(root, "../outside.bin", "", 0)
            report = validate_receipt(inbound, manifest)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("traversal", report["files"][0]["errors"][0])


if __name__ == "__main__":
    unittest.main()
