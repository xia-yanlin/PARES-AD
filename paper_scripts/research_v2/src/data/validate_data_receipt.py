"""Validate a controlled-data receipt manifest without exposing file contents.

The manifest is a CSV with these required columns:
file_id,relative_path,cohort,data_class,expected_bytes,expected_sha256

Expected hashes may be blank when the provider does not supply them. The report
contains paths/hashes/size/status only; it never loads or prints table values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


REQUIRED_COLUMNS = {
    "file_id",
    "relative_path",
    "cohort",
    "data_class",
    "expected_bytes",
    "expected_sha256",
}


@dataclass(frozen=True)
class FileResult:
    file_id: str
    relative_path: str
    status: str
    actual_bytes: int | None
    actual_sha256: str | None
    errors: tuple[str, ...]


def safe_relative_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("relative_path must be a non-empty relative path without traversal")
    if ":" in pure.parts[0]:
        raise ValueError("relative_path must not contain a drive prefix")
    return Path(*pure.parts)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"manifest missing columns: {', '.join(missing)}")
        rows = list(reader)
    file_ids = [row["file_id"] for row in rows]
    if not rows:
        raise ValueError("manifest contains no files")
    if len(set(file_ids)) != len(file_ids):
        raise ValueError("manifest file_id values must be unique")
    return rows


def validate_one(root: Path, row: dict[str, str]) -> FileResult:
    errors: list[str] = []
    actual_bytes: int | None = None
    actual_sha256: str | None = None
    try:
        relative = safe_relative_path(row["relative_path"])
    except ValueError as exc:
        return FileResult(row["file_id"], row["relative_path"], "FAIL", None, None, (str(exc),))

    target = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root not in target.parents:
        return FileResult(row["file_id"], relative.as_posix(), "FAIL", None, None, ("resolved path escaped input root",))
    if not target.is_file():
        return FileResult(row["file_id"], relative.as_posix(), "FAIL", None, None, ("file missing",))

    actual_bytes = target.stat().st_size
    expected_bytes = row["expected_bytes"].strip()
    if expected_bytes:
        try:
            if actual_bytes != int(expected_bytes):
                errors.append("byte size mismatch")
        except ValueError:
            errors.append("expected_bytes is not an integer")

    actual_sha256 = sha256_file(target)
    expected_sha256 = row["expected_sha256"].strip().lower()
    if expected_sha256:
        if len(expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha256):
            errors.append("expected_sha256 is malformed")
        elif actual_sha256 != expected_sha256:
            errors.append("SHA-256 mismatch")

    return FileResult(
        row["file_id"],
        relative.as_posix(),
        "PASS" if not errors else "FAIL",
        actual_bytes,
        actual_sha256,
        tuple(errors),
    )


def validate_receipt(input_root: Path, manifest_path: Path) -> dict[str, object]:
    if not input_root.is_dir():
        raise ValueError("input root is not a directory")
    rows = load_manifest(manifest_path)
    results = [validate_one(input_root, row) for row in rows]
    passed = sum(result.status == "PASS" for result in results)
    report = {
        "schema_version": "1.0",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed == len(results) else "FAIL",
        "file_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "files": [asdict(result) for result in results],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_receipt(args.input_root, args.manifest)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "file_count", "passed_count", "failed_count")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
