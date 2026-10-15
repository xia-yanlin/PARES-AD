#!/usr/bin/env python3
"""Freeze the complete FW2 inner-domain benchmark job manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY = V2_ROOT / "configs" / "model_registry_v2_1.json"
SPLIT_REGISTRY = V2_ROOT / "data_registry" / "nested_domain_split_registry.csv"
OUTPUT_CSV = V2_ROOT / "data_registry" / "fw2_inner_benchmark_job_manifest.csv"
OUTPUT_JSON = V2_ROOT / "data_registry" / "fw2_inner_benchmark_job_definition.json"
OUTPUT_LOG = V2_ROOT / "logs" / "fw2_job_manifest_validation.json"

HEADERS = [
    "job_id",
    "model_kind",
    "family",
    "configuration_id",
    "seed",
    "outer_domain",
    "inner_validation_domain",
    "outer_test_access",
    "output_scope",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_jobs(model_registry: dict, split_pairs: list[dict[str, str]]) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    counter = 1
    seeds = list(model_registry["common_training"]["seeds"])
    for pair in split_pairs:
        outer = pair["outer_domain"]
        inner = pair["inner_validation_domain"]
        if outer == inner:
            raise ValueError("Outer and inner validation domains must differ")
        for family in model_registry["neural_families"]:
            for configuration in family["configurations"]:
                for seed in seeds:
                    jobs.append(
                        {
                            "job_id": f"FW2J{counter:05d}",
                            "model_kind": "neural",
                            "family": family["family"],
                            "configuration_id": configuration["id"],
                            "seed": int(seed),
                            "outer_domain": outer,
                            "inner_validation_domain": inner,
                            "outer_test_access": "forbidden",
                            "output_scope": "private inner-validation predictions and checkpoint",
                        }
                    )
                    counter += 1
        for family in model_registry["strong_references"]:
            for configuration in family["configurations"]:
                jobs.append(
                    {
                        "job_id": f"FW2J{counter:05d}",
                        "model_kind": "reference",
                        "family": family["family"],
                        "configuration_id": configuration["id"],
                        "seed": int(seeds[0]),
                        "outer_domain": outer,
                        "inner_validation_domain": inner,
                        "outer_test_access": "forbidden",
                        "output_scope": "private inner-validation predictions; no representation audit",
                    }
                )
                counter += 1
    return jobs


def build_payloads() -> dict[str, bytes]:
    with MODEL_REGISTRY.open("r", encoding="utf-8") as handle:
        model_registry = json.load(handle)
    with SPLIT_REGISTRY.open("r", encoding="utf-8", newline="") as handle:
        split_pairs = list(csv.DictReader(handle))
    if len(split_pairs) != 42:
        raise ValueError(f"Expected 42 frozen outer-inner pairs, found {len(split_pairs)}")
    jobs = build_jobs(model_registry, split_pairs)
    manifest_payload = csv_bytes(jobs)

    neural_count = sum(job["model_kind"] == "neural" for job in jobs)
    reference_count = sum(job["model_kind"] == "reference" for job in jobs)
    expected_neural = 42 * 4 * 3 * 5
    expected_reference = 42 * 3 * 3
    if neural_count != expected_neural or reference_count != expected_reference:
        raise AssertionError("FW2 job counts differ from the frozen design")
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise AssertionError("FW2 job IDs are not unique")

    definition = {
        "freeze_id": "CRUA-AD-FW2-INNER-BENCHMARK-JOBS-v2.1",
        "frozen_date": "2026-07-18",
        "status": "FROZEN",
        "scope": "inner-domain benchmark only",
        "outer_test_access": "FORBIDDEN",
        "job_counts": {
            "total": len(jobs),
            "neural": neural_count,
            "reference": reference_count,
            "outer_inner_pairs": len(split_pairs),
        },
        "source_hashes": {
            "model_registry_sha256": sha256_file(MODEL_REGISTRY),
            "nested_split_registry_sha256": sha256_file(SPLIT_REGISTRY),
        },
        "manifest": {
            "path": "data_registry/fw2_inner_benchmark_job_manifest.csv",
            "sha256": sha256_bytes(manifest_payload),
        },
        "run_policy": {
            "participant_outputs": "outputs/private/fw2/jobs/<job_id>/",
            "existing_complete_job": "verify and skip; never overwrite",
            "outer_rows": "excluded before preprocessing, early stopping, calibration, fitting, and evaluation",
            "tracked_progress": "counts, technical failures, runtime, and hashes only; no performance metrics",
        },
    }
    definition_payload = (json.dumps(definition, indent=2, sort_keys=True) + "\n").encode("utf-8")
    log = {
        "check_id": "FW2_JOB_MANIFEST_FREEZE",
        "date": "2026-07-18",
        "status": "PASS",
        "checks": {
            "exactly_42_outer_inner_pairs": True,
            "outer_inner_domains_differ": True,
            "unique_job_ids": True,
            "all_outer_access_forbidden": True,
            "expected_neural_jobs": neural_count == expected_neural,
            "expected_reference_jobs": reference_count == expected_reference,
        },
        "counts": definition["job_counts"],
        "artifacts": {
            "manifest_sha256": sha256_bytes(manifest_payload),
            "definition_sha256": sha256_bytes(definition_payload),
        },
        "privacy": "No participant identifiers, rows, predictions, representations, or performance metrics are present.",
    }
    return {
        "manifest": manifest_payload,
        "definition": definition_payload,
        "log": (json.dumps(log, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }


def write_or_verify(path: Path, payload: bytes, verify: bool) -> None:
    if verify:
        if not path.exists() or path.read_bytes() != payload:
            raise RuntimeError(f"Frozen artifact missing or changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    payloads = build_payloads()
    write_or_verify(OUTPUT_CSV, payloads["manifest"], args.verify)
    write_or_verify(OUTPUT_JSON, payloads["definition"], args.verify)
    write_or_verify(OUTPUT_LOG, payloads["log"], args.verify)
    print(("Verified" if args.verify else "Wrote") + f" FW2 job manifest: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
