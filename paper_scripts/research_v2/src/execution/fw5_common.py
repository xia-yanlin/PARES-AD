"""Shared immutable contracts for FW5 outer refits and predictions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


V2_ROOT = Path(__file__).resolve().parents[2]
HISS_ROOT = V2_ROOT.parent
COHORT_PATH = HISS_ROOT / "data" / "processed" / "baseline_cohort_with_splits.csv"
MEMBERSHIP_PATH = V2_ROOT / "outputs" / "private" / "nested_domain_splits.csv"
MAPPING_PATH = V2_ROOT / "data_registry" / "noncombat_roi_mapping_v2_1.json"
MODEL_REGISTRY_PATH = V2_ROOT / "configs" / "model_registry_v2_1.json"
PLAN_PATH = V2_ROOT / "configs" / "fw5_outer_evaluation_plan_v2_1.json"
AGGREGATION_PLAN_PATH = V2_ROOT / "configs" / "fw5_aggregation_plan_v2_1.json"
MANIFEST_PATH = V2_ROOT / "outputs" / "private" / "fw5" / "manifests" / "fw5_formal_job_manifest.csv"
DEFINITION_PATH = V2_ROOT / "data_registry" / "fw5_formal_job_definition.json"
OUTPUT_ROOT = V2_ROOT / "outputs" / "private" / "fw5" / "jobs"
AGGREGATION_ROOT = V2_ROOT / "outputs" / "private" / "fw5" / "aggregation"
FW2_ELIGIBILITY_PATH = V2_ROOT / "outputs" / "private" / "fw2" / "aggregation" / "fw2_inner_selection_v2_1" / "performance_eligible_sets.json"
FW2_SUMMARIES_PATH = V2_ROOT / "outputs" / "private" / "fw2" / "aggregation" / "fw2_inner_selection_v2_1" / "configuration_summaries.csv"
FW3_READABILITY_PATH = V2_ROOT / "outputs" / "private" / "fw3" / "aggregation" / "fw3_inner_readability_v2_1" / "configuration_readability_profiles.csv"
FW4_SELECTION_PATH = V2_ROOT / "outputs" / "private" / "fw4" / "aggregation" / "fw4_inner_validated_use_v2_1" / "audit_selection_profiles.csv"
FW4_CONFIG_PATH = V2_ROOT / "outputs" / "private" / "fw4" / "aggregation" / "fw4_inner_validated_use_v2_1" / "configuration_condition_profiles.csv"
FW4_DOMAIN_PATH = V2_ROOT / "outputs" / "private" / "fw4" / "aggregation" / "fw4_inner_validated_use_v2_1" / "domain_condition_profiles.csv"
FW4_COMPLETION_PATH = V2_ROOT / "outputs" / "private" / "fw4" / "aggregation" / "fw4_inner_validated_use_v2_1" / "completion.json"
RUNNER_PATH = V2_ROOT / "src" / "execution" / "run_fw5_job.py"
COMPLETION_RUNNER_PATH = V2_ROOT / "src" / "execution" / "run_fw5_to_completion.py"
AGGREGATOR_PATH = V2_ROOT / "src" / "execution" / "aggregate_fw5_outer.py"
COMMON_PATH = V2_ROOT / "src" / "execution" / "fw5_common.py"
MANIFEST_BUILDER_PATH = V2_ROOT / "src" / "execution" / "build_fw5_manifest.py"
FW2_RUNNER_PATH = V2_ROOT / "src" / "execution" / "run_fw2_job.py"
NONCOMBAT_FEATURES_PATH = V2_ROOT / "src" / "data" / "noncombat_features.py"
NEURAL_MODELS_PATH = V2_ROOT / "src" / "models" / "neural.py"
METRICS_PATH = V2_ROOT / "src" / "statistics" / "metrics.py"
DOMAIN_BUILDER_PATH = V2_ROOT / "src" / "data" / "build_acquisition_domains.py"
DOMAIN_DEFINITION_PATH = V2_ROOT / "data_registry" / "acquisition_domain_definition.json"

MANIFEST_COLUMNS = [
    "fw5_job_id",
    "outer_domain",
    "family",
    "configuration_id",
    "seed",
    "performance_only_selected",
    "audit_guided_selected",
    "outer_test_access",
    "output_scope",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"Expected explicit boolean text, received {value!r}")
    return text == "true"


def validate_manifest_row(row: dict[str, str]) -> None:
    if list(row) != MANIFEST_COLUMNS:
        raise ValueError("FW5 manifest columns differ from the frozen schema")
    if row["outer_test_access"] != "permitted_after_freeze":
        raise ValueError("FW5 outer access is not bound to the completed freeze")
    if row["output_scope"] != "private_outer_prediction":
        raise ValueError("FW5 outputs must remain private")
    if int(row["seed"]) not in {7, 42, 123, 202, 314}:
        raise ValueError("FW5 manifest contains an unfrozen seed")
    parse_bool(row["performance_only_selected"])
    parse_bool(row["audit_guided_selected"])


def verify_frozen_sources() -> dict[str, Any]:
    definition = json.loads(DEFINITION_PATH.read_text(encoding="utf-8"))
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    aggregation = json.loads(AGGREGATION_PLAN_PATH.read_text(encoding="utf-8"))
    if definition.get("status") != "FROZEN_BEFORE_OUTER_ACCESS":
        raise RuntimeError("FW5 formal definition is not frozen")
    if plan.get("status") != "FROZEN_BEFORE_OUTER_ACCESS" or aggregation.get("status") != "FROZEN_BEFORE_OUTER_ACCESS":
        raise RuntimeError("FW5 plans were not frozen before outer access")
    expected = {
        "formal_manifest": sha256_file(MANIFEST_PATH),
        "outer_evaluation_plan": sha256_file(PLAN_PATH),
        "aggregation_plan": sha256_file(AGGREGATION_PLAN_PATH),
        "fw2_eligibility": sha256_file(FW2_ELIGIBILITY_PATH),
        "fw4_selection": sha256_file(FW4_SELECTION_PATH),
        "fw4_completion": sha256_file(FW4_COMPLETION_PATH),
        "model_registry": sha256_file(MODEL_REGISTRY_PATH),
        "cohort": sha256_file(COHORT_PATH),
        "membership": sha256_file(MEMBERSHIP_PATH),
        "mapping": sha256_file(MAPPING_PATH),
        "domain_definition": sha256_file(DOMAIN_DEFINITION_PATH),
        "fw2_summaries": sha256_file(FW2_SUMMARIES_PATH),
        "fw3_readability": sha256_file(FW3_READABILITY_PATH),
        "fw4_configuration_profiles": sha256_file(FW4_CONFIG_PATH),
        "fw4_domain_profiles": sha256_file(FW4_DOMAIN_PATH),
        "job_runner": sha256_file(RUNNER_PATH),
        "completion_runner": sha256_file(COMPLETION_RUNNER_PATH),
        "aggregator": sha256_file(AGGREGATOR_PATH),
        "common": sha256_file(COMMON_PATH),
        "manifest_builder": sha256_file(MANIFEST_BUILDER_PATH),
        "fw2_training_helpers": sha256_file(FW2_RUNNER_PATH),
        "noncombat_features": sha256_file(NONCOMBAT_FEATURES_PATH),
        "neural_models": sha256_file(NEURAL_MODELS_PATH),
        "metrics": sha256_file(METRICS_PATH),
        "domain_builder": sha256_file(DOMAIN_BUILDER_PATH),
    }
    if definition.get("source_sha256") != expected:
        raise RuntimeError("FW5 frozen source hashes differ from current authorities")
    return definition


def verify_completed_directory(directory: Path, expected_job_id: str | None = None) -> bool:
    completion_path = directory / "completion.json"
    if not completion_path.is_file():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if completion.get("status") != "PASS" or completion.get("outer_test_predictions_generated") is not True:
        return False
    if expected_job_id is not None and completion.get("fw5_job_id") != expected_job_id:
        return False
    hashes = completion.get("artifact_sha256")
    return isinstance(hashes, dict) and bool(hashes) and all(
        (directory / name).is_file() and sha256_file(directory / name) == expected
        for name, expected in hashes.items()
    )
