#!/usr/bin/env python3
"""Freeze the private FW5 manifest without reading any outer row or outcome."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pandas as pd

from research_v2.src.execution.fw5_common import (
    AGGREGATION_PLAN_PATH,
    AGGREGATOR_PATH,
    COMMON_PATH,
    COHORT_PATH,
    COMPLETION_RUNNER_PATH,
    DEFINITION_PATH,
    DOMAIN_BUILDER_PATH,
    DOMAIN_DEFINITION_PATH,
    FW2_ELIGIBILITY_PATH,
    FW2_SUMMARIES_PATH,
    FW3_READABILITY_PATH,
    FW4_CONFIG_PATH,
    FW4_COMPLETION_PATH,
    FW4_DOMAIN_PATH,
    FW4_SELECTION_PATH,
    FW2_RUNNER_PATH,
    MANIFEST_COLUMNS,
    MANIFEST_BUILDER_PATH,
    MANIFEST_PATH,
    MEMBERSHIP_PATH,
    METRICS_PATH,
    MODEL_REGISTRY_PATH,
    MAPPING_PATH,
    NEURAL_MODELS_PATH,
    NONCOMBAT_FEATURES_PATH,
    PLAN_PATH,
    RUNNER_PATH,
    load_csv_rows,
    sha256_file,
    validate_manifest_row,
)


def build_manifest_rows(eligibility: dict, selection: pd.DataFrame, seeds: list[int]) -> list[dict[str, object]]:
    selected = selection.copy()
    selected["performance_only_selected"] = selected["performance_only_selected"].astype(str).str.lower().eq("true")
    selected["audit_guided_selected"] = selected["audit_guided_selected"].astype(str).str.lower().eq("true")
    rows: list[dict[str, object]] = []
    unit_index = 0
    for entry in eligibility["outer_loop_eligibility"]:
        outer = str(entry["outer_domain"])
        outer_rows = selected[selected["outer_domain"].eq(outer)]
        expected = set(map(str, entry["eligible_configuration_ids"]))
        if set(outer_rows["configuration_id"].astype(str)) != expected:
            raise RuntimeError("FW4 selection candidates differ from FW2 frozen eligibility")
        if int(outer_rows["performance_only_selected"].sum()) != 1 or int(outer_rows["audit_guided_selected"].sum()) != 1:
            raise RuntimeError("FW5 requires one frozen selection per rule and outer loop")
        for _, unit in outer_rows.sort_values(["family", "configuration_id"]).iterrows():
            unit_index += 1
            for seed_index, seed in enumerate(seeds, start=1):
                rows.append(
                    {
                        "fw5_job_id": f"FW5J{(unit_index - 1) * len(seeds) + seed_index:06d}",
                        "outer_domain": outer,
                        "family": str(unit["family"]),
                        "configuration_id": str(unit["configuration_id"]),
                        "seed": int(seed),
                        "performance_only_selected": bool(unit["performance_only_selected"]),
                        "audit_guided_selected": bool(unit["audit_guided_selected"]),
                        "outer_test_access": "permitted_after_freeze",
                        "output_scope": "private_outer_prediction",
                    }
                )
    if unit_index != 58 or len(rows) != 290:
        raise RuntimeError("FW5 formal scope must contain 58 units and 290 seed jobs")
    return rows


def verify_existing() -> None:
    definition = json.loads(DEFINITION_PATH.read_text(encoding="utf-8"))
    rows = load_csv_rows(MANIFEST_PATH)
    if len(rows) != 290 or any(validate_manifest_row(row) is not None for row in rows):
        raise RuntimeError("Existing FW5 manifest is invalid")
    if sha256_file(MANIFEST_PATH) != definition["source_sha256"]["formal_manifest"]:
        raise RuntimeError("Existing FW5 manifest hash differs from its definition")
    print(f"PASS: verified existing frozen FW5 manifest at {MANIFEST_PATH}")


def main() -> None:
    if MANIFEST_PATH.exists() or DEFINITION_PATH.exists():
        if not MANIFEST_PATH.exists() or not DEFINITION_PATH.exists():
            raise RuntimeError("FW5 manifest freeze is partial")
        verify_existing()
        return
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    aggregation = json.loads(AGGREGATION_PLAN_PATH.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN_BEFORE_OUTER_ACCESS" or aggregation.get("status") != "FROZEN_BEFORE_OUTER_ACCESS":
        raise RuntimeError("FW5 plans must be frozen before manifest creation")
    eligibility = json.loads(FW2_ELIGIBILITY_PATH.read_text(encoding="utf-8"))
    selection = pd.read_csv(FW4_SELECTION_PATH, dtype={"configuration_id": str})
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    seeds = list(map(int, registry["common_training"]["seeds"]))
    rows = build_manifest_rows(eligibility, selection, seeds)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source_sha256 = {
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
    definition = {
        "freeze_id": "CRUA-AD-FW5-FORMAL-JOBS-v2.1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_OUTER_ACCESS",
        "job_counts": {
            "eligible_outer_configuration_units": 58,
            "seeds_per_unit": 5,
            "formal_jobs": 290,
            "outer_domains": 7,
            "performance_only_selected_units": 7,
            "audit_guided_selected_units": 7,
        },
        "manifest": {
            "path": "outputs/private/fw5/manifests/fw5_formal_job_manifest.csv",
            "sha256": source_sha256["formal_manifest"],
        },
        "source_sha256": source_sha256,
        "outer_access": "AUTHORIZED_ONLY_AFTER_THIS_FREEZE",
        "selection_recomputation": "FORBIDDEN",
        "privacy": "Manifest is private because it contains frozen configuration identifiers; definition contains counts and hashes only.",
    }
    DEFINITION_PATH.write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_existing()
    print(f"PASS: {DEFINITION_PATH}")


if __name__ == "__main__":
    main()
