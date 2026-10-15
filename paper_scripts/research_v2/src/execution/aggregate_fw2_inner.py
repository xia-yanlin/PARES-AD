#!/usr/bin/env python3
"""Freeze FW2 inner-only performance summaries and eligibility sets.

The script reads only hash-verified private FW2 ``summary.json`` files. It does
not read prediction CSVs, the participant cohort, or any outer-domain outcome.
All participant-derived performance outputs remain beneath ``outputs/private``;
the tracked log contains only completion counts, checks, and artifact hashes.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

from research_v2.src.execution.run_fw2_job import (
    FORMAL_ROOT,
    JOB_MANIFEST_PATH,
    MODEL_REGISTRY_PATH,
    sha256_file,
    verify_completed_directory,
)


V2_ROOT = Path(__file__).resolve().parents[2]
SELECTION_RULES_PATH = V2_ROOT / "configs" / "audit_selection_rules_v2_1.json"
JOB_DEFINITION_PATH = V2_ROOT / "data_registry" / "fw2_inner_benchmark_job_definition.json"
OUTPUT_ROOT = V2_ROOT / "outputs" / "private" / "fw2" / "aggregation"
OUTPUT_DIRECTORY = OUTPUT_ROOT / "fw2_inner_selection_v2_1"
TRACKED_LOG = V2_ROOT / "logs" / "fw2_inner_aggregation_freeze.json"


def standard_error(values: Sequence[float]) -> float:
    """Return the sample standard error across held-out acquisition domains."""

    numeric = [float(value) for value in values]
    if len(numeric) < 2 or not all(math.isfinite(value) for value in numeric):
        raise ValueError("At least two finite domain values are required")
    return stdev(numeric) / math.sqrt(len(numeric))


def one_standard_error_set(
    domain_aurocs: dict[str, Sequence[float]],
) -> dict[str, object]:
    """Apply the frozen one-SE rule after seeds have been averaged by domain."""

    if not domain_aurocs:
        raise ValueError("No configuration AUROCs supplied")
    lengths = {len(values) for values in domain_aurocs.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("Every configuration must cover the same two or more domains")
    means = {config_id: mean(values) for config_id, values in domain_aurocs.items()}
    best_configuration_id = sorted(means, key=lambda item: (-means[item], item))[0]
    best_mean = means[best_configuration_id]
    best_se = standard_error(domain_aurocs[best_configuration_id])
    threshold = best_mean - best_se
    tolerance = 1e-12
    eligible = sorted(
        config_id
        for config_id, value in means.items()
        if value + tolerance >= threshold
    )
    return {
        "best_configuration_id": best_configuration_id,
        "best_mean_inner_domain_auroc": best_mean,
        "best_standard_error_across_inner_domains": best_se,
        "eligibility_threshold": threshold,
        "eligible_configuration_ids": eligible,
        "configuration_mean_inner_domain_auroc": dict(sorted(means.items())),
    }


def select_performance_only(
    eligible_configuration_ids: Iterable[str],
    calibration_distance: dict[str, float],
    parameter_count: dict[str, int],
) -> str:
    """Apply the frozen calibration, complexity, and lexical tie-break order."""

    eligible = sorted(set(eligible_configuration_ids))
    if not eligible:
        raise ValueError("The performance-eligible set cannot be empty")
    for config_id in eligible:
        if config_id not in calibration_distance or config_id not in parameter_count:
            raise ValueError(f"Missing tie-break value for {config_id}")
        if not math.isfinite(float(calibration_distance[config_id])):
            raise ValueError(f"Non-finite calibration distance for {config_id}")
        if int(parameter_count[config_id]) <= 0:
            raise ValueError(f"Invalid parameter count for {config_id}")
    return min(
        eligible,
        key=lambda config_id: (
            float(calibration_distance[config_id]),
            int(parameter_count[config_id]),
            config_id,
        ),
    )


def write_csv(path: Path, rows: list[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verified_existing_directory(directory: Path) -> bool:
    completion_path = directory / "completion.json"
    if not completion_path.exists():
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS":
        return False
    return all(
        (directory / relative).exists()
        and sha256_file(directory / relative) == expected_hash
        for relative, expected_hash in completion.get("artifact_sha256", {}).items()
    )


def load_inputs() -> tuple[list[dict[str, str]], dict, dict, dict]:
    with JOB_MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    rules = json.loads(SELECTION_RULES_PATH.read_text(encoding="utf-8"))
    definition = json.loads(JOB_DEFINITION_PATH.read_text(encoding="utf-8"))
    if len(manifest) != int(definition["job_counts"]["total"]):
        raise RuntimeError("FW2 manifest count does not match its frozen definition")
    if sha256_file(JOB_MANIFEST_PATH) != definition["manifest"]["sha256"]:
        raise RuntimeError("FW2 manifest hash does not match its frozen definition")
    if "inner held-out acquisition domain" not in rules["performance_matching"]["one_standard_error_unit"]:
        raise RuntimeError("Selection rules do not define the required domain-level one-SE unit")
    return manifest, registry, rules, definition


def load_verified_job_metrics(manifest: list[dict[str, str]]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for manifest_row in manifest:
        job_id = manifest_row["job_id"]
        if job_id in seen:
            raise RuntimeError(f"Duplicate manifest job: {job_id}")
        seen.add(job_id)
        directory = FORMAL_ROOT / job_id
        if not verify_completed_directory(directory):
            raise RuntimeError(f"Job directory is absent or fails hash verification: {job_id}")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        job = summary["job"]
        for field in [
            "job_id",
            "model_kind",
            "family",
            "configuration_id",
            "seed",
            "outer_domain",
            "inner_validation_domain",
            "outer_test_access",
        ]:
            if str(job[field]) != str(manifest_row[field]):
                raise RuntimeError(f"Manifest/summary mismatch for {job_id}: {field}")
        if summary.get("mode") != "formal":
            raise RuntimeError(f"Non-formal summary in formal output: {job_id}")
        if summary.get("outer_test_predictions_generated") is not False:
            raise RuntimeError(f"Outer prediction boundary violated: {job_id}")
        if summary["split_checks"].get("outer_rows_loaded_into_analysis_frame") is not False:
            raise RuntimeError(f"Outer rows were loaded into the analysis frame: {job_id}")
        if not all(
            summary["split_checks"].get(key) is True
            for key in [
                "fit_calibration_disjoint",
                "inner_validation_disjoint",
                "outer_test_disjoint",
            ]
        ):
            raise RuntimeError(f"Split check failed: {job_id}")
        metrics = summary["inner_validation_metrics_private"]["calibrated"]
        numeric_metrics = {
            "auroc": float(metrics["auroc"]),
            "auprc": float(metrics["auprc"]),
            "brier": float(metrics["brier"]),
            "calibration_intercept": float(metrics["calibration_intercept"]),
            "calibration_slope": float(metrics["calibration_slope"]),
            "ece_10": float(metrics["ece_10"]),
            "fit_seconds": float(summary["training"]["fit_seconds"]),
        }
        if not all(math.isfinite(value) for value in numeric_metrics.values()):
            raise RuntimeError(f"Non-finite metric in {job_id}")
        if not 0.0 <= numeric_metrics["auroc"] <= 1.0:
            raise RuntimeError(f"AUROC is outside [0, 1] in {job_id}")
        parameter_value = summary["training"].get("parameter_count")
        if job["model_kind"] == "neural":
            if parameter_value is None or int(parameter_value) <= 0:
                raise RuntimeError(f"Missing neural parameter count in {job_id}")
            parameter_value = int(parameter_value)
        elif parameter_value is not None:
            raise RuntimeError(f"Unexpected reference parameter count in {job_id}")
        rows.append(
            {
                "job_id": job_id,
                "model_kind": job["model_kind"],
                "family": job["family"],
                "configuration_id": job["configuration_id"],
                "seed": int(job["seed"]),
                "outer_domain": job["outer_domain"],
                "inner_validation_domain": job["inner_validation_domain"],
                "calibrated_auroc": numeric_metrics["auroc"],
                "calibrated_auprc": numeric_metrics["auprc"],
                "calibrated_brier": numeric_metrics["brier"],
                "calibration_intercept": numeric_metrics["calibration_intercept"],
                "calibration_slope": numeric_metrics["calibration_slope"],
                "absolute_calibration_slope_distance": abs(numeric_metrics["calibration_slope"] - 1.0),
                "ece_10": numeric_metrics["ece_10"],
                "fit_seconds": numeric_metrics["fit_seconds"],
                "parameter_count": parameter_value,
                "inner_validation_n": int(metrics["n"]),
                "completion_sha256": sha256_file(directory / "completion.json"),
            }
        )
    return rows


def aggregate_domain_rows(job_rows: list[dict], registry: dict) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in job_rows:
        key = (
            row["model_kind"],
            row["outer_domain"],
            row["inner_validation_domain"],
            row["family"],
            row["configuration_id"],
        )
        grouped[key].append(row)

    expected_neural_seeds = sorted(int(seed) for seed in registry["common_training"]["seeds"])
    domain_rows: list[dict] = []
    for key, rows in sorted(grouped.items()):
        model_kind, outer_domain, inner_domain, family, config_id = key
        expected_seeds = expected_neural_seeds if model_kind == "neural" else [7]
        if sorted(row["seed"] for row in rows) != expected_seeds:
            raise RuntimeError(f"Incomplete seed coverage for {outer_domain}/{inner_domain}/{config_id}")
        parameter_values = {row["parameter_count"] for row in rows}
        if len(parameter_values) != 1:
            raise RuntimeError(f"Parameter count varies across seeds for {config_id}")
        domain_rows.append(
            {
                "model_kind": model_kind,
                "outer_domain": outer_domain,
                "inner_validation_domain": inner_domain,
                "family": family,
                "configuration_id": config_id,
                "seed_count": len(rows),
                "mean_calibrated_auroc": mean(row["calibrated_auroc"] for row in rows),
                "mean_calibrated_auprc": mean(row["calibrated_auprc"] for row in rows),
                "mean_calibrated_brier": mean(row["calibrated_brier"] for row in rows),
                "mean_absolute_calibration_slope_distance": mean(
                    row["absolute_calibration_slope_distance"] for row in rows
                ),
                "mean_ece_10": mean(row["ece_10"] for row in rows),
                "mean_fit_seconds": mean(row["fit_seconds"] for row in rows),
                "parameter_count": next(iter(parameter_values)),
            }
        )
    return domain_rows


def aggregate_configuration_rows(
    domain_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    outer_inner_domains: dict[str, set[str]] = defaultdict(set)
    for row in domain_rows:
        key = (row["model_kind"], row["outer_domain"], row["family"], row["configuration_id"])
        grouped[key].append(row)
        outer_inner_domains[row["outer_domain"]].add(row["inner_validation_domain"])
    if len(outer_inner_domains) != 7 or any(len(domains) != 6 for domains in outer_inner_domains.values()):
        raise RuntimeError("Expected seven outer domains with six inner domains each")

    summaries: list[dict] = []
    for key, rows in sorted(grouped.items()):
        model_kind, outer_domain, family, config_id = key
        if len(rows) != len(outer_inner_domains[outer_domain]):
            raise RuntimeError(f"Incomplete inner-domain coverage for {outer_domain}/{config_id}")
        parameter_values = {row["parameter_count"] for row in rows}
        if len(parameter_values) != 1:
            raise RuntimeError(f"Parameter count varies across domains for {config_id}")
        aurocs = [row["mean_calibrated_auroc"] for row in rows]
        summaries.append(
            {
                "model_kind": model_kind,
                "outer_domain": outer_domain,
                "family": family,
                "configuration_id": config_id,
                "inner_domain_count": len(rows),
                "macro_mean_calibrated_auroc": mean(aurocs),
                "sample_sd_calibrated_auroc_across_domains": stdev(aurocs),
                "standard_error_calibrated_auroc_across_domains": standard_error(aurocs),
                "macro_mean_calibrated_auprc": mean(row["mean_calibrated_auprc"] for row in rows),
                "macro_mean_calibrated_brier": mean(row["mean_calibrated_brier"] for row in rows),
                "macro_mean_absolute_calibration_slope_distance": mean(
                    row["mean_absolute_calibration_slope_distance"] for row in rows
                ),
                "macro_mean_ece_10": mean(row["mean_ece_10"] for row in rows),
                "total_fit_seconds": sum(row["mean_fit_seconds"] * row["seed_count"] for row in rows),
                "parameter_count": next(iter(parameter_values)),
            }
        )

    selections: list[dict] = []
    neural_by_outer: dict[str, list[dict]] = defaultdict(list)
    for row in domain_rows:
        if row["model_kind"] == "neural":
            neural_by_outer[row["outer_domain"]].append(row)
    for outer_domain, rows in sorted(neural_by_outer.items()):
        by_config: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_config[row["configuration_id"]].append(row)
        if len(by_config) != 12:
            raise RuntimeError(f"Expected 12 neural configurations for {outer_domain}")
        aurocs = {
            config_id: [row["mean_calibrated_auroc"] for row in sorted(config_rows, key=lambda item: item["inner_validation_domain"])]
            for config_id, config_rows in by_config.items()
        }
        eligibility = one_standard_error_set(aurocs)
        calibration_distance = {
            config_id: mean(row["mean_absolute_calibration_slope_distance"] for row in config_rows)
            for config_id, config_rows in by_config.items()
        }
        parameters = {
            config_id: int(next(iter({row["parameter_count"] for row in config_rows})))
            for config_id, config_rows in by_config.items()
        }
        selected = select_performance_only(
            eligibility["eligible_configuration_ids"],
            calibration_distance,
            parameters,
        )
        selections.append(
            {
                "outer_domain": outer_domain,
                **eligibility,
                "eligible_set_size": len(eligibility["eligible_configuration_ids"]),
                "performance_only_selected_configuration_id": selected,
                "mean_absolute_calibration_slope_distance": dict(sorted(calibration_distance.items())),
                "parameter_count": dict(sorted(parameters.items())),
                "outer_test_outcomes_used": False,
            }
        )
    return summaries, selections


def build_private_freeze(
    temporary: Path,
    job_rows: list[dict],
    domain_rows: list[dict],
    summary_rows: list[dict],
    selections: list[dict],
    source_hashes: dict[str, str],
) -> dict[str, str]:
    temporary.mkdir(parents=True)
    write_csv(temporary / "job_metrics.csv", job_rows, list(job_rows[0]))
    write_csv(temporary / "domain_seed_averages.csv", domain_rows, list(domain_rows[0]))
    write_csv(temporary / "configuration_summaries.csv", summary_rows, list(summary_rows[0]))
    selection_payload = {
        "freeze_id": "CRUA-AD-FW2-INNER-PERFORMANCE-ELIGIBILITY-v2.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN",
        "scope": "Twelve auditable neural configurations; seven outer loops; six inner held-out acquisition domains per outer loop",
        "selection_unit": "Seeds averaged within configuration and inner domain before equal-domain macro averaging",
        "calibration_tie_break": "Equal-domain mean of seed-level absolute calibrated slope distance from 1",
        "outer_domain_prohibition": "No outer-domain labels, predictions, metrics, probes, audit effects, or calibration outcomes were read or used.",
        "source_sha256": source_hashes,
        "outer_loop_eligibility": selections,
        "privacy": "Private participant-derived aggregate performance; no participant rows or identifiers.",
    }
    (temporary / "performance_eligible_sets.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_names = [
        "job_metrics.csv",
        "domain_seed_averages.csv",
        "configuration_summaries.csv",
        "performance_eligible_sets.json",
    ]
    hashes = {name: sha256_file(temporary / name) for name in artifact_names}
    completion = {
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": hashes,
        "checks": {
            "all_manifest_jobs_hash_verified": True,
            "all_manifest_summary_fields_match": True,
            "all_split_checks_pass": True,
            "all_outer_test_predictions_absent": True,
            "seeds_averaged_before_domain_standard_error": True,
            "seven_outer_loop_eligible_sets_frozen": len(selections) == 7,
        },
        "privacy": "Private participant-derived aggregate performance; no participant rows or identifiers.",
    }
    (temporary / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashes


def write_tracked_log(
    *,
    job_rows: list[dict],
    domain_rows: list[dict],
    summary_rows: list[dict],
    selections: list[dict],
    artifact_hashes: dict[str, str],
) -> None:
    payload = {
        "check_id": "FW2_INNER_AGGREGATION_AND_ELIGIBILITY_FREEZE",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "status": "PASS",
        "counts": {
            "hash_verified_manifest_jobs": len(job_rows),
            "neural_jobs": sum(row["model_kind"] == "neural" for row in job_rows),
            "reference_jobs": sum(row["model_kind"] == "reference" for row in job_rows),
            "outer_inner_domain_pairs": len({(row["outer_domain"], row["inner_validation_domain"]) for row in job_rows}),
            "seed_averaged_domain_rows": len(domain_rows),
            "configuration_summary_rows": len(summary_rows),
            "outer_loop_eligibility_sets": len(selections),
        },
        "checks": {
            "all_private_job_hashes_valid": True,
            "all_outer_test_predictions_absent": True,
            "selection_uses_inner_domains_only": True,
            "performance_outputs_remain_private": True,
        },
        "private_artifact_sha256": artifact_hashes,
        "known_instrumentation_gaps": {
            "inference_time_recorded_during_fw2": False,
            "peak_memory_recorded_during_fw2": False,
        },
        "performance_reporting": "Omitted from tracked log; metrics and selected configuration identifiers remain in Git-ignored private artifacts.",
        "privacy": "No participant identifiers, rows, predictions, representations, or performance metrics are present.",
    }
    TRACKED_LOG.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT_DIRECTORY.exists():
        if not verified_existing_directory(OUTPUT_DIRECTORY):
            raise RuntimeError(f"Existing aggregation directory is incomplete or invalid: {OUTPUT_DIRECTORY}")
        print(f"PASS: verified existing immutable FW2 aggregation at {OUTPUT_DIRECTORY}")
        return

    manifest, registry, _rules, _definition = load_inputs()
    job_rows = load_verified_job_metrics(manifest)
    domain_rows = aggregate_domain_rows(job_rows, registry)
    summary_rows, selections = aggregate_configuration_rows(domain_rows)
    source_hashes = {
        "job_manifest": sha256_file(JOB_MANIFEST_PATH),
        "job_definition": sha256_file(JOB_DEFINITION_PATH),
        "model_registry": sha256_file(MODEL_REGISTRY_PATH),
        "selection_rules": sha256_file(SELECTION_RULES_PATH),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_ROOT / f".{OUTPUT_DIRECTORY.name}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        artifact_hashes = build_private_freeze(
            temporary,
            job_rows,
            domain_rows,
            summary_rows,
            selections,
            source_hashes,
        )
        temporary.replace(OUTPUT_DIRECTORY)
    except Exception:
        if temporary.exists() and temporary.parent == OUTPUT_ROOT:
            shutil.rmtree(temporary)
        raise
    if not verified_existing_directory(OUTPUT_DIRECTORY):
        raise RuntimeError("New aggregation directory failed its immutable hash check")
    write_tracked_log(
        job_rows=job_rows,
        domain_rows=domain_rows,
        summary_rows=summary_rows,
        selections=selections,
        artifact_hashes=artifact_hashes,
    )
    print(f"PASS: {OUTPUT_DIRECTORY}")
    print(f"PASS: {TRACKED_LOG}")


if __name__ == "__main__":
    main()
