#!/usr/bin/env python3
"""Refit one frozen FW5 seed job and predict its outer acquisition domain."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from research_v2.src.data.noncombat_features import FrozenROITransformer, build_structural_covariance_adjacency
from research_v2.src.execution.fw5_common import (
    COHORT_PATH,
    DEFINITION_PATH,
    MANIFEST_PATH,
    MAPPING_PATH,
    MEMBERSHIP_PATH,
    MODEL_REGISTRY_PATH,
    OUTPUT_ROOT,
    load_csv_rows,
    sha256_file,
    validate_manifest_row,
    verify_completed_directory,
    verify_frozen_sources,
)
from research_v2.src.execution.run_fw2_job import (
    deterministic_fit_calibration_indices,
    find_configuration,
    model_scores,
    save_preprocessor,
    train_neural,
)
from research_v2.src.statistics.metrics import fit_platt, sigmoid


def load_job(job_id: str) -> dict[str, str]:
    definition = verify_frozen_sources()
    if definition.get("outer_access") != "AUTHORIZED_ONLY_AFTER_THIS_FREEZE":
        raise RuntimeError("FW5 outer access is not authorized")
    matches = [row for row in load_csv_rows(MANIFEST_PATH) if row["fw5_job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one FW5 manifest row for {job_id}, found {len(matches)}")
    validate_manifest_row(matches[0])
    return matches[0]


def prepare_outer_partitions(job: dict[str, str]) -> dict[str, Any]:
    membership = pd.read_csv(MEMBERSHIP_PATH, dtype={"RID": str})
    outer = membership[membership["outer_domain"].eq(job["outer_domain"])].copy()
    if outer["RID"].duplicated().any():
        raise RuntimeError("FW5 outer membership contains duplicate subjects")
    train_rids = set(outer.loc[outer["outer_role"].eq("train_pool"), "RID"])
    test_rids = set(outer.loc[outer["outer_role"].eq("test"), "RID"])
    if not train_rids or not test_rids or train_rids & test_rids:
        raise RuntimeError("FW5 frozen outer train/test membership is invalid")
    cohort = pd.read_csv(COHORT_PATH, low_memory=False)
    cohort["RID"] = cohort["RID"].astype(str)
    allowed = train_rids | test_rids
    analysis = cohort[cohort["RID"].isin(allowed)].copy()
    if len(analysis) != len(allowed) or analysis["RID"].duplicated().any():
        raise RuntimeError("FW5 cohort and frozen membership do not align")
    train_pool = analysis[analysis["RID"].isin(train_rids)].sort_values("RID", kind="stable").reset_index(drop=True)
    outer_test = analysis[analysis["RID"].isin(test_rids)].sort_values("RID", kind="stable").reset_index(drop=True)
    fit_indices, calibration_indices = deterministic_fit_calibration_indices(
        train_pool["AD_label"].to_numpy(dtype=int),
        seed=int(job["seed"]),
        calibration_fraction=0.15,
    )
    fit = train_pool.iloc[fit_indices].reset_index(drop=True)
    calibration = train_pool.iloc[calibration_indices].reset_index(drop=True)
    if set(fit["RID"]) & set(calibration["RID"]) or set(train_pool["RID"]) & set(outer_test["RID"]):
        raise RuntimeError("FW5 partitions overlap")
    return {"fit": fit, "calibration": calibration, "outer_test": outer_test}


def run_job(job_id: str, *, device_name: str) -> Path:
    job = load_job(job_id)
    final_directory = OUTPUT_ROOT / job_id
    if final_directory.exists():
        if verify_completed_directory(final_directory, expected_job_id=job_id):
            print(f"SKIP verified existing FW5 job: {final_directory}")
            return final_directory
        raise RuntimeError(f"Existing FW5 output is incomplete or invalid: {final_directory}")
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["nodes"]
    configuration = find_configuration(registry, {**job, "model_kind": "neural"})
    common = registry["common_training"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    started = time.perf_counter()
    partitions = prepare_outer_partitions(job)
    fit_frame = partitions["fit"]
    calibration_frame = partitions["calibration"]
    outer_frame = partitions["outer_test"]
    transformer = FrozenROITransformer.fit(fit_frame, mapping, np.arange(len(fit_frame)))
    fit_features = transformer.transform(fit_frame)
    calibration_features = transformer.transform(calibration_frame)
    outer_features = transformer.transform(outer_frame)
    fit_labels = fit_frame["AD_label"].to_numpy(dtype=int, copy=True)
    calibration_labels = calibration_frame["AD_label"].to_numpy(dtype=int, copy=True)

    adjacency = None
    if job["family"] in {"gcn", "gat"}:
        adjacency = build_structural_covariance_adjacency(fit_features, np.arange(len(fit_features)), top_k=int(configuration["top_k"]))
    model, training = train_neural(
        job["family"], configuration, fit_features, fit_labels, calibration_features, calibration_labels,
        adjacency, seed=int(job["seed"]), max_epochs=int(common["max_epochs"]), common=common, device=device,
    )
    calibration_scores = model_scores(model, calibration_features, adjacency, device)
    outer_scores = model_scores(model, outer_features, adjacency, device)
    calibrator = fit_platt(calibration_scores, calibration_labels)
    calibrated = calibrator.predict(outer_scores)
    raw_probability = sigmoid(outer_scores)
    if not np.isfinite(calibrated).all() or not np.isfinite(raw_probability).all():
        raise RuntimeError("FW5 outer probabilities are non-finite")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_ROOT / f".{job_id}.tmp-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir()
    try:
        predictions = pd.DataFrame(
            {
                "fw5_job_id": job_id,
                "outer_domain": job["outer_domain"],
                "family": job["family"],
                "configuration_id": job["configuration_id"],
                "seed": int(job["seed"]),
                "RID": outer_frame["RID"].astype(str),
                "AD_label": outer_frame["AD_label"].to_numpy(dtype=int),
                "raw_score": outer_scores,
                "raw_probability_AD": raw_probability,
                "calibrated_probability_AD": calibrated,
            }
        )
        predictions.to_csv(temporary / "outer_predictions.csv", index=False, lineterminator="\n")
        save_preprocessor(temporary / "preprocessor.npz", transformer)
        if adjacency is not None:
            np.save(temporary / "adjacency.npy", adjacency)
        checkpoint = {
            "fw5_job_id": job_id,
            "outer_domain": job["outer_domain"],
            "family": job["family"],
            "configuration_id": job["configuration_id"],
            "seed": int(job["seed"]),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
        torch.save(checkpoint, temporary / "checkpoint.pt")
        summary = {
            "fw5_job_id": job_id,
            "status": "PASS",
            "job": job,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "split_counts": {"model_fit_n": len(fit_frame), "calibration_n": len(calibration_frame), "outer_test_n": len(outer_frame)},
            "split_checks": {"fit_calibration_disjoint": True, "outer_test_disjoint": True},
            "training": training,
            "platt": {"coefficient": calibrator.coefficient, "intercept": calibrator.intercept},
            "elapsed_seconds": time.perf_counter() - started,
            "checks": {"selection_recomputed_from_outer": False, "outer_recalibration": False, "outer_test_predictions_generated": True},
            "source_sha256": {
                "definition": sha256_file(DEFINITION_PATH),
                "manifest": sha256_file(MANIFEST_PATH),
                "cohort": sha256_file(COHORT_PATH),
                "membership": sha256_file(MEMBERSHIP_PATH),
                "mapping": sha256_file(MAPPING_PATH),
                "model_registry": sha256_file(MODEL_REGISTRY_PATH),
            },
            "privacy": "Private outer predictions and training metadata.",
        }
        (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts = {path.name: sha256_file(path) for path in sorted(temporary.iterdir())}
        completion = {
            "fw5_job_id": job_id,
            "status": "PASS",
            "outer_test_predictions_generated": True,
            "selection_recomputed_from_outer": False,
            "artifact_sha256": artifacts,
        }
        (temporary / "completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.rename(final_directory)
    except Exception:
        if temporary.exists() and temporary.parent == OUTPUT_ROOT:
            shutil.rmtree(temporary)
        raise
    if not verify_completed_directory(final_directory, expected_job_id=job_id):
        raise RuntimeError("New FW5 job failed immutable verification")
    print(f"PASS: {final_directory}")
    return final_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    if args.verify_only:
        load_job(args.job_id)
        directory = OUTPUT_ROOT / args.job_id
        if not verify_completed_directory(directory, expected_job_id=args.job_id):
            raise RuntimeError(f"FW5 job failed verify-only: {directory}")
        print(f"PASS verified only: {directory}")
        return
    run_job(args.job_id, device_name=args.device)


if __name__ == "__main__":
    main()
