#!/usr/bin/env python3
"""Run one frozen FW2 inner-domain benchmark job.

All participant-derived artifacts are written beneath ``outputs/private``.
Formal jobs are immutable: a verified completed directory is skipped, and an
existing incomplete or mismatched directory fails closed rather than being
overwritten.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from research_v2.src.data.noncombat_features import (
    FrozenROITransformer,
    build_structural_covariance_adjacency,
)
from research_v2.src.models import build_neural_model, build_reference_model, count_parameters
from research_v2.src.representations import ModelMetadata, TorchModelInputs, TorchRepresentationAdapter
from research_v2.src.statistics.metrics import binary_metrics, fit_platt, logit, sigmoid


V2_ROOT = Path(__file__).resolve().parents[2]
HISS_ROOT = V2_ROOT.parent
COHORT_PATH = HISS_ROOT / "data" / "processed" / "baseline_cohort_with_splits.csv"
MEMBERSHIP_PATH = V2_ROOT / "outputs" / "private" / "nested_domain_splits.csv"
MAPPING_PATH = V2_ROOT / "data_registry" / "noncombat_roi_mapping_v2_1.json"
MODEL_REGISTRY_PATH = V2_ROOT / "configs" / "model_registry_v2_1.json"
JOB_MANIFEST_PATH = V2_ROOT / "data_registry" / "fw2_inner_benchmark_job_manifest.csv"
JOB_DEFINITION_PATH = V2_ROOT / "data_registry" / "fw2_inner_benchmark_job_definition.json"
FORMAL_ROOT = V2_ROOT / "outputs" / "private" / "fw2" / "jobs"
PILOT_ROOT = V2_ROOT / "outputs" / "private" / "fw2" / "pilots"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_fit_calibration_indices(
    labels: np.ndarray,
    *,
    seed: int,
    calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Fit/calibration split requires both classes")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=calibration_fraction,
        random_state=seed,
    )
    fit_indices, calibration_indices = next(splitter.split(np.zeros(len(labels)), labels))
    if set(fit_indices) & set(calibration_indices):
        raise AssertionError("Fit and calibration indices overlap")
    if len(fit_indices) + len(calibration_indices) != len(labels):
        raise AssertionError("Fit/calibration split does not cover the training pool")
    for subset in [fit_indices, calibration_indices]:
        if set(np.unique(labels[subset])) != {0, 1}:
            raise ValueError("Fit and calibration subsets must each contain both classes")
    return np.asarray(fit_indices), np.asarray(calibration_indices)


def load_job(job_id: str) -> dict[str, str]:
    with JOB_MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest row for {job_id}, found {len(matches)}")
    job = matches[0]
    if job["outer_test_access"] != "forbidden":
        raise ValueError("Job manifest does not forbid outer-test access")
    return job


def find_configuration(registry: dict, job: dict[str, str]) -> dict[str, Any]:
    collection = registry["neural_families"] if job["model_kind"] == "neural" else registry["strong_references"]
    families = [entry for entry in collection if entry["family"] == job["family"]]
    if len(families) != 1:
        raise ValueError(f"Family is not uniquely registered: {job['family']}")
    configs = [entry for entry in families[0]["configurations"] if entry["id"] == job["configuration_id"]]
    if len(configs) != 1:
        raise ValueError(f"Configuration is not uniquely registered: {job['configuration_id']}")
    return configs[0]


def verify_completed_directory(directory: Path) -> bool:
    completion_path = directory / "completion.json"
    if not completion_path.exists():
        return False
    with completion_path.open("r", encoding="utf-8") as handle:
        completion = json.load(handle)
    if completion.get("status") != "PASS":
        return False
    for relative, expected_hash in completion.get("artifact_sha256", {}).items():
        path = directory / relative
        if not path.exists() or sha256_file(path) != expected_hash:
            return False
    return True


def job_output_directory(job: dict[str, str], *, mode: str, max_epochs: int) -> Path:
    root = FORMAL_ROOT if mode == "formal" else PILOT_ROOT
    name = job["job_id"] if mode == "formal" else f"{job['job_id']}_e{max_epochs}"
    return root / name


def save_preprocessor(path: Path, transformer: FrozenROITransformer) -> None:
    np.savez_compressed(
        path,
        cortical_mask=transformer.cortical_mask,
        thickness_median=transformer.thickness_median,
        thickness_mean=transformer.thickness_mean,
        thickness_scale=transformer.thickness_scale,
        volume_median=transformer.volume_median,
        icv_median=np.asarray([transformer.icv_median]),
        volume_beta=transformer.volume_beta,
        volume_residual_mean=transformer.volume_residual_mean,
        volume_residual_scale=transformer.volume_residual_scale,
    )


def model_scores(model: Any, features: np.ndarray, adjacency: np.ndarray | None, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        graph = None if adjacency is None else torch.as_tensor(adjacency, dtype=torch.float32, device=device)
        return model(x, graph).detach().cpu().numpy().reshape(-1)


def train_neural(
    family: str,
    configuration: dict[str, Any],
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    adjacency: np.ndarray | None,
    *,
    seed: int,
    max_epochs: int,
    common: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_neural_model(
        family,
        configuration,
        n_nodes=fit_features.shape[1],
        n_features=fit_features.shape[2],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    positives = float(fit_labels.sum())
    negatives = float(len(fit_labels) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    )
    dataset = TensorDataset(
        torch.as_tensor(fit_features, dtype=torch.float32),
        torch.as_tensor(fit_labels, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(common["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    graph = None if adjacency is None else torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    patience = int(common["early_stopping"]["patience"])
    minimum_delta = float(common["early_stopping"]["minimum_delta"])
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    last_loss = float("nan")
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features, graph).reshape(-1)
            loss = loss_fn(logits, batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Neural training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses))
        calibration_scores = model_scores(model, calibration_features, adjacency, device)
        calibration_auc = float(roc_auc_score(calibration_labels, calibration_scores))
        if calibration_auc > best_auc + minimum_delta:
            best_auc = calibration_auc
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No valid neural checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_calibration_subset_auroc_private": best_auc,
        "last_training_loss_private": last_loss,
        "fit_seconds": elapsed,
        "parameter_count": count_parameters(model),
    }


def reference_raw_scores(model: Any, features: np.ndarray) -> np.ndarray:
    flat = features.reshape(len(features), -1)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(flat), dtype=float).reshape(-1)
    if hasattr(model, "predict_proba"):
        return logit(np.asarray(model.predict_proba(flat)[:, 1], dtype=float))
    return logit(np.asarray(model.predict(flat), dtype=float))


def prepare_partitions(job: dict[str, str], seed: int) -> dict[str, Any]:
    membership = pd.read_csv(MEMBERSHIP_PATH, dtype={"RID": str})
    outer = membership[membership["outer_domain"].eq(job["outer_domain"])].copy()
    role_column = f"inner_role__{job['inner_validation_domain']}"
    if role_column not in outer:
        raise ValueError("Inner validation role is missing from frozen membership")
    train_rids = set(outer.loc[outer[role_column].eq("training"), "RID"])
    validation_rids = set(outer.loc[outer[role_column].eq("validation"), "RID"])
    outer_test_rids = set(outer.loc[outer["outer_role"].eq("test"), "RID"])
    if train_rids & validation_rids or train_rids & outer_test_rids or validation_rids & outer_test_rids:
        raise AssertionError("Frozen train, inner-validation, and outer-test sets overlap")

    allowed_rids = train_rids | validation_rids
    cohort = pd.read_csv(COHORT_PATH, low_memory=False)
    cohort["RID"] = cohort["RID"].astype(str)
    analysis = cohort[cohort["RID"].isin(allowed_rids)].copy()
    if len(analysis) != len(allowed_rids):
        raise ValueError("Cohort and frozen membership do not align")
    train_pool = analysis[analysis["RID"].isin(train_rids)].reset_index(drop=True)
    validation = analysis[analysis["RID"].isin(validation_rids)].reset_index(drop=True)
    fit_indices, calibration_indices = deterministic_fit_calibration_indices(
        train_pool["AD_label"].to_numpy(int),
        seed=seed,
        calibration_fraction=0.15,
    )
    fit = train_pool.iloc[fit_indices].reset_index(drop=True)
    calibration = train_pool.iloc[calibration_indices].reset_index(drop=True)
    if set(fit["RID"]) & set(calibration["RID"]):
        raise AssertionError("Model-fit and calibration RIDs overlap")
    return {
        "fit": fit,
        "calibration": calibration,
        "validation": validation,
        "outer_test_n": len(outer_test_rids),
        "checks": {
            "outer_rows_loaded_into_analysis_frame": False,
            "fit_calibration_disjoint": True,
            "inner_validation_disjoint": True,
            "outer_test_disjoint": True,
        },
    }


def run_job(job: dict[str, str], *, mode: str, pilot_epochs: int | None, device_name: str) -> Path:
    with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    with MAPPING_PATH.open("r", encoding="utf-8") as handle:
        mapping_artifact = json.load(handle)
    configuration = find_configuration(registry, job)
    seed = int(job["seed"])
    common = registry["common_training"]
    max_epochs = int(common["max_epochs"])
    if mode == "pilot":
        if pilot_epochs is None or not 1 <= pilot_epochs <= max_epochs:
            raise ValueError("Pilot epochs must be between 1 and the frozen maximum")
        max_epochs = pilot_epochs

    final_directory = job_output_directory(job, mode=mode, max_epochs=max_epochs)
    if final_directory.exists():
        if verify_completed_directory(final_directory):
            print(f"SKIP verified existing {mode} job: {final_directory}")
            return final_directory
        raise RuntimeError(f"Existing job directory is incomplete or mismatched: {final_directory}")

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    partitions = prepare_partitions(job, seed)
    fit_frame = partitions["fit"]
    calibration_frame = partitions["calibration"]
    validation_frame = partitions["validation"]
    mapping = list(mapping_artifact["nodes"])
    transformer = FrozenROITransformer.fit(fit_frame, mapping, np.arange(len(fit_frame)))
    fit_features = transformer.transform(fit_frame)
    calibration_features = transformer.transform(calibration_frame)
    validation_features = transformer.transform(validation_frame)
    fit_labels = fit_frame["AD_label"].to_numpy(dtype=int, copy=True)
    calibration_labels = calibration_frame["AD_label"].to_numpy(dtype=int, copy=True)
    validation_labels = validation_frame["AD_label"].to_numpy(dtype=int, copy=True)

    adjacency = None
    training_metadata: dict[str, Any]
    checkpoint_payload = None
    if job["model_kind"] == "neural":
        if job["family"] in {"gcn", "gat"}:
            adjacency = build_structural_covariance_adjacency(
                fit_features,
                np.arange(len(fit_features)),
                top_k=int(configuration["top_k"]),
            )
        model, training_metadata = train_neural(
            job["family"],
            configuration,
            fit_features,
            fit_labels,
            calibration_features,
            calibration_labels,
            adjacency,
            seed=seed,
            max_epochs=max_epochs,
            common=common,
            device=device,
        )
        calibration_scores = model_scores(model, calibration_features, adjacency, device)
        validation_scores = model_scores(model, validation_features, adjacency, device)
        checkpoint_payload = {
            "job_id": job["job_id"],
            "family": job["family"],
            "configuration_id": job["configuration_id"],
            "seed": seed,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }
    else:
        model = build_reference_model(job["family"], configuration, seed=seed)
        started = time.perf_counter()
        model.fit(fit_features.reshape(len(fit_features), -1), fit_labels)
        training_metadata = {"fit_seconds": time.perf_counter() - started, "parameter_count": None}
        calibration_scores = reference_raw_scores(model, calibration_features)
        validation_scores = reference_raw_scores(model, validation_features)

    calibrator = fit_platt(calibration_scores, calibration_labels)
    raw_probabilities = sigmoid(validation_scores)
    calibrated_probabilities = calibrator.predict(validation_scores)
    if not np.isfinite(raw_probabilities).all() or not np.isfinite(calibrated_probabilities).all():
        raise RuntimeError("Validation probabilities contain non-finite values")

    root = final_directory.parent
    name = final_directory.name
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise RuntimeError(f"Temporary job directory already exists: {temporary}")
    temporary.mkdir()

    predictions = pd.DataFrame(
        {
            "RID": validation_frame["RID"].astype(str),
            "AD_label": validation_labels,
            "raw_score": validation_scores,
            "raw_probability_AD": raw_probabilities,
            "calibrated_probability_AD": calibrated_probabilities,
        }
    )
    predictions.to_csv(temporary / "predictions.csv", index=False, lineterminator="\n")
    save_preprocessor(temporary / "preprocessor.npz", transformer)
    if adjacency is not None:
        np.save(temporary / "adjacency.npy", adjacency)
    if checkpoint_payload is not None:
        torch.save(checkpoint_payload, temporary / "checkpoint.pt")
        metadata = ModelMetadata(
            model_id=job["job_id"],
            family=job["family"],
            configuration_id=job["configuration_id"],
            seed=seed,
            fold=f"{job['outer_domain']}__{job['inner_validation_domain']}",
            checkpoint_sha256=sha256_file(temporary / "checkpoint.pt"),
            input_manifest_sha256=sha256_file(MAPPING_PATH),
            parameter_count=int(training_metadata["parameter_count"]),
        )
        adapter = TorchRepresentationAdapter(model, metadata, device=str(device))
        contract_n = min(16, len(validation_frame))
        contract = adapter.validate_contract(
            TorchModelInputs(validation_features[:contract_n], adjacency),
            validation_frame["RID"].astype(str).iloc[:contract_n].tolist(),
            layer_name="penultimate",
        )
    else:
        contract = None

    summary = {
        "job": job,
        "mode": mode,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "max_epochs": max_epochs,
        "split_counts": {
            "model_fit_n": len(fit_frame),
            "calibration_n": len(calibration_frame),
            "inner_validation_n": len(validation_frame),
            "reserved_outer_test_n": partitions["outer_test_n"],
        },
        "split_checks": partitions["checks"],
        "training": training_metadata,
        "platt": {"coefficient": calibrator.coefficient, "intercept": calibrator.intercept},
        "inner_validation_metrics_private": {
            "raw": binary_metrics(validation_labels, raw_probabilities),
            "calibrated": binary_metrics(validation_labels, calibrated_probabilities),
        },
        "representation_contract": contract,
        "outer_test_predictions_generated": False,
        "input_sha256": {
            "cohort": sha256_file(COHORT_PATH),
            "membership": sha256_file(MEMBERSHIP_PATH),
            "mapping": sha256_file(MAPPING_PATH),
            "model_registry": sha256_file(MODEL_REGISTRY_PATH),
            "job_manifest": sha256_file(JOB_MANIFEST_PATH),
            "job_definition": sha256_file(JOB_DEFINITION_PATH),
        },
        "privacy": "Private controlled artifact; contains participant-derived metrics but no outer-test outcomes.",
    }
    (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_names = [path.name for path in temporary.iterdir() if path.name != "completion.json"]
    artifact_hashes = {name: sha256_file(temporary / name) for name in sorted(artifact_names)}
    completion = {
        "job_id": job["job_id"],
        "mode": mode,
        "status": "PASS",
        "artifact_sha256": artifact_hashes,
        "outer_test_predictions_generated": False,
    }
    (temporary / "completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.rename(final_directory)
    print(f"PASS: {final_directory}")
    return final_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--mode", choices=["formal", "pilot"], default="formal")
    parser.add_argument("--pilot-epochs", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    job = load_job(args.job_id)
    if args.verify_only:
        if args.mode == "pilot" and args.pilot_epochs is None:
            raise ValueError("Pilot verification requires --pilot-epochs")
        if args.mode == "formal":
            with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
                max_epochs = int(json.load(handle)["common_training"]["max_epochs"])
        else:
            max_epochs = int(args.pilot_epochs)
        directory = job_output_directory(job, mode=args.mode, max_epochs=max_epochs)
        if not verify_completed_directory(directory):
            raise RuntimeError(f"Job output did not pass verification: {directory}")
        print(f"PASS verified only: {directory}")
        return
    run_job(job, mode=args.mode, pilot_epochs=args.pilot_epochs, device_name=args.device)


if __name__ == "__main__":
    main()
