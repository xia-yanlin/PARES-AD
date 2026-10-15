#!/usr/bin/env python3
"""Run the frozen independent-participant confirmation of PARES-AD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_v2.src.analysis.build_ad_mri_explanation_system import (
    CORE_CONCEPTS,
    CONCEPT_ORIENTATION,
    RANDOM_SEED,
    calibration_slope,
    fit_monotone_pathology_model,
)
from research_v2.src.audit.concepts import build_concept_target
from research_v2.src.data.build_acquisition_domains import (
    canonicalize_field_strength,
    canonicalize_manufacturer,
    canonicalize_phase,
)
from research_v2.src.data.noncombat_features import FrozenROITransformer
from research_v2.src.execution.fw5_common import verify_completed_directory
from research_v2.src.execution.run_fw2_job import find_configuration, model_scores
from research_v2.src.models.neural import build_neural_model


HISS_ROOT = Path(__file__).resolve().parents[3]
V2_ROOT = HISS_ROOT / "research_v2"
RAW_FS = HISS_ROOT / "data" / "raw" / "tabular" / "ADSP_PHC_T1_FS_06Jul2026.csv"
DXSUM = HISS_ROOT / "data" / "raw" / "tabular" / "DXSUM_05Jul2026.csv"
BASELINE = HISS_ROOT / "data" / "processed" / "baseline_cohort_with_splits.csv"
BASELINE_CONCEPTS = HISS_ROOT / "data" / "processed" / "baseline_cohort_with_concepts.csv"
ASSIGNMENTS = V2_ROOT / "outputs" / "private" / "domain_assignments.csv"
MANIFEST = V2_ROOT / "outputs" / "private" / "fw5" / "manifests" / "fw5_formal_job_manifest.csv"
JOBS_ROOT = V2_ROOT / "outputs" / "private" / "fw5" / "jobs"
MAPPING = V2_ROOT / "data_registry" / "noncombat_roi_mapping_v2_1.json"
REGISTRY = V2_ROOT / "configs" / "model_registry_v2_1.json"
DEVELOPMENT_CASES = V2_ROOT / "outputs" / "private" / "ad_mri_explanation_system_2026-07-23" / "case_explanations.csv"
PROTOCOL = V2_ROOT / "protocols" / "pares_ad_independent_confirmation_freeze_2026-07-23.md"
DEFAULT_OUTPUT = V2_ROOT / "confirmation_results_2026-07-23"
BOOTSTRAP_REPLICATES = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rid(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("Int64")
    return values.astype(str)


def build_confirmation_cohort(mapping: list[dict[str, Any]]) -> pd.DataFrame:
    fs = pd.read_csv(RAW_FS, low_memory=False)
    diagnosis = pd.read_csv(DXSUM, low_memory=False)
    baseline_rids = set(canonical_rid(pd.read_csv(BASELINE, usecols=["RID"])["RID"]))
    fs["_RID"] = canonical_rid(fs["RID"])
    diagnosis["_RID"] = canonical_rid(diagnosis["RID"])
    fs["_VISCODE2"] = fs["VISCODE2"].fillna("").astype(str).str.strip().str.lower()
    diagnosis["_VISCODE2"] = diagnosis["VISCODE2"].fillna("").astype(str).str.strip().str.lower()
    diagnosis["_DIAGNOSIS"] = pd.to_numeric(diagnosis["DIAGNOSIS"], errors="coerce")

    def unique_ad_cn(values: pd.Series) -> float:
        unique = sorted(set(int(value) for value in values.dropna() if value in {1, 3}))
        return float(unique[0]) if len(unique) == 1 else float("nan")

    visit_diagnosis = (
        diagnosis.groupby(["_RID", "_VISCODE2"], as_index=False)["_DIAGNOSIS"]
        .agg(unique_ad_cn)
        .rename(columns={"_DIAGNOSIS": "DIAGNOSIS_CONFIRM"})
    )
    merged = fs.merge(visit_diagnosis, on=["_RID", "_VISCODE2"], how="left", validate="many_to_one")
    allowed_phase = merged["PHASE"].fillna("").astype(str).str.strip().str.upper().isin({"ADNI1", "ADNI3", "ADNI4"})
    merged = merged[allowed_phase].copy()
    merged["phase"] = merged["PHASE"].map(canonicalize_phase)
    merged["manufacturer"] = merged["ScannerManufacturer"].map(canonicalize_manufacturer)
    merged["field_strength"] = merged["FieldSgth"].map(canonicalize_field_strength)
    merged["outer_domain"] = merged["phase"] + "_" + merged["manufacturer"]
    eligible_domains = set(pd.read_csv(MANIFEST)["outer_domain"])
    merged["_scan_date"] = pd.to_datetime(merged["PHC_SCANDATE"], errors="coerce")
    required = ["EstimatedTotalIntraCranialVol"]
    required.extend(node["volume_col"] for node in mapping)
    required.extend(node["thickness_col"] for node in mapping if node["thickness_col"] is not None)
    values = merged[required].apply(pd.to_numeric, errors="coerce")
    valid_roi = np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1)
    eligible = merged[
        ~merged["_RID"].isin(baseline_rids)
        & merged["DIAGNOSIS_CONFIRM"].isin([1.0, 3.0])
        & merged["outer_domain"].isin(eligible_domains)
        & merged["_scan_date"].notna()
        & valid_roi
    ].copy()
    earliest = eligible.groupby("_RID")["_scan_date"].transform("min")
    first = eligible[eligible["_scan_date"].eq(earliest)].copy()
    tie_counts = first.groupby("_RID").size()
    first = first[first["_RID"].isin(tie_counts[tie_counts.eq(1)].index)].copy()
    first["AD_label"] = first["DIAGNOSIS_CONFIRM"].eq(3).astype(int)
    first = first.sort_values(["outer_domain", "_RID"], kind="stable").reset_index(drop=True)
    if len(first) != 882 or first["_RID"].nunique() != 882:
        raise RuntimeError(f"Frozen confirmation receipt changed: rows={len(first)}, subjects={first['_RID'].nunique()}")
    if int(first["AD_label"].sum()) != 342:
        raise RuntimeError("Frozen confirmation diagnosis receipt changed")
    return first


def restore_transformer(path: Path, mapping: list[dict[str, Any]]) -> FrozenROITransformer:
    arrays = np.load(path)
    return FrozenROITransformer(
        mapping=tuple(mapping),
        cortical_mask=arrays["cortical_mask"],
        thickness_median=arrays["thickness_median"],
        thickness_mean=arrays["thickness_mean"],
        thickness_scale=arrays["thickness_scale"],
        volume_median=arrays["volume_median"],
        icv_median=float(arrays["icv_median"][0]),
        volume_beta=arrays["volume_beta"],
        volume_residual_mean=arrays["volume_residual_mean"],
        volume_residual_scale=arrays["volume_residual_scale"],
    )


def run_black_boxes(
    confirmation: pd.DataFrame,
    manifest: pd.DataFrame,
    mapping: list[dict[str, Any]],
    registry: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    device = torch.device("cpu")
    for record in manifest.to_dict("records"):
        job_id = str(record["fw5_job_id"])
        directory = JOBS_ROOT / job_id
        if not verify_completed_directory(directory, expected_job_id=job_id):
            raise RuntimeError(f"FW5 artifact verification failed: {job_id}")
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
        for key in ["outer_domain", "family", "configuration_id"]:
            if str(checkpoint[key]) != str(record[key]):
                raise RuntimeError(f"Checkpoint/manifest mismatch for {job_id}: {key}")
        if int(checkpoint["seed"]) != int(record["seed"]):
            raise RuntimeError(f"Checkpoint/manifest seed mismatch for {job_id}")
        evaluation = confirmation[confirmation["outer_domain"].eq(record["outer_domain"])]
        transformer = restore_transformer(directory / "preprocessor.npz", mapping)
        features = transformer.transform(evaluation)
        configuration = find_configuration(registry, {**record, "model_kind": "neural"})
        model = build_neural_model(
            str(record["family"]), configuration, n_nodes=features.shape[1], n_features=features.shape[2]
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        adjacency = np.load(directory / "adjacency.npy") if (directory / "adjacency.npy").exists() else None
        scores = model_scores(model, features, adjacency, device)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        coefficient = float(summary["platt"]["coefficient"])
        intercept = float(summary["platt"]["intercept"])
        probability = expit(coefficient * scores + intercept)
        frames.append(
            pd.DataFrame(
                {
                    "_RID": evaluation["_RID"].to_numpy(),
                    "outer_domain": record["outer_domain"],
                    "family": record["family"],
                    "configuration_id": record["configuration_id"],
                    "seed": int(record["seed"]),
                    "AD_label": evaluation["AD_label"].to_numpy(dtype=int),
                    "probability_AD": probability,
                }
            )
        )
    seed_predictions = pd.concat(frames, ignore_index=True)
    key = ["_RID", "outer_domain", "family", "configuration_id", "AD_label"]
    output = seed_predictions.groupby(key, as_index=False).agg(
        seed_count=("seed", "size"),
        model_probability_AD=("probability_AD", "mean"),
        between_seed_probability_sd=("probability_AD", lambda values: float(pd.Series(values).std(ddof=1))),
    )
    if not output["seed_count"].eq(5).all():
        raise RuntimeError("Every confirmation model-case prediction requires five seeds")
    return output


def build_pathology_reference(confirmation: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_CONCEPTS, low_memory=False, dtype={"RID": str})
    assignments = pd.read_csv(ASSIGNMENTS, dtype={"RID": str})
    baseline = baseline.merge(
        assignments[["RID", "acquisition_domain"]], on="RID", validate="one_to_one"
    )
    frames: list[pd.DataFrame] = []
    for domain, evaluation in confirmation.groupby("outer_domain", sort=True):
        train = baseline[~baseline["acquisition_domain"].eq(domain)].reset_index(drop=True)
        evaluation = evaluation.reset_index(drop=True)
        train_values: list[np.ndarray] = []
        evaluation_values: list[np.ndarray] = []
        for concept_id in CORE_CONCEPTS:
            target = build_concept_target(concept_id, train, evaluation)
            if not target.train_valid.all() or not target.evaluation_valid.all():
                raise RuntimeError(f"Invalid primary pathology concept rows for {domain}/{concept_id}")
            orientation = CONCEPT_ORIENTATION[concept_id]
            train_values.append(orientation * target.train_values.astype(float))
            evaluation_values.append(orientation * target.evaluation_values.astype(float))
        x_train = np.column_stack(train_values)
        x_evaluation = np.column_stack(evaluation_values)
        intercept, coefficients = fit_monotone_pathology_model(
            x_train, train["AD_label"].to_numpy(dtype=int), l2_penalty=0.1
        )
        if (coefficients < 0).any():
            raise RuntimeError("Pathology reference monotonicity failed")
        pathology_logit = intercept + x_evaluation @ coefficients
        frames.append(
            pd.DataFrame(
                {
                    "_RID": evaluation["_RID"].to_numpy(),
                    "outer_domain": domain,
                    "pathology_probability_AD": expit(pathology_logit),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def add_confidence_comparator(cases: pd.DataFrame) -> pd.DataFrame:
    development = pd.read_csv(DEVELOPMENT_CASES)
    output = cases.copy()
    output["confidence_error_risk"] = np.nan
    baseline_columns = ["model_probability_AD", "model_confidence_distance", "model_predicted_class"]
    for (family, configuration_id), evaluation_config in output.groupby(["family", "configuration_id"]):
        development_config = development[
            development["family"].eq(family) & development["configuration_id"].eq(configuration_id)
        ].copy()
        development_config["model_predicted_class"] = development_config["model_probability_AD"].ge(0.5).astype(int)
        development_config["model_confidence_distance"] = (development_config["model_probability_AD"] - 0.5).abs()
        for domain in sorted(evaluation_config["outer_domain"].unique()):
            train = development_config[~development_config["outer_domain"].eq(domain)]
            evaluation_index = evaluation_config[evaluation_config["outer_domain"].eq(domain)].index
            if train["model_error"].nunique() != 2:
                raise RuntimeError(f"Confidence comparator lacks both error classes for {family}/{configuration_id}/{domain}")
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2_000, random_state=RANDOM_SEED),
            )
            estimator.fit(train[baseline_columns], train["model_error"].astype(int))
            output.loc[evaluation_index, "confidence_error_risk"] = estimator.predict_proba(
                output.loc[evaluation_index, baseline_columns]
            )[:, 1]
    if output["confidence_error_risk"].isna().any():
        raise RuntimeError("Confidence comparator predictions are incomplete")
    return output


def review_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    conflict_errors = 0
    confidence_errors = 0
    reviewed = 0
    for _, group in frame.groupby(["family", "configuration_id"], sort=True):
        conflict = group["pathology_conflict"].to_numpy(dtype=bool)
        errors = group["model_error"].to_numpy(dtype=int)
        k = int(conflict.sum())
        if k == 0:
            continue
        ranking = np.argsort(-group["confidence_error_risk"].to_numpy(dtype=float), kind="stable")
        conflict_errors += int(errors[conflict].sum())
        confidence_errors += int(errors[ranking[:k]].sum())
        reviewed += k
    return {
        "reviewed_predictions": reviewed,
        "conflict_errors_captured": conflict_errors,
        "confidence_errors_captured": confidence_errors,
        "conflict_review_yield": conflict_errors / reviewed if reviewed else float("nan"),
        "confidence_review_yield": confidence_errors / reviewed if reviewed else float("nan"),
        "review_yield_gain": (conflict_errors - confidence_errors) / reviewed if reviewed else float("nan"),
    }


def state_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    conflict = frame["pathology_conflict"].to_numpy(dtype=bool)
    errors = frame["model_error"].to_numpy(dtype=int)
    conflict_rate = float(errors[conflict].mean()) if conflict.any() else float("nan")
    aligned_rate = float(errors[~conflict].mean()) if (~conflict).any() else float("nan")
    return {
        "conflict_predictions": int(conflict.sum()),
        "aligned_predictions": int((~conflict).sum()),
        "conflict_error_rate": conflict_rate,
        "aligned_error_rate": aligned_rate,
        "error_risk_ratio": conflict_rate / aligned_rate if aligned_rate > 0 else float("inf"),
    }


def bootstrap(cases: pd.DataFrame) -> dict[str, float]:
    subjects = np.asarray(sorted(cases["_RID"].unique()), dtype=object)
    blocks = {subject: cases[cases["_RID"].eq(subject)] for subject in subjects}
    generator = np.random.default_rng(RANDOM_SEED)
    gains = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    ratios = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = generator.choice(subjects, size=len(subjects), replace=True)
        pieces = []
        for draw_index, subject in enumerate(sampled):
            piece = blocks[subject].copy()
            piece["_bootstrap_subject"] = draw_index
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        gains[replicate] = float(review_statistics(sample)["review_yield_gain"])
        ratios[replicate] = float(state_statistics(sample)["error_risk_ratio"])
    return {
        "review_yield_gain_ci95_low": float(np.quantile(gains, 0.025)),
        "review_yield_gain_ci95_high": float(np.quantile(gains, 0.975)),
        "error_risk_ratio_ci95_low": float(np.quantile(ratios, 0.025)),
        "error_risk_ratio_ci95_high": float(np.quantile(ratios, 0.975)),
    }


def metric_row(scope: str, value: str, group: pd.DataFrame) -> dict[str, Any]:
    review = review_statistics(group)
    state = state_statistics(group)
    labels = group["AD_label"].to_numpy(dtype=int)
    probability = group["model_probability_AD"].to_numpy(dtype=float)
    return {
        "scope": scope,
        "value": value,
        "model_case_predictions": len(group),
        "unique_participants": group["_RID"].nunique(),
        "review_burden": state["conflict_predictions"] / len(group),
        **review,
        **state,
        "auroc": roc_auc_score(labels, probability) if len(np.unique(labels)) == 2 else np.nan,
        "auprc": average_precision_score(labels, probability) if len(np.unique(labels)) == 2 else np.nan,
        "brier": brier_score_loss(labels, probability),
    }


def analyze(cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [metric_row("overall", "all", cases)]
    for (family, configuration_id), group in cases.groupby(["family", "configuration_id"], sort=True):
        rows.append(metric_row("configuration", f"{family}|{configuration_id}", group))
    for family, group in cases.groupby("family", sort=True):
        rows.append(metric_row("family", family, group))
    for domain, group in cases.groupby("outer_domain", sort=True):
        rows.append(metric_row("domain", domain, group))
    results = pd.DataFrame(rows)
    overall = results[results["scope"].eq("overall")].iloc[0].to_dict()
    intervals = bootstrap(cases)
    config_rows = results[results["scope"].eq("configuration")]
    family_rows = results[results["scope"].eq("family")]
    config_direction = int((config_rows["conflict_error_rate"] > config_rows["aligned_error_rate"]).sum())
    family_direction = int((family_rows["conflict_error_rate"] > family_rows["aligned_error_rate"]).sum())
    conflict_participants = int(cases.loc[cases["pathology_conflict"], "_RID"].nunique())
    gates = {
        "artifact_and_receipt_integrity": bool(len(cases) > 0),
        "review_burden_3_to_15_percent": bool(0.03 <= overall["review_burden"] <= 0.15),
        "at_least_50_conflict_participants": bool(conflict_participants >= 50),
        "review_yield_gain_at_least_5pp": bool(overall["review_yield_gain"] >= 0.05),
        "review_yield_gain_ci_lower_above_zero": bool(intervals["review_yield_gain_ci95_low"] > 0),
        "error_risk_ratio_at_least_2": bool(overall["error_risk_ratio"] >= 2.0),
        "error_risk_ratio_ci_lower_above_one": bool(intervals["error_risk_ratio_ci95_low"] > 1.0),
        "direction_at_least_7_of_9_configurations": bool(config_direction >= 7),
        "direction_all_3_families": bool(family_direction == 3),
    }
    summary = {
        "confirmation_id": "PARES-AD-INDEPENDENT-CONFIRM-v1",
        "status": "PASS" if all(gates.values()) else "STOP",
        "data_receipt": {
            "independent_participants": int(cases["_RID"].nunique()),
            "ad": int(cases.drop_duplicates("_RID")["AD_label"].sum()),
            "cn": int(cases["_RID"].nunique() - cases.drop_duplicates("_RID")["AD_label"].sum()),
            "model_case_predictions": len(cases),
            "configurations": int(cases.groupby(["family", "configuration_id"]).ngroups),
            "families": int(cases["family"].nunique()),
            "outer_domains": int(cases["outer_domain"].nunique()),
        },
        "primary_endpoint": {
            "review_burden": overall["review_burden"],
            "reviewed_predictions": int(overall["reviewed_predictions"]),
            "conflict_errors_captured": int(overall["conflict_errors_captured"]),
            "confidence_errors_captured": int(overall["confidence_errors_captured"]),
            "conflict_review_yield": overall["conflict_review_yield"],
            "confidence_review_yield": overall["confidence_review_yield"],
            "absolute_review_yield_gain": overall["review_yield_gain"],
            "ci95_low": intervals["review_yield_gain_ci95_low"],
            "ci95_high": intervals["review_yield_gain_ci95_high"],
        },
        "corroboration": {
            "conflict_unique_participants": conflict_participants,
            "conflict_error_rate": overall["conflict_error_rate"],
            "aligned_error_rate": overall["aligned_error_rate"],
            "error_risk_ratio": overall["error_risk_ratio"],
            "error_risk_ratio_ci95_low": intervals["error_risk_ratio_ci95_low"],
            "error_risk_ratio_ci95_high": intervals["error_risk_ratio_ci95_high"],
            "configurations_with_higher_conflict_error": config_direction,
            "families_with_higher_conflict_error": family_direction,
        },
        "gates": gates,
    }
    return results, summary


def render_report(summary: dict[str, Any]) -> str:
    primary = summary["primary_endpoint"]
    corroboration = summary["corroboration"]
    receipt = summary["data_receipt"]
    gate_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in summary["gates"].items()
    )
    return f"""# PARES-AD independent confirmation

## Verdict

**{summary['status']} under the frozen protocol.**

The confirmation used {receipt['independent_participants']} participants absent from the entire 729-person baseline cohort ({receipt['ad']} AD, {receipt['cn']} CN), {receipt['model_case_predictions']} model-case predictions, nine model configurations, three neural families, and seven held-out acquisition domains. No model was retrained or recalibrated.

## Primary practical endpoint

At the same {primary['review_burden']:.2%} review workload, the AD-pathology conflict rule captured {primary['conflict_errors_captured']} errors versus {primary['confidence_errors_captured']} under the frozen confidence-only comparator. Error yield was {primary['conflict_review_yield']:.2%} versus {primary['confidence_review_yield']:.2%}: an absolute gain of {primary['absolute_review_yield_gain']:.2%} (95% participant-cluster bootstrap interval {primary['ci95_low']:.2%} to {primary['ci95_high']:.2%}).

## Corroboration

Conflict predictions had error rate {corroboration['conflict_error_rate']:.2%} versus {corroboration['aligned_error_rate']:.2%}, risk ratio {corroboration['error_risk_ratio']:.2f} (95% interval {corroboration['error_risk_ratio_ci95_low']:.2f} to {corroboration['error_risk_ratio_ci95_high']:.2f}). The direction held in {corroboration['configurations_with_higher_conflict_error']}/9 configurations and {corroboration['families_with_higher_conflict_error']}/3 families; {corroboration['conflict_unique_participants']} unique participants received at least one conflict flag.

## Frozen gates

{gate_lines}

## Claim boundary

If PASS, the evidence supports an AD/MRI-specific second-reader claim: disagreement between a black-box prediction and an exact, monotone four-feature AD structural reference improves error-focused review triage over model confidence alone in an independent ADNI participant set. It does not establish clinician benefit, causal mechanism, a new diagnosis, external-cohort transport, or cross-disease validity.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    expected_hashes = {
        RAW_FS: "5093ebfcfd1330eb88390aafe3c3bc9470063e537864d11824363f828c0cddd8",
        DXSUM: "d7c484fcaf56d4f7d83a4f862a805bfafc4146ccd91e3baab02e7bb8b455d42f",
        BASELINE: "246db0323d0ba9e0fdb3987bc0b15ea834c06943fa083df7caedd03e7faa5613",
        BASELINE_CONCEPTS: "ffbe5879de554abdacd29fa0a75685311aed7c839876b2a75dff16c7498c4ce5",
        ASSIGNMENTS: "52813725ce19cdf1452dcee71848292df467529e200c5bf14fe77ab8e91ede4c",
        MANIFEST: "7a6c618f7985b430b6a74e3ab48a66c69a18e183b441cc6c48375fef66b5e4d9",
        MAPPING: "8ed9e565afb3f46e6bc4ec1a4e9d088d470184842a2317d43e6b8580dc64c3db",
        REGISTRY: "10c5d5c38ca1d07a679218204ec1bb4a31d31eeeb3ecbdcc00a728bb53aa90c7",
        DEVELOPMENT_CASES: "a7ad3530154e71d0789188e498b60479ddc6af612d46e1840979f769b4d7acfa",
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen source hash changed: {path}")
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST, dtype={"seed": int})
    if len(manifest) != 290:
        raise RuntimeError("Frozen FW5 manifest must contain 290 jobs")
    confirmation = build_confirmation_cohort(mapping)
    predictions = run_black_boxes(confirmation, manifest, mapping, registry)
    pathology = build_pathology_reference(confirmation)
    cases = predictions.merge(pathology, on=["_RID", "outer_domain"], validate="many_to_one")
    cases["model_predicted_class"] = cases["model_probability_AD"].ge(0.5).astype(int)
    cases["model_confidence_distance"] = (cases["model_probability_AD"] - 0.5).abs()
    cases["pathology_predicted_class"] = cases["pathology_probability_AD"].ge(0.5).astype(int)
    cases["pathology_conflict"] = cases["model_predicted_class"].ne(cases["pathology_predicted_class"])
    cases["model_error"] = cases["model_predicted_class"].ne(cases["AD_label"]).astype(int)
    cases = add_confidence_comparator(cases)
    results, summary = analyze(cases)
    summary["runtime_seconds"] = time.perf_counter() - started
    summary["protocol"] = {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)}
    summary["source_sha256"] = {str(path): expected for path, expected in expected_hashes.items()}
    summary["privacy"] = {"participant_identifiers_written": False, "participant_level_rows_written": False}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "confirmation_results.csv", index=False, lineterminator="\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "runtime_seconds": summary["runtime_seconds"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
