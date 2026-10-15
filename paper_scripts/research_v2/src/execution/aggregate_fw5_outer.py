#!/usr/bin/env python3
"""Freeze private FW5 outer endpoints from immutable seed-level predictions."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from research_v2.src.data.build_acquisition_domains import (
    canonicalize_field_strength,
    canonicalize_manufacturer,
    canonicalize_phase,
)
from research_v2.src.execution.fw5_common import (
    AGGREGATION_PLAN_PATH,
    AGGREGATION_ROOT,
    COHORT_PATH,
    DEFINITION_PATH,
    DOMAIN_DEFINITION_PATH,
    FW2_SUMMARIES_PATH,
    FW3_READABILITY_PATH,
    FW4_CONFIG_PATH,
    FW4_DOMAIN_PATH,
    FW4_SELECTION_PATH,
    MANIFEST_PATH,
    MEMBERSHIP_PATH,
    OUTPUT_ROOT,
    PLAN_PATH,
    V2_ROOT,
    load_csv_rows,
    parse_bool,
    sha256_file,
    verify_completed_directory,
    verify_frozen_sources,
)
from research_v2.src.statistics.metrics import binary_metrics


OUTPUT_DIRECTORY = AGGREGATION_ROOT / "fw5_outer_endpoints_v2_1"
TRACKED_LOG = V2_ROOT / "logs" / "fw5_outer_evaluation_freeze.json"


def verified_existing_directory(directory: Path) -> bool:
    completion_path = directory / "completion.json"
    if not completion_path.is_file():
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS" or completion.get("selection_recomputed_from_outer") is not False:
        return False
    hashes = completion.get("artifact_sha256", {})
    return bool(hashes) and all((directory / name).is_file() and sha256_file(directory / name) == expected for name, expected in hashes.items())


def load_seed_predictions(manifest: list[dict[str, str]]) -> pd.DataFrame:
    frames = []
    for row in manifest:
        directory = OUTPUT_ROOT / row["fw5_job_id"]
        if not verify_completed_directory(directory, row["fw5_job_id"]):
            raise RuntimeError(f"FW5 job is missing or invalid: {row['fw5_job_id']}")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["checks"].get("selection_recomputed_from_outer") is not False or summary["checks"].get("outer_recalibration") is not False:
            raise RuntimeError("FW5 job violates the frozen outer contract")
        frame = pd.read_csv(directory / "outer_predictions.csv", dtype={"RID": str})
        for field in ["fw5_job_id", "outer_domain", "family", "configuration_id", "seed"]:
            if not frame[field].astype(str).eq(str(row[field])).all():
                raise RuntimeError(f"FW5 manifest/prediction mismatch: {field}")
        frame["performance_only_selected"] = parse_bool(row["performance_only_selected"])
        frame["audit_guided_selected"] = parse_bool(row["audit_guided_selected"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate_seed_predictions(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    key = ["outer_domain", "family", "configuration_id", "RID", "AD_label", "performance_only_selected", "audit_guided_selected"]
    configuration = seed_predictions.groupby(key, as_index=False).agg(
        seed_count=("seed", "size"),
        mean_raw_probability_AD=("raw_probability_AD", "mean"),
        mean_calibrated_probability_AD=("calibrated_probability_AD", "mean"),
        between_seed_calibrated_probability_sd=("calibrated_probability_AD", lambda values: float(pd.Series(values).std(ddof=1))),
    )
    if not configuration["seed_count"].eq(5).all():
        raise RuntimeError("FW5 configuration predictions require exactly five seeds")
    return configuration


def metric_row(labels: pd.Series, probabilities: pd.Series) -> dict[str, Any]:
    return binary_metrics(labels.to_numpy(dtype=int), probabilities.to_numpy(dtype=float), threshold=0.5)


def configuration_metrics(configuration: pd.DataFrame, inner: pd.DataFrame) -> pd.DataFrame:
    inner_lookup = inner.set_index(["outer_domain", "configuration_id"])
    rows = []
    for key, group in configuration.groupby(["outer_domain", "family", "configuration_id"], sort=True):
        metrics = metric_row(group["AD_label"], group["mean_calibrated_probability_AD"])
        inner_row = inner_lookup.loc[(key[0], key[2])]
        rows.append(
            {
                "outer_domain": key[0], "family": key[1], "configuration_id": key[2],
                **metrics,
                "inner_macro_auroc": float(inner_row["macro_mean_calibrated_auroc"]),
                "inner_macro_brier": float(inner_row["macro_mean_calibrated_brier"]),
                "outer_auroc_loss": float(inner_row["macro_mean_calibrated_auroc"] - metrics["auroc"]),
                "performance_only_selected": bool(group["performance_only_selected"].iloc[0]),
                "audit_guided_selected": bool(group["audit_guided_selected"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def rule_predictions(configuration: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for rule, column in [("performance_only", "performance_only_selected"), ("audit_guided", "audit_guided_selected")]:
        selected = configuration[configuration[column]].copy()
        selected.insert(0, "selection_rule", rule)
        frames.append(selected)
    rules = pd.concat(frames, ignore_index=True)
    performance_keys = set(
        map(tuple, rules[rules["selection_rule"].eq("performance_only")][["outer_domain", "RID", "AD_label"]].to_numpy())
    )
    audit_keys = set(
        map(tuple, rules[rules["selection_rule"].eq("audit_guided")][["outer_domain", "RID", "AD_label"]].to_numpy())
    )
    if performance_keys != audit_keys:
        raise RuntimeError("FW5 rules do not have paired outer subject coverage")
    return rules


def rule_domain_metrics(rules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rule, outer), group in rules.groupby(["selection_rule", "outer_domain"], sort=True):
        rows.append({"selection_rule": rule, "outer_domain": outer, **metric_row(group["AD_label"], group["mean_calibrated_probability_AD"])})
    return pd.DataFrame(rows)


def _metric(labels: np.ndarray, probabilities: np.ndarray, name: str) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if name == "auroc":
        return float(roc_auc_score(labels, probabilities))
    if name == "auprc":
        return float(average_precision_score(labels, probabilities))
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(labels, probabilities >= 0.5))
    if name == "brier":
        return float(brier_score_loss(labels, probabilities))
    if name == "ece_10":
        boundaries = np.linspace(0.0, 1.0, 11)
        value = 0.0
        for index in range(10):
            mask = (probabilities >= boundaries[index]) & (
                probabilities <= boundaries[index + 1]
                if index == 9
                else probabilities < boundaries[index + 1]
            )
            if mask.any():
                value += mask.mean() * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
        return float(value)
    raise ValueError(f"Unknown bootstrap metric: {name}")


def e1_bootstrap(rules: pd.DataFrame, *, replicates: int, seed: int) -> np.ndarray:
    pivot = rules.pivot(index=["outer_domain", "RID", "AD_label"], columns="selection_rule", values="mean_calibrated_probability_AD").reset_index()
    domains = sorted(pivot["outer_domain"].unique())
    generator = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_domains = generator.choice(domains, size=len(domains), replace=True)
        domain_differences = []
        for outer in sampled_domains:
            group = pivot[pivot["outer_domain"].eq(outer)]
            positions = []
            for label in [0, 1]:
                label_positions = np.flatnonzero(group["AD_label"].to_numpy() == label)
                positions.extend(generator.choice(label_positions, size=len(label_positions), replace=True))
            sample = group.iloc[positions]
            audit = _metric(sample["AD_label"].to_numpy(int), sample["audit_guided"].to_numpy(float), "auroc")
            performance = _metric(sample["AD_label"].to_numpy(int), sample["performance_only"].to_numpy(float), "auroc")
            domain_differences.append(audit - performance)
        differences[replicate] = float(np.mean(domain_differences))
    return differences


def e2_bootstrap(rules: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    pivot = rules.pivot(index=["outer_domain", "RID", "AD_label"], columns="selection_rule", values="mean_calibrated_probability_AD").reset_index()
    strata = [group.index.to_numpy() for _, group in pivot.groupby(["outer_domain", "AD_label"], sort=True)]
    generator = np.random.default_rng(seed)
    metric_names = ["auroc", "auprc", "balanced_accuracy", "brier", "ece_10"]
    values = {name: np.empty(replicates, dtype=float) for name in metric_names}
    for replicate in range(replicates):
        selected = np.concatenate([generator.choice(indices, size=len(indices), replace=True) for indices in strata])
        sample = pivot.loc[selected]
        for name in metric_names:
            audit = _metric(sample["AD_label"].to_numpy(int), sample["audit_guided"].to_numpy(float), name)
            performance = _metric(sample["AD_label"].to_numpy(int), sample["performance_only"].to_numpy(float), name)
            values[name][replicate] = audit - performance
    return pd.DataFrame(values)


def add_domain_intervals(domain_metrics: pd.DataFrame, rules: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    output = domain_metrics.copy()
    generator = np.random.default_rng(seed)
    for outer, group in rules.groupby("outer_domain", sort=True):
        pivot = group.pivot(index=["RID", "AD_label"], columns="selection_rule", values="mean_calibrated_probability_AD").reset_index()
        label_positions = {label: np.flatnonzero(pivot["AD_label"].to_numpy() == label) for label in [0, 1]}
        values = {(rule, metric): np.empty(replicates, dtype=float) for rule in ["performance_only", "audit_guided"] for metric in ["auroc", "auprc"]}
        differences = {metric: np.empty(replicates, dtype=float) for metric in ["auroc", "auprc"]}
        for replicate in range(replicates):
            positions = np.concatenate([generator.choice(indices, size=len(indices), replace=True) for indices in label_positions.values()])
            sample = pivot.iloc[positions]
            for metric in ["auroc", "auprc"]:
                for rule in ["performance_only", "audit_guided"]:
                    values[(rule, metric)][replicate] = _metric(sample["AD_label"].to_numpy(int), sample[rule].to_numpy(float), metric)
                differences[metric][replicate] = values[("audit_guided", metric)][replicate] - values[("performance_only", metric)][replicate]
        for rule in ["performance_only", "audit_guided"]:
            mask = output["outer_domain"].eq(outer) & output["selection_rule"].eq(rule)
            for metric in ["auroc", "auprc"]:
                observed_audit = float(output.loc[output["outer_domain"].eq(outer) & output["selection_rule"].eq("audit_guided"), metric].iloc[0])
                observed_performance = float(output.loc[output["outer_domain"].eq(outer) & output["selection_rule"].eq("performance_only"), metric].iloc[0])
                output.loc[mask, f"paired_{metric}_difference"] = observed_audit - observed_performance
                output.loc[mask, f"{metric}_ci95_low"] = float(np.quantile(values[(rule, metric)], 0.025))
                output.loc[mask, f"{metric}_ci95_high"] = float(np.quantile(values[(rule, metric)], 0.975))
                output.loc[mask, f"paired_{metric}_difference_ci95_low"] = float(np.quantile(differences[metric], 0.025))
                output.loc[mask, f"paired_{metric}_difference_ci95_high"] = float(np.quantile(differences[metric], 0.975))
    return output


def association_row(endpoint: str, component: str, frame: pd.DataFrame, x: str, y: str, covariates: list[str] | None = None) -> dict[str, Any]:
    columns = [x, y] + (covariates or [])
    complete = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    row: dict[str, Any] = {"endpoint": endpoint, "component": component, "n": len(complete), "status": "not_estimable", "spearman_r": np.nan, "standardized_coefficient": np.nan, "hc3_standard_error": np.nan}
    if len(complete) < 10 or complete[x].nunique() < 2 or complete[y].nunique() < 2:
        return row
    row["spearman_r"] = float(spearmanr(complete[x], complete[y]).statistic)
    design_columns = [x] + (covariates or [])
    matrix = complete[design_columns].to_numpy(float)
    scales = matrix.std(axis=0, ddof=1)
    if (scales <= 0).any():
        return row
    matrix = (matrix - matrix.mean(axis=0)) / scales
    outcome = complete[y].to_numpy(float)
    outcome_scale = outcome.std(ddof=1)
    if outcome_scale <= 0:
        return row
    outcome = (outcome - outcome.mean()) / outcome_scale
    design = np.column_stack([np.ones(len(matrix)), matrix])
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ coefficients
    hat = np.sum(design * (design @ np.linalg.pinv(design.T @ design)), axis=1)
    adjusted = residuals / np.clip(1.0 - hat, 1e-8, None)
    meat = design.T @ ((adjusted[:, None] ** 2) * design)
    bread = np.linalg.pinv(design.T @ design)
    covariance = bread @ meat @ bread
    row.update({"status": "estimable", "standardized_coefficient": float(coefficients[1]), "hc3_standard_error": float(math.sqrt(max(covariance[1, 1], 0.0)))})
    return row


def e3_profiles(inner: pd.DataFrame, readability: pd.DataFrame, fw4: pd.DataFrame) -> pd.DataFrame:
    matched = inner[inner["model_kind"].eq("neural")].copy()
    best = matched.groupby("outer_domain")["macro_mean_calibrated_auroc"].transform("max")
    matched = matched[matched["macro_mean_calibrated_auroc"].ge(best - 0.02)]
    keys = matched[["outer_domain", "family", "configuration_id", "macro_mean_calibrated_auroc"]]
    readable = readability[readability["representation_layer"].eq("penultimate") & readability["probe_family"].eq("linear")].merge(keys, on=["outer_domain", "family", "configuration_id"], how="inner", validate="many_to_one")
    encoding = readable.rename(columns={"concept_id": "component_id", "profile_mean_primary_selectivity": "component_estimate", "profile_variance_primary_selectivity": "component_inner_domain_variance", "estimable_unit_count": "valid_inner_domain_count"})
    encoding["profile_source"] = "FW3_encoding"
    encoding["component_type"] = "linear_penultimate_selectivity"
    usable_columns = [
        "outer_domain",
        "family",
        "configuration_id",
        "condition_id",
        "mean_excess_auroc_drop_interpretable_domains",
        "between_inner_domain_excess_variance_valid_domains",
        "interpretable_inner_domain_count",
    ]
    usable = fw4[usable_columns].merge(keys, on=["outer_domain", "family", "configuration_id"], how="inner", validate="many_to_one").rename(columns={"condition_id": "component_id", "mean_excess_auroc_drop_interpretable_domains": "component_estimate", "between_inner_domain_excess_variance_valid_domains": "component_inner_domain_variance", "interpretable_inner_domain_count": "valid_inner_domain_count"})
    usable["profile_source"] = "FW4_validated_use"
    usable["component_type"] = "random_corrected_interpretable_use"
    columns = ["profile_source", "component_type", "outer_domain", "family", "configuration_id", "component_id", "component_estimate", "component_inner_domain_variance", "valid_inner_domain_count", "macro_mean_calibrated_auroc"]
    return pd.concat([encoding[columns], usable[columns]], ignore_index=True)


def e4_associations(readability: pd.DataFrame, domain: pd.DataFrame) -> pd.DataFrame:
    encoding = readability[(readability["representation_layer"].eq("penultimate")) & (readability["probe_family"].eq("linear"))]
    encoding = encoding[["outer_domain", "family", "configuration_id", "concept_id", "profile_mean_primary_selectivity"]]
    condition_to_concept = {"M_C001": "C001", "M_C002": "C002", "M_C003": "C003", "M_C004": "C004", "M_C005": "C005", "J_C006": "C006", "C_C007": "C007", "M_D001": "D001", "M_D002": "D002", "M_D003": "D003", "M_D004": "D004", "M_S001": "S001", "M_S002": "S002", "M_S003": "S003", "M_N003": "N003"}
    use = domain[domain["domain_interpretable"].astype(str).str.lower().eq("true")].copy()
    use["concept_id"] = use["condition_id"].map(condition_to_concept)
    use = use.dropna(subset=["concept_id"]).groupby(["outer_domain", "family", "configuration_id", "concept_id"], as_index=False)["mean_excess_auroc_drop"].mean()
    merged = encoding.merge(use, on=["outer_domain", "family", "configuration_id", "concept_id"], how="inner")
    return pd.DataFrame([association_row("E4", "encoding_selectivity_vs_validated_use", merged, "profile_mean_primary_selectivity", "mean_excess_auroc_drop")])


def e5_e6_associations(metrics: pd.DataFrame, inner: pd.DataFrame, fw4: pd.DataFrame, readability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inner_columns = inner[["outer_domain", "family", "configuration_id", "macro_mean_absolute_calibration_slope_distance", "parameter_count"]].copy()
    inner_columns["log_parameter_count"] = np.log(inner_columns["parameter_count"].astype(float))
    merged = metrics.merge(inner_columns, on=["outer_domain", "family", "configuration_id"], how="left", validate="one_to_one")
    joint = fw4[fw4["condition_id"].eq("J_C006")][["outer_domain", "family", "configuration_id", "between_inner_domain_excess_variance_valid_domains"]]
    e5_frame = merged.merge(joint, on=["outer_domain", "family", "configuration_id"], how="left", validate="one_to_one")
    e5 = pd.DataFrame([association_row("E5", "joint_use_variance_vs_outer_auroc_loss", e5_frame, "between_inner_domain_excess_variance_valid_domains", "outer_auroc_loss", ["inner_macro_auroc", "macro_mean_absolute_calibration_slope_distance", "log_parameter_count"])])
    acquisition_rows = []
    for concept, condition in [("S001", "M_S001"), ("S002", "M_S002"), ("S003", "M_S003")]:
        encoding = readability[(readability["representation_layer"].eq("penultimate")) & (readability["probe_family"].eq("linear")) & (readability["concept_id"].eq(concept))][["outer_domain", "family", "configuration_id", "profile_mean_primary_selectivity"]]
        frame = merged.merge(encoding, on=["outer_domain", "family", "configuration_id"], how="left", validate="one_to_one")
        acquisition_rows.append(association_row("E6", f"{concept}_encoding_vs_outer_auroc_loss", frame, "profile_mean_primary_selectivity", "outer_auroc_loss"))
        use = fw4[fw4["condition_id"].eq(condition)][["outer_domain", "family", "configuration_id", "mean_excess_auroc_drop_interpretable_domains"]]
        frame = merged.merge(use, on=["outer_domain", "family", "configuration_id"], how="left", validate="one_to_one")
        acquisition_rows.append(association_row("E6", f"{concept}_validated_use_vs_outer_auroc_loss", frame, "mean_excess_auroc_drop_interpretable_domains", "outer_auroc_loss"))
    return e5, pd.DataFrame(acquisition_rows)


def taxonomy(metrics: pd.DataFrame, selection: pd.DataFrame, cohort: pd.DataFrame, plan: dict) -> pd.DataFrame:
    frozen = selection[["outer_domain", "family", "configuration_id", "invalid_primary_condition_count", "validated_acquisition_use_violation_count"]]
    rows = metrics.merge(frozen, on=["outer_domain", "family", "configuration_id"], how="left", validate="one_to_one")
    cohort = cohort.copy()
    if "acquisition_domain" not in cohort:
        membership = pd.read_csv(MEMBERSHIP_PATH, dtype={"RID": str})
        assignments = membership[["RID", "acquisition_domain"]].drop_duplicates("RID")
        cohort["RID"] = cohort["RID"].astype(str)
        cohort = cohort.merge(assignments, on="RID", how="left", validate="one_to_one")
    prevalence = cohort.groupby("acquisition_domain")["AD_label"].mean().to_dict()
    non_outer_prevalence = {
        outer: float(cohort.loc[~cohort["acquisition_domain"].eq(outer), "AD_label"].mean())
        for outer in rows["outer_domain"].unique()
    }
    threshold = float(plan["taxonomy"]["material_auroc_loss"])
    labels = []
    for _, row in rows.iterrows():
        categories = []
        if row["invalid_primary_condition_count"] > 0:
            categories.append("audit_invalid")
        if row["validated_acquisition_use_violation_count"] > 0:
            categories.append("acquisition_reliance")
        if row["outer_auroc_loss"] <= threshold and row["brier"] > row["inner_macro_brier"]:
            categories.append("calibration_only_drift")
        if abs(float(prevalence[row["outer_domain"]]) - non_outer_prevalence[row["outer_domain"]]) >= 0.05 and row["outer_auroc_loss"] > threshold:
            categories.append("prevalence_composition_sensitivity")
        if row["outer_auroc_loss"] > threshold and not categories:
            categories.append("unexplained_residual_failure")
        if row["outer_auroc_loss"] <= threshold and "calibration_only_drift" not in categories:
            categories.append("no_material_failure")
        labels.append("|".join(categories))
    rows["failure_categories"] = labels
    rows["disease_evidence_loss_assigned"] = False
    return rows


def main() -> None:
    if OUTPUT_DIRECTORY.exists():
        if not verified_existing_directory(OUTPUT_DIRECTORY):
            raise RuntimeError("Existing FW5 aggregation is invalid")
        print(f"PASS: verified existing immutable FW5 aggregation at {OUTPUT_DIRECTORY}")
        return
    definition = verify_frozen_sources()
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    manifest = load_csv_rows(MANIFEST_PATH)
    if len(manifest) != 290:
        raise RuntimeError("FW5 aggregation requires all 290 jobs")
    seed_predictions = load_seed_predictions(manifest)
    configuration = aggregate_seed_predictions(seed_predictions)
    inner = pd.read_csv(FW2_SUMMARIES_PATH, low_memory=False)
    config_metrics = configuration_metrics(configuration, inner)
    rules = rule_predictions(configuration)
    replicates = int(plan["uncertainty"]["bootstrap_replicates"])
    domain_metrics = add_domain_intervals(rule_domain_metrics(rules), rules, replicates=replicates, seed=int(plan["uncertainty"]["random_seed"]) + 2)
    performance = domain_metrics[domain_metrics["selection_rule"].eq("performance_only")].set_index("outer_domain")
    audit = domain_metrics[domain_metrics["selection_rule"].eq("audit_guided")].set_index("outer_domain")
    observed = float((audit["auroc"] - performance["auroc"]).mean())
    bootstrap = e1_bootstrap(rules, replicates=replicates, seed=int(plan["uncertainty"]["random_seed"]))
    domain_sizes = audit["n"].astype(float)
    e1 = {
        "endpoint": "E1",
        "estimand": "audit-guided minus performance-only equal-domain macro AUROC",
        "observed_difference": observed,
        "performance_only_macro_auroc": float(performance["auroc"].mean()),
        "audit_guided_macro_auroc": float(audit["auroc"].mean()),
        "domain_size_weighted_difference": float(np.average(audit["auroc"] - performance["auroc"], weights=domain_sizes)),
        "ci95_percentile": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "bootstrap_replicates": replicates,
        "domains_favoring_audit": int((audit["auroc"] > performance["auroc"]).sum()),
        "domains_equal": int((audit["auroc"] == performance["auroc"]).sum()),
        "domains_favoring_performance": int((audit["auroc"] < performance["auroc"]).sum()),
    }
    pooled_rows = []
    for rule, group in rules.groupby("selection_rule"):
        pooled_rows.append({"selection_rule": rule, **metric_row(group["AD_label"], group["mean_calibrated_probability_AD"])})
    e2 = pd.DataFrame(pooled_rows)
    e2_boot = e2_bootstrap(rules, replicates=replicates, seed=int(plan["uncertainty"]["random_seed"]) + 1)
    for name in e2_boot:
        e2[f"{name}_difference_ci95_low"] = float(e2_boot[name].quantile(0.025))
        e2[f"{name}_difference_ci95_high"] = float(e2_boot[name].quantile(0.975))
    readability = pd.read_csv(FW3_READABILITY_PATH, low_memory=False)
    fw4 = pd.read_csv(FW4_CONFIG_PATH, low_memory=False)
    fw4_domain = pd.read_csv(FW4_DOMAIN_PATH, low_memory=False)
    selection = pd.read_csv(FW4_SELECTION_PATH, low_memory=False)
    e3 = e3_profiles(inner, readability, fw4)
    e4 = e4_associations(readability, fw4_domain)
    e5, e6 = e5_e6_associations(config_metrics, inner, fw4, readability)
    cohort = pd.read_csv(COHORT_PATH, low_memory=False)
    e7 = taxonomy(config_metrics, selection, cohort, plan)
    sensitivities = []
    metadata = cohort[["RID", "ScannerManufacturer", "FieldSgth", "PHASE"]].copy().rename(
        columns={
            "ScannerManufacturer": "manufacturer_canonical",
            "FieldSgth": "field_strength_canonical",
            "PHASE": "phase_canonical",
        }
    )
    metadata["manufacturer_canonical"] = metadata["manufacturer_canonical"].map(canonicalize_manufacturer)
    metadata["field_strength_canonical"] = metadata["field_strength_canonical"].map(canonicalize_field_strength)
    metadata["phase_canonical"] = metadata["phase_canonical"].map(canonicalize_phase)
    metadata["RID"] = metadata["RID"].astype(str)
    for rule, group in rules.merge(metadata, on="RID", how="left", validate="many_to_one").groupby("selection_rule"):
        for variable in ["manufacturer_canonical", "field_strength_canonical", "phase_canonical"]:
            for level, subgroup in group.groupby(variable):
                if len(subgroup) >= 20 and subgroup["AD_label"].nunique() == 2 and subgroup["AD_label"].value_counts().min() >= 5:
                    sensitivities.append({"selection_rule": rule, "sensitivity_variable": variable, "level": level, **metric_row(subgroup["AD_label"], subgroup["mean_calibrated_probability_AD"])})
    sensitivity = pd.DataFrame(sensitivities)
    stable_domains = set(json.loads(DOMAIN_DEFINITION_PATH.read_text(encoding="utf-8"))["eligible_stable_domains"])
    stable_domain_metrics = domain_metrics[domain_metrics["outer_domain"].isin(stable_domains)].copy()
    if stable_domain_metrics["outer_domain"].nunique() != 5:
        raise RuntimeError("FW5 P3 sensitivity requires five frozen stable outer domains")
    stable_summary = stable_domain_metrics.groupby("selection_rule", as_index=False).agg(
        stable_domain_count=("outer_domain", "nunique"),
        macro_auroc=("auroc", "mean"),
        macro_auprc=("auprc", "mean"),
        macro_brier=("brier", "mean"),
        macro_ece_10=("ece_10", "mean"),
    )

    AGGREGATION_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = AGGREGATION_ROOT / f".{OUTPUT_DIRECTORY.name}.tmp-{os.getpid()}-{time.time_ns()}"
    temporary.mkdir()
    try:
        tables = {
            "configuration_outer_predictions.csv": configuration,
            "configuration_outer_metrics.csv": config_metrics,
            "rule_outer_predictions.csv": rules,
            "rule_domain_metrics.csv": domain_metrics,
            "e2_pooled_metrics.csv": e2,
            "e3_performance_matched_profiles.csv": e3,
            "e4_encoding_use_associations.csv": e4,
            "e5_profile_stability_associations.csv": e5,
            "e6_acquisition_failure_associations.csv": e6,
            "e7_failure_taxonomy.csv": e7,
            "acquisition_sensitivity_metrics.csv": sensitivity,
            "stable_domain_rule_summary.csv": stable_summary,
        }
        for name, frame in tables.items():
            frame.to_csv(temporary / name, index=False, lineterminator="\n")
        (temporary / "e1_primary_endpoint.json").write_text(json.dumps(e1, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact_names = sorted(path.name for path in temporary.iterdir())
        hashes = {name: sha256_file(temporary / name) for name in artifact_names}
        completion = {
            "freeze_id": "CRUA-AD-FW5-OUTER-ENDPOINTS-v2.1",
            "status": "PASS",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": {name.removesuffix(".csv"): len(frame) for name, frame in tables.items()},
            "checks": {
                "all_290_seed_jobs_hash_valid": True,
                "five_seeds_averaged_per_configuration_subject": True,
                "paired_rule_subject_coverage": True,
                "frozen_selection_recomputed_from_outer": False,
                "outer_recalibration": False,
                "all_poor_performing_valid_runs_retained": True,
                "E1_to_E7_materialized": True,
            },
            "source_sha256": {
                "definition": sha256_file(DEFINITION_PATH), "manifest": sha256_file(MANIFEST_PATH),
                "outer_evaluation_plan": sha256_file(PLAN_PATH), "aggregation_plan": sha256_file(AGGREGATION_PLAN_PATH),
            },
            "artifact_sha256": hashes,
            "selection_recomputed_from_outer": False,
            "privacy": "Private outer endpoint values and selected identifiers.",
        }
        (temporary / "completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.rename(OUTPUT_DIRECTORY)
    except Exception:
        if temporary.exists() and temporary.parent == AGGREGATION_ROOT:
            shutil.rmtree(temporary)
        raise
    if not verified_existing_directory(OUTPUT_DIRECTORY):
        raise RuntimeError("FW5 aggregation failed immutable verification")
    completion = json.loads((OUTPUT_DIRECTORY / "completion.json").read_text(encoding="utf-8"))
    tracked = {
        "check_id": "FW5_OUTER_EVALUATION_FREEZE",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "status": "PASS",
        "counts": completion["counts"],
        "checks": completion["checks"],
        "private_artifact_sha256": completion["artifact_sha256"],
        "source_sha256": completion["source_sha256"],
        "endpoint_reporting": "Omitted; all outer metrics and selected identifiers remain private.",
        "privacy": "Tracked log contains counts, checks, and hashes only.",
    }
    TRACKED_LOG.write_text(json.dumps(tracked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: {OUTPUT_DIRECTORY}")
    print(f"PASS: {TRACKED_LOG}")


if __name__ == "__main__":
    main()
