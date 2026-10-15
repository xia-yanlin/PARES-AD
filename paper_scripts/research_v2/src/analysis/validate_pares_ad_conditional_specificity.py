#!/usr/bin/env python3
"""Run the user-resumed frozen PARES-AD specificity and blind-spot tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from research_v2.src.analysis.audit_pares_ftp_feasibility import (
    DEFAULT_RSCRIPT,
    export_blind_metadata,
    link_ftp,
)
from research_v2.src.analysis.build_ad_mri_explanation_system import (
    RANDOM_SEED,
    fit_monotone_pathology_model,
)
from research_v2.src.analysis.confirm_pares_ad_independent import (
    BASELINE_CONCEPTS,
    HISS_ROOT,
    MANIFEST,
    MAPPING,
    PROTOCOL as CONFIRMATION_PROTOCOL,
    REGISTRY,
    add_confidence_comparator,
    build_confirmation_cohort,
    build_pathology_reference,
    run_black_boxes,
    sha256_file,
)
from research_v2.src.audit.concepts import _complete_mean, _complete_sum, _log_icv_residual, _zscore_from_training


V2_ROOT = HISS_ROOT / "research_v2"
ASSIGNMENTS = V2_ROOT / "outputs" / "private" / "domain_assignments.csv"
DEMOGRAPHICS = HISS_ROOT / "data" / "raw" / "tabular" / "PTDEMOG_05Jul2026.csv"
PROTOCOL = V2_ROOT / "protocols" / "pares_ad_biological_specificity_freeze_2026-07-23.md"
DEFAULT_OUTPUT = V2_ROOT / "outputs" / "private" / "pares_ad_biological_specificity_2026-07-23"
BOOTSTRAP_REPLICATES = 2_000
PILOT_REPLICATES = 100
MAX_RUNTIME_SECONDS = 30 * 60


def canonical_rid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def build_control_features(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    caudate = _log_icv_residual(
        train,
        evaluation,
        lambda frame: _complete_sum(frame, ("Left_Caudate", "Right_Caudate")),
    )
    pallidum = _log_icv_residual(
        train,
        evaluation,
        lambda frame: _complete_sum(frame, ("Left_Pallidum", "Right_Pallidum")),
    )
    pericalcarine = _zscore_from_training(
        _complete_mean(train, ("lh_pericalcarine_thickness", "rh_pericalcarine_thickness")),
        _complete_mean(evaluation, ("lh_pericalcarine_thickness", "rh_pericalcarine_thickness")),
    )
    precentral = _zscore_from_training(
        _complete_mean(train, ("lh_precentral_thickness", "rh_precentral_thickness")),
        _complete_mean(evaluation, ("lh_precentral_thickness", "rh_precentral_thickness")),
    )
    components = (caudate, pallidum, pericalcarine, precentral)
    if not all(component[2].all() and component[3].all() for component in components):
        raise RuntimeError("Frozen control reference contains invalid rows")
    train_features = np.column_stack([-component[0] for component in components])
    evaluation_features = np.column_stack([-component[1] for component in components])
    return train_features, evaluation_features


def build_control_reference(confirmation: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_CONCEPTS, low_memory=False, dtype={"RID": str})
    assignments = pd.read_csv(ASSIGNMENTS, dtype={"RID": str})
    baseline = baseline.merge(
        assignments[["RID", "acquisition_domain"]], on="RID", validate="one_to_one"
    )
    frames: list[pd.DataFrame] = []
    for domain, evaluation in confirmation.groupby("outer_domain", sort=True):
        train = baseline[~baseline["acquisition_domain"].eq(domain)].reset_index(drop=True)
        evaluation = evaluation.reset_index(drop=True)
        x_train, x_evaluation = build_control_features(train, evaluation)
        intercept, coefficients = fit_monotone_pathology_model(
            x_train, train["AD_label"].to_numpy(dtype=int), l2_penalty=0.1
        )
        if (coefficients < 0).any():
            raise RuntimeError("Control reference monotonicity failed")
        control_logit = intercept + x_evaluation @ coefficients
        frames.append(
            pd.DataFrame(
                {
                    "_RID": evaluation["_RID"].to_numpy(),
                    "outer_domain": domain,
                    "control_logit": control_logit,
                    "control_probability_AD": expit(control_logit),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def earliest_demographics(confirmation: pd.DataFrame) -> pd.DataFrame:
    demographics = pd.read_csv(
        DEMOGRAPHICS,
        usecols=["RID", "VISDATE", "USERDATE", "PTGENDER", "PTEDUCAT"],
        low_memory=False,
    )
    demographics["_RID"] = canonical_rid(demographics["RID"])
    demographics["_date"] = pd.to_datetime(demographics["VISDATE"], errors="coerce")
    missing_date = demographics["_date"].isna()
    demographics.loc[missing_date, "_date"] = pd.to_datetime(
        demographics.loc[missing_date, "USERDATE"], errors="coerce"
    )
    demographics["sex"] = demographics["PTGENDER"].map(
        lambda value: "M"
        if str(value).strip().upper() in {"1", "1.0", "M", "MALE"}
        else "F"
        if str(value).strip().upper() in {"2", "2.0", "F", "FEMALE"}
        else np.nan
    )
    demographics["education"] = pd.to_numeric(demographics["PTEDUCAT"], errors="coerce")
    demographics.loc[demographics["education"].lt(0), "education"] = np.nan
    demographics = demographics[demographics["_RID"].isin(set(confirmation["_RID"]))]

    rows: list[dict[str, Any]] = []
    for rid, group in demographics.groupby("_RID", sort=False):
        ordered = group.sort_values(["_date"], kind="stable", na_position="last")
        valid_sex = ordered.loc[ordered["sex"].notna(), "sex"]
        valid_education = ordered.loc[ordered["education"].notna(), "education"]
        rows.append(
            {
                "_RID": rid,
                "sex": valid_sex.iloc[0] if not valid_sex.empty else np.nan,
                "education": float(valid_education.iloc[0]) if not valid_education.empty else np.nan,
            }
        )
    output = pd.DataFrame(rows)
    if output["_RID"].nunique() != confirmation["_RID"].nunique():
        raise RuntimeError("Demographic linkage is incomplete")
    output["sex_missing"] = output["sex"].isna().astype(int)
    output["sex"] = output["sex"].fillna("UNKNOWN")
    output["education_missing"] = output["education"].isna().astype(int)
    output["education"] = output["education"].fillna(output["education"].median())
    return output


def prepare_cases(confirmation: pd.DataFrame) -> pd.DataFrame:
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST, dtype={"seed": int})
    if len(manifest) != 290:
        raise RuntimeError("Frozen FW5 manifest must contain 290 jobs")
    predictions = run_black_boxes(confirmation, manifest, mapping, registry)
    pathology = build_pathology_reference(confirmation)
    control = build_control_reference(confirmation)
    cases = predictions.merge(pathology, on=["_RID", "outer_domain"], validate="many_to_one")
    cases = cases.merge(control, on=["_RID", "outer_domain"], validate="many_to_one")
    case_covariates = confirmation[
        ["_RID", "outer_domain", "PHC_Age_T1", "phase", "manufacturer", "field_strength"]
    ].copy()
    case_covariates = case_covariates.merge(
        earliest_demographics(confirmation), on="_RID", validate="one_to_one"
    )
    cases = cases.merge(case_covariates, on=["_RID", "outer_domain"], validate="many_to_one")
    cases["model_predicted_class"] = cases["model_probability_AD"].ge(0.5).astype(int)
    cases["model_confidence_distance"] = (cases["model_probability_AD"] - 0.5).abs()
    cases["pathology_predicted_class"] = cases["pathology_probability_AD"].ge(0.5).astype(int)
    cases["control_predicted_class"] = cases["control_probability_AD"].ge(0.5).astype(int)
    cases["pathology_conflict"] = cases["model_predicted_class"].ne(cases["pathology_predicted_class"])
    cases["control_conflict"] = cases["model_predicted_class"].ne(cases["control_predicted_class"])
    cases["model_error"] = cases["model_predicted_class"].ne(cases["AD_label"]).astype(int)
    cases = add_confidence_comparator(cases)
    if len(cases) != 7_091 or cases["_RID"].nunique() != 882:
        raise RuntimeError("Frozen model-case receipt changed")
    return cases


def state_risk(frame: pd.DataFrame, conflict_column: str) -> dict[str, float | int]:
    conflict = frame[conflict_column].to_numpy(dtype=bool)
    errors = frame["model_error"].to_numpy(dtype=int)
    conflict_rate = float(errors[conflict].mean()) if conflict.any() else math.nan
    aligned_rate = float(errors[~conflict].mean()) if (~conflict).any() else math.nan
    return {
        "predictions": int(len(frame)),
        "unique_participants": int(frame["_RID"].nunique()),
        "conflict_predictions": int(conflict.sum()),
        "conflict_unique_participants": int(frame.loc[conflict, "_RID"].nunique()),
        "conflict_error_rate": conflict_rate,
        "aligned_error_rate": aligned_rate,
        "error_risk_ratio": conflict_rate / aligned_rate if aligned_rate > 0 else math.inf,
    }


def cluster_state_bootstrap(
    cases: pd.DataFrame, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    participant_rows = list(cases.groupby("_RID", sort=True).indices.values())
    generator = np.random.default_rng(RANDOM_SEED)
    ad_ratios = np.empty(replicates, dtype=float)
    control_ratios = np.empty(replicates, dtype=float)
    high_ratios = np.empty(replicates, dtype=float)
    high_mask = cases["high_confidence"].to_numpy(dtype=bool)
    errors = cases["model_error"].to_numpy(dtype=int)
    ad_conflict = cases["pathology_conflict"].to_numpy(dtype=bool)
    control_conflict = cases["control_conflict"].to_numpy(dtype=bool)

    def ratio(mask: np.ndarray, conflict: np.ndarray, weights: np.ndarray) -> float:
        selected = weights * mask
        conflict_n = float(selected[conflict].sum())
        aligned_n = float(selected[~conflict].sum())
        if conflict_n <= 0 or aligned_n <= 0:
            return math.nan
        conflict_rate = float((selected[conflict] * errors[conflict]).sum() / conflict_n)
        aligned_rate = float((selected[~conflict] * errors[~conflict]).sum() / aligned_n)
        return conflict_rate / aligned_rate if aligned_rate > 0 else math.inf

    for replicate in range(replicates):
        sampled = generator.integers(0, len(participant_rows), size=len(participant_rows))
        counts = np.bincount(sampled, minlength=len(participant_rows))
        weights = np.zeros(len(cases), dtype=float)
        for participant_index, row_indices in enumerate(participant_rows):
            weights[row_indices] = counts[participant_index]
        all_rows = np.ones(len(cases), dtype=bool)
        ad_ratios[replicate] = ratio(all_rows, ad_conflict, weights)
        control_ratios[replicate] = ratio(all_rows, control_conflict, weights)
        high_ratios[replicate] = ratio(high_mask, ad_conflict, weights)
    valid_ad = ad_ratios[np.isfinite(ad_ratios)]
    valid_control = control_ratios[np.isfinite(control_ratios)]
    valid_high = high_ratios[np.isfinite(high_ratios)]
    differences = ad_ratios - control_ratios
    valid_differences = differences[np.isfinite(differences)]
    if min(len(valid_ad), len(valid_control), len(valid_high), len(valid_differences)) < 0.95 * replicates:
        raise RuntimeError("Too many invalid participant-cluster bootstrap replicates")
    return {
        "ad_rr_ci95": [float(np.quantile(valid_ad, 0.025)), float(np.quantile(valid_ad, 0.975))],
        "control_rr_ci95": [
            float(np.quantile(valid_control, 0.025)),
            float(np.quantile(valid_control, 0.975)),
        ],
        "ad_minus_control_rr_ci95": [
            float(np.quantile(valid_differences, 0.025)),
            float(np.quantile(valid_differences, 0.975)),
        ],
        "high_confidence_ad_rr_ci95": [
            float(np.quantile(valid_high, 0.025)),
            float(np.quantile(valid_high, 0.975)),
        ],
    }


def build_adjusted_design(cases: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.DataFrame(
        {
            "pathology_conflict": cases["pathology_conflict"].astype(float),
            "confidence_error_risk": cases["confidence_error_risk"].astype(float),
            "predicted_class": cases["model_predicted_class"].astype(float),
            "age": pd.to_numeric(cases["PHC_Age_T1"], errors="coerce"),
            "education": cases["education"].astype(float),
            "education_missing": cases["education_missing"].astype(float),
            "sex_missing": cases["sex_missing"].astype(float),
        }
    )
    for column in ["confidence_error_risk", "age", "education"]:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Adjusted covariate is incomplete: {column}")
        frame[column] = (values - values.mean()) / values.std(ddof=0)
    categorical = pd.DataFrame(
        {
            "sex": cases["sex"].astype(str),
            "phase": cases["phase"].astype(str),
            "manufacturer": cases["manufacturer"].astype(str),
            "field_strength": cases["field_strength"].astype(str),
            "model_configuration": cases["family"].astype(str) + "|" + cases["configuration_id"].astype(str),
            "outer_domain": cases["outer_domain"].astype(str),
        }
    )
    dummies = pd.get_dummies(categorical, drop_first=True, dtype=float)
    frame = pd.concat([frame, dummies], axis=1)
    candidates = [np.ones(len(frame), dtype=float)] + [
        frame[column].to_numpy(dtype=float) for column in frame.columns
    ]
    names = ["intercept"] + frame.columns.tolist()
    selected: list[np.ndarray] = []
    selected_names: list[str] = []
    rank = 0
    for name, candidate in zip(names, candidates):
        trial = np.column_stack([*selected, candidate]) if selected else candidate.reshape(-1, 1)
        trial_rank = int(np.linalg.matrix_rank(trial))
        if trial_rank > rank:
            selected.append(candidate)
            selected_names.append(name)
            rank = trial_rank
    if "pathology_conflict" not in selected_names:
        raise RuntimeError("Adjusted design lost the primary conflict coefficient")
    return np.column_stack(selected), cases["model_error"].to_numpy(dtype=int), selected_names


def adjusted_conflict_bootstrap(
    cases: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    *,
    solver: str = "liblinear",
    return_estimates: bool = False,
) -> dict[str, Any]:
    design, outcomes, names = build_adjusted_design(cases)
    conflict_index = names.index("pathology_conflict")
    participant_rows = list(cases.groupby("_RID", sort=True).indices.values())
    estimator = LogisticRegression(
        C=1e6,
        fit_intercept=False,
        max_iter=1_000,
        solver=solver,
        tol=1e-8,
    )
    estimator.fit(design, outcomes)
    observed = float(math.exp(estimator.coef_[0, conflict_index]))
    generator = np.random.default_rng(RANDOM_SEED + 1)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = generator.integers(0, len(participant_rows), size=len(participant_rows))
        counts = np.bincount(sampled, minlength=len(participant_rows))
        weights = np.zeros(len(cases), dtype=float)
        for participant_index, row_indices in enumerate(participant_rows):
            weights[row_indices] = counts[participant_index]
        estimator.fit(design, outcomes, sample_weight=weights)
        estimates[replicate] = math.exp(estimator.coef_[0, conflict_index])
    valid = estimates[np.isfinite(estimates)]
    if len(valid) < 0.95 * replicates:
        raise RuntimeError("Too many invalid adjusted bootstrap replicates")
    result = {
        "odds_ratio": observed,
        "ci95": [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))],
        "retained_design_columns": names,
        "solver": solver,
    }
    if return_estimates:
        result["bootstrap_estimates"] = estimates.tolist()
    return result


def ftp_receipt(confirmation: pd.DataFrame, cases: pd.DataFrame, rscript: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pares_ftp_") as temporary:
        directory = Path(temporary)
        export_blind_metadata(directory, rscript)
        tau_meta = pd.read_csv(directory / "tau_meta.csv", low_memory=False)
        linked = link_ftp(confirmation, tau_meta)
    linked_ids = set(linked["_RID"])
    eligible_cases = cases[cases["_RID"].isin(linked_ids)]
    conflict_participants = int(
        eligible_cases.loc[eligible_cases["pathology_conflict"], "_RID"].nunique()
    )
    receipt = {
        "participants": int(linked["_RID"].nunique()),
        "ad": int(linked["AD_label"].sum()),
        "cn": int(len(linked) - linked["AD_label"].sum()),
        "outer_domains": int(linked["outer_domain"].nunique()),
        "median_absolute_gap_days": float(linked["absolute_gap_days"].median()),
        "pathology_conflict_participants": conflict_participants,
    }
    gates = {
        "participants_at_least_150": receipt["participants"] >= 150,
        "ad_at_least_35": receipt["ad"] >= 35,
        "cn_at_least_75": receipt["cn"] >= 75,
        "outer_domains_at_least_4": receipt["outer_domains"] >= 4,
        "pathology_conflict_participants_at_least_40": conflict_participants >= 40,
    }
    return {
        "receipt": receipt,
        "gates": gates,
        "pass": all(gates.values()),
        "tau_values_exported_or_analyzed": False,
    }


def aggregate_rows(
    ad: dict[str, Any], control: dict[str, Any], high: dict[str, Any], intervals: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for reference, state, ci_key in [
        ("AD", ad, "ad_rr_ci95"),
        ("non_AD_control", control, "control_rr_ci95"),
        ("AD_high_confidence", high, "high_confidence_ad_rr_ci95"),
    ]:
        rows.append(
            {
                "reference": reference,
                **state,
                "error_risk_ratio_ci95_low": intervals[ci_key][0],
                "error_risk_ratio_ci95_high": intervals[ci_key][1],
            }
        )
    return pd.DataFrame(rows)


def render_report(summary: dict[str, Any]) -> str:
    ftp = summary["ftp_feasibility"]
    ad = summary["anatomical_specificity"]["ad_reference"]
    control = summary["anatomical_specificity"]["control_reference"]
    difference = summary["anatomical_specificity"]["ad_minus_control_rr"]
    adjusted = summary["conditional_error_association"]
    high = summary["confidence_blind_spot"]
    return f"""# PARES-AD frozen deepening result

## Verdict

**{summary['status']}**

The FTP molecular-specificity analysis stopped outcome-blind: {ftp['receipt']['participants']} participants met the basic FTP linkage rules, but only {ftp['receipt']['pathology_conflict_participants']} had a frozen AD-reference conflict, below the required 40. No tau value was exported or analyzed, and amyloid/PACC were not used as rescue outcomes.

## Non-molecular falsification and conditional evidence

The AD-reference conflict error-risk ratio was {ad['error_risk_ratio']:.2f} (95% participant-cluster interval {ad['ci95'][0]:.2f}–{ad['ci95'][1]:.2f}); the equal-complexity non-AD anatomical control ratio was {control['error_risk_ratio']:.2f} ({control['ci95'][0]:.2f}–{control['ci95'][1]:.2f}). Their risk-ratio difference was {difference['estimate']:.2f} ({difference['ci95'][0]:.2f}–{difference['ci95'][1]:.2f}).

After frozen confidence risk, predicted class, demographic, acquisition, configuration and outer-domain covariates, AD-reference conflict had error odds ratio {adjusted['odds_ratio']:.2f} ({adjusted['ci95'][0]:.2f}–{adjusted['ci95'][1]:.2f}).

Among fixed high-confidence predictions, conflict error risk was {high['conflict_error_rate']:.2%} versus {high['aligned_error_rate']:.2%}, risk ratio {high['error_risk_ratio']:.2f} ({high['ci95'][0]:.2f}–{high['ci95'][1]:.2f}); {high['conflict_unique_participants']} unique participants had at least one high-confidence conflict. Conflict marked {high['fraction_high_confidence_errors_with_conflict']:.2%} of high-confidence model-case errors.

## Claim boundary

{summary['claim_boundary']}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rscript", type=Path, default=DEFAULT_RSCRIPT)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if not args.rscript.is_file():
        raise RuntimeError(f"Rscript not found: {args.rscript}")
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    confirmation = build_confirmation_cohort(mapping)
    cases = prepare_cases(confirmation)
    cases["high_confidence"] = cases["model_probability_AD"].le(0.10) | cases[
        "model_probability_AD"
    ].ge(0.90)

    pilot_started = time.perf_counter()
    cluster_state_bootstrap(cases, replicates=PILOT_REPLICATES)
    adjusted_conflict_bootstrap(cases, replicates=PILOT_REPLICATES)
    pilot_seconds = time.perf_counter() - pilot_started
    projected_total = (time.perf_counter() - started) + pilot_seconds * (
        BOOTSTRAP_REPLICATES / PILOT_REPLICATES
    )
    if args.pilot_only:
        print(json.dumps({"pilot_seconds": pilot_seconds, "projected_total_seconds": projected_total}, indent=2))
        return

    state_intervals = cluster_state_bootstrap(cases)
    adjusted = adjusted_conflict_bootstrap(cases)
    ad = state_risk(cases, "pathology_conflict")
    control = state_risk(cases, "control_conflict")
    high_cases = cases[cases["high_confidence"]].copy()
    high = state_risk(high_cases, "pathology_conflict")
    high_errors = high_cases["model_error"].eq(1)
    high["fraction_high_confidence_errors_with_conflict"] = float(
        high_cases.loc[high_errors, "pathology_conflict"].mean()
    )
    ftp = ftp_receipt(confirmation, cases, args.rscript)

    ad_ci = state_intervals["ad_rr_ci95"]
    control_ci = state_intervals["control_rr_ci95"]
    high_ci = state_intervals["high_confidence_ad_rr_ci95"]
    difference_estimate = float(ad["error_risk_ratio"] - control["error_risk_ratio"])
    difference_ci = state_intervals["ad_minus_control_rr_ci95"]
    conditional_gate = adjusted["odds_ratio"] >= 2.0 and adjusted["ci95"][0] > 1.0
    high_gate = (
        high["error_risk_ratio"] >= 2.0
        and high_ci[0] > 1.0
        and high["conflict_unique_participants"] >= 30
    )
    control_separation = difference_estimate > 0 and difference_ci[0] > 0
    nonmolecular_pass = bool(conditional_gate and high_gate and control_separation)
    status = "MOLECULAR_STOP_NONMOLECULAR_SUPPORTED" if nonmolecular_pass else "STOP"
    if nonmolecular_pass:
        claim_boundary = (
            "PARES-AD retains conditional and high-confidence error-warning value beyond the frozen "
            "confidence covariates and exceeds an equal-complexity non-AD anatomical control, but "
            "AD molecular specificity was not estimable and superiority to confidence ranking remains unsupported."
        )
    else:
        claim_boundary = (
            "Only the previously replicated unadjusted anatomical-concordance warning is retained; "
            "AD molecular specificity or conditional/high-confidence specificity was not fully established."
        )
    summary: dict[str, Any] = {
        "analysis": "PARES-AD-BIOSPECIFICITY-v1",
        "status": status,
        "runtime_seconds": time.perf_counter() - started,
        "pilot_runtime": {
            "replicates": PILOT_REPLICATES,
            "seconds": pilot_seconds,
            "projected_total_seconds": projected_total,
            "cap_seconds": MAX_RUNTIME_SECONDS,
        },
        "data_receipt": {
            "participants": int(cases["_RID"].nunique()),
            "model_case_predictions": int(len(cases)),
            "configurations": int(cases.groupby(["family", "configuration_id"]).ngroups),
            "families": int(cases["family"].nunique()),
            "outer_domains": int(cases["outer_domain"].nunique()),
        },
        "ftp_feasibility": ftp,
        "anatomical_specificity": {
            "ad_reference": {**ad, "ci95": ad_ci},
            "control_reference": {**control, "ci95": control_ci},
            "ad_minus_control_rr": {"estimate": difference_estimate, "ci95": difference_ci},
            "formal_gate": control_separation,
        },
        "conditional_error_association": {
            "odds_ratio": adjusted["odds_ratio"],
            "ci95": adjusted["ci95"],
            "formal_gate": bool(conditional_gate),
        },
        "confidence_blind_spot": {**high, "ci95": high_ci, "formal_gate": bool(high_gate)},
        "gates": {
            "ftp_feasibility": bool(ftp["pass"]),
            "ad_exceeds_equal_complexity_control": bool(control_separation),
            "adjusted_conflict_or_at_least_2_ci_above_1": bool(conditional_gate),
            "high_confidence_rr_at_least_2_ci_above_1_at_least_30_participants": bool(high_gate),
            "nonmolecular_specificity_supported": nonmolecular_pass,
            "full_ad_biological_specificity_supported": False,
        },
        "outcome_access": {
            "tau_values_exported_or_analyzed": False,
            "amyloid_values_exported_or_analyzed": False,
            "pacc_values_exported_or_analyzed": False,
        },
        "claim_boundary": claim_boundary,
        "privacy": {"participant_identifiers_written": False, "participant_level_rows_written": False},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
        "confirmation_protocol": {
            "path": str(CONFIRMATION_PROTOCOL),
            "sha256": sha256_file(CONFIRMATION_PROTOCOL),
        },
        "source_sha256": {
            str(BASELINE_CONCEPTS): sha256_file(BASELINE_CONCEPTS),
            str(ASSIGNMENTS): sha256_file(ASSIGNMENTS),
            str(DEMOGRAPHICS): sha256_file(DEMOGRAPHICS),
            str(MAPPING): sha256_file(MAPPING),
            str(REGISTRY): sha256_file(REGISTRY),
            str(MANIFEST): sha256_file(MANIFEST),
        },
    }
    aggregate = aggregate_rows(ad, control, high, state_intervals)
    forbidden = {"rid", "ptid", "subject_id", "participant_id"}
    if forbidden.intersection(column.lower() for column in aggregate.columns):
        raise RuntimeError("Participant identifier leaked into aggregate output")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output / "aggregate_results.csv", index=False, lineterminator="\n")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {"status": status, "runtime_seconds": summary["runtime_seconds"], "output": str(output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
