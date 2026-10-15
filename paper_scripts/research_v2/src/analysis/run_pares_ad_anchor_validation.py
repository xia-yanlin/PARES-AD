#!/usr/bin/env python3
"""Run the user-resumed PARES-AD molecular and clinical anchor validation."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from research_v2.src.analysis.audit_pares_ftp_feasibility import (
    DEFAULT_RSCRIPT,
    export_blind_metadata,
)
from research_v2.src.analysis.build_ad_mri_explanation_system import RANDOM_SEED
from research_v2.src.analysis.confirm_pares_ad_independent import (
    MAPPING,
    build_confirmation_cohort,
)
from research_v2.src.analysis.validate_pares_ad_conditional_specificity import (
    BOOTSTRAP_REPLICATES,
    DEFAULT_OUTPUT,
    PROTOCOL,
    earliest_demographics,
    prepare_cases,
)
from research_v2.src.analysis.confirm_pares_ad_independent import sha256_file


def canonical_rid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def nearest_anchor(
    confirmation: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    window_days: int,
    extra_mask: pd.Series,
) -> pd.DataFrame:
    source = anchors.copy()
    source["_RID"] = canonical_rid(source["RID"])
    source["_anchor_date"] = pd.to_datetime(source[date_column], errors="coerce")
    source["_anchor_value"] = pd.to_numeric(source[value_column], errors="coerce")
    source = source[extra_mask & source["_anchor_date"].notna() & np.isfinite(source["_anchor_value"])].copy()
    linked = confirmation[
        ["_RID", "_scan_date", "AD_label", "outer_domain", "PHC_Age_T1", "phase", "manufacturer", "field_strength"]
    ].merge(source[["_RID", "_anchor_date", "_anchor_value"]], on="_RID", how="inner", validate="one_to_many")
    linked["absolute_gap_days"] = (linked["_anchor_date"] - linked["_scan_date"]).abs().dt.days
    linked = linked[linked["absolute_gap_days"].le(window_days)].copy()
    linked["anchor_after_mri"] = linked["_anchor_date"].gt(linked["_scan_date"]).astype(int)
    linked = linked.sort_values(
        ["_RID", "absolute_gap_days", "anchor_after_mri", "_anchor_date"], kind="stable"
    ).drop_duplicates("_RID", keep="first")
    return linked.reset_index(drop=True)


def participant_reference(cases: pd.DataFrame, confirmation: pd.DataFrame) -> pd.DataFrame:
    reference = cases[
        ["_RID", "outer_domain", "pathology_probability_AD", "control_logit", "control_probability_AD"]
    ].drop_duplicates(["_RID", "outer_domain"])
    if reference["_RID"].nunique() != 882 or len(reference) != 882:
        raise RuntimeError("Participant reference is not unique")
    reference["pathology_logit"] = np.log(
        np.clip(reference["pathology_probability_AD"], 1e-8, 1 - 1e-8)
        / np.clip(1 - reference["pathology_probability_AD"], 1e-8, 1 - 1e-8)
    )
    demographics = earliest_demographics(confirmation)
    return reference.merge(demographics, on="_RID", validate="one_to_one")


def covariate_design(frame: pd.DataFrame) -> np.ndarray:
    numeric = pd.DataFrame(
        {
            "age": pd.to_numeric(frame["PHC_Age_T1"], errors="coerce"),
            "education": pd.to_numeric(frame["education"], errors="coerce"),
            "education_missing": frame["education_missing"].astype(float),
            "sex_missing": frame["sex_missing"].astype(float),
            "anchor_gap": pd.to_numeric(frame["absolute_gap_days"], errors="coerce"),
            "diagnosis": frame["AD_label"].astype(float),
        }
    )
    for column in ("age", "education", "anchor_gap"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or values.std(ddof=0) <= 0:
            raise RuntimeError(f"Invalid anchor covariate: {column}")
        numeric[column] = (values - values.mean()) / values.std(ddof=0)
    categorical = pd.DataFrame(
        {
            "sex": frame["sex"].astype(str),
            "phase": frame["phase"].astype(str),
            "manufacturer": frame["manufacturer"].astype(str),
            "field_strength": frame["field_strength"].astype(str),
        }
    )
    design = pd.concat([numeric, pd.get_dummies(categorical, drop_first=True, dtype=float)], axis=1)
    return np.column_stack([np.ones(len(frame)), design.to_numpy(dtype=float)])


def crossfit_continuous(
    frame: pd.DataFrame, outcome: np.ndarray, score_columns: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    values = np.asarray(outcome, dtype=float)
    values = (values - values.mean()) / values.std(ddof=0)
    design = covariate_design(frame)
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    residual = np.empty(len(frame), dtype=float)
    score_predictions = {column: np.empty(len(frame), dtype=float) for column in score_columns}
    standardized_scores = {
        column: (frame[column].to_numpy(dtype=float) - frame[column].mean()) / frame[column].std(ddof=0)
        for column in score_columns
    }
    for train_index, test_index in folds.split(frame):
        beta, *_ = np.linalg.lstsq(design[train_index], values[train_index], rcond=None)
        residual[test_index] = values[test_index] - design[test_index] @ beta
        train_residual = values[train_index] - design[train_index] @ beta
        for column, score in standardized_scores.items():
            denominator = float(score[train_index] @ score[train_index])
            coefficient = float(score[train_index] @ train_residual / denominator)
            score_predictions[column][test_index] = coefficient * score[test_index]
    baseline_sse = float(residual @ residual)
    delta_r2 = {
        column: 1.0 - float(np.sum((residual - prediction) ** 2)) / baseline_sse
        for column, prediction in score_predictions.items()
    }
    standardized_residual = (residual - residual.mean()) / residual.std(ddof=0)
    return standardized_residual, score_predictions, delta_r2


def continuous_bootstrap(
    residual: np.ndarray,
    predictions: dict[str, np.ndarray],
    cases: pd.DataFrame,
    participant_frame: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    generator = np.random.default_rng(RANDOM_SEED + 20)
    participant_ids = participant_frame["_RID"].to_numpy(object)
    participant_position = {rid: index for index, rid in enumerate(participant_ids)}
    case_positions = {
        rid: group.index.to_numpy(dtype=int)
        for rid, group in cases.reset_index(drop=True).groupby("_RID", sort=False)
    }
    ad_delta = np.empty(replicates)
    control_delta = np.empty(replicates)
    specificity_delta = np.empty(replicates)
    ad_signed = np.empty(replicates)
    control_signed = np.empty(replicates)
    signed_difference = np.empty(replicates)
    ad_conflict = cases["pathology_conflict"].to_numpy(dtype=bool)
    control_conflict = cases["control_conflict"].to_numpy(dtype=bool)
    ad_signed_values = cases["ad_signed_anchor"].to_numpy(dtype=float)
    control_signed_values = cases["control_signed_anchor"].to_numpy(dtype=float)
    base_sse = residual**2
    ad_sse = (residual - predictions["pathology_logit"]) ** 2
    control_sse = (residual - predictions["control_logit"]) ** 2
    for replicate in range(replicates):
        sampled = generator.integers(0, len(participant_ids), size=len(participant_ids))
        counts = np.bincount(sampled, minlength=len(participant_ids)).astype(float)
        denominator = float(counts @ base_sse)
        ad_delta[replicate] = 1.0 - float(counts @ ad_sse) / denominator
        control_delta[replicate] = 1.0 - float(counts @ control_sse) / denominator
        specificity_delta[replicate] = ad_delta[replicate] - control_delta[replicate]
        case_weights = np.zeros(len(cases), dtype=float)
        for rid, positions in case_positions.items():
            case_weights[positions] = counts[participant_position[rid]]
        ad_denominator = float(case_weights[ad_conflict].sum())
        control_denominator = float(case_weights[control_conflict].sum())
        ad_signed[replicate] = float(case_weights[ad_conflict] @ ad_signed_values[ad_conflict] / ad_denominator)
        control_signed[replicate] = float(
            case_weights[control_conflict] @ control_signed_values[control_conflict] / control_denominator
        )
        signed_difference[replicate] = ad_signed[replicate] - control_signed[replicate]

    def interval(values: np.ndarray) -> list[float]:
        finite = values[np.isfinite(values)]
        if len(finite) < 0.95 * len(values):
            raise RuntimeError("Too many invalid anchor bootstrap replicates")
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    return {
        "delta_r2_ad_ci95": interval(ad_delta),
        "delta_r2_control_ci95": interval(control_delta),
        "delta_r2_specificity_ci95": interval(specificity_delta),
        "ad_signed_adjudication_ci95": interval(ad_signed),
        "control_signed_adjudication_ci95": interval(control_signed),
        "signed_adjudication_difference_ci95": interval(signed_difference),
    }


def continuous_anchor_analysis(
    linked: pd.DataFrame,
    references: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    outcome_transform,
) -> dict[str, Any]:
    participants = references.merge(
        linked[["_RID", "outer_domain", "_anchor_value", "absolute_gap_days"]],
        on=["_RID", "outer_domain"],
        validate="one_to_one",
    )
    outcome = outcome_transform(participants["_anchor_value"].to_numpy(dtype=float))
    residual, predictions, delta_r2 = crossfit_continuous(
        participants, outcome, ("pathology_logit", "control_logit")
    )
    participants["anchor_residual"] = residual
    linked_cases = cases[cases["_RID"].isin(set(participants["_RID"]))].merge(
        participants[["_RID", "anchor_residual"]], on="_RID", validate="many_to_one"
    ).reset_index(drop=True)
    ad_direction = np.where(
        linked_cases["pathology_predicted_class"].eq(1)
        & linked_cases["model_predicted_class"].eq(0),
        1.0,
        -1.0,
    )
    control_direction = np.where(
        linked_cases["control_predicted_class"].eq(1)
        & linked_cases["model_predicted_class"].eq(0),
        1.0,
        -1.0,
    )
    linked_cases["ad_signed_anchor"] = ad_direction * linked_cases["anchor_residual"]
    linked_cases["control_signed_anchor"] = control_direction * linked_cases["anchor_residual"]
    ad_conflict = linked_cases["pathology_conflict"]
    control_conflict = linked_cases["control_conflict"]
    observed_ad_signed = float(linked_cases.loc[ad_conflict, "ad_signed_anchor"].mean())
    observed_control_signed = float(linked_cases.loc[control_conflict, "control_signed_anchor"].mean())
    intervals = continuous_bootstrap(residual, predictions, linked_cases, participants)
    return {
        "receipt": {
            "participants": int(len(participants)),
            "ad": int(participants["AD_label"].sum()),
            "cn": int(len(participants) - participants["AD_label"].sum()),
            "outer_domains": int(participants["outer_domain"].nunique()),
            "median_absolute_gap_days": float(participants["absolute_gap_days"].median()),
            "ad_conflict_participants": int(
                linked_cases.loc[ad_conflict, "_RID"].nunique()
            ),
        },
        "incremental_r2": {
            "ad_reference": delta_r2["pathology_logit"],
            "control_reference": delta_r2["control_logit"],
            "specificity": delta_r2["pathology_logit"] - delta_r2["control_logit"],
            "ad_reference_ci95": intervals["delta_r2_ad_ci95"],
            "control_reference_ci95": intervals["delta_r2_control_ci95"],
            "specificity_ci95": intervals["delta_r2_specificity_ci95"],
        },
        "signed_adjudication": {
            "ad_reference": observed_ad_signed,
            "control_reference": observed_control_signed,
            "specificity": observed_ad_signed - observed_control_signed,
            "ad_reference_ci95": intervals["ad_signed_adjudication_ci95"],
            "control_reference_ci95": intervals["control_signed_adjudication_ci95"],
            "specificity_ci95": intervals["signed_adjudication_difference_ci95"],
        },
    }


def amyloid_analysis(
    linked: pd.DataFrame,
    references: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    participants = references.merge(
        linked[["_RID", "outer_domain", "_anchor_value", "absolute_gap_days"]],
        on=["_RID", "outer_domain"],
        validate="one_to_one",
    )
    design = covariate_design(participants)
    outcome = participants["_anchor_value"].to_numpy(dtype=int)
    standardized_scores = {}
    full_designs = {}
    results: dict[str, Any] = {}
    for column in ("pathology_logit", "control_logit"):
        score = participants[column].to_numpy(dtype=float)
        standardized_scores[column] = (score - score.mean()) / score.std(ddof=0)
        full_designs[column] = np.column_stack([design, standardized_scores[column]])
        estimator = LogisticRegression(C=1e6, fit_intercept=False, solver="liblinear", max_iter=2_000)
        estimator.fit(full_designs[column], outcome)
        results[column] = {"odds_ratio_per_sd": float(math.exp(estimator.coef_[0, -1]))}
    generator = np.random.default_rng(RANDOM_SEED + 30)
    estimates = {column: np.empty(replicates) for column in standardized_scores}
    log_specificity = np.empty(replicates)
    for replicate in range(replicates):
        sampled = generator.integers(0, len(participants), size=len(participants))
        weights = np.bincount(sampled, minlength=len(participants)).astype(float)
        log_odds = {}
        for column in standardized_scores:
            estimator = LogisticRegression(
                C=1e6, fit_intercept=False, solver="liblinear", max_iter=2_000
            )
            estimator.fit(full_designs[column], outcome, sample_weight=weights)
            log_odds[column] = float(estimator.coef_[0, -1])
            estimates[column][replicate] = math.exp(log_odds[column])
        log_specificity[replicate] = log_odds["pathology_logit"] - log_odds["control_logit"]
    for column in standardized_scores:
        results[column]["odds_ratio_per_sd_ci95"] = [
            float(np.quantile(estimates[column], 0.025)),
            float(np.quantile(estimates[column], 0.975)),
        ]
    results["pathology_vs_control_log_odds_difference"] = {
        "estimate": float(
            math.log(results["pathology_logit"]["odds_ratio_per_sd"])
            - math.log(results["control_logit"]["odds_ratio_per_sd"])
        ),
        "ci95": [
            float(np.quantile(log_specificity, 0.025)),
            float(np.quantile(log_specificity, 0.975)),
        ],
    }
    results["receipt"] = {
        "participants": int(len(participants)),
        "amyloid_positive": int(outcome.sum()),
        "amyloid_negative": int(len(outcome) - outcome.sum()),
        "outer_domains": int(participants["outer_domain"].nunique()),
        "median_absolute_gap_days": float(participants["absolute_gap_days"].median()),
    }
    return results


def aggregate_rows(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tau = summary["tau_primary"]
    for reference in ("ad_reference", "control_reference", "specificity"):
        rows.append(
            {
                "analysis": "tau_incremental_r2",
                "reference": reference,
                "estimate": tau["incremental_r2"][reference],
                "ci95_low": tau["incremental_r2"][f"{reference}_ci95"][0],
                "ci95_high": tau["incremental_r2"][f"{reference}_ci95"][1],
            }
        )
        rows.append(
            {
                "analysis": "tau_signed_adjudication",
                "reference": reference,
                "estimate": tau["signed_adjudication"][reference],
                "ci95_low": tau["signed_adjudication"][f"{reference}_ci95"][0],
                "ci95_high": tau["signed_adjudication"][f"{reference}_ci95"][1],
            }
        )
    pacc = summary["pacc_corroboration"]
    for reference in ("ad_reference", "control_reference", "specificity"):
        rows.append(
            {
                "analysis": "pacc_incremental_r2",
                "reference": reference,
                "estimate": pacc["incremental_r2"][reference],
                "ci95_low": pacc["incremental_r2"][f"{reference}_ci95"][0],
                "ci95_high": pacc["incremental_r2"][f"{reference}_ci95"][1],
            }
        )
    for reference in ("pathology_logit", "control_logit"):
        rows.append(
            {
                "analysis": "amyloid_adjusted_association",
                "reference": reference,
                "estimate": summary["amyloid_corroboration"][reference]["odds_ratio_per_sd"],
                "ci95_low": summary["amyloid_corroboration"][reference]["odds_ratio_per_sd_ci95"][0],
                "ci95_high": summary["amyloid_corroboration"][reference]["odds_ratio_per_sd_ci95"][1],
            }
        )
    return pd.DataFrame(rows)


def render_report(summary: dict[str, Any]) -> str:
    tau = summary["tau_primary"]
    r2 = tau["incremental_r2"]
    signed = tau["signed_adjudication"]
    gates = summary["tau_gates"]
    nonmolecular = summary["nonmolecular_results"]
    return f"""# PARES-AD biological and conditional validation

## Verdict

**{summary['status']}**

The equal-complexity anatomical falsification and adjusted-error tests support specificity beyond generic classifier disagreement: AD-reference minus control-reference error-risk ratio was {nonmolecular['anatomical_specificity']['ad_minus_control_rr']['estimate']:.2f} (95% interval {nonmolecular['anatomical_specificity']['ad_minus_control_rr']['ci95'][0]:.2f}–{nonmolecular['anatomical_specificity']['ad_minus_control_rr']['ci95'][1]:.2f}), and adjusted conflict error odds ratio was {nonmolecular['conditional_error_association']['odds_ratio']:.2f} ({nonmolecular['conditional_error_association']['ci95'][0]:.2f}–{nonmolecular['conditional_error_association']['ci95'][1]:.2f}).

In {tau['receipt']['participants']} FTP-linked participants, AD-reference incremental tau R² was {r2['ad_reference']:.3f}; AD-minus-control specificity was {r2['specificity']:.3f} ({r2['specificity_ci95'][0]:.3f}–{r2['specificity_ci95'][1]:.3f}). Signed tau adjudication for AD-reference conflicts was {signed['ad_reference']:.3f} SD ({signed['ad_reference_ci95'][0]:.3f}–{signed['ad_reference_ci95'][1]:.3f}); AD-minus-control adjudication was {signed['specificity']:.3f} SD ({signed['specificity_ci95'][0]:.3f}–{signed['specificity_ci95'][1]:.3f}).

Frozen substantive tau gates: {sum(gates.values())}/{len(gates)} passed. The sample contained {tau['receipt']['ad_conflict_participants']} participants with at least one AD-reference conflict, so molecular estimates are reported as precision-limited rather than adequately powered.

Amyloid and PACC are non-rescuing corroboration only. No participant-level identifier or anchor value was written.

## Claim boundary

{summary['claim_boundary']}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rscript", type=Path, default=DEFAULT_RSCRIPT)
    args = parser.parse_args()
    started = time.perf_counter()
    prior_summary_path = args.output / "summary.json"
    if not prior_summary_path.is_file():
        raise RuntimeError("Run the frozen non-molecular analysis before anchor validation")
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    nonmolecular = prior_summary.get("nonmolecular_results", prior_summary)
    if "gates" not in nonmolecular:
        required = {
            "anatomical_specificity",
            "conditional_error_association",
            "confidence_blind_spot",
        }
        if not required.issubset(nonmolecular):
            raise RuntimeError("Prior summary does not contain frozen non-molecular results")
        nonmolecular["gates"] = {
            "ad_exceeds_equal_complexity_control": bool(
                nonmolecular["anatomical_specificity"]["formal_gate"]
            ),
            "adjusted_conflict_or_at_least_2_ci_above_1": bool(
                nonmolecular["conditional_error_association"]["formal_gate"]
            ),
            "high_confidence_rr_at_least_2_ci_above_1_at_least_30_participants": bool(
                nonmolecular["confidence_blind_spot"]["formal_gate"]
            ),
        }
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    confirmation = build_confirmation_cohort(mapping)
    cases = prepare_cases(confirmation)
    references = participant_reference(cases, confirmation)
    references = references.merge(
        confirmation[
            ["_RID", "outer_domain", "PHC_Age_T1", "phase", "manufacturer", "field_strength", "AD_label"]
        ],
        on=["_RID", "outer_domain"],
        validate="one_to_one",
    )
    with tempfile.TemporaryDirectory(prefix="pares_anchor_values_") as temporary:
        directory = Path(temporary)
        export_blind_metadata(directory, args.rscript, include_values=True)
        tau_values = pd.read_csv(directory / "tau_values.csv", low_memory=False)
        amy_values = pd.read_csv(directory / "amy_values.csv", low_memory=False)
        pacc_values = pd.read_csv(directory / "pacc_values.csv", low_memory=False)

        tau_numeric = pd.to_numeric(tau_values["META_TEMPORAL_SUVR"], errors="coerce")
        tau_mask = (
            tau_values["TRACER"].fillna("").astype(str).str.strip().str.upper().eq("FTP")
            & pd.to_numeric(tau_values["qc_flag"], errors="coerce").eq(2)
            & tau_numeric.gt(0)
        )
        tau_linked = nearest_anchor(
            confirmation,
            tau_values,
            date_column="SCANDATE",
            value_column="META_TEMPORAL_SUVR",
            window_days=365,
            extra_mask=tau_mask,
        )
        amy_mask = pd.to_numeric(amy_values["qc_flag"], errors="coerce").eq(2) & pd.to_numeric(
            amy_values["AMYLOID_STATUS"], errors="coerce"
        ).isin([0, 1])
        amy_linked = nearest_anchor(
            confirmation,
            amy_values,
            date_column="SCANDATE",
            value_column="AMYLOID_STATUS",
            window_days=730,
            extra_mask=amy_mask,
        )
        pacc_numeric = pd.to_numeric(pacc_values["mPACCtrailsB"], errors="coerce")
        pacc_linked = nearest_anchor(
            confirmation,
            pacc_values,
            date_column="PACC_VISDATE",
            value_column="mPACCtrailsB",
            window_days=180,
            extra_mask=np.isfinite(pacc_numeric),
        )

        tau = continuous_anchor_analysis(
            tau_linked, references, cases, outcome_transform=lambda values: np.log(values)
        )
        amyloid = amyloid_analysis(amy_linked, references)
        pacc = continuous_anchor_analysis(
            pacc_linked, references, cases, outcome_transform=lambda values: -values
        )

    r2 = tau["incremental_r2"]
    signed = tau["signed_adjudication"]
    tau_gates = {
        "delta_r2_ad_at_least_0p02": r2["ad_reference"] >= 0.02,
        "delta_r2_specificity_at_least_0p01_ci_above_zero": (
            r2["specificity"] >= 0.01 and r2["specificity_ci95"][0] > 0
        ),
        "signed_adjudication_at_least_0p20_ci_above_zero": (
            signed["ad_reference"] >= 0.20 and signed["ad_reference_ci95"][0] > 0
        ),
        "signed_specificity_at_least_0p10_ci_above_zero": (
            signed["specificity"] >= 0.10 and signed["specificity_ci95"][0] > 0
        ),
    }
    nonmolecular_gates = {
        "ad_exceeds_equal_complexity_control": bool(
            nonmolecular["gates"]["ad_exceeds_equal_complexity_control"]
        ),
        "conditional_error_association": bool(
            nonmolecular["gates"]["adjusted_conflict_or_at_least_2_ci_above_1"]
        ),
        "high_confidence_blind_spot": bool(
            nonmolecular["gates"]["high_confidence_rr_at_least_2_ci_above_1_at_least_30_participants"]
        ),
    }
    biological_supported = all(tau_gates.values())
    pacc_clinical_specificity = (
        pacc["incremental_r2"]["ad_reference"] >= 0.02
        and pacc["incremental_r2"]["specificity"] >= 0.01
        and pacc["incremental_r2"]["specificity_ci95"][0] > 0
    )
    anatomical_conditional_supported = (
        nonmolecular_gates["ad_exceeds_equal_complexity_control"]
        and nonmolecular_gates["conditional_error_association"]
    )
    if biological_supported and anatomical_conditional_supported:
        status = "MOLECULAR_ANATOMICAL_SPECIFICITY_SUPPORTED_WITH_PRECISION_LIMIT"
        claim_boundary = (
            "PARES-AD pathology conflict is supported as AD-specific anatomical counterevidence by molecular, "
            "equal-complexity anatomical-control, and confidence-adjusted evidence within ADNI. It complements "
            "but does not outperform confidence ranking; high-confidence and external-clinical utility remain unproven."
        )
    elif anatomical_conditional_supported and pacc_clinical_specificity:
        status = "ANATOMICAL_AND_CLINICAL_SPECIFICITY_SUPPORTED_MOLECULAR_NOT_SUPPORTED"
        claim_boundary = (
            "PARES-AD pathology conflict is supported within ADNI as AD-pattern-specific anatomical "
            "counterevidence: it exceeds an equal-complexity non-AD reference, retains error association "
            "beyond confidence and routine covariates, and the AD reference adds clinical-severity information "
            "beyond the control. FTP tau and amyloid do not establish molecular AD specificity; confidence "
            "ranking superiority, high-confidence utility, clinician benefit, and external transport remain unsupported."
        )
    else:
        status = "PARTIAL_SPECIFICITY_ONLY"
        claim_boundary = (
            "PARES-AD has anatomical-control and confidence-adjusted specificity within ADNI, but the full frozen "
            "molecular conjunction did not pass. It remains a disease-oriented, independently replicated warning "
            "rather than proven molecular AD evidence or a decision-improving tool."
        )
    nonmolecular_evidence = {
        "data_receipt": nonmolecular["data_receipt"],
        "anatomical_specificity": nonmolecular["anatomical_specificity"],
        "conditional_error_association": nonmolecular["conditional_error_association"],
        "confidence_blind_spot": nonmolecular["confidence_blind_spot"],
        "runtime_seconds": nonmolecular["runtime_seconds"],
    }
    summary = {
        "analysis": "PARES-AD-BIOSPECIFICITY-v1-user-resumed",
        "status": status,
        "runtime_seconds": time.perf_counter() - started,
        "nonmolecular_results": nonmolecular_evidence,
        "tau_primary": tau,
        "tau_gates": tau_gates,
        "nonmolecular_gates": nonmolecular_gates,
        "amyloid_corroboration": amyloid,
        "pacc_corroboration": pacc,
        "full_biological_specificity_supported": biological_supported,
        "pacc_clinical_specificity_supported": pacc_clinical_specificity,
        "anatomical_and_conditional_specificity_supported": anatomical_conditional_supported,
        "supported_claims": [
            "The frozen AD-reference conflict exceeds an equal-complexity non-AD anatomical conflict control.",
            "AD-reference conflict retains error association beyond confidence, predicted class, demographics, acquisition, model configuration, and outer domain.",
            "The AD reference adds cross-fitted clinical-severity information beyond the equal-complexity control.",
        ],
        "not_supported_claims": [
            "FTP-tau molecular specificity of the AD reference or its conflict state.",
            "Amyloid-specific corroboration with an interval excluding the null.",
            "High-confidence conflict value with a participant-cluster interval excluding the null.",
            "Superior equal-workload error ranking versus confidence, clinician benefit, or external-program transport.",
        ],
        "claim_boundary": claim_boundary,
        "outcome_access": {
            "tau_values_analyzed_in_temporary_directory": True,
            "amyloid_values_analyzed_in_temporary_directory": True,
            "pacc_values_analyzed_in_temporary_directory": True,
            "participant_level_anchor_values_written": False,
        },
        "privacy": {"participant_identifiers_written": False, "participant_level_rows_written": False},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
    }
    aggregate = aggregate_rows(summary)
    forbidden = {"rid", "ptid", "subject_id", "participant_id", "anchor_value"}
    if forbidden.intersection(column.lower() for column in aggregate.columns):
        raise RuntimeError("Participant or anchor value leaked into aggregate output")
    aggregate.to_csv(args.output / "aggregate_results.csv", index=False, lineterminator="\n")
    prior_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"status": status, "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
