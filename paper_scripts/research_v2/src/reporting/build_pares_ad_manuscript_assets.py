#!/usr/bin/env python3
"""Build frozen aggregate tables and figure data for the PARES-AD manuscript.

This module does not fit a new classifier or change any cohort definition.  It
replays the frozen confirmation cohort constructor and writes aggregate-only
reporting artifacts.  No participant identifier is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_v2.src.analysis.confirm_pares_ad_independent import (
    MAPPING,
    build_confirmation_cohort,
    canonical_rid,
)
from research_v2.src.analysis.validate_pares_ad_conditional_specificity import (
    earliest_demographics,
)


ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "research_v2"
MANUSCRIPT = ROOT / "manuscript"
TABLE_DIR = MANUSCRIPT / "tables"
SUPP_DIR = MANUSCRIPT / "supplementary_tables"
FIG_DATA_DIR = MANUSCRIPT / "figures" / "data"

BASELINE = ROOT / "data" / "processed" / "baseline_cohort_with_concepts.csv"
PREDICTIONS = (
    V2
    / "outputs"
    / "private"
    / "fw5"
    / "aggregation"
    / "fw5_outer_endpoints_v2_1"
    / "configuration_outer_predictions.csv"
)
MODEL_PROFILES = (
    V2
    / "outputs"
    / "private"
    / "ad_mri_explanation_system_2026-07-23"
    / "model_reliability_profiles.csv"
)
REFERENCE_PROFILES = (
    V2
    / "outputs"
    / "private"
    / "ad_mri_explanation_system_2026-07-23"
    / "pathology_reference_profiles.csv"
)
CASE_EXPLANATIONS = (
    V2
    / "outputs"
    / "private"
    / "ad_mri_explanation_system_2026-07-23"
    / "case_explanations.csv"
)
CONFIRMATION_RESULTS = V2 / "confirmation_results_2026-07-23" / "confirmation_results.csv"
CONFIRMATION_SUMMARY = V2 / "confirmation_results_2026-07-23" / "summary.json"
BIOSPECIFICITY_RESULTS = (
    V2 / "pares_ad_biospecificity_nonmolecular_2026-07-23" / "aggregate_results.csv"
)
BIOSPECIFICITY_SUMMARY = (
    V2 / "pares_ad_biospecificity_nonmolecular_2026-07-23" / "summary.json"
)


def pct(count: int, total: int) -> str:
    return f"{count} ({100.0 * count / total:.1f}%)"


def mean_sd(values: pd.Series, digits: int = 1) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return f"{numeric.mean():.{digits}f} ({numeric.std(ddof=1):.{digits}f})"


def median_iqr(values: pd.Series, digits: int = 1) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    q1, median, q3 = numeric.quantile([0.25, 0.5, 0.75])
    return f"{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def characteristic_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(PREDICTIONS, dtype={"RID": str})
    development_rids = set(canonical_rid(predictions["RID"]))
    baseline = pd.read_csv(BASELINE, low_memory=False, dtype={"RID": str})
    baseline["_RID"] = canonical_rid(baseline["RID"])
    development = baseline[baseline["_RID"].isin(development_rids)].copy()
    if len(development) != 650 or development["_RID"].nunique() != 650:
        raise RuntimeError("Development cohort receipt changed")

    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    confirmation = build_confirmation_cohort(mapping)
    demographics = earliest_demographics(confirmation)
    confirmation = confirmation.merge(demographics, on="_RID", validate="one_to_one")

    dev_n = len(development)
    val_n = len(confirmation)
    dev_female = int(pd.to_numeric(development["PTGENDER"], errors="coerce").eq(2).sum())
    val_female = int(confirmation["sex"].eq("F").sum())
    dev_education = pd.to_numeric(development["PTEDUCAT"], errors="coerce")
    val_education = confirmation.loc[confirmation["education_missing"].eq(0), "education"]

    rows = [
        ("Participants", str(dev_n), str(val_n)),
        ("Clinical AD", pct(int(development["AD_label"].sum()), dev_n), pct(int(confirmation["AD_label"].sum()), val_n)),
        ("Cognitively normal", pct(int((1 - development["AD_label"]).sum()), dev_n), pct(int((1 - confirmation["AD_label"]).sum()), val_n)),
        ("Age, years, mean (SD)", mean_sd(development["PHC_Age_T1"]), mean_sd(confirmation["PHC_Age_T1"])),
        ("Female", pct(dev_female, dev_n), pct(val_female, val_n)),
        ("Education, years, median [IQR]", median_iqr(dev_education), median_iqr(val_education)),
        ("Education missing", pct(int(dev_education.isna().sum()), dev_n), pct(int(confirmation["education_missing"].sum()), val_n)),
        ("ADNI1", pct(int(development["PHASE"].astype(str).str.upper().eq("ADNI1").sum()), dev_n), pct(int(confirmation["phase"].eq("ADNI1").sum()), val_n)),
        ("ADNI3", pct(int(development["PHASE"].astype(str).str.upper().eq("ADNI3").sum()), dev_n), pct(int(confirmation["phase"].eq("ADNI3").sum()), val_n)),
        ("ADNI4", pct(int(development["PHASE"].astype(str).str.upper().eq("ADNI4").sum()), dev_n), pct(int(confirmation["phase"].eq("ADNI4").sum()), val_n)),
        ("1.5 T", pct(int(pd.to_numeric(development["FieldSgth"], errors="coerce").eq(1.5).sum()), dev_n), pct(int(confirmation["field_strength"].eq("1.5T").sum()), val_n)),
        ("3.0 T", pct(int(pd.to_numeric(development["FieldSgth"], errors="coerce").eq(3.0).sum()), dev_n), pct(int(confirmation["field_strength"].eq("3.0T").sum()), val_n)),
    ]
    table = pd.DataFrame(rows, columns=["Characteristic", "Development", "Independent validation"])
    flow = pd.DataFrame(
        [
            {"track": "Development", "stage": "Original baseline cohort", "participants": 729},
            {"track": "Development", "stage": "Complete frozen outer predictions", "participants": 650},
            {"track": "Independent validation", "stage": "Frozen eligible first scans", "participants": 882},
            {"track": "Independent validation", "stage": "Overlap with 729-person baseline cohort", "participants": 0},
            {"track": "PACC corroboration", "stage": "Linked independent participants", "participants": 876},
        ]
    )
    return table, flow


def supplementary_table_s1() -> pd.DataFrame:
    profiles = pd.read_csv(MODEL_PROFILES)
    columns = [
        "family",
        "configuration_id",
        "n_cases",
        "n_outer_domains",
        "outer_domains",
        "auroc",
        "auprc",
        "brier",
        "calibration_slope",
        "mean_between_seed_probability_sd",
        "disease_behavior_r2",
        "disease_behavior_spearman",
        "full_behavior_r2",
        "acquisition_incremental_r2",
        "empirical_residual_bound_coverage",
        "fraction_unexplained_abstain",
        "fraction_pathology_conflict",
        "pathology_conflict_vs_aligned_error_risk_ratio",
        "explanation_review_value_pass",
    ]
    return profiles[columns].copy()


def supplementary_table_s2() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_PROFILES).copy()


def endpoint_row(
    analysis: str,
    estimand: str,
    estimate: float,
    low: float,
    high: float,
    sample: str,
    status: str,
) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "estimand": estimand,
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "sample": sample,
        "status": status,
    }


def supplementary_table_s3() -> pd.DataFrame:
    bio = json.loads(BIOSPECIFICITY_SUMMARY.read_text(encoding="utf-8"))
    conf = json.loads(CONFIRMATION_SUMMARY.read_text(encoding="utf-8"))
    nonmol = bio["nonmolecular_results"]
    anatomy = nonmol["anatomical_specificity"]
    pacc = bio["pacc_corroboration"]["incremental_r2"]
    tau = bio["tau_primary"]
    amyloid = bio["amyloid_corroboration"]
    rows: list[dict[str, Any]] = []

    for label, item, status in [
        ("AD reference", anatomy["ad_reference"], "supported"),
        ("Matched control", anatomy["control_reference"], "not supported"),
    ]:
        rows.append(endpoint_row("Anatomical specificity", f"Conflict error risk ratio: {label}", item["error_risk_ratio"], *item["ci95"], "882 participants; 7,091 predictions", status))
    item = anatomy["ad_minus_control_rr"]
    rows.append(endpoint_row("Anatomical specificity", "AD-minus-control risk-ratio difference", item["estimate"], *item["ci95"], "882 participants; 7,091 predictions", "supported"))

    item = nonmol["conditional_error_association"]
    rows.append(endpoint_row("Conditional association", "Adjusted conflict-error odds ratio", item["odds_ratio"], *item["ci95"], "882 participants; 7,091 predictions", "supported association"))

    for label, estimate_key, ci_key, status in [
        ("AD reference", "ad_reference", "ad_reference_ci95", "supported"),
        ("Matched control", "control_reference", "control_reference_ci95", "not supported"),
        ("AD-minus-control", "specificity", "specificity_ci95", "supported"),
    ]:
        rows.append(endpoint_row("PACC corroboration", f"Incremental R2: {label}", pacc[estimate_key], *pacc[ci_key], "876 participants", status))

    tau_r2 = tau["incremental_r2"]
    for label, estimate_key, ci_key in [
        ("AD reference", "ad_reference", "ad_reference_ci95"),
        ("Matched control", "control_reference", "control_reference_ci95"),
        ("AD-minus-control", "specificity", "specificity_ci95"),
    ]:
        rows.append(endpoint_row("FTP tau", f"Incremental R2: {label}", tau_r2[estimate_key], *tau_r2[ci_key], "295 participants", "not supported"))
    tau_signed = tau["signed_adjudication"]
    rows.append(endpoint_row("FTP tau", "Signed conflict adjudication: AD reference", tau_signed["ad_reference"], *tau_signed["ad_reference_ci95"], "295 participants", "not supported"))

    rows.append(endpoint_row("Amyloid", "Odds ratio per SD: AD reference", amyloid["pathology_logit"]["odds_ratio_per_sd"], *amyloid["pathology_logit"]["odds_ratio_per_sd_ci95"], "366 participants", "not supported"))
    rows.append(endpoint_row("Amyloid", "AD-minus-control log-odds difference", amyloid["pathology_vs_control_log_odds_difference"]["estimate"], *amyloid["pathology_vs_control_log_odds_difference"]["ci95"], "366 participants", "not supported"))

    high = nonmol["confidence_blind_spot"]
    rows.append(endpoint_row("High-confidence subset", "Conflict error risk ratio", high["error_risk_ratio"], *high["ci95"], "718 participants; 4,751 predictions", "not supported"))
    primary = conf["primary_endpoint"]
    rows.append(endpoint_row("Equal-workload review", "Conflict-minus-confidence review-yield difference", primary["absolute_review_yield_gain"], primary["ci95_low"], primary["ci95_high"], "882 participants; 687 reviewed predictions", "not supported"))
    return pd.DataFrame(rows)


def anonymized_cases() -> pd.DataFrame:
    cases = pd.read_csv(CASE_EXPLANATIONS)
    cases = cases[
        cases["family"].eq("flat_mlp") & cases["configuration_id"].eq("mlp_h64")
    ].sort_values("case_key", kind="stable")
    selectors = [
        ("Case A", "Aligned; correct", cases["explanation_covered"] & ~cases["pathology_conflict"] & cases["model_error"].eq(0)),
        ("Case B", "Mismatched; incorrect", cases["explanation_covered"] & cases["pathology_conflict"] & cases["model_error"].eq(1)),
        ("Case C", "Mismatched; correct", cases["explanation_covered"] & cases["pathology_conflict"] & cases["model_error"].eq(0)),
        ("Case D", "Residual above bound", ~cases["explanation_covered"]),
    ]
    rows: list[dict[str, Any]] = []
    for public_name, category, mask in selectors:
        eligible = cases[mask]
        if eligible.empty:
            raise RuntimeError(f"No frozen case available for {category}")
        row = eligible.iloc[0]
        rows.append(
            {
                "case": public_name,
                "category": category,
                "model_probability_AD": row["model_probability_AD"],
                "reference_probability_AD": row["pathology_probability_AD"],
                "hippocampal_contribution": row["C001_pathology_logodds_contribution"],
                "entorhinal_contribution": row["C002_pathology_logodds_contribution"],
                "ventricular_contribution": row["C003_pathology_logodds_contribution"],
                "temporoparietal_contribution": row["C004_pathology_logodds_contribution"],
                "disease_behavior_logit": row["disease_behavior_centered_logit"],
                "acquisition_behavior_logit": row["acquisition_behavior_logit"],
                "unexplained_logit": row["unexplained_model_logit"],
                "residual_bound_90": row["residual_bound_90"],
                "explanation_covered": bool(row["explanation_covered"]),
                "mismatched": bool(row["pathology_conflict"]),
                "model_error": int(row["model_error"]),
                "reference_label": int(row["reference_label"]),
            }
        )
    return pd.DataFrame(rows)


def write_manifest(outputs: dict[str, str]) -> None:
    manifest = {
        "artifact": "PARES-AD manuscript aggregate assets",
        "privacy": "Aggregate tables only; no participant identifier is written.",
        "frozen_sources": [
            str(BASELINE),
            str(PREDICTIONS),
            str(MODEL_PROFILES),
            str(REFERENCE_PROFILES),
            str(CASE_EXPLANATIONS),
            str(CONFIRMATION_RESULTS),
            str(CONFIRMATION_SUMMARY),
            str(BIOSPECIFICITY_RESULTS),
            str(BIOSPECIFICITY_SUMMARY),
        ],
        "transformations": [
            "Replay frozen confirmation cohort eligibility without refitting classifiers",
            "Aggregate demographics at participant level",
            "Copy frozen model/reference profiles",
            "Flatten frozen endpoint JSON into supplementary table rows",
            "Select four examples by fixed lexicographic rules and remove internal case keys",
        ],
        "outputs": outputs,
    }
    (MANUSCRIPT / "PARES_AD_MANUSCRIPT_ASSET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    for directory in [TABLE_DIR, SUPP_DIR, FIG_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    table1, flow = characteristic_table()
    s1 = supplementary_table_s1()
    s2 = supplementary_table_s2()
    s3 = supplementary_table_s3()
    case_table = anonymized_cases()
    confirmation = pd.read_csv(CONFIRMATION_RESULTS)
    model_profiles = pd.read_csv(MODEL_PROFILES)
    reference_profiles = pd.read_csv(REFERENCE_PROFILES)
    biospecificity = pd.read_csv(BIOSPECIFICITY_RESULTS)

    artifacts = {
        TABLE_DIR / "Table1_participant_characteristics.csv": table1,
        SUPP_DIR / "Table_S1_model_and_functional_coverage.csv": s1,
        SUPP_DIR / "Table_S2_transparent_reference_by_domain.csv": s2,
        SUPP_DIR / "Table_S3_corroboration_and_boundary_endpoints.csv": s3,
        FIG_DATA_DIR / "Fig1_cohort_flow.csv": flow,
        FIG_DATA_DIR / "Fig2_model_and_reference_profiles.csv": model_profiles,
        FIG_DATA_DIR / "Fig2_reference_profiles.csv": reference_profiles,
        FIG_DATA_DIR / "Fig3_anonymized_cases.csv": case_table,
        FIG_DATA_DIR / "Fig4_error_enrichment.csv": confirmation,
        FIG_DATA_DIR / "Fig5_corroboration.csv": s3[s3["analysis"].isin(["Anatomical specificity", "Conditional association", "PACC corroboration"])],
        FIG_DATA_DIR / "Fig6_claim_boundaries.csv": s3[s3["analysis"].isin(["FTP tau", "Amyloid", "High-confidence subset", "Equal-workload review"])],
        FIG_DATA_DIR / "Fig6_biospecificity_raw_endpoints.csv": biospecificity,
    }
    for path, frame in artifacts.items():
        frame.to_csv(path, index=False, lineterminator="\n")
    write_manifest({str(path.relative_to(ROOT)): str(frame.shape) for path, frame in artifacts.items()})
    print(json.dumps({"status": "ok", "outputs": len(artifacts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
