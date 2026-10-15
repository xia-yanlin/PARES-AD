#!/usr/bin/env python3
"""Build the AD/MRI pathology–acquisition explanation and reliability system."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_v2.src.audit.concepts import build_concept_target


RANDOM_SEED = 20_260_723
BOOTSTRAP_REPLICATES = 2_000
CORE_CONCEPTS = ("C001", "C002", "C003", "C004")
CONCEPT_LABELS = {
    "C001": "hippocampal_atrophy",
    "C002": "entorhinal_thinning",
    "C003": "ventricular_enlargement",
    "C004": "temporoparietal_thinning",
}
CONCEPT_ORIENTATION = {"C001": -1.0, "C002": -1.0, "C003": 1.0, "C004": -1.0}
PHASE_LEVELS = ("ADNI1", "ADNI3", "ADNI4")
MANUFACTURER_LEVELS = ("Siemens", "GE", "Philips")
FIELD_LEVELS = ("1.5T", "3.0T")
CASE_STATE_ORDER = (
    "UNEXPLAINED_ABSTAIN",
    "ACQUISITION_UNVALIDATED_ABSTAIN",
    "ACQUISITION_SENSITIVE",
    "PATHOLOGY_CONFLICT",
    "PATHOLOGY_ALIGNED",
)
FORBIDDEN_OUTPUT_COLUMNS = {
    "rid",
    "ptid",
    "subject_id",
    "participant_id",
    "subjid",
    "phc_loniuid",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logit(probability: np.ndarray | pd.Series, epsilon: float = 1e-4) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    values = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(values / (1.0 - values))


def assert_private_identifiers_absent(frame: pd.DataFrame) -> None:
    normalized = {str(column).strip().lower() for column in frame.columns}
    forbidden = normalized.intersection(FORBIDDEN_OUTPUT_COLUMNS)
    if forbidden:
        raise ValueError(f"Participant identifier columns are forbidden: {sorted(forbidden)}")


def conformal_absolute_quantile(residuals: np.ndarray, coverage: float = 0.90) -> float:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Conformal calibration requires finite residuals")
    rank = min(len(values), int(math.ceil((len(values) + 1) * coverage)))
    return float(np.partition(values, rank - 1)[rank - 1])


def fit_monotone_pathology_model(
    features: np.ndarray,
    labels: np.ndarray,
    l2_penalty: float = 0.1,
) -> tuple[float, np.ndarray]:
    """Fit logistic loss with nonnegative disease-severity coefficients."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(CORE_CONCEPTS):
        raise ValueError("Pathology model requires four disease-severity columns")
    if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Pathology model inputs are invalid")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Pathology model requires both binary classes")
    prevalence = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    initial = np.r_[math.log(prevalence / (1 - prevalence)), np.full(x.shape[1], 0.1)]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        linear = intercept + x @ coefficients
        loss = float(np.mean(np.logaddexp(0.0, linear) - y * linear))
        loss += 0.5 * l2_penalty * float(coefficients @ coefficients)
        error = expit(linear) - y
        gradient = np.r_[
            float(error.mean()),
            x.T @ error / len(x) + l2_penalty * coefficients,
        ]
        return loss, gradient

    result = minimize(
        lambda parameters: objective(parameters)[0],
        initial,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        bounds=[(None, None), *[(0.0, None)] * x.shape[1]],
        options={"maxiter": 2_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Monotone pathology optimization failed: {result.message}")
    coefficients = np.asarray(result.x[1:], dtype=float)
    if (coefficients < -1e-10).any():
        raise RuntimeError("Monotone pathology coefficients violated their bounds")
    return float(result.x[0]), coefficients


def acquisition_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for source, levels in [
        ("phase", PHASE_LEVELS),
        ("manufacturer", MANUFACTURER_LEVELS),
        ("field_strength", FIELD_LEVELS),
    ]:
        values = frame[source].astype(str).to_numpy()
        unknown = sorted(set(values) - set(levels))
        if unknown:
            raise ValueError(f"Unknown acquisition levels for {source}: {unknown}")
        for level in levels:
            columns.append((values == level).astype(float))
            names.append(f"{source}={level}")
    return np.column_stack(columns), names


def fit_two_stage_behavior(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> dict[str, np.ndarray | float]:
    disease_columns = [f"{concept}_severity" for concept in CORE_CONCEPTS]
    x_disease_train = train[disease_columns].to_numpy(dtype=float)
    x_disease_eval = evaluation[disease_columns].to_numpy(dtype=float)
    y_train = train["model_logit"].to_numpy(dtype=float)
    disease_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    disease_model.fit(x_disease_train, y_train)
    disease_train = disease_model.predict(x_disease_train)
    disease_eval = disease_model.predict(x_disease_eval)

    residual_train = y_train - disease_train
    x_acquisition_train, acquisition_names = acquisition_design(train)
    x_acquisition_eval, _ = acquisition_design(evaluation)
    acquisition_model = Ridge(alpha=1.0)
    acquisition_model.fit(x_acquisition_train, residual_train)
    acquisition_train = acquisition_model.predict(x_acquisition_train)
    acquisition_eval = acquisition_model.predict(x_acquisition_eval)
    full_eval = disease_eval + acquisition_eval
    return {
        "training_mean_logit": float(y_train.mean()),
        "disease_eval": np.asarray(disease_eval, dtype=float),
        "acquisition_eval": np.asarray(acquisition_eval, dtype=float),
        "full_eval": np.asarray(full_eval, dtype=float),
        "acquisition_feature_names": np.asarray(acquisition_names, dtype=object),
        "acquisition_coefficients": np.asarray(acquisition_model.coef_, dtype=float),
    }


def nested_training_residuals(train: pd.DataFrame) -> np.ndarray:
    pieces: list[np.ndarray] = []
    domains = sorted(train["outer_domain"].unique())
    if len(domains) < 3:
        raise ValueError("At least three training domains are needed for residual calibration")
    for calibration_domain in domains:
        calibration = train[train["outer_domain"].eq(calibration_domain)]
        fitting = train[~train["outer_domain"].eq(calibration_domain)]
        decomposition = fit_two_stage_behavior(fitting, calibration)
        pieces.append(
            np.abs(
                calibration["model_logit"].to_numpy(dtype=float)
                - np.asarray(decomposition["full_eval"], dtype=float)
            )
        )
    return np.concatenate(pieces)


def reliability_state(
    covered: bool,
    acquisition_component: float,
    disease_component: float,
    pathology_conflict: bool,
) -> tuple[str, bool]:
    acquisition_dominant = bool(
        abs(acquisition_component) >= 0.25
        and abs(acquisition_component) >= abs(disease_component)
    )
    if not covered:
        return "UNEXPLAINED_ABSTAIN", acquisition_dominant
    if acquisition_dominant:
        return "ACQUISITION_SENSITIVE", acquisition_dominant
    if pathology_conflict:
        return "PATHOLOGY_CONFLICT", acquisition_dominant
    return "PATHOLOGY_ALIGNED", acquisition_dominant


def selective_curve(
    errors: np.ndarray,
    predicted_error_risk: np.ndarray,
    coverages: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(errors, dtype=int)
    risk = np.asarray(predicted_error_risk, dtype=float)
    if coverages is None:
        coverages = np.arange(0.50, 1.001, 0.01)
    order = np.argsort(risk, kind="stable")
    selective_risk: list[float] = []
    for coverage in coverages:
        accepted_n = max(1, int(math.floor(len(y) * float(coverage))))
        selective_risk.append(float(y[order[:accepted_n]].mean()))

    def risk_at(coverage: float) -> float:
        accepted_n = max(1, int(math.floor(len(y) * coverage)))
        return float(y[order[:accepted_n]].mean())

    return {
        "aurc_50_100": float(np.mean(selective_risk)),
        "risk_at_80": risk_at(0.80),
        "risk_at_90": risk_at(0.90),
    }


def build_concept_and_pathology_rows(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    target_rids: set[str],
    target_domains: list[str],
) -> pd.DataFrame:
    cohort = cohort.copy()
    cohort["RID"] = cohort["RID"].astype(str)
    assignments = assignments.copy()
    assignments["RID"] = assignments["RID"].astype(str)
    assignment_columns = [
        "RID",
        "phase",
        "manufacturer",
        "field_strength",
        "acquisition_domain",
        "label",
    ]
    merged = cohort.merge(
        assignments[assignment_columns], on="RID", validate="one_to_one"
    )
    if not merged["AD_label"].astype(int).eq(merged["label"].astype(int)).all():
        raise ValueError("Cohort and acquisition-assignment labels disagree")

    rows: list[pd.DataFrame] = []
    for domain in target_domains:
        train = merged[~merged["acquisition_domain"].eq(domain)].reset_index(drop=True)
        evaluation = merged[
            merged["acquisition_domain"].eq(domain) & merged["RID"].isin(target_rids)
        ].reset_index(drop=True)
        if evaluation.empty:
            raise ValueError(f"No target subjects found for acquisition domain {domain}")
        train_values: list[np.ndarray] = []
        evaluation_values: list[np.ndarray] = []
        for concept_id in CORE_CONCEPTS:
            target = build_concept_target(concept_id, train, evaluation)
            if not target.train_valid.all() or not target.evaluation_valid.all():
                raise ValueError(f"Primary concept {concept_id} has invalid target rows")
            orientation = CONCEPT_ORIENTATION[concept_id]
            train_values.append(orientation * target.train_values.astype(float))
            evaluation_values.append(orientation * target.evaluation_values.astype(float))
        x_train = np.column_stack(train_values)
        x_evaluation = np.column_stack(evaluation_values)
        intercept, coefficients = fit_monotone_pathology_model(
            x_train, train["AD_label"].to_numpy(dtype=int)
        )
        contributions = x_evaluation * coefficients
        pathology_logit = intercept + contributions.sum(axis=1)
        frame = pd.DataFrame(
            {
                "_RID": evaluation["RID"].astype(str),
                "outer_domain": domain,
                "reference_label": evaluation["AD_label"].astype(int),
                "phase": evaluation["phase"].astype(str),
                "manufacturer": evaluation["manufacturer"].astype(str),
                "field_strength": evaluation["field_strength"].astype(str),
                "pathology_intercept": intercept,
                "pathology_logit": pathology_logit,
                "pathology_probability_AD": expit(pathology_logit),
            }
        )
        for index, concept_id in enumerate(CORE_CONCEPTS):
            frame[f"{concept_id}_severity"] = x_evaluation[:, index]
            frame[f"{concept_id}_pathology_logodds_contribution"] = contributions[
                :, index
            ]
            frame[f"{concept_id}_pathology_coefficient"] = coefficients[index]
        contribution_columns = [
            f"{concept_id}_pathology_logodds_contribution"
            for concept_id in CORE_CONCEPTS
        ]
        contribution_matrix = frame[contribution_columns].to_numpy(dtype=float)
        support_indices = np.argmax(contribution_matrix, axis=1)
        counter_indices = np.argmin(contribution_matrix, axis=1)
        frame["strongest_pathology_support"] = [
            CONCEPT_LABELS[CORE_CONCEPTS[index]] for index in support_indices
        ]
        frame["strongest_pathology_counterevidence"] = [
            CONCEPT_LABELS[CORE_CONCEPTS[index]] for index in counter_indices
        ]
        reconstruction = (
            frame["pathology_intercept"].to_numpy()
            + frame[contribution_columns].sum(axis=1).to_numpy()
        )
        if not np.allclose(reconstruction, pathology_logit, atol=1e-10, rtol=0):
            raise RuntimeError("Transparent pathology contributions do not sum exactly")
        rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    if output["_RID"].duplicated().any():
        raise ValueError("Concept/pathology output has duplicate subjects")
    return output


def build_pathology_reference_profiles(concept_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain, group in concept_rows.groupby("outer_domain", sort=True):
        labels = group["reference_label"].to_numpy(dtype=int)
        probability = group["pathology_probability_AD"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "outer_domain": domain,
            "n_cases": int(len(group)),
            "n_ad": int(labels.sum()),
            "n_cn": int(len(labels) - labels.sum()),
            "auroc": float(roc_auc_score(labels, probability)),
            "auprc": float(average_precision_score(labels, probability)),
            "brier": float(brier_score_loss(labels, probability)),
            "calibration_slope": calibration_slope(labels, probability),
            "intercept": float(group["pathology_intercept"].iloc[0]),
        }
        for concept_id in CORE_CONCEPTS:
            values = group[f"{concept_id}_pathology_coefficient"].unique()
            if len(values) != 1:
                raise RuntimeError(
                    f"Pathology coefficient varies within outer domain: {domain}/{concept_id}"
                )
            row[f"{concept_id}_coefficient"] = float(values[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("outer_domain").reset_index(drop=True)


def decompose_black_box_behavior(
    joined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_groups: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    for (family, configuration_id), group in joined.groupby(
        ["family", "configuration_id"], sort=True
    ):
        domains = sorted(group["outer_domain"].unique())
        if len(domains) < 4 or len(group) < 300:
            raise ValueError(
                f"Insufficient multi-domain evidence for {family}/{configuration_id}"
            )
        for domain in domains:
            evaluation = group[group["outer_domain"].eq(domain)].copy()
            train = group[~group["outer_domain"].eq(domain)].copy()
            decomposition = fit_two_stage_behavior(train, evaluation)
            calibration_residuals = nested_training_residuals(train)
            bound = conformal_absolute_quantile(calibration_residuals, 0.90)
            evaluation["disease_behavior_logit"] = np.asarray(
                decomposition["disease_eval"], dtype=float
            )
            evaluation["disease_behavior_centered_logit"] = (
                evaluation["disease_behavior_logit"]
                - float(decomposition["training_mean_logit"])
            )
            evaluation["acquisition_behavior_logit"] = np.asarray(
                decomposition["acquisition_eval"], dtype=float
            )
            evaluation["reconstructed_model_logit"] = np.asarray(
                decomposition["full_eval"], dtype=float
            )
            evaluation["unexplained_model_logit"] = (
                evaluation["model_logit"] - evaluation["reconstructed_model_logit"]
            )
            evaluation["absolute_unexplained_model_logit"] = evaluation[
                "unexplained_model_logit"
            ].abs()
            evaluation["residual_bound_90"] = bound
            evaluation["residual_to_bound_ratio"] = (
                evaluation["absolute_unexplained_model_logit"] / max(bound, 1e-12)
            )
            evaluation["explanation_covered"] = evaluation[
                "absolute_unexplained_model_logit"
            ].le(bound)
            evaluation["pathology_conflict"] = (
                evaluation["model_probability_AD"].ge(0.5)
                != evaluation["pathology_probability_AD"].ge(0.5)
            )
            states: list[str] = []
            dominance: list[bool] = []
            for record in evaluation[
                [
                    "explanation_covered",
                    "acquisition_behavior_logit",
                    "disease_behavior_centered_logit",
                    "pathology_conflict",
                ]
            ].to_dict("records"):
                state, dominant = reliability_state(
                    bool(record["explanation_covered"]),
                    float(record["acquisition_behavior_logit"]),
                    float(record["disease_behavior_centered_logit"]),
                    bool(record["pathology_conflict"]),
                )
                states.append(state)
                dominance.append(dominant)
            evaluation["acquisition_dominant"] = dominance
            evaluation["reliability_state"] = states
            output_groups.append(evaluation)
            for name, coefficient in zip(
                decomposition["acquisition_feature_names"],
                decomposition["acquisition_coefficients"],
            ):
                coefficient_rows.append(
                    {
                        "family": family,
                        "configuration_id": configuration_id,
                        "evaluation_outer_domain": domain,
                        "acquisition_feature": str(name),
                        "coefficient": float(coefficient),
                    }
                )
    return (
        pd.concat(output_groups, ignore_index=True),
        pd.DataFrame(coefficient_rows),
    )


def apply_acquisition_validation_gate(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    validation_rows: list[dict[str, Any]] = []
    for (family, configuration_id), group in output.groupby(
        ["family", "configuration_id"], sort=True
    ):
        model_logit = group["model_logit"].to_numpy(dtype=float)
        disease_r2 = float(
            r2_score(model_logit, group["disease_behavior_logit"].to_numpy(dtype=float))
        )
        full_r2 = float(
            r2_score(
                model_logit,
                group["reconstructed_model_logit"].to_numpy(dtype=float),
            )
        )
        validation_rows.append(
            {
                "family": family,
                "configuration_id": configuration_id,
                "acquisition_component_model_validated": bool(full_r2 - disease_r2 > 0),
            }
        )
    validation = pd.DataFrame(validation_rows)
    output = output.merge(
        validation,
        on=["family", "configuration_id"],
        validate="many_to_one",
    )
    invalid_acquisition_state = (
        output["reliability_state"].eq("ACQUISITION_SENSITIVE")
        & ~output["acquisition_component_model_validated"]
    )
    output.loc[
        invalid_acquisition_state, "reliability_state"
    ] = "ACQUISITION_UNVALIDATED_ABSTAIN"
    return output


def crossfit_error_risk(frame: pd.DataFrame) -> pd.DataFrame:
    baseline_columns = [
        "model_probability_AD",
        "model_confidence_distance",
        "model_predicted_class",
    ]
    explanation_columns = [
        "model_pathology_probability_gap",
        "absolute_acquisition_behavior_logit",
        "residual_to_bound_ratio_clipped",
        "pathology_conflict_numeric",
        "acquisition_dominant_numeric",
        "unexplained_abstain_numeric",
    ]
    output = frame.copy()
    output["model_error"] = (
        output["model_predicted_class"].astype(int)
        != output["reference_label"].astype(int)
    ).astype(int)
    output["confidence_error_risk"] = np.nan
    output["explanation_error_risk"] = np.nan
    for domain in sorted(output["outer_domain"].unique()):
        train = output[~output["outer_domain"].eq(domain)]
        evaluation = output[output["outer_domain"].eq(domain)]
        y_train = train["model_error"].to_numpy(dtype=int)
        if len(np.unique(y_train)) != 2:
            raise ValueError(f"Error-risk training lacks both classes before {domain}")
        for columns, target in [
            (baseline_columns, "confidence_error_risk"),
            (baseline_columns + explanation_columns, "explanation_error_risk"),
        ]:
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                ),
            )
            estimator.fit(train[columns].to_numpy(dtype=float), y_train)
            output.loc[evaluation.index, target] = estimator.predict_proba(
                evaluation[columns].to_numpy(dtype=float)
            )[:, 1]
    if output[["confidence_error_risk", "explanation_error_risk"]].isna().any().any():
        raise RuntimeError("Cross-fitted error-risk predictions are incomplete")
    return output


def paired_selective_bootstrap(
    errors: np.ndarray,
    confidence_risk: np.ndarray,
    explanation_risk: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(errors, dtype=int)
    confidence = np.asarray(confidence_risk, dtype=float)
    explanation = np.asarray(explanation_risk, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(y), size=len(y))
        confidence_curve = selective_curve(y[sample], confidence[sample])
        explanation_curve = selective_curve(y[sample], explanation[sample])
        differences[replicate] = (
            explanation_curve["aurc_50_100"] - confidence_curve["aurc_50_100"]
        )
    return {
        "mean": float(differences.mean()),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
    }


def clustered_state_risk_ratio(
    cases: pd.DataFrame,
    state: str,
    reference_state: str = "PATHOLOGY_ALIGNED",
) -> dict[str, float | int]:
    relevant = cases[
        cases["reliability_state"].isin([state, reference_state])
    ].copy()
    if relevant.empty or not relevant["reliability_state"].eq(state).any():
        return {
            "state_n": 0,
            "reference_n": int(
                relevant["reliability_state"].eq(reference_state).sum()
            ),
            "state_error_rate": float("nan"),
            "reference_error_rate": float("nan"),
            "risk_ratio": float("nan"),
            "cluster_bootstrap_ci95_low": float("nan"),
            "cluster_bootstrap_ci95_high": float("nan"),
        }
    state_mask = relevant["reliability_state"].eq(state)
    reference_mask = relevant["reliability_state"].eq(reference_state)
    state_rate = float(relevant.loc[state_mask, "model_error"].mean())
    reference_rate = float(relevant.loc[reference_mask, "model_error"].mean())
    observed = state_rate / reference_rate if reference_rate > 0 else float("nan")

    per_case_rows: list[dict[str, float | str]] = []
    for case_key, group in relevant.groupby("case_key", sort=True):
        group_state = group["reliability_state"].eq(state)
        group_reference = group["reliability_state"].eq(reference_state)
        per_case_rows.append(
            {
                "case_key": case_key,
                "state_errors": float(
                    group.loc[group_state, "model_error"].sum()
                ),
                "state_n": float(group_state.sum()),
                "reference_errors": float(
                    group.loc[group_reference, "model_error"].sum()
                ),
                "reference_n": float(group_reference.sum()),
            }
        )
    per_case = pd.DataFrame(per_case_rows)
    values = per_case[
        ["state_errors", "state_n", "reference_errors", "reference_n"]
    ].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    ratios: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        if sampled[1] <= 0 or sampled[3] <= 0 or sampled[2] <= 0:
            continue
        sampled_state_rate = sampled[0] / sampled[1]
        sampled_reference_rate = sampled[2] / sampled[3]
        ratios.append(float(sampled_state_rate / sampled_reference_rate))
    if not ratios:
        low = high = float("nan")
    else:
        low = float(np.quantile(ratios, 0.025))
        high = float(np.quantile(ratios, 0.975))
    return {
        "state_n": int(state_mask.sum()),
        "reference_n": int(reference_mask.sum()),
        "state_error_rate": state_rate,
        "reference_error_rate": reference_rate,
        "risk_ratio": observed,
        "cluster_bootstrap_ci95_low": low,
        "cluster_bootstrap_ci95_high": high,
    }


def calibration_slope(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int)
    x = logit(probabilities).reshape(-1, 1)
    estimator = LogisticRegression(C=1e6, max_iter=2_000)
    estimator.fit(x, y)
    return float(estimator.coef_[0, 0])


def build_model_profiles(
    cases_internal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows: list[dict[str, Any]] = []
    risk_groups: list[pd.DataFrame] = []
    for (family, configuration_id), group in cases_internal.groupby(
        ["family", "configuration_id"], sort=True
    ):
        group = group.copy().reset_index(drop=True)
        group["model_predicted_class"] = group["model_probability_AD"].ge(0.5).astype(int)
        group["model_confidence_distance"] = (
            group["model_probability_AD"] - 0.5
        ).abs()
        group["model_pathology_probability_gap"] = (
            group["model_probability_AD"] - group["pathology_probability_AD"]
        ).abs()
        group["absolute_acquisition_behavior_logit"] = group[
            "acquisition_behavior_logit"
        ].abs()
        group["residual_to_bound_ratio_clipped"] = group[
            "residual_to_bound_ratio"
        ].clip(0, 10)
        group["pathology_conflict_numeric"] = group["pathology_conflict"].astype(int)
        group["acquisition_dominant_numeric"] = group[
            "acquisition_dominant"
        ].astype(int)
        group["unexplained_abstain_numeric"] = (
            group["reliability_state"].str.endswith("ABSTAIN").astype(int)
        )
        group = crossfit_error_risk(group)
        risk_groups.append(group)

        labels = group["reference_label"].to_numpy(dtype=int)
        probability = group["model_probability_AD"].to_numpy(dtype=float)
        model_logits = group["model_logit"].to_numpy(dtype=float)
        disease_logits = group["disease_behavior_logit"].to_numpy(dtype=float)
        full_logits = group["reconstructed_model_logit"].to_numpy(dtype=float)
        disease_r2 = float(r2_score(model_logits, disease_logits))
        full_r2 = float(r2_score(model_logits, full_logits))
        correlation = spearmanr(model_logits, disease_logits).statistic
        confidence_curve = selective_curve(
            group["model_error"].to_numpy(),
            group["confidence_error_risk"].to_numpy(),
        )
        explanation_curve = selective_curve(
            group["model_error"].to_numpy(),
            group["explanation_error_risk"].to_numpy(),
        )
        bootstrap = paired_selective_bootstrap(
            group["model_error"].to_numpy(),
            group["confidence_error_risk"].to_numpy(),
            group["explanation_error_risk"].to_numpy(),
        )
        relative_reduction = (
            (confidence_curve["aurc_50_100"] - explanation_curve["aurc_50_100"])
            / confidence_curve["aurc_50_100"]
            if confidence_curve["aurc_50_100"] > 0
            else float("nan")
        )
        decision_pass = bool(
            np.isfinite(relative_reduction)
            and relative_reduction >= 0.05
            and bootstrap["ci95_high"] < 0
            and explanation_curve["risk_at_80"] < confidence_curve["risk_at_80"]
            and explanation_curve["risk_at_90"] <= confidence_curve["risk_at_90"]
        )
        usable = bool(
            full_r2 > 0
            and group["explanation_covered"].mean() >= 0.80
            and group["outer_domain"].nunique() >= 4
            and len(group) >= 300
        )
        row: dict[str, Any] = {
            "family": family,
            "configuration_id": configuration_id,
            "n_cases": int(len(group)),
            "n_outer_domains": int(group["outer_domain"].nunique()),
            "outer_domains": ";".join(sorted(group["outer_domain"].unique())),
            "auroc": float(roc_auc_score(labels, probability)),
            "auprc": float(average_precision_score(labels, probability)),
            "brier": float(brier_score_loss(labels, probability)),
            "calibration_slope": calibration_slope(labels, probability),
            "mean_between_seed_probability_sd": float(
                group["between_seed_probability_sd"].mean()
            ),
            "disease_behavior_r2": disease_r2,
            "disease_behavior_spearman": float(correlation),
            "full_behavior_r2": full_r2,
            "acquisition_incremental_r2": full_r2 - disease_r2,
            "acquisition_component_model_validated": bool(full_r2 - disease_r2 > 0),
            "mean_absolute_unexplained_logit": float(
                group["absolute_unexplained_model_logit"].mean()
            ),
            "empirical_residual_bound_coverage": float(
                group["explanation_covered"].mean()
            ),
            "black_box_explanation_usable": usable,
            "confidence_aurc_50_100": confidence_curve["aurc_50_100"],
            "explanation_aurc_50_100": explanation_curve["aurc_50_100"],
            "explanation_relative_aurc_reduction": relative_reduction,
            "explanation_minus_confidence_bootstrap_mean": bootstrap["mean"],
            "explanation_minus_confidence_ci95_low": bootstrap["ci95_low"],
            "explanation_minus_confidence_ci95_high": bootstrap["ci95_high"],
            "confidence_risk_at_80": confidence_curve["risk_at_80"],
            "explanation_risk_at_80": explanation_curve["risk_at_80"],
            "confidence_risk_at_90": confidence_curve["risk_at_90"],
            "explanation_risk_at_90": explanation_curve["risk_at_90"],
            "explanation_review_value_pass": decision_pass,
        }
        for state in CASE_STATE_ORDER:
            mask = group["reliability_state"].eq(state)
            row[f"fraction_{state.lower()}"] = float(mask.mean())
            row[f"error_rate_{state.lower()}"] = (
                float(group.loc[mask, "model_error"].mean()) if mask.any() else np.nan
            )
        aligned_error = row["error_rate_pathology_aligned"]
        conflict_error = row["error_rate_pathology_conflict"]
        row["pathology_conflict_vs_aligned_error_risk_ratio"] = (
            float(conflict_error / aligned_error)
            if np.isfinite(conflict_error)
            and np.isfinite(aligned_error)
            and aligned_error > 0
            else np.nan
        )
        profile_rows.append(row)
    return (
        pd.DataFrame(profile_rows).sort_values(["family", "configuration_id"]),
        pd.concat(risk_groups, ignore_index=True),
    )


def prepare_inputs(
    cohort_path: Path,
    assignments_path: Path,
    predictions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(cohort_path, low_memory=False, dtype={"RID": str})
    assignments = pd.read_csv(assignments_path, dtype={"RID": str})
    predictions = pd.read_csv(predictions_path, dtype={"RID": str})
    if cohort["RID"].isna().any() or cohort["RID"].duplicated().any():
        raise ValueError("Cohort RID must be complete and unique")
    if assignments["RID"].isna().any() or assignments["RID"].duplicated().any():
        raise ValueError("Domain assignments RID must be complete and unique")
    prediction_key = ["outer_domain", "family", "configuration_id", "RID"]
    if predictions.duplicated(prediction_key).any():
        raise ValueError("Predictions contain duplicate model-domain-subject keys")
    required_prediction = {
        *prediction_key,
        "AD_label",
        "mean_calibrated_probability_AD",
        "between_seed_calibrated_probability_sd",
    }
    missing = sorted(required_prediction - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction columns missing: {missing}")
    return cohort, assignments, predictions


def build_system(
    cohort: pd.DataFrame,
    assignments: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_rids = set(predictions["RID"].astype(str))
    target_domains = sorted(predictions["outer_domain"].unique())
    concept_rows = build_concept_and_pathology_rows(
        cohort, assignments, target_rids, target_domains
    )
    joined = predictions.rename(
        columns={
            "mean_calibrated_probability_AD": "model_probability_AD",
            "between_seed_calibrated_probability_sd": "between_seed_probability_sd",
        }
    ).merge(
        concept_rows,
        left_on=["RID", "outer_domain"],
        right_on=["_RID", "outer_domain"],
        validate="many_to_one",
    )
    if len(joined) != len(predictions):
        raise RuntimeError("Predictions and pathology concepts did not join completely")
    if not joined["AD_label"].astype(int).eq(joined["reference_label"].astype(int)).all():
        raise ValueError("Prediction and cohort labels disagree")
    joined["model_logit"] = logit(joined["model_probability_AD"])
    decomposed, acquisition_coefficients = decompose_black_box_behavior(joined)
    decomposed = apply_acquisition_validation_gate(decomposed)
    profiles, cases_with_risk = build_model_profiles(decomposed)
    pathology_profiles = build_pathology_reference_profiles(concept_rows)

    ordered_rids = sorted(cases_with_risk["_RID"].astype(str).unique(), key=lambda x: int(x))
    case_keys = {rid: f"CASE{index:04d}" for index, rid in enumerate(ordered_rids, 1)}
    cases_with_risk["case_key"] = cases_with_risk["_RID"].map(case_keys)
    contribution_columns = [
        f"{concept_id}_pathology_logodds_contribution" for concept_id in CORE_CONCEPTS
    ]
    severity_columns = [f"{concept_id}_severity" for concept_id in CORE_CONCEPTS]
    output_columns = [
        "case_key",
        "outer_domain",
        "family",
        "configuration_id",
        "reference_label",
        "model_probability_AD",
        "pathology_probability_AD",
        *severity_columns,
        *contribution_columns,
        "strongest_pathology_support",
        "strongest_pathology_counterevidence",
        "disease_behavior_centered_logit",
        "acquisition_behavior_logit",
        "unexplained_model_logit",
        "residual_bound_90",
        "residual_to_bound_ratio",
        "explanation_covered",
        "pathology_conflict",
        "acquisition_dominant",
        "acquisition_component_model_validated",
        "reliability_state",
        "between_seed_probability_sd",
        "model_error",
        "confidence_error_risk",
        "explanation_error_risk",
    ]
    cases = (
        cases_with_risk[output_columns]
        .sort_values(["family", "configuration_id", "outer_domain", "case_key"])
        .reset_index(drop=True)
    )
    assert_private_identifiers_absent(cases)
    assert_private_identifiers_absent(profiles)
    assert_private_identifiers_absent(acquisition_coefficients)
    return (
        cases,
        profiles.reset_index(drop=True),
        pathology_profiles,
        acquisition_coefficients,
    )


def receipt(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def build_summary(
    cases: pd.DataFrame,
    profiles: pd.DataFrame,
    pathology_profiles: pd.DataFrame,
    acquisition_coefficients: pd.DataFrame,
    receipts: dict[str, Any],
    protocol_receipt: dict[str, Any],
) -> dict[str, Any]:
    state_counts = {
        str(key): int(value)
        for key, value in cases["reliability_state"].value_counts().sort_index().items()
    }
    usable_count = int(profiles["black_box_explanation_usable"].sum())
    decision_count = int(profiles["explanation_review_value_pass"].sum())
    positive_acquisition = int((profiles["acquisition_incremental_r2"] > 0).sum())
    state_risk = {
        state: clustered_state_risk_ratio(cases, state)
        for state in CASE_STATE_ORDER
        if state != "PATHOLOGY_ALIGNED"
    }
    model_conflict_ratios = profiles[
        "pathology_conflict_vs_aligned_error_risk_ratio"
    ].dropna()
    pathology_join = cases[
        ["case_key", "outer_domain", "reference_label", "pathology_probability_AD"]
    ].drop_duplicates(["case_key", "outer_domain"])
    pathology_labels = pathology_join["reference_label"].to_numpy(dtype=int)
    pathology_probability = pathology_join["pathology_probability_AD"].to_numpy(
        dtype=float
    )
    return {
        "system_id": "PARES_AD_V1",
        "system_name": (
            "Pathology-Acquisition Reliability Explanation System for Alzheimer's MRI"
        ),
        "scope": "AD_vs_CN_T1_FreeSurfer_within_ADNI_acquisition_domains",
        "inputs": receipts,
        "protocol": protocol_receipt,
        "data_receipt": {
            "anonymized_unique_cases": int(cases["case_key"].nunique()),
            "case_model_explanations": int(len(cases)),
            "model_configurations": int(len(profiles)),
            "model_families": int(profiles["family"].nunique()),
            "outer_domains": int(cases["outer_domain"].nunique()),
        },
        "transparent_pathology_component": {
            "concepts": [CONCEPT_LABELS[concept] for concept in CORE_CONCEPTS],
            "exact_additive_logodds_decomposition": True,
            "monotone_nonnegative_coefficients": True,
            "global_outer_domain_auroc": float(
                roc_auc_score(pathology_labels, pathology_probability)
            ),
            "global_outer_domain_auprc": float(
                average_precision_score(pathology_labels, pathology_probability)
            ),
            "global_outer_domain_brier": float(
                brier_score_loss(pathology_labels, pathology_probability)
            ),
            "domain_profile_rows": int(len(pathology_profiles)),
        },
        "behavioral_decomposition": {
            "usable_model_configurations": usable_count,
            "unusable_model_configurations": int(len(profiles) - usable_count),
            "configurations_with_positive_acquisition_incremental_r2": positive_acquisition,
            "state_counts": state_counts,
            "mean_full_behavior_r2": float(profiles["full_behavior_r2"].mean()),
            "mean_disease_behavior_r2": float(profiles["disease_behavior_r2"].mean()),
            "mean_empirical_bound_coverage": float(
                profiles["empirical_residual_bound_coverage"].mean()
            ),
            "acquisition_coefficient_rows": int(len(acquisition_coefficients)),
            "state_error_risk_vs_pathology_aligned": state_risk,
            "pathology_conflict_higher_error_in_configurations": int(
                (
                    profiles["error_rate_pathology_conflict"]
                    > profiles["error_rate_pathology_aligned"]
                ).sum()
            ),
            "median_configuration_pathology_conflict_risk_ratio": float(
                model_conflict_ratios.median()
            ),
            "configuration_pathology_conflict_risk_ratio_range": [
                float(model_conflict_ratios.min()),
                float(model_conflict_ratios.max()),
            ],
        },
        "decision_value": {
            "configurations_passing_all_frozen_rules": decision_count,
            "configurations_failing_or_not_passing": int(len(profiles) - decision_count),
            "human_benefit_claim_permitted": False,
        },
        "privacy": {
            "participant_identifiers_written": False,
            "case_keys_have_no_mapping_artifact": True,
        },
        "probability_update": {
            "system_implementation_complete": 1.0,
            "single_disease_empirical_framework_value": {
                "lower": 0.55 if usable_count >= 5 else 0.30,
                "upper": 0.75 if usable_count >= 5 else 0.50,
            },
            "single_disease_hiss_competitiveness": {
                "lower": 0.35
                if decision_count >= 1
                else 0.30
                if state_risk["PATHOLOGY_CONFLICT"][
                    "cluster_bootstrap_ci95_low"
                ]
                > 1
                else 0.20,
                "upper": 0.55
                if decision_count >= 1
                else 0.45
                if state_risk["PATHOLOGY_CONFLICT"][
                    "cluster_bootstrap_ci95_low"
                ]
                > 1
                else 0.35,
            },
        },
        "claim_boundary": {
            "allowed": (
                "Patient-level AD pathology support/counterevidence is exact for the "
                "transparent reference; black-box disease/acquisition decomposition "
                "is a domain-held-out functional reconstruction with abstention."
            ),
            "not_allowed": (
                "Causal internal mechanism, biological AD diagnosis, scanner causation, "
                "clinical benefit, or cross-disease generalization."
            ),
        },
    }


def render_report(summary: dict[str, Any], profiles: pd.DataFrame) -> str:
    receipt_data = summary["data_receipt"]
    behavior = summary["behavioral_decomposition"]
    decision = summary["decision_value"]
    pathology = summary["transparent_pathology_component"]
    conflict_risk = behavior["state_error_risk_vs_pathology_aligned"][
        "PATHOLOGY_CONFLICT"
    ]
    usable = profiles[profiles["black_box_explanation_usable"]]
    usable_names = (
        ", ".join(
            f"{row.family}/{row.configuration_id}"
            for row in usable.itertuples(index=False)
        )
        if len(usable)
        else "none"
    )
    decision_models = profiles[profiles["explanation_review_value_pass"]]
    decision_names = (
        ", ".join(
            f"{row.family}/{row.configuration_id}"
            for row in decision_models.itertuples(index=False)
        )
        if len(decision_models)
        else "none"
    )
    return f"""# PARES-AD: AD/MRI pathology–acquisition explanation and reliability system

## Result

The implemented system produced {receipt_data['case_model_explanations']} anonymized case–model explanations for {receipt_data['anonymized_unique_cases']} ADNI subjects across {receipt_data['model_configurations']} model configurations, {receipt_data['model_families']} architecture families, and {receipt_data['outer_domains']} held-out acquisition domains.

This is an AD/MRI explanation system, not the earlier evidence certificate. Each case contains exact signed contributions from hippocampal atrophy, entorhinal thinning, ventricular enlargement, and temporoparietal thinning in a transparent monotone pathology model. The black-box prediction is then separately reconstructed into disease-associated behavior, acquisition-associated behavior, and an unclaimed residual.

The transparent pathology reference achieved outer-domain AUROC {pathology['global_outer_domain_auroc']:.4f}, AUPRC {pathology['global_outer_domain_auprc']:.4f}, and Brier {pathology['global_outer_domain_brier']:.4f}.

## What the system can answer

For a given anonymized case and model it reports:

- strongest AD pathology support and strongest counterevidence;
- transparent pathology probability versus black-box probability;
- acquisition-associated log-odds contribution after disease evidence;
- unexplained residual and whether it exceeds the frozen residual bound;
- one of `PATHOLOGY_ALIGNED`, `PATHOLOGY_CONFLICT`, `ACQUISITION_SENSITIVE`, `ACQUISITION_UNVALIDATED_ABSTAIN`, or `UNEXPLAINED_ABSTAIN`.

## Empirical system validity

- Usable domain-held-out black-box reconstructions: {behavior['usable_model_configurations']}/{receipt_data['model_configurations']}.
- Mean disease-only behavior R²: {behavior['mean_disease_behavior_r2']:.4f}.
- Mean full disease-plus-acquisition behavior R²: {behavior['mean_full_behavior_r2']:.4f}.
- Mean empirical residual-bound coverage: {behavior['mean_empirical_bound_coverage']:.3f}.
- Models with positive acquisition incremental R²: {behavior['configurations_with_positive_acquisition_incremental_r2']}/{receipt_data['model_configurations']}.
- Reliability states: {json.dumps(behavior['state_counts'], sort_keys=True)}.

`PATHOLOGY_CONFLICT` cases had error rate {conflict_risk['state_error_rate']:.3f} versus {conflict_risk['reference_error_rate']:.3f} for `PATHOLOGY_ALIGNED`, a clustered risk ratio of {conflict_risk['risk_ratio']:.2f} (95% subject-cluster bootstrap interval {conflict_risk['cluster_bootstrap_ci95_low']:.2f}–{conflict_risk['cluster_bootstrap_ci95_high']:.2f}). The conflict group had higher error in {behavior['pathology_conflict_higher_error_in_configurations']}/{receipt_data['model_configurations']} configurations; configuration-specific risk ratios ranged from {behavior['configuration_pathology_conflict_risk_ratio_range'][0]:.2f} to {behavior['configuration_pathology_conflict_risk_ratio_range'][1]:.2f}.

Models passing the frozen reconstruction gate: {usable_names}.

## Review/deferral value

Explanation-aware review passed all frozen incremental selective-risk rules for {decision['configurations_passing_all_frozen_rules']}/{receipt_data['model_configurations']} configurations: {decision_names}.

This endpoint is normative and offline. It does not show that clinicians benefited. Models that fail remain valid descriptive pathology comparisons where stated, but cannot be called decision-improving.

## Practical value

The system solves a concrete single-disease problem: it prevents an AD probability from being presented as a unitary “AD explanation.” It distinguishes anatomical support, anatomical counterevidence, acquisition-associated behavior, and unexplained behavior, and it refuses black-box explanations that do not transport to a held-out acquisition domain.

## Boundaries

The pathology contribution is exact for the transparent reference model. The black-box decomposition is a functional surrogate, not a causal account of hidden representations. Acquisition association is not proof that changing a scanner would change the prediction. No clinical deployment or cross-disease claim is made.
"""


def parse_args() -> argparse.Namespace:
    v2 = Path(__file__).resolve().parents[2]
    hiss = v2.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=hiss / "data" / "processed" / "baseline_cohort_with_concepts.csv",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=v2 / "outputs" / "private" / "domain_assignments.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=v2
        / "outputs"
        / "private"
        / "fw5"
        / "aggregation"
        / "fw5_outer_endpoints_v2_1"
        / "configuration_outer_predictions.csv",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=v2
        / "protocols"
        / "ad_mri_explanation_reliability_system_freeze_2026-07-23.md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=v2
        / "outputs"
        / "private"
        / "ad_mri_explanation_system_2026-07-23",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort, assignments, predictions = prepare_inputs(
        args.cohort, args.assignments, args.predictions
    )
    cases, profiles, pathology_profiles, acquisition_coefficients = build_system(
        cohort, assignments, predictions
    )
    receipts = {
        "cohort": receipt(args.cohort, len(cohort)),
        "domain_assignments": receipt(args.assignments, len(assignments)),
        "outer_predictions": receipt(args.predictions, len(predictions)),
    }
    summary = build_summary(
        cases,
        profiles,
        pathology_profiles,
        acquisition_coefficients,
        receipts,
        receipt(args.protocol),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(
        args.output_dir / "case_explanations.csv", index=False, lineterminator="\n"
    )
    profiles.to_csv(
        args.output_dir / "model_reliability_profiles.csv",
        index=False,
        lineterminator="\n",
    )
    pathology_profiles.to_csv(
        args.output_dir / "pathology_reference_profiles.csv",
        index=False,
        lineterminator="\n",
    )
    with (args.output_dir / "system_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    (args.output_dir / "report.md").write_text(
        render_report(summary, profiles), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "case_model_explanations": len(cases),
                "model_configurations": len(profiles),
                "usable_reconstructions": int(
                    profiles["black_box_explanation_usable"].sum()
                ),
                "decision_value_passes": int(
                    profiles["explanation_review_value_pass"].sum()
                ),
                "output_dir": str(args.output_dir.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
