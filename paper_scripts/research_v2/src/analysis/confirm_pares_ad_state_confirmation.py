"""Outcome-blind confirmation of the frozen PARES-AD evidence-state contract."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_v2.src.analysis.build_ad_mri_explanation_system import (
    CORE_CONCEPTS,
    CONCEPT_LABELS,
    CONCEPT_ORIENTATION,
    MANUFACTURER_LEVELS,
    PHASE_LEVELS,
    FIELD_LEVELS,
    apply_acquisition_validation_gate,
    build_concept_and_pathology_rows,
    build_model_profiles,
    clustered_state_risk_ratio,
    conformal_absolute_quantile,
    decompose_black_box_behavior,
    fit_monotone_pathology_model,
    logit,
    nested_training_residuals,
    reliability_state,
    prepare_inputs,
)
from research_v2.src.analysis.confirm_pares_ad_independent import (
    ASSIGNMENTS,
    BASELINE,
    BASELINE_CONCEPTS,
    DXSUM,
    MANIFEST,
    MAPPING,
    RAW_FS,
    REGISTRY,
    JOBS_ROOT,
    run_black_boxes,
    build_confirmation_cohort,
    sha256_file,
)
from research_v2.src.audit.concepts import build_concept_target


ROOT = Path(__file__).resolve().parents[3]
V2_ROOT = ROOT / "research_v2"
DEV_PREDICTIONS = V2_ROOT / "outputs" / "private" / "fw5" / "aggregation" / "fw5_outer_endpoints_v2_1" / "configuration_outer_predictions.csv"
PROTOCOL = V2_ROOT / "protocols" / "pares_ad_state_confirmation_freeze_2026-09-27.md"
OUTPUT = V2_ROOT / "outputs" / "private" / "pares_ad_state_confirmation_2026-09-27"
STATES = [
    "PATHOLOGY_ALIGNED",
    "PATHOLOGY_CONFLICT",
    "UNEXPLAINED_ABSTAIN",
    "ACQUISITION_SENSITIVE",
    "ACQUISITION_UNVALIDATED_ABSTAIN",
]


def load_development_joined() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort, assignments, predictions = prepare_inputs(
        BASELINE_CONCEPTS, ASSIGNMENTS, DEV_PREDICTIONS
    )
    concept_rows = build_concept_and_pathology_rows(
        cohort,
        assignments,
        set(predictions["RID"].astype(str)),
        sorted(predictions["outer_domain"].unique()),
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
    joined["model_logit"] = logit(joined["model_probability_AD"])
    joined["model_predicted_class"] = joined["model_probability_AD"].ge(0.5).astype(int)
    joined["pathology_predicted_class"] = joined["pathology_probability_AD"].ge(0.5).astype(int)
    joined["pathology_conflict"] = joined["model_predicted_class"].ne(
        joined["pathology_predicted_class"]
    )
    joined["model_error"] = joined["model_predicted_class"].ne(joined["AD_label"]).astype(int)
    return joined, cohort, assignments, predictions


def confirmation_concept_rows(
    confirmation: pd.DataFrame,
    baseline: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    baseline = baseline.copy()
    baseline["RID"] = baseline["RID"].astype(str)
    assignments = assignments.copy()
    assignments["RID"] = assignments["RID"].astype(str)
    merged = baseline.merge(
        assignments[
            ["RID", "phase", "manufacturer", "field_strength", "acquisition_domain", "label"]
        ],
        on="RID",
        validate="one_to_one",
    )
    rows: list[pd.DataFrame] = []
    for domain, evaluation in confirmation.groupby("outer_domain", sort=True):
        train = merged[~merged["acquisition_domain"].eq(domain)].reset_index(drop=True)
        evaluation = evaluation.reset_index(drop=True)
        train_values: list[np.ndarray] = []
        evaluation_values: list[np.ndarray] = []
        for concept_id in CORE_CONCEPTS:
            target = build_concept_target(concept_id, train, evaluation)
            if not target.train_valid.all() or not target.evaluation_valid.all():
                raise RuntimeError(f"Invalid confirmation concept rows for {domain}/{concept_id}")
            orientation = CONCEPT_ORIENTATION[concept_id]
            train_values.append(orientation * target.train_values.astype(float))
            evaluation_values.append(orientation * target.evaluation_values.astype(float))
        x_train = np.column_stack(train_values)
        x_eval = np.column_stack(evaluation_values)
        intercept, coefficients = fit_monotone_pathology_model(
            x_train, train["AD_label"].to_numpy(dtype=int), l2_penalty=0.1
        )
        contributions = x_eval * coefficients
        pathology_logit = intercept + contributions.sum(axis=1)
        frame = pd.DataFrame(
            {
                "_RID": evaluation["_RID"].astype(str).to_numpy(),
                "outer_domain": domain,
                "phase": evaluation["phase"].astype(str).to_numpy(),
                "manufacturer": evaluation["manufacturer"].astype(str).to_numpy(),
                "field_strength": evaluation["field_strength"].astype(str).to_numpy(),
                "pathology_probability_AD": expit(pathology_logit),
            }
        )
        frame["pathology_intercept"] = float(intercept)
        for index, concept_id in enumerate(CORE_CONCEPTS):
            frame[f"{concept_id}_severity"] = x_eval[:, index]
            frame[f"{concept_id}_pathology_logodds_contribution"] = contributions[:, index]
            frame[f"{concept_id}_pathology_coefficient"] = coefficients[index]
        contribution_columns = [
            f"{concept_id}_pathology_logodds_contribution" for concept_id in CORE_CONCEPTS
        ]
        contribution_matrix = frame[contribution_columns].to_numpy(dtype=float)
        frame["strongest_pathology_support"] = [
            CONCEPT_LABELS[CORE_CONCEPTS[index]] for index in np.argmax(contribution_matrix, axis=1)
        ]
        frame["strongest_pathology_counterevidence"] = [
            CONCEPT_LABELS[CORE_CONCEPTS[index]] for index in np.argmin(contribution_matrix, axis=1)
        ]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def acquisition_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = []
    for source, levels in [
        ("phase", PHASE_LEVELS),
        ("manufacturer", MANUFACTURER_LEVELS),
        ("field_strength", FIELD_LEVELS),
    ]:
        values = frame[source].astype(str).to_numpy()
        if set(values) - set(levels):
            raise RuntimeError(f"Unknown acquisition levels in {source}")
        columns.extend((values == level).astype(float) for level in levels)
    return np.column_stack(columns)


def fit_behavior_bundle(train: pd.DataFrame) -> dict[str, object]:
    disease_columns = [f"{concept}_severity" for concept in CORE_CONCEPTS]
    x_disease = train[disease_columns].to_numpy(dtype=float)
    y = train["model_logit"].to_numpy(dtype=float)
    disease_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    disease_model.fit(x_disease, y)
    disease_train = disease_model.predict(x_disease)
    acquisition_model = Ridge(alpha=1.0)
    acquisition_model.fit(acquisition_matrix(train), y - disease_train)
    return {
        "disease_model": disease_model,
        "acquisition_model": acquisition_model,
        "training_mean_logit": float(y.mean()),
        "bound": float(conformal_absolute_quantile(nested_training_residuals(train), 0.90)),
    }


def apply_bundle(bundle: dict[str, object], frame: pd.DataFrame) -> pd.DataFrame:
    disease_columns = [f"{concept}_severity" for concept in CORE_CONCEPTS]
    x_disease = frame[disease_columns].to_numpy(dtype=float)
    disease = np.asarray(bundle["disease_model"].predict(x_disease), dtype=float)
    acquisition = np.asarray(
        bundle["acquisition_model"].predict(acquisition_matrix(frame)), dtype=float
    )
    reconstructed = disease + acquisition
    residual = frame["model_logit"].to_numpy(dtype=float) - reconstructed
    bound = float(bundle["bound"])
    output = frame.copy()
    output["disease_behavior_centered_logit"] = disease - float(bundle["training_mean_logit"])
    output["acquisition_behavior_logit"] = acquisition
    output["residual_to_bound_ratio"] = np.abs(residual) / max(bound, 1e-12)
    output["explanation_covered"] = np.abs(residual) <= bound
    return output


def assign_confirmation_states(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
    acquisition_validity: dict[tuple[str, str], bool],
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for (family, configuration_id), dev_config in development.groupby(
        ["family", "configuration_id"], sort=True
    ):
        conf_config = confirmation[
            confirmation["family"].eq(family)
            & confirmation["configuration_id"].eq(configuration_id)
        ]
        for domain, dev_domain in dev_config.groupby("outer_domain", sort=True):
            conf_domain = conf_config[conf_config["outer_domain"].eq(domain)].copy()
            if conf_domain.empty:
                continue
            bundle = fit_behavior_bundle(dev_config[~dev_config["outer_domain"].eq(domain)])
            scored = apply_bundle(bundle, conf_domain)
            states: list[str] = []
            for record in scored[
                [
                    "explanation_covered",
                    "acquisition_behavior_logit",
                    "disease_behavior_centered_logit",
                    "pathology_conflict",
                ]
            ].to_dict("records"):
                state, _ = reliability_state(
                    bool(record["explanation_covered"]),
                    float(record["acquisition_behavior_logit"]),
                    float(record["disease_behavior_centered_logit"]),
                    bool(record["pathology_conflict"]),
                )
                states.append(state)
            scored["reliability_state"] = states
            if not acquisition_validity[(family, configuration_id)]:
                mask = scored["reliability_state"].eq("ACQUISITION_SENSITIVE")
                scored.loc[mask, "reliability_state"] = "ACQUISITION_UNVALIDATED_ABSTAIN"
            groups.append(scored)
    output = pd.concat(groups, ignore_index=True)
    return output


def development_acquisition_validity(development: pd.DataFrame) -> dict[tuple[str, str], bool]:
    decomposed, _ = decompose_black_box_behavior(development)
    profiles, _ = build_model_profiles(apply_acquisition_validation_gate(decomposed))
    return {
        (str(row.family), str(row.configuration_id)): bool(row.acquisition_component_model_validated)
        for row in profiles.itertuples()
    }


def summarize_states(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for state in STATES:
        relevant = frame[frame["reliability_state"].isin([state, "PATHOLOGY_ALIGNED"])].copy()
        risk = clustered_state_risk_ratio(
            relevant.assign(case_key=relevant["_RID"].astype(str)), state
        )
        rows.append(
            {
                "state": state,
                "n_predictions": int((frame["reliability_state"] == state).sum()),
                "n_participants": int(frame.loc[frame["reliability_state"] == state, "_RID"].nunique()),
                "fraction": float((frame["reliability_state"] == state).mean()),
                **risk,
            }
        )
    state_results = pd.DataFrame(rows)

    direction_rows: list[dict[str, object]] = []
    for level in ["configuration_id", "family", "outer_domain"]:
        for value, group in frame.groupby(level, sort=True):
            for state in ["PATHOLOGY_CONFLICT", "UNEXPLAINED_ABSTAIN"]:
                risk = clustered_state_risk_ratio(
                    group.assign(case_key=group["_RID"].astype(str)), state
                )
                direction_rows.append(
                    {
                        "level": level,
                        "value": str(value),
                        "state": state,
                        "risk_ratio": risk["risk_ratio"],
                        "state_n": risk["state_n"],
                        "reference_n": risk["reference_n"],
                    }
                )
    return state_results, pd.DataFrame(direction_rows)


def summarize_concept_specific_states(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate which AD-pattern contribution is the strongest counterevidence.

    The output is aggregate only.  Participant-level rows remain in memory and
    are never written to the confirmation output directory.
    """
    rows: list[dict[str, object]] = []
    labels = sorted(set(CONCEPT_LABELS.values()))
    base = frame.copy()
    base["case_key"] = base["_RID"].astype(str)
    for state in ["PATHOLOGY_CONFLICT", "UNEXPLAINED_ABSTAIN"]:
        for label in labels:
            relevant = base[base["reliability_state"].isin([state, "PATHOLOGY_ALIGNED"])].copy()
            marker = f"{state}::{label}"
            relevant["reliability_state"] = np.where(
                relevant["reliability_state"].eq(state)
                & relevant["strongest_pathology_counterevidence"].eq(label),
                marker,
                "PATHOLOGY_ALIGNED",
            )
            risk = clustered_state_risk_ratio(relevant, marker)
            rows.append(
                {
                    "state": state,
                    "counterevidence": label,
                    **risk,
                }
            )
    return pd.DataFrame(rows)


def summarize_confidence_state_interaction(frame: pd.DataFrame) -> pd.DataFrame:
    """Test whether pathology conflict adds risk within confidence strata.

    This is a CDSS-oriented complementarity analysis: confidence defines how
    surprising the model output is, while the frozen PARES state defines a
    structural reason for concern.  Only aggregate rows are written.
    """
    scored = frame.copy()
    probability = scored["model_probability_AD"].to_numpy(dtype=float)
    scored["confidence_stratum"] = np.select(
        [
            (probability <= 0.10) | (probability >= 0.90),
            ((probability > 0.10) & (probability <= 0.25))
            | ((probability >= 0.75) & (probability < 0.90)),
        ],
        ["HIGH_CONFIDENCE", "INTERMEDIATE_CONFIDENCE"],
        default="LOW_CONFIDENCE",
    )
    rows: list[dict[str, object]] = []
    for stratum in [
        "HIGH_CONFIDENCE",
        "INTERMEDIATE_CONFIDENCE",
        "LOW_CONFIDENCE",
    ]:
        subset = scored[
            scored["confidence_stratum"].eq(stratum)
            & scored["reliability_state"].isin(
                ["PATHOLOGY_ALIGNED", "PATHOLOGY_CONFLICT"]
            )
        ].copy()
        subset["case_key"] = subset["_RID"].astype(str)
        risk = clustered_state_risk_ratio(subset, "PATHOLOGY_CONFLICT")
        rows.append(
            {
                "confidence_stratum": stratum,
                "n_predictions": int(len(subset)),
                "n_participants": int(subset["_RID"].nunique()),
                **risk,
            }
        )
    return pd.DataFrame(rows)


def summarize_high_confidence_conflict_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    """Break down high-confidence structural conflicts by counterevidence."""
    probability = frame["model_probability_AD"].to_numpy(dtype=float)
    high = frame[
        ((probability <= 0.10) | (probability >= 0.90))
        & frame["reliability_state"].eq("PATHOLOGY_CONFLICT")
    ].copy()
    grouped = (
        high.groupby("strongest_pathology_counterevidence", sort=True)
        .agg(
            n_predictions=("model_error", "size"),
            errors=("model_error", "sum"),
            n_participants=("_RID", "nunique"),
        )
        .reset_index()
    )
    grouped["error_rate"] = grouped["errors"] / grouped["n_predictions"]
    return grouped


def outcome_blind_ranking_summary(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
    acquisition_validity: dict[tuple[str, str], bool],
) -> dict[str, object]:
    """Fit a development-only state/reason error lookup and test it盲态."""
    dev_input = development.drop(columns=["AD_label"]).copy()
    dev_assigned = assign_confirmation_states(development, dev_input, acquisition_validity)
    dev_outcome = development[["_RID", "outer_domain", "family", "configuration_id", "AD_label"]].copy()
    dev_assigned = dev_assigned.merge(
        dev_outcome,
        on=["_RID", "outer_domain", "family", "configuration_id"],
        validate="one_to_one",
    )
    dev_assigned["model_error"] = dev_assigned["model_predicted_class"].ne(dev_assigned["AD_label"]).astype(int)
    dev_assigned["reason_key"] = (
        dev_assigned["reliability_state"].astype(str)
        + "::"
        + dev_assigned["strongest_pathology_counterevidence"].astype(str)
    )
    lookup = dev_assigned.groupby("reason_key", sort=True)["model_error"].agg(["sum", "count"])
    lookup["smoothed_error_rate"] = (lookup["sum"] + 0.5) / (lookup["count"] + 1.0)

    scored = confirmation.copy()
    scored["reason_key"] = (
        scored["reliability_state"].astype(str)
        + "::"
        + scored["strongest_pathology_counterevidence"].astype(str)
    )
    global_rate = float(dev_assigned["model_error"].mean())
    scored["evidence_risk_score"] = scored["reason_key"].map(lookup["smoothed_error_rate"]).fillna(global_rate)
    scored["confidence_risk_score"] = -np.abs(scored["model_probability_AD"].to_numpy(dtype=float) - 0.5)
    scored["conflict_risk_score"] = scored["pathology_conflict"].astype(float)
    budgets = (
        scored.groupby("configuration_id", sort=True)["pathology_conflict"]
        .sum()
        .astype(int)
        .to_dict()
    )

    def capture(column: str) -> dict[str, float | int]:
        selected: list[pd.DataFrame] = []
        for configuration_id, group in scored.groupby("configuration_id", sort=True):
            budget = int(budgets[configuration_id])
            ranked = group.sort_values(
                [column, "_RID"], ascending=[False, True]
            )
            selected.append(ranked.head(budget))
        top = pd.concat(selected, ignore_index=True)
        return {
            "review_n": int(len(top)),
            "captured_errors": int(top["model_error"].sum()),
            "yield": float(top["model_error"].mean()),
            "total_errors": int(scored["model_error"].sum()),
            "score_source": column,
        }

    result = {
        "review_fraction": float(sum(budgets.values()) / len(scored)),
        "confirmation_predictions": int(len(scored)),
        "budget_rule": "within each configuration, review the same number of predictions as the raw pathology-conflict rule",
        "development_lookup_cells": int(len(lookup)),
        "development_only_score": True,
        "methods": {
            "evidence_risk": "Laplace-smoothed error rate by frozen reliability state and strongest AD counterevidence, fit on development only",
            "confidence_risk": "negative absolute distance from probability 0.5",
            "conflict_risk": "PATHOLOGY_CONFLICT indicator with deterministic tie break",
        },
        "rankings": [
            capture("evidence_risk_score"),
            capture("conflict_risk_score"),
            capture("confidence_risk_score"),
        ],
    }
    high_confidence = scored[
        scored["model_probability_AD"].le(0.10)
        | scored["model_probability_AD"].ge(0.90)
    ].copy()
    hc_rows: list[dict[str, object]] = []
    for state in [
        "PATHOLOGY_ALIGNED",
        "PATHOLOGY_CONFLICT",
        "UNEXPLAINED_ABSTAIN",
        "ACQUISITION_SENSITIVE",
        "ACQUISITION_UNVALIDATED_ABSTAIN",
    ]:
        part = high_confidence[high_confidence["reliability_state"].eq(state)]
        hc_rows.append(
            {
                "state": state,
                "n_predictions": int(len(part)),
                "errors": int(part["model_error"].sum()),
                "error_rate": float(part["model_error"].mean()) if len(part) else float("nan"),
            }
        )
    result["high_confidence_subgroup"] = {
        "definition": "model probability <= 0.10 or >= 0.90",
        "n_predictions": int(len(high_confidence)),
        "rows": hc_rows,
    }
    return result


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    development, _, _, _ = load_development_joined()
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST, dtype={"seed": int})
    confirmation = build_confirmation_cohort(mapping)
    confirmation_predictions = run_black_boxes(confirmation, manifest, mapping, registry)
    confirmation_concepts = confirmation_concept_rows(
        confirmation,
        pd.read_csv(BASELINE_CONCEPTS, low_memory=False, dtype={"RID": str}),
        pd.read_csv(ASSIGNMENTS, dtype={"RID": str}),
    )
    confirmation_joined = confirmation_predictions.merge(
        confirmation_concepts,
        left_on=["_RID", "outer_domain"],
        right_on=["_RID", "outer_domain"],
        validate="many_to_one",
    )
    confirmation_joined["model_logit"] = logit(confirmation_joined["model_probability_AD"])
    confirmation_joined["model_predicted_class"] = confirmation_joined["model_probability_AD"].ge(0.5).astype(int)
    confirmation_joined["pathology_predicted_class"] = confirmation_joined["pathology_probability_AD"].ge(0.5).astype(int)
    confirmation_joined["pathology_conflict"] = confirmation_joined["model_predicted_class"].ne(
        confirmation_joined["pathology_predicted_class"]
    )

    # State assignment is deliberately performed after dropping the confirmation outcome label.
    outcome = confirmation_joined[
        ["_RID", "outer_domain", "family", "configuration_id", "AD_label"]
    ].copy()
    state_input = confirmation_joined.drop(columns=["AD_label"]).copy()
    validity = development_acquisition_validity(development)
    assigned = assign_confirmation_states(development, state_input, validity)
    assigned = assigned.merge(
        outcome,
        on=["_RID", "outer_domain", "family", "configuration_id"],
        validate="one_to_one",
    )
    assigned["model_error"] = assigned["model_predicted_class"].ne(assigned["AD_label"]).astype(int)
    state_results, direction_results = summarize_states(assigned)
    concept_state_results = summarize_concept_specific_states(assigned)
    confidence_state_results = summarize_confidence_state_interaction(assigned)
    high_confidence_reason_results = summarize_high_confidence_conflict_reasons(assigned)
    ranking_results = outcome_blind_ranking_summary(development, assigned, validity)

    state_results.to_csv(OUTPUT / "state_results.csv", index=False, lineterminator="\n")
    direction_results.to_csv(OUTPUT / "state_direction_results.csv", index=False, lineterminator="\n")
    concept_state_results.to_csv(
        OUTPUT / "state_concept_results.csv", index=False, lineterminator="\n"
    )
    confidence_state_results.to_csv(
        OUTPUT / "confidence_state_interaction.csv", index=False, lineterminator="\n"
    )
    high_confidence_reason_results.to_csv(
        OUTPUT / "high_confidence_conflict_reasons.csv", index=False, lineterminator="\n"
    )
    source_paths = [
        RAW_FS, DXSUM, BASELINE, BASELINE_CONCEPTS, ASSIGNMENTS, MANIFEST,
        MAPPING, REGISTRY, DEV_PREDICTIONS, PROTOCOL,
    ]
    summary = {
        "analysis": "PARES-AD-STATE-CONFIRM-v1",
        "status": "COMPLETED",
        "data_receipt": {
            "development_participants": int(development["RID"].nunique()),
            "development_predictions": int(len(development)),
            "confirmation_participants": int(assigned["_RID"].nunique()),
            "confirmation_predictions": int(len(assigned)),
            "configurations": int(assigned["configuration_id"].nunique()),
            "families": int(assigned["family"].nunique()),
            "outer_domains": int(assigned["outer_domain"].nunique()),
        },
        "state_results": state_results.to_dict(orient="records"),
        "concept_specific_state_results": concept_state_results.to_dict(orient="records"),
        "confidence_state_interaction": confidence_state_results.to_dict(orient="records"),
        "high_confidence_conflict_reasons": high_confidence_reason_results.to_dict(orient="records"),
        "outcome_blind_ranking": ranking_results,
        "state_assignment": {
            "outcome_blind": True,
            "acquisition_validity_from_development_only": True,
            "states": STATES,
        },
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "privacy": {"participant_identifiers_written": False, "participant_level_rows_written": False},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "outcome_blind_ranking.json").write_text(
        json.dumps(ranking_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_table = [
        "| state | n_predictions | n_participants | fraction | state_error_rate | reference_error_rate | risk_ratio | ci95_low | ci95_high |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in state_results.to_dict(orient="records"):
        report_table.append(
            "| {state} | {n_predictions} | {n_participants} | {fraction:.4f} | {state_error_rate:.4f} | {reference_error_rate:.4f} | {risk_ratio:.4f} | {cluster_bootstrap_ci95_low:.4f} | {cluster_bootstrap_ci95_high:.4f} |".format(**row)
        )
    report_lines = [
        "# PARES-AD evidence-state confirmation",
        "",
        "State assignment used the development-fitted reference, behavior reconstruction, residual bound and acquisition-validation gate before confirmation labels were joined for error evaluation.",
        "",
        *report_table,
        "",
        "The pathology-conflict state is the primary confirmation state. The two acquisition states remain descriptive and cannot be interpreted as mechanisms unless their development validity gate is true.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "output": str(OUTPUT), "runtime_seconds": summary["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
