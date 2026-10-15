"""Development-fitted test of PARES-AD error-risk increment beyond confidence.

The analysis fits three frozen risk models on the development cohort and
evaluates them once on the development-isolated 882-participant confirmation
cohort.  Confirmation outcomes are used only for final metric computation.
Only aggregate outputs are written.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_v2.src.analysis.confirm_pares_ad_independent import (
    ASSIGNMENTS,
    BASELINE_CONCEPTS,
    MANIFEST,
    MAPPING,
    REGISTRY,
    build_confirmation_cohort,
    run_black_boxes,
)
from research_v2.src.analysis.confirm_pares_ad_state_confirmation import (
    CORE_CONCEPTS,
    OUTPUT as STATE_OUTPUT,
    STATES,
    assign_confirmation_states,
    confirmation_concept_rows,
    development_acquisition_validity,
    load_development_joined,
    logit,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "research_v2" / "outputs" / "private" / "pares_ad_error_risk_increment_2026-09-28"
RANDOM_SEED = 20260928
BOOTSTRAP_REPLICATES = 2_000
CONCEPT_LABELS = [
    "entorhinal_thinning",
    "hippocampal_atrophy",
    "temporoparietal_thinning",
    "ventricular_enlargement",
]
MODEL_ORDER = [
    "confidence_only",
    "confidence_plus_conflict",
    "confidence_plus_state_reason",
]


def assign_development_states(
    development: pd.DataFrame,
    validity: dict[tuple[str, str], bool],
) -> pd.DataFrame:
    state_input = development.drop(columns=["AD_label"]).copy()
    assigned = assign_confirmation_states(development, state_input, validity)
    outcome = development[
        ["_RID", "outer_domain", "family", "configuration_id", "AD_label"]
    ].copy()
    assigned = assigned.merge(
        outcome,
        on=["_RID", "outer_domain", "family", "configuration_id"],
        validate="one_to_one",
    )
    assigned["model_error"] = assigned["model_predicted_class"].ne(
        assigned["AD_label"]
    ).astype(int)
    return assigned


def build_confirmation_assigned(
    development: pd.DataFrame,
    validity: dict[tuple[str, str], bool],
) -> pd.DataFrame:
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST, dtype={"seed": int})
    confirmation = build_confirmation_cohort(mapping)
    predictions = run_black_boxes(confirmation, manifest, mapping, registry)
    concepts = confirmation_concept_rows(
        confirmation,
        pd.read_csv(BASELINE_CONCEPTS, low_memory=False, dtype={"RID": str}),
        pd.read_csv(ASSIGNMENTS, dtype={"RID": str}),
    )
    joined = predictions.merge(
        concepts,
        on=["_RID", "outer_domain"],
        validate="many_to_one",
    )
    joined["model_logit"] = logit(joined["model_probability_AD"])
    joined["model_predicted_class"] = joined["model_probability_AD"].ge(0.5).astype(int)
    joined["pathology_predicted_class"] = joined["pathology_probability_AD"].ge(0.5).astype(int)
    joined["pathology_conflict"] = joined["model_predicted_class"].ne(
        joined["pathology_predicted_class"]
    )
    outcome = joined[
        ["_RID", "outer_domain", "family", "configuration_id", "AD_label"]
    ].copy()
    assigned = assign_confirmation_states(
        development,
        joined.drop(columns=["AD_label"]).copy(),
        validity,
    )
    assigned = assigned.merge(
        outcome,
        on=["_RID", "outer_domain", "family", "configuration_id"],
        validate="one_to_one",
    )
    assigned["model_error"] = assigned["model_predicted_class"].ne(
        assigned["AD_label"]
    ).astype(int)
    return assigned


def design_matrix(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    absolute_logit = np.abs(frame["model_logit"].to_numpy(dtype=float))
    matrix = pd.DataFrame(
        {
            "absolute_logit": absolute_logit,
            "absolute_logit_squared": absolute_logit**2,
            "predicted_ad": frame["model_predicted_class"].to_numpy(dtype=float),
        },
        index=frame.index,
    )
    if model_name == "confidence_only":
        return matrix
    matrix["pathology_conflict"] = frame["pathology_conflict"].astype(float)
    if model_name == "confidence_plus_conflict":
        return matrix
    if model_name != "confidence_plus_state_reason":
        raise ValueError(model_name)
    for state in STATES[1:]:
        matrix[f"state::{state}"] = frame["reliability_state"].eq(state).astype(float)
    reason = (
        frame["reliability_state"].astype(str)
        + "::"
        + frame["strongest_pathology_counterevidence"].astype(str)
    )
    for state in STATES:
        for concept in CONCEPT_LABELS:
            key = f"{state}::{concept}"
            matrix[f"reason::{key}"] = reason.eq(key).astype(float)
    return matrix


def metric_row(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    labels = np.asarray(labels, dtype=int)
    return {
        "auroc": float(roc_auc_score(labels, clipped)),
        "average_precision": float(average_precision_score(labels, clipped)),
        "brier": float(np.mean((clipped - labels) ** 2)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
    }


def fit_predict(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    labels = development["model_error"].to_numpy(dtype=int)
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        x_dev = design_matrix(development, model_name)
        x_confirmation = design_matrix(confirmation, model_name)
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2_000, solver="lbfgs"),
        )
        estimator.fit(x_dev, labels)
        predicted = estimator.predict_proba(x_confirmation)[:, 1]
        predictions[model_name] = predicted
        rows.append(
            {
                "model": model_name,
                "development_predictions": int(len(development)),
                "confirmation_predictions": int(len(confirmation)),
                "features": int(x_dev.shape[1]),
                **metric_row(confirmation["model_error"].to_numpy(dtype=int), predicted),
            }
        )
    return predictions, pd.DataFrame(rows)


def subgroup_metrics(
    confirmation: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level in ["outer_domain", "family", "configuration_id"]:
        for value, indices in confirmation.groupby(level, sort=True).groups.items():
            index = np.asarray(list(indices), dtype=int)
            labels = confirmation.loc[index, "model_error"].to_numpy(dtype=int)
            if np.unique(labels).size < 2:
                continue
            for model_name in MODEL_ORDER:
                rows.append(
                    {
                        "level": level,
                        "value": str(value),
                        "model": model_name,
                        "n_predictions": int(len(index)),
                        **metric_row(labels, predictions[model_name][index]),
                    }
                )
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    confirmation: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> dict[str, object]:
    groups = [
        np.asarray(list(indices), dtype=int)
        for _, indices in confirmation.groupby("_RID", sort=True).groups.items()
    ]
    labels_all = confirmation["model_error"].to_numpy(dtype=int)
    rng = np.random.default_rng(RANDOM_SEED)
    draws: dict[str, dict[str, list[float]]] = {
        comparator: {metric: [] for metric in ["auroc", "average_precision", "brier", "log_loss"]}
        for comparator in MODEL_ORDER[1:]
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[position] for position in sampled_groups])
        labels = labels_all[indices]
        if np.unique(labels).size < 2:
            continue
        baseline = metric_row(labels, predictions["confidence_only"][indices])
        for comparator in MODEL_ORDER[1:]:
            compared = metric_row(labels, predictions[comparator][indices])
            for metric in draws[comparator]:
                draws[comparator][metric].append(compared[metric] - baseline[metric])

    result: dict[str, object] = {
        "replicates_requested": BOOTSTRAP_REPLICATES,
        "cluster": "participant RID",
        "contrast": "comparator minus confidence_only; positive is better for AUROC/AP, negative is better for Brier/log-loss",
        "comparisons": [],
    }
    observed = {
        model: metric_row(labels_all, predictions[model]) for model in MODEL_ORDER
    }
    comparisons: list[dict[str, object]] = []
    for comparator in MODEL_ORDER[1:]:
        metrics: dict[str, object] = {}
        for metric, values in draws[comparator].items():
            array = np.asarray(values, dtype=float)
            metrics[metric] = {
                "difference": float(observed[comparator][metric] - observed["confidence_only"][metric]),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
            }
        comparisons.append({"comparator": comparator, "metrics": metrics})
    result["comparisons"] = comparisons
    result["replicates_completed"] = min(
        len(values)
        for comparator in draws.values()
        for values in comparator.values()
    )
    return result


def direction_summary(subgroups: pd.DataFrame) -> pd.DataFrame:
    baseline = subgroups[subgroups["model"].eq("confidence_only")]
    rows: list[dict[str, object]] = []
    for comparator in MODEL_ORDER[1:]:
        compared = subgroups[subgroups["model"].eq(comparator)]
        joined = baseline.merge(
            compared,
            on=["level", "value", "n_predictions"],
            suffixes=("_baseline", "_comparator"),
            validate="one_to_one",
        )
        for level, group in joined.groupby("level", sort=True):
            rows.append(
                {
                    "level": level,
                    "comparator": comparator,
                    "groups": int(len(group)),
                    "auroc_improved": int((group["auroc_comparator"] > group["auroc_baseline"]).sum()),
                    "average_precision_improved": int(
                        (group["average_precision_comparator"] > group["average_precision_baseline"]).sum()
                    ),
                    "brier_improved": int((group["brier_comparator"] < group["brier_baseline"]).sum()),
                    "log_loss_improved": int(
                        (group["log_loss_comparator"] < group["log_loss_baseline"]).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    development, _, _, _ = load_development_joined()
    validity = development_acquisition_validity(development)
    development_assigned = assign_development_states(development, validity)
    confirmation_assigned = build_confirmation_assigned(development, validity).reset_index(drop=True)
    predictions, metrics = fit_predict(development_assigned, confirmation_assigned)
    subgroups = subgroup_metrics(confirmation_assigned, predictions)
    directions = direction_summary(subgroups)
    bootstrap = paired_cluster_bootstrap(confirmation_assigned, predictions)

    metrics.to_csv(OUTPUT / "confirmation_metrics.csv", index=False, lineterminator="\n")
    subgroups.to_csv(OUTPUT / "subgroup_metrics.csv", index=False, lineterminator="\n")
    directions.to_csv(OUTPUT / "direction_summary.csv", index=False, lineterminator="\n")
    (OUTPUT / "paired_cluster_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "analysis": "PARES-AD-ERROR-RISK-INCREMENT-v1",
        "status": "COMPLETED",
        "development_fit_only": True,
        "confirmation_outcomes_used_for_evaluation_only": True,
        "development_predictions": int(len(development_assigned)),
        "confirmation_participants": int(confirmation_assigned["_RID"].nunique()),
        "confirmation_predictions": int(len(confirmation_assigned)),
        "models": MODEL_ORDER,
        "metrics": metrics.to_dict(orient="records"),
        "direction_summary": directions.to_dict(orient="records"),
        "paired_cluster_bootstrap": bootstrap,
        "privacy": {
            "participant_identifiers_written": False,
            "participant_level_rows_written": False,
        },
        "runtime_seconds": time.perf_counter() - started,
        "state_confirmation_output": str(STATE_OUTPUT),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(OUTPUT),
                "runtime_seconds": summary["runtime_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
