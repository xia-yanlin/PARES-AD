"""Performance-matched non-canonical anatomical references for PARES-AD.

The candidate pool and selection rule are development-only.  The selected
references are then replayed on the development-isolated confirmation cohort.
No confirmation outcome is used to choose a candidate or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from research_v2.src.analysis.build_ad_mri_explanation_system import (
    CORE_CONCEPTS,
    CONCEPT_ORIENTATION,
    build_concept_and_pathology_rows,
    fit_monotone_pathology_model,
    logit,
)
from research_v2.src.analysis.confirm_pares_ad_independent import (
    ASSIGNMENTS,
    BASELINE,
    BASELINE_CONCEPTS,
    MANIFEST,
    MAPPING,
    REGISTRY,
    build_confirmation_cohort,
    run_black_boxes,
)
from research_v2.src.analysis.build_ad_mri_explanation_system import prepare_inputs


ROOT = Path(__file__).resolve().parents[3]
V2_ROOT = ROOT / "research_v2"
OUTPUT = V2_ROOT / "outputs" / "private" / "pares_ad_performance_matched_reference_2026-10-05"
DEV_PREDICTIONS = V2_ROOT / "outputs" / "private" / "fw5" / "aggregation" / "fw5_outer_endpoints_v2_1" / "configuration_outer_predictions.csv"
SEED = 20261005
BOOTSTRAPS = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def region_pool() -> list[tuple[str, str]]:
    cortical = [
        "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
        "frontalpole", "fusiform", "insula", "isthmuscingulate", "lateraloccipital",
        "lateralorbitofrontal", "lingual", "medialorbitofrontal", "parahippocampal",
        "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis",
        "pericalcarine", "postcentral", "posteriorcingulate", "precentral",
        "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
        "superiorparietal", "superiortemporal", "temporalpole", "transversetemporal",
    ]
    subcortical = [
        "Accumbens_area", "Amygdala", "Caudate", "Cerebellum_Cortex",
        "Cerebellum_White_Matter", "Pallidum", "Putamen", "Thalamus_Proper",
        "VentralDC", "choroid_plexus",
    ]
    return [("cortical", name) for name in cortical] + [("subcortical", name) for name in subcortical]


def raw_region(frame: pd.DataFrame, kind: str, name: str) -> np.ndarray:
    if kind == "cortical":
        cols = [f"lh_{name}_thickness", f"rh_{name}_thickness"]
        return frame[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy(float)
    cols = [f"Left_{name}", f"Right_{name}"]
    values = frame[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).to_numpy(float)
    return values


def standardized_feature(train: pd.DataFrame, evaluation: pd.DataFrame, kind: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    train_raw = raw_region(train, kind, name)
    eval_raw = raw_region(evaluation, kind, name)
    if kind == "subcortical":
        train_icv = pd.to_numeric(train["EstimatedTotalIntraCranialVol"], errors="coerce").to_numpy(float)
        eval_icv = pd.to_numeric(evaluation["EstimatedTotalIntraCranialVol"], errors="coerce").to_numpy(float)
        valid = np.isfinite(train_raw) & np.isfinite(train_icv) & (train_raw > 0) & (train_icv > 0)
        design = np.column_stack([np.ones(valid.sum()), np.log(train_icv[valid])])
        beta, *_ = np.linalg.lstsq(design, np.log(train_raw[valid]), rcond=None)
        train_raw = np.where(valid, np.log(train_raw) - (beta[0] + beta[1] * np.log(train_icv)), np.nan)
        valid_eval = np.isfinite(eval_raw) & np.isfinite(eval_icv) & (eval_raw > 0) & (eval_icv > 0)
        eval_raw = np.where(valid_eval, np.log(eval_raw) - (beta[0] + beta[1] * np.log(eval_icv)), np.nan)
    train_valid = np.isfinite(train_raw)
    eval_valid = np.isfinite(eval_raw)
    if not train_valid.all() or not eval_valid.all():
        raise ValueError(f"Missing candidate feature {kind}/{name}")
    mean = float(train_raw.mean())
    scale = float(train_raw.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Degenerate candidate feature {kind}/{name}")
    train_z = (train_raw - mean) / scale
    eval_z = (eval_raw - mean) / scale
    direction = 1.0 if np.corrcoef(train_z, train["AD_label"].to_numpy(float))[0, 1] >= 0 else -1.0
    return direction * train_z, direction * eval_z


def fit_reference(train: pd.DataFrame, evaluation: pd.DataFrame, candidate: tuple[tuple[str, str], ...]) -> np.ndarray:
    train_features, eval_features = [], []
    for kind, name in candidate:
        tr, ev = standardized_feature(train, evaluation, kind, name)
        train_features.append(tr)
        eval_features.append(ev)
    x_train = np.column_stack(train_features)
    x_eval = np.column_stack(eval_features)
    intercept, coefficients = fit_monotone_pathology_model(
        x_train, train["AD_label"].to_numpy(int), l2_penalty=0.1
    )
    return expit(intercept + x_eval @ coefficients)


def candidate_predictions(cohort: pd.DataFrame, assignments: pd.DataFrame, candidates: list[tuple[tuple[str, str], ...]]) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    merged = cohort.merge(assignments[["RID", "acquisition_domain"]], on="RID", validate="one_to_one")
    for index, candidate in enumerate(candidates):
        rows = []
        for domain, evaluation in merged.groupby("acquisition_domain", sort=True):
            train = merged[~merged["acquisition_domain"].eq(domain)].reset_index(drop=True)
            evaluation = evaluation.reset_index(drop=True)
            probability = fit_reference(train, evaluation, candidate)
            rows.append(pd.DataFrame({"RID": evaluation["RID"].astype(str), "outer_domain": domain, "probability": probability}))
        outputs[str(index)] = pd.concat(rows, ignore_index=True)
    return outputs


def metrics(pred: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, float]:
    labels = cohort.set_index("RID").loc[pred["RID"].astype(str), "AD_label"].to_numpy(int)
    p = pred["probability"].to_numpy(float)
    return {"auroc": float(roc_auc_score(labels, p)), "auprc": float(average_precision_score(labels, p)), "brier": float(brier_score_loss(labels, p))}


def rr_and_bootstrap(frame: pd.DataFrame, conflict_column: str, seed: int) -> dict[str, object]:
    frame = frame.reset_index(drop=True)
    y = frame["model_error"].to_numpy(int)
    conflict = frame[conflict_column].to_numpy(bool)
    def rr(indices: np.ndarray) -> float:
        return float(y[indices][conflict[indices]].mean() / y[indices][~conflict[indices]].mean())
    observed = rr(np.arange(len(frame)))
    groups = [np.asarray(list(ix), int) for _, ix in frame.groupby("RID", sort=True).groups.items()]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in sampled])
        if conflict[idx].sum() and (~conflict[idx]).sum() and y[idx][conflict[idx]].sum() >= 0 and y[idx][~conflict[idx]].sum() >= 0:
            values.append(rr(idx))
    return {"rr": observed, "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))], "n_conflict": int(conflict.sum()), "conflict_error_rate": float(y[conflict].mean()), "aligned_error_rate": float(y[~conflict].mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE_CONCEPTS, low_memory=False, dtype={"RID": str})
    assignments = pd.read_csv(ASSIGNMENTS, dtype={"RID": str})
    baseline["RID"] = pd.to_numeric(baseline["RID"], errors="coerce").astype("Int64").astype(str)
    assignments["RID"] = pd.to_numeric(assignments["RID"], errors="coerce").astype("Int64").astype(str)
    cohort, assignment_subset, dev_predictions = prepare_inputs(BASELINE_CONCEPTS, ASSIGNMENTS, DEV_PREDICTIONS)
    concept_rows = build_concept_and_pathology_rows(cohort, assignment_subset, set(dev_predictions["RID"].astype(str)), sorted(dev_predictions["outer_domain"].unique()))
    dev = dev_predictions.rename(columns={"mean_calibrated_probability_AD": "model_probability_AD"}).merge(concept_rows, left_on=["RID", "outer_domain"], right_on=["_RID", "outer_domain"], validate="many_to_one")
    dev["model_predicted_class"] = dev["model_probability_AD"].ge(.5).astype(int)
    dev["model_error"] = dev["model_predicted_class"].ne(dev["AD_label"]).astype(int)
    dev["ad_reference_probability"] = dev["pathology_probability_AD"]
    pool = region_pool()
    rng = np.random.default_rng(SEED)
    candidates = []
    for combo in combinations(pool, 4):
        if len(candidates) >= 10000:
            break
        candidates.append(combo)
    rng.shuffle(candidates)
    candidates = candidates[:1000]
    dev_candidate_outputs = candidate_predictions(cohort, assignment_subset, candidates)
    dev_metrics = []
    ad_metrics = metrics(pd.DataFrame({"RID": dev["RID"].astype(str), "probability": dev["pathology_probability_AD"].to_numpy()}), cohort)
    for index, candidate in enumerate(candidates):
        pred = dev_candidate_outputs[str(index)]
        score = metrics(pred, cohort)
        distance = math.sqrt(((score["auroc"] - ad_metrics["auroc"]) / .01) ** 2 + ((score["brier"] - ad_metrics["brier"]) / .01) ** 2)
        dev_metrics.append({"candidate_id": index, "regions": [f"{k}:{n}" for k, n in candidate], **score, "distance": distance})
    metrics_frame = pd.DataFrame(dev_metrics).sort_values(["distance", "candidate_id"]).reset_index(drop=True)
    selected = metrics_frame.head(10)
    selected_ids = selected["candidate_id"].astype(int).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(args.output / "development_candidate_metrics.csv", index=False)
    selected.to_csv(args.output / "selected_candidates.csv", index=False)
    mapping_nodes = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    confirmation = build_confirmation_cohort(mapping_nodes)
    manifest = pd.read_csv(MANIFEST, dtype={"seed": int})
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    model_predictions = run_black_boxes(confirmation, manifest, mapping_nodes, registry)
    model_predictions = model_predictions.rename(columns={"model_probability_AD": "model_probability_AD"})
    conf_assign = pd.read_csv(ASSIGNMENTS, dtype={"RID": str})
    conf_assign["RID"] = pd.to_numeric(conf_assign["RID"], errors="coerce").astype("Int64").astype(str)
    conf_base = confirmation.copy()
    rows = []
    for candidate_id in selected_ids:
        candidate = candidates[candidate_id]
        for domain, evaluation in conf_base.groupby("outer_domain", sort=True):
            train = baseline.merge(assignments[["RID", "acquisition_domain"]], on="RID", validate="one_to_one")
            train = train[~train["acquisition_domain"].eq(domain)].reset_index(drop=True)
            evaluation = evaluation.reset_index(drop=True)
            p = fit_reference(train, evaluation, candidate)
            rows.append(pd.DataFrame({"_RID": evaluation["_RID"].astype(str), "outer_domain": domain, "null_probability": p, "candidate_id": candidate_id}))
    null = pd.concat(rows, ignore_index=True)
    model_predictions = model_predictions.assign(_candidate_join=1).merge(
        pd.DataFrame({"candidate_id": selected_ids, "_candidate_join": 1}),
        on="_candidate_join", how="inner",
    ).drop(columns=["_candidate_join"])
    cases = model_predictions.merge(null, on=["_RID", "outer_domain", "candidate_id"], validate="many_to_one")
    cases["model_predicted_class"] = cases["model_probability_AD"].ge(.5).astype(int)
    cases["model_error"] = cases["model_predicted_class"].ne(cases["AD_label"]).astype(int)
    cases["null_predicted_class"] = cases["null_probability"].ge(.5).astype(int)
    cases["null_conflict"] = cases["model_predicted_class"].ne(cases["null_predicted_class"])
    results = []
    for candidate_id, group in cases.groupby("candidate_id", sort=True):
        row = rr_and_bootstrap(group.rename(columns={"_RID": "RID"}), "null_conflict", SEED + int(candidate_id))
        row["candidate_id"] = int(candidate_id)
        results.append(row)
    receipt = {"seed": SEED, "candidate_pool": len(candidates), "selected_ids": selected_ids, "development_ad_reference_metrics": ad_metrics, "selection_rule": "minimum scaled Euclidean distance in development AUROC and Brier; top 10 retained", "baseline_sha256": sha256(BASELINE_CONCEPTS), "assignments_sha256": sha256(ASSIGNMENTS)}
    pd.DataFrame(results).to_csv(args.output / "confirmation_null_risk.csv", index=False)
    cases.to_csv(args.output / "confirmation_null_cases.csv", index=False)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "development_ad": ad_metrics, "selected": selected.to_dict("records"), "confirmation": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
