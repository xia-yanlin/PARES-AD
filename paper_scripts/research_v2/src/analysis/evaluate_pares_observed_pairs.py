"""Frozen natural-pair behavioral evaluation; no training of MRI classifiers."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from research_v2.src.analysis.confirm_pares_ad_independent import (
    RAW_FS, MANIFEST, MAPPING, REGISTRY, run_black_boxes,
    canonicalize_phase, canonicalize_manufacturer, canonicalize_field_strength,
)
from research_v2.src.analysis.confirm_pares_ad_state_confirmation import (
    load_development_joined, confirmation_concept_rows,
)
from research_v2.src.analysis.validate_pares_ad_conditional_specificity import build_control_features
from research_v2.src.analysis.build_ad_mri_explanation_system import CORE_CONCEPTS, logit

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "research_v2"
OUT = V2 / "outputs/private/pares_ad_observed_pairs_2026-10-09"
PROTOCOL = V2 / "protocols/pares_ad_observed_pair_evaluation_2026-10-09.json"
CACHE = V2 / "outputs/private/pares_ad_comparator_transport_2026-10-08/adni_scores_private.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / "evaluation.json").exists():
        raise RuntimeError("Evaluation already complete; do not overwrite")
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    torch.set_num_threads(1)
    feasibility = json.loads((OUT / "feasibility.json").read_text(encoding="utf-8"))
    assert feasibility["status"] == "FEASIBLE_FOR_PRESPECIFIED_EVALUATION"
    assert sha(OUT / "pair_manifest_private.csv") == feasibility["pair_manifest_sha256"]
    assert sha(RAW_FS) == feasibility["source_sha256"][str(RAW_FS.relative_to(ROOT))]
    pairs = pd.read_csv(OUT / "pair_manifest_private.csv", dtype={"_RID": str})
    raw = pd.read_csv(RAW_FS, low_memory=False)
    pieces = []
    for visit, field in [("anchor", "anchor_raw_row"), ("followup", "followup_raw_row")]:
        frame = raw.iloc[pairs[field].to_numpy(int)].copy().reset_index(drop=True)
        assert np.array_equal(pd.to_numeric(frame.RID).astype("Int64").astype(str), pairs._RID)
        frame["participant"] = pairs._RID.to_numpy()
        frame["_RID"] = [f"pair{i:04d}_{visit}" for i in range(len(pairs))]
        frame["visit"] = visit
        frame["phase"] = frame.PHASE.map(canonicalize_phase)
        frame["manufacturer"] = frame.ScannerManufacturer.map(canonicalize_manufacturer)
        frame["field_strength"] = frame.FieldSgth.map(canonicalize_field_strength)
        frame["outer_domain"] = frame.phase + "_" + frame.manufacturer
        assert np.array_equal(frame.outer_domain, pairs.outer_domain)
        # The inherited inference helper requires this field, but never uses
        # labels for prediction. -1 explicitly denotes unscored/unlabelled;
        # it is dropped from outputs, not treated as a clinical diagnosis.
        frame["AD_label"] = -1
        pieces.append(frame)
    scans = pd.concat(pieces, ignore_index=True)
    metadata = scans[["_RID", "participant", "visit", "outer_domain"]].copy()
    print("Preparing frozen source-only behavior models and scan features.", flush=True)
    development, baseline, assignments, _ = load_development_joined()
    concepts = confirmation_concept_rows(scans, baseline, assignments)
    base_domains = baseline.merge(assignments[["RID", "acquisition_domain"]], on="RID", validate="one_to_one")
    control_names = [f"control_{i}" for i in range(4)]
    source_control = []
    scan_control = []
    source_keys = development[["RID", "outer_domain"]].drop_duplicates()
    for domain in sorted(source_keys.outer_domain.unique()):
        train = base_domains[base_domains.acquisition_domain.ne(domain)].reset_index(drop=True)
        source_eval = source_keys[source_keys.outer_domain.eq(domain)].merge(baseline, on="RID", validate="one_to_one")
        _, feature = build_control_features(train, source_eval)
        block = source_eval[["RID", "outer_domain"]].copy()
        block[control_names] = feature
        source_control.append(block)
        scan_eval = scans[scans.outer_domain.eq(domain)].reset_index(drop=True)
        if len(scan_eval):
            _, feature = build_control_features(train, scan_eval)
            block = scan_eval[["_RID", "outer_domain"]].copy()
            block[control_names] = feature
            scan_control.append(block)
    development = development.merge(pd.concat(source_control), on=["RID", "outer_domain"], validate="many_to_one")
    concepts = concepts.merge(pd.concat(scan_control), on=["_RID", "outer_domain"], validate="one_to_one")
    development["reference_logit"] = logit(development.pathology_probability_AD)
    concepts["reference_logit"] = logit(concepts.pathology_probability_AD)
    ad_names = [f"{c}_severity" for c in CORE_CONCEPTS]
    designs = {"AD4": ad_names, "REF1": ["reference_logit"], "CONTROL4": control_names}
    estimates = []
    for (family, config), group in development.groupby(["family", "configuration_id"], sort=True):
        for domain in sorted(group.outer_domain.unique()):
            train = group[group.outer_domain.ne(domain)]
            evaluation = concepts[concepts.outer_domain.eq(domain)]
            if evaluation.empty: continue
            block = evaluation[["_RID", "outer_domain"]].copy()
            block["family"] = family
            block["configuration_id"] = config
            for name, columns in designs.items():
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                estimator.fit(train[columns].to_numpy(float), train.model_logit.to_numpy(float))
                block[f"estimate_{name}"] = estimator.predict(evaluation[columns].to_numpy(float))
            estimates.append(block)
    estimates = pd.concat(estimates, ignore_index=True)
    estimates.to_csv(OUT / "source_fitted_scan_estimates_private.csv", index=False, float_format="%.17g")
    frozen = {"protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)), "estimates_sha256": sha(OUT / "source_fitted_scan_estimates_private.csv"), "new_scan_predictions_started": True}
    (OUT / "evaluation_started.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print("Source estimates frozen. Applying existing MRI checkpoints to both visits.", flush=True)
    manifest = pd.read_csv(MANIFEST, dtype={"seed": int})
    mapping = json.loads(MAPPING.read_text())["nodes"]
    registry = json.loads(REGISTRY.read_text())
    predictions = []
    for domain, jobs in manifest.groupby("outer_domain", sort=True):
        evaluation = scans[scans.outer_domain.eq(domain)].copy()
        if evaluation.empty: continue
        predictions.append(run_black_boxes(evaluation, jobs, mapping, registry))
        print(f"Completed source-domain checkpoint group: {domain} ({len(evaluation)} scans, {len(jobs)} checkpoints)", flush=True)
    predictions = pd.concat(predictions, ignore_index=True).drop(columns=["AD_label"])
    predictions = predictions.merge(metadata, on=["_RID", "outer_domain"], validate="many_to_one")
    prediction_keys = ["_RID", "outer_domain", "family", "configuration_id"]
    assert not predictions.duplicated(prediction_keys).any()
    assert predictions.seed_count.eq(5).all()
    assert len(predictions) == len(estimates)
    joint = predictions.merge(estimates, on=prediction_keys, validate="one_to_one")
    joint["model_logit"] = logit(joint.model_probability_AD)
    saved = pd.read_csv(CACHE, dtype={"_RID": str}).rename(columns={"_RID": "participant", "model_probability_AD": "cached_probability"})
    anchor_check = joint[joint.visit.eq("anchor")].merge(saved[["participant", "outer_domain", "family", "configuration_id", "cached_probability"]], on=["participant", "outer_domain", "family", "configuration_id"], validate="one_to_one")
    assert len(anchor_check) == len(joint) // 2
    max_replay_error = float(np.max(np.abs(anchor_check.model_probability_AD - anchor_check.cached_probability)))
    assert max_replay_error < 1e-6, max_replay_error
    pair_keys = ["participant", "outer_domain", "family", "configuration_id"]
    values = ["model_logit"] + [f"estimate_{name}" for name in designs]
    wide = joint[joint.visit.eq("anchor")][pair_keys + values].merge(joint[joint.visit.eq("followup")][pair_keys + values], on=pair_keys, suffixes=("_anchor", "_followup"), validate="one_to_one")
    wide["observed_delta"] = wide.model_logit_followup - wide.model_logit_anchor
    methods = list(designs) + ["ZERO"]
    for name in methods:
        wide[f"predicted_delta_{name}"] = 0. if name == "ZERO" else wide[f"estimate_{name}_followup"] - wide[f"estimate_{name}_anchor"]
        wide[f"signed_error_{name}"] = wide[f"predicted_delta_{name}"] - wide.observed_delta
        wide[f"absolute_error_{name}"] = wide[f"signed_error_{name}"].abs()
    assert wide.participant.nunique() == len(pairs) == 402
    losses = [f"absolute_error_{name}" for name in methods]
    people = wide.groupby("participant", sort=True)[losses].mean()
    mae = {name: float(people[f"absolute_error_{name}"].mean()) for name in methods}
    rng = np.random.default_rng(20261010)
    participant_contrasts = np.column_stack([people.absolute_error_REF1 - people.absolute_error_AD4, people.absolute_error_CONTROL4 - people.absolute_error_AD4])
    bootstrap = np.empty((2000, 2))
    for b in range(2000):
        idx = rng.integers(0, len(people), size=len(people))
        bootstrap[b] = participant_contrasts[idx].mean(axis=0)
    intervals = np.quantile(bootstrap, [.025, .975], axis=0)
    aggregates = []
    for scope, group_columns in [("configuration", ["family", "configuration_id"]), ("domain", ["outer_domain"])]:
        for key, group in wide.groupby(group_columns, sort=True):
            if not isinstance(key, tuple): key = (key,)
            result = group.groupby("participant")[losses].mean().mean()
            aggregates.append(dict(scope=scope, value="|".join(key), people=int(group.participant.nunique()), prediction_pairs=len(group), **{f"mae_{name}":float(result[f"absolute_error_{name}"]) for name in methods}))
    aggregated = pd.DataFrame(aggregates)
    configs = aggregated[aggregated.scope.eq("configuration")]
    contrasts = {}
    for col, comparator in enumerate(["REF1", "CONTROL4"]):
        gain = mae[comparator] - mae["AD4"]
        direction = int((configs.mae_AD4 < configs[f"mae_{comparator}"]).sum())
        contrasts[comparator] = {"mae_reduction":gain, "relative_reduction":gain/mae[comparator], "ci95":intervals[:, col].tolist(), "configurations_with_lower_ad4_error":direction, "criterion_pass":bool(intervals[0, col] > 0 and gain/mae[comparator] >= .10 and direction >= 7)}
    summary = {
        "status":"PASS_BOUNDED_WITHIN_PERSON_RECONSTRUCTION" if all(c["criterion_pass"] for c in contrasts.values()) else "NOT_SUPPORTED_PRESPECIFIED_INCREMENT",
        "people":len(people), "prediction_pairs":len(wide), "scan_predictions":len(joint),
        "mae":mae, "contrasts":contrasts,
        "observed_mean_absolute_logit_change":float(wide.assign(abs_delta=wide.observed_delta.abs()).groupby("participant").abs_delta.mean().mean()),
        "signed_bias":{name:float(wide.groupby("participant")[f"signed_error_{name}"].mean().mean()) for name in methods},
        "verification":{"anchor_probability_max_absolute_replay_error":max_replay_error,"all_seed_counts_five":True,"all_402_people_retained":True,"paired_method_keys_identical":True},
        "limitations":["Within previously used ADNI participants, not external or new participant confirmation","Observed scan variation includes measurement variability; no causal or biological progression claim","Fixed-prediction bootstrap omits source-fit uncertainty","No human report-utility result","Sparse ADNI4 Siemens retained; no domain-specific inference"],
        "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [PROTOCOL,Path(__file__),MANIFEST,MAPPING,REGISTRY,RAW_FS,OUT / "pair_manifest_private.csv",CACHE]},
        "source_estimate_freeze":frozen,
    }
    joint.to_csv(OUT / "scan_predictions_private.csv", index=False, float_format="%.17g")
    wide.to_csv(OUT / "paired_results_private.csv", index=False, float_format="%.17g")
    aggregated.to_csv(OUT / "group_results_aggregate.csv", index=False)
    np.save(OUT / "paired_contrast_bootstrap.npy", bootstrap)
    (OUT / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["source_sha256","source_estimate_freeze"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
