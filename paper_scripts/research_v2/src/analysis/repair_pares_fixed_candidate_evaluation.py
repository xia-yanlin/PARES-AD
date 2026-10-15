"""Repair evaluation populations for ten already selected references.

Protocol is saved before execution. No selection, new ROI search, MRI inference,
or confirmation access. Old results are read-only; outputs go to a new folder.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from research_v2.src.analysis.run_pares_performance_matched_reference import (
    candidate_predictions, BASELINE_CONCEPTS, ASSIGNMENTS, DEV_PREDICTIONS,
)
from research_v2.src.analysis.build_ad_mri_explanation_system import build_concept_and_pathology_rows

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "research_v2"
OLD = V2 / "outputs/private/pares_ad_performance_matched_reference_2026-10-05"
OUT = V2 / "outputs/private/pares_ad_fixed_candidate_repair_2026-10-09"
PROTOCOL = V2 / "protocols/pares_ad_fixed_candidate_repair_2026-10-09.json"
SOURCE_SAVED = V2 / "outputs/private/pares_ad_comparator_transport_2026-10-08/source_full_private.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(y, p):
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)))


def main():
    if (OUT / "summary.json").exists():
        raise RuntimeError("Completed repair exists; do not overwrite or rerun without a reason")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    old_receipt = json.loads((OLD / "receipt.json").read_text(encoding="utf-8"))
    assert sha(BASELINE_CONCEPTS) == old_receipt["baseline_sha256"]
    assert sha(ASSIGNMENTS) == old_receipt["assignments_sha256"]
    selected = pd.read_csv(OLD / "selected_candidates.csv")
    assert selected.candidate_id.tolist() == protocol["fixed_candidates"] == old_receipt["selected_ids"]
    candidates = [tuple(tuple(v.split(":")) for v in ast.literal_eval(s)) for s in selected.regions]
    baseline = pd.read_csv(BASELINE_CONCEPTS, dtype={"RID": str}, low_memory=False)
    assignments = pd.read_csv(ASSIGNMENTS, dtype={"RID": str})
    dev = pd.read_csv(DEV_PREDICTIONS, dtype={"RID": str})
    target = dev[["RID", "outer_domain", "AD_label"]].drop_duplicates()
    assert len(target) == target.RID.nunique() == 650 and len(dev) == 5184
    merged = baseline.merge(assignments[["RID", "acquisition_domain"]], on="RID", validate="one_to_one")
    assert len(merged) == 729 and merged.acquisition_domain.nunique() == 9
    print("Replaying source-domain fits for the same 10 selected candidates; no search.", flush=True)
    probabilities = candidate_predictions(baseline, assignments, candidates)
    ad = build_concept_and_pathology_rows(baseline, assignments, set(baseline.RID), sorted(merged.acquisition_domain.unique()))
    keys = ["RID", "outer_domain"]
    all_rows = ad.rename(columns={"_RID": "RID", "reference_label": "AD_label", "pathology_probability_AD": "AD_reference"})[keys + ["AD_label", "AD_reference"]]
    assert len(all_rows) == 729 and not all_rows.duplicated(keys).any()
    replay_errors = []
    for i, row in selected.iterrows():
        candidate_id = int(row.candidate_id)
        frame = probabilities[str(i)].rename(columns={"probability": f"candidate_{candidate_id}"})
        all_rows = all_rows.merge(frame, on=keys, validate="one_to_one")
        observed = metric(all_rows.AD_label, all_rows[f"candidate_{candidate_id}"])
        err = max(abs(observed[k] - float(row[k])) for k in observed)
        assert err < 1e-10, (candidate_id, observed)
        replay_errors.append(err)
        print(f"Candidate {candidate_id}: historical source metric replay verified", flush=True)
    primary = target.merge(all_rows, on=keys + ["AD_label"], validate="one_to_one")
    repeated = dev[keys + ["AD_label"]].merge(all_rows, on=keys + ["AD_label"], validate="many_to_one")
    assert len(primary) == 650 and len(repeated) == 5184
    source_saved = pd.read_csv(SOURCE_SAVED, dtype={"_RID": str})
    source_saved = source_saved.rename(columns={"_RID": "RID"}).drop_duplicates(keys)
    check = primary.merge(source_saved[keys + ["pathology_probability_AD"]], on=keys, validate="one_to_one")
    max_ad_error = float(np.max(np.abs(check.AD_reference - check.pathology_probability_AD)))
    assert max_ad_error < 1e-10
    observed_repeat = metric(repeated.AD_label, repeated.AD_reference)
    assert all(abs(observed_repeat[k] - old_receipt["development_ad_reference_metrics"][k]) < 1e-10 for k in observed_repeat)
    metrics = []
    cohorts = {"primary_650_people": primary, "secondary_5184_prediction_rows": repeated, "secondary_729_people": all_rows}
    for name, frame in cohorts.items():
        ad_metric = metric(frame.AD_label, frame.AD_reference)
        for reference in ["AD_reference"] + [f"candidate_{i}" for i in protocol["fixed_candidates"]]:
            result = metric(frame.AD_label, frame[reference])
            gap_auc = result["auroc"] - ad_metric["auroc"]
            gap_brier = result["brier"] - ad_metric["brier"]
            metrics.append(dict(population=name, reference=reference, people=int(frame.RID.nunique()), rows=len(frame), domains=int(frame.outer_domain.nunique()),
                                **result, delta_auroc_candidate_minus_ad=gap_auc, delta_brier_candidate_minus_ad=gap_brier,
                                both_observed_gaps_within_001=bool(abs(gap_auc) <= .01 and abs(gap_brier) <= .01)))
    print("Computing the prespecified paired interval for historical lead candidate 624 on the same 650 people.", flush=True)
    primary = primary.sort_values(keys).reset_index(drop=True)
    y = primary.AD_label.to_numpy(int)
    p = primary.AD_reference.to_numpy(float)
    q = primary.candidate_624.to_numpy(float)
    rng = np.random.default_rng(20261009)
    gaps = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) != 2:
            raise RuntimeError("Non-estimable bootstrap sample; report before changing protocol")
        gaps.append([roc_auc_score(y[idx], q[idx]) - roc_auc_score(y[idx], p[idx]),
                     np.mean((y[idx] - q[idx])**2 - (y[idx] - p[idx])**2)])
    ci = np.quantile(gaps, [.025, .975], axis=0)
    results = pd.DataFrame(metrics)
    lead = results[(results.population == "primary_650_people") & (results.reference == "candidate_624")].iloc[0].to_dict()
    sources = [PROTOCOL, Path(__file__), BASELINE_CONCEPTS, ASSIGNMENTS, DEV_PREDICTIONS, SOURCE_SAVED,
               OLD / "selected_candidates.csv", OLD / "receipt.json",
               V2 / "src/analysis/run_pares_performance_matched_reference.py",
               V2 / "src/analysis/build_ad_mri_explanation_system.py"]
    summary = {
        "status": "COMPLETED_COMMON_POPULATION_REPAIR",
        "role": protocol["role"], "protocol_sha256": sha(PROTOCOL),
        "historical_lead_candidate": 624, "lead_common_population": lead,
        "lead_delta_auroc_ci95": ci[:, 0].tolist(), "lead_delta_brier_ci95": ci[:, 1].tolist(),
        "matching_counts_among_fixed_ten": {name: int(results[(results.population == name) & (results.reference != "AD_reference")].both_observed_gaps_within_001.sum()) for name in cohorts},
        "verification": {"original_ten_candidate_metrics_reproduced_max_error": max(replay_errors), "AD_saved_source_prediction_max_error": max_ad_error, "same_keys_labels_weights_per_comparison": True},
        "limitations": ["Previously selected candidates and known confirmation outcomes; posthoc repair, not independent confirmation", "Observed 0.01 gaps are descriptive, not formal equivalence", "Intervals conditional on fixed fitted predictions and original selection; no source refitting uncertainty", "No change to existing candidate confirmation RRs", "No disease specificity or human benefit inferred automatically"],
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "common_population_metrics.csv", index=False)
    all_rows.to_csv(OUT / "source_reference_predictions_private.csv", index=False, float_format="%.17g")
    np.save(OUT / "lead_performance_gap_bootstrap.npy", np.asarray(gaps))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
