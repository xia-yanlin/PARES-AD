"""Outcome-free feasibility of a single prespecified observed MRI pair per person."""
from pathlib import Path
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from research_v2.src.analysis.confirm_pares_ad_independent import (
    RAW_FS, DXSUM, MAPPING, BASELINE, canonical_rid, build_confirmation_cohort,
    canonicalize_phase, canonicalize_manufacturer, canonicalize_field_strength,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research_v2/protocols/pares_ad_observed_pair_feasibility_2026-10-09.json"
OUT = ROOT / "research_v2/outputs/private/pares_ad_observed_pairs_2026-10-09"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / "feasibility.json").exists():
        raise RuntimeError("Feasibility already recorded; do not silently replace")
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    mapping = json.loads(MAPPING.read_text())["nodes"]
    anchor = build_confirmation_cohort(mapping).copy()
    diagnosis = pd.read_csv(DXSUM, low_memory=False)
    diagnosis["_RID"] = canonical_rid(diagnosis.RID)
    diagnosis["_VISCODE2"] = diagnosis.VISCODE2.fillna("").astype(str).str.strip().str.lower()
    diagnosis["_dx"] = pd.to_numeric(diagnosis.DIAGNOSIS, errors="coerce")
    full_dx = diagnosis.groupby(["_RID", "_VISCODE2"])._dx.agg(lambda x: tuple(sorted(set(x.dropna())))).rename("full_dx")
    diagnosis_check = anchor.merge(full_dx, on=["_RID", "_VISCODE2"], validate="many_to_one")
    assert diagnosis_check.full_dx.map(lambda x: len(x) == 1 and x[0] in (1., 3.)).all()
    fs = pd.read_csv(RAW_FS, low_memory=False)
    fs["raw_row"] = np.arange(len(fs))
    fs["_RID"] = canonical_rid(fs.RID)
    fs["_scan_date"] = pd.to_datetime(fs.PHC_SCANDATE, errors="coerce")
    fs = fs[fs._RID.isin(anchor._RID)].copy()
    # Canonical functions fail closed. Restrict recognized phases first, as in
    # original confirmation, without looking at follow-up clinical outcomes.
    fs = fs[fs.PHASE.fillna("").astype(str).str.strip().str.upper().isin(["ADNI1", "ADNI3", "ADNI4"])].copy()
    fs["phase"] = fs.PHASE.map(canonicalize_phase)
    fs["manufacturer"] = fs.ScannerManufacturer.map(canonicalize_manufacturer)
    fs["field_strength"] = fs.FieldSgth.map(canonicalize_field_strength)
    fs["outer_domain"] = fs.phase + "_" + fs.manufacturer
    required = ["EstimatedTotalIntraCranialVol"] + [x["volume_col"] for x in mapping] + [x["thickness_col"] for x in mapping if x["thickness_col"] is not None]
    assert len(required) == 163
    values = fs[required].apply(pd.to_numeric, errors="coerce")
    fs["valid_inputs"] = np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1)
    anchor = anchor.merge(fs[["_RID", "_scan_date", "raw_row"]], on=["_RID", "_scan_date"], validate="one_to_one")
    assert len(anchor) == 882
    pairs = []
    flow = dict(anchor_people=882, with_time_window=0, with_metadata_match=0, with_valid_inputs=0, excluded_best_distance_tie=0)
    metadata = ["phase", "manufacturer", "field_strength", "ScannerBatchName"]
    for a in anchor.to_dict("records"):
        group = fs[fs._RID.eq(a["_RID"])].copy()
        group["days"] = (group._scan_date - a["_scan_date"]).dt.days
        group = group[group.days.between(180, 730)].copy()
        if group.empty: continue
        flow["with_time_window"] += 1
        for col in metadata:
            if pd.isna(a[col]) or str(a[col]).strip() == "":
                group = group.iloc[:0]
                break
            group = group[group[col].notna() & group[col].eq(a[col])]
        if group.empty: continue
        flow["with_metadata_match"] += 1
        group = group[group.valid_inputs].copy()
        if group.empty: continue
        flow["with_valid_inputs"] += 1
        group["distance"] = (group.days - 365).abs()
        best = group[group.distance.eq(group.distance.min())]
        if len(best) != 1:
            flow["excluded_best_distance_tie"] += 1
            continue
        b = best.iloc[0]
        pairs.append(dict(_RID=a["_RID"], outer_domain=a["outer_domain"], anchor_raw_row=int(a["raw_row"]), followup_raw_row=int(b.raw_row), days=int(b.days), anchor_label=int(a["AD_label"])))
    columns = ["_RID", "outer_domain", "anchor_raw_row", "followup_raw_row", "days", "anchor_label"]
    pairs = pd.DataFrame(pairs, columns=columns)
    counts = pairs.groupby("outer_domain").size().reindex(sorted(anchor.outer_domain.unique()), fill_value=0)
    baseline_ids = set(canonical_rid(pd.read_csv(BASELINE, usecols=["RID"]).RID))
    assert not set(pairs._RID) & baseline_ids
    passed = len(pairs) >= 100 and int(counts.ge(20).sum()) >= 3
    receipt = {
        "status": "FEASIBLE_FOR_PRESPECIFIED_EVALUATION" if passed else "STOP_NOT_FEASIBLE",
        "flow": flow, "eligible_people": len(pairs), "one_pair_per_person": bool(not pairs._RID.duplicated().any()),
        "people_by_domain": {k: int(v) for k, v in counts.items()},
        "domains_with_at_least_20": int(counts.ge(20).sum()),
        "interval_days": {"min": int(pairs.days.min()), "median": float(pairs.days.median()), "max": int(pairs.days.max())} if len(pairs) else None,
        "baseline_AD": int(pairs.anchor_label.sum()), "baseline_CN": int(len(pairs) - pairs.anchor_label.sum()),
        "source_participant_overlap": 0, "ambiguous_original_anchor_diagnoses": 0,
        "new_classifier_inference_run": False, "ROI_changes_or_model_change_results_inspected": False,
        "limitations": ["Already-used ADNI participants, not new independent people", "Matching batch and vendor metadata does not prove physical scanner identity", "Cross-sectional FreeSurfer measurements, not guaranteed biological change", "Operational floor is not a power calculation"],
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [RAW_FS, DXSUM, MAPPING, BASELINE, PROTOCOL, Path(__file__)]},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT / "pair_manifest_private.csv", index=False)
    receipt["pair_manifest_sha256"] = sha(OUT / "pair_manifest_private.csv")
    (OUT / "feasibility.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
