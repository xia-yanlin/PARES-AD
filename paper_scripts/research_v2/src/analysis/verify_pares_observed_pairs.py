"""Independent checks of pair identity and weighted paired-change statistics."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research_v2/outputs/private/pares_ad_observed_pairs_2026-10-09"


def main():
    receipt = json.loads((OUT / "evaluation.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(OUT / "pair_manifest_private.csv", dtype={"_RID": str})
    raw = pd.read_csv(ROOT / "data/raw/tabular/ADSP_PHC_T1_FS_06Jul2026.csv", low_memory=False,
                      usecols=["RID", "PHC_SCANDATE", "PHASE", "ScannerManufacturer", "FieldSgth", "ScannerBatchName"])
    a = raw.iloc[pairs.anchor_raw_row.to_numpy()].reset_index(drop=True)
    b = raw.iloc[pairs.followup_raw_row.to_numpy()].reset_index(drop=True)
    assert np.array_equal(pd.to_numeric(a.RID).astype("Int64").astype(str), pairs._RID)
    assert np.array_equal(a.RID, b.RID)
    # Protocol matches canonical manufacturer, not spelling-identical raw
    # vendor strings (e.g. SIEMENS and Siemens Healthineers are both Siemens).
    def manufacturer(value):
        value = str(value).strip().upper()
        if "SIEMENS" in value: return "Siemens"
        if "PHILIPS" in value: return "Philips"
        if value == "GE" or value.startswith("GE ") or "GE MEDICAL" in value: return "GE"
        raise ValueError("Unknown manufacturer")
    assert a.ScannerManufacturer.map(manufacturer).equals(b.ScannerManufacturer.map(manufacturer))
    for field in ["PHASE", "FieldSgth", "ScannerBatchName"]:
        assert a[field].notna().all() and a[field].equals(b[field])
    days = (pd.to_datetime(b.PHC_SCANDATE) - pd.to_datetime(a.PHC_SCANDATE)).dt.days
    assert np.array_equal(days, pairs.days) and days.between(180,730).all()
    scans = pd.read_csv(OUT / "scan_predictions_private.csv", dtype={"participant": str})
    wide = pd.read_csv(OUT / "paired_results_private.csv", dtype={"participant": str})
    assert scans.seed_count.eq(5).all() and scans.visit.value_counts().nunique() == 1
    assert "AD_label" not in scans and "AD_label" not in wide
    keys = ["participant", "outer_domain", "family", "configuration_id"]
    visit = scans.pivot(index=keys, columns="visit", values="model_probability_AD")
    p0 = np.clip(visit.anchor.to_numpy(), 1e-4, 1-1e-4)
    p1 = np.clip(visit.followup.to_numpy(), 1e-4, 1-1e-4)
    delta = np.log(p1/(1-p1)) - np.log(p0/(1-p0))
    aligned = wide.set_index(keys).loc[visit.index].copy()
    assert np.allclose(aligned.observed_delta, delta, atol=1e-11, rtol=0)
    errors = pd.DataFrame(index=visit.index)
    for method in ["AD4", "REF1", "CONTROL4", "ZERO"]:
        predicted = np.zeros(len(aligned)) if method == "ZERO" else aligned[f"estimate_{method}_followup"].to_numpy() - aligned[f"estimate_{method}_anchor"].to_numpy()
        errors[method] = abs(delta - predicted)
    people = errors.groupby(level="participant", sort=True).mean()
    assert len(people) == len(pairs) == 402
    means = people.mean().to_dict()
    assert all(abs(means[k] - receipt["mae"][k]) < 1e-11 for k in means)
    # Recreate paired resamples directly from the per-person loss matrix.
    rng = np.random.default_rng(20261010)
    boot = []
    for _ in range(2000):
        ix = rng.choice(np.arange(len(people)), size=len(people), replace=True)
        sample_means = people.iloc[ix].mean()
        boot.append([sample_means.REF1-sample_means.AD4, sample_means.CONTROL4-sample_means.AD4])
    boot = np.asarray(boot)
    saved_boot = np.load(OUT / "paired_contrast_bootstrap.npy")
    assert np.allclose(boot, saved_boot, atol=1e-11, rtol=0)
    intervals = np.quantile(boot, [.025,.975], axis=0)
    for i, method in enumerate(["REF1", "CONTROL4"]):
        assert np.allclose(intervals[:,i], receipt["contrasts"][method]["ci95"], atol=1e-11, rtol=0)
    for path, digest in receipt["source_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    result = {"status":"PASS","people":len(people),"paired_prediction_rows":len(wide),"same_person_same_metadata_date_checks":True,"independent_probability_to_delta_replay":True,"independent_mae_and_participant_bootstrap":True,"maximum_bootstrap_difference":float(np.max(abs(boot-saved_boot))),"unlabelled_followup_not_scored_as_diagnosis":True,"scope":"Verifies recorded pair identities, fixed prediction losses, resampling and file provenance; not clinical or causal validity"}
    (OUT / "verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
