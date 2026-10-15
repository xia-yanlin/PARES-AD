#!/usr/bin/env python3
"""Audit frozen FTP coverage without exporting or analyzing tau values."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research_v2.src.analysis.confirm_pares_ad_independent import (
    HISS_ROOT,
    MANIFEST,
    MAPPING,
    REGISTRY,
    JOBS_ROOT,
    build_confirmation_cohort,
    build_pathology_reference,
    run_black_boxes,
)


ARCHIVE = HISS_ROOT / "data" / "raw" / "archives" / "Data___Database.zip"
INNER_ARCHIVE = "ADNIMERGE2.tar.gz"
RDA_MEMBERS = (
    "ADNIMERGE2/data/UCBERKELEY_TAU_6MM.rda",
    "ADNIMERGE2/data/UCBERKELEY_AMY_6MM.rda",
    "ADNIMERGE2/data/PACC.rda",
)
R_HELPER = HISS_ROOT / "research_v2" / "src" / "analysis" / "build_pares_anchor_receipt.R"
DEFAULT_RSCRIPT = Path(r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe")


def canonical_rid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def export_blind_metadata(
    directory: Path, rscript: Path, *, include_values: bool = False
) -> None:
    inner_path = directory / INNER_ARCHIVE
    with zipfile.ZipFile(ARCHIVE) as outer:
        with outer.open(INNER_ARCHIVE) as source, inner_path.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
    with tarfile.open(inner_path, mode="r:gz") as inner:
        members = {member.name: member for member in inner.getmembers()}
        for name in RDA_MEMBERS:
            member = members.get(name)
            if member is None:
                raise RuntimeError(f"Missing frozen anchor member: {name}")
            source = inner.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read frozen anchor member: {name}")
            destination = directory / Path(name).name
            with source, destination.open("wb") as handle:
                while chunk := source.read(1024 * 1024):
                    handle.write(chunk)
    inner_path.unlink()
    command = [str(rscript), str(R_HELPER), str(directory)]
    if include_values:
        command.append("values")
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def link_ftp(confirmation: pd.DataFrame, tau_meta: pd.DataFrame) -> pd.DataFrame:
    tau = tau_meta.copy()
    tau["_RID"] = canonical_rid(tau["RID"])
    tau["_pet_date"] = pd.to_datetime(tau["SCANDATE"], errors="coerce")
    tau["_qc"] = pd.to_numeric(tau["qc_flag"], errors="coerce")
    available = tau["anchor_available"].astype(str).str.strip().str.lower().isin({"true", "t", "1"})
    tau = tau[
        tau["TRACER"].fillna("").astype(str).str.strip().str.upper().eq("FTP")
        & tau["_qc"].eq(2)
        & tau["_pet_date"].notna()
        & available
    ].copy()
    linked = confirmation[["_RID", "_scan_date", "AD_label", "outer_domain"]].merge(
        tau[["_RID", "_pet_date"]], on="_RID", how="inner", validate="one_to_many"
    )
    linked["absolute_gap_days"] = (linked["_pet_date"] - linked["_scan_date"]).abs().dt.days
    linked = linked[linked["absolute_gap_days"].le(365)].copy()
    linked["pet_after_mri"] = linked["_pet_date"].gt(linked["_scan_date"]).astype(int)
    linked = linked.sort_values(
        ["_RID", "absolute_gap_days", "pet_after_mri", "_pet_date"], kind="stable"
    ).drop_duplicates("_RID", keep="first")
    return linked.reset_index(drop=True)


def base_receipt(linked: pd.DataFrame) -> dict[str, Any]:
    receipt = {
        "participants": int(linked["_RID"].nunique()),
        "ad": int(linked["AD_label"].sum()),
        "cn": int(len(linked) - linked["AD_label"].sum()),
        "outer_domains": int(linked["outer_domain"].nunique()),
        "median_absolute_gap_days": float(linked["absolute_gap_days"].median()),
    }
    gates = {
        "participants_at_least_150": receipt["participants"] >= 150,
        "ad_at_least_35": receipt["ad"] >= 35,
        "cn_at_least_75": receipt["cn"] >= 75,
        "outer_domains_at_least_4": receipt["outer_domains"] >= 4,
    }
    return {"receipt": receipt, "gates": gates, "pass": all(gates.values())}


def conflict_receipt(confirmation: pd.DataFrame, linked: pd.DataFrame) -> dict[str, Any]:
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifest = pd.read_csv(MANIFEST, dtype={"seed": int})
    if len(manifest) != 290:
        raise RuntimeError("Frozen FW5 manifest must contain 290 jobs")
    predictions = run_black_boxes(confirmation, manifest, mapping, registry)
    pathology = build_pathology_reference(confirmation)
    cases = predictions.merge(pathology, on=["_RID", "outer_domain"], validate="many_to_one")
    cases["model_predicted_class"] = cases["model_probability_AD"].ge(0.5)
    cases["pathology_predicted_class"] = cases["pathology_probability_AD"].ge(0.5)
    cases["pathology_conflict"] = cases["model_predicted_class"].ne(cases["pathology_predicted_class"])
    linked_ids = set(linked["_RID"])
    eligible = cases[cases["_RID"].isin(linked_ids)]
    conflict_participants = int(eligible.loc[eligible["pathology_conflict"], "_RID"].nunique())
    return {
        "ftp_linked_model_case_predictions": int(len(eligible)),
        "ftp_linked_conflict_participants": conflict_participants,
        "conflict_participants_at_least_40": conflict_participants >= 40,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rscript", type=Path, default=DEFAULT_RSCRIPT)
    parser.add_argument("--skip-conflict-replay", action="store_true")
    args = parser.parse_args()
    if not args.rscript.is_file():
        raise RuntimeError(f"Rscript not found: {args.rscript}")
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["nodes"]
    confirmation = build_confirmation_cohort(mapping)
    with tempfile.TemporaryDirectory(prefix="pares_ftp_") as temporary:
        directory = Path(temporary)
        export_blind_metadata(directory, args.rscript)
        tau_meta = pd.read_csv(directory / "tau_meta.csv", low_memory=False)
        linked = link_ftp(confirmation, tau_meta)
    result: dict[str, Any] = {
        "audit": "PARES-AD FTP outcome-blind feasibility",
        "tau_values_exported_or_analyzed": False,
        **base_receipt(linked),
    }
    if result["pass"] and not args.skip_conflict_replay:
        conflict = conflict_receipt(confirmation, linked)
        result["conflict_receipt"] = conflict
        result["pass"] = bool(conflict["conflict_participants_at_least_40"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
