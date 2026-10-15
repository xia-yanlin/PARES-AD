#!/usr/bin/env python3
"""Freeze acquisition-domain definitions from the existing ADNI cohort.

The source cohort is controlled participant-level data.  This module writes
only aggregate counts to the tracked registry and keeps the RID-level mapping
under ``outputs/private/``, which is excluded from Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


V2_ROOT = Path(__file__).resolve().parents[2]
HISS_ROOT = V2_ROOT.parent
DEFAULT_SOURCE = HISS_ROOT / "data" / "processed" / "baseline_cohort_with_splits.csv"
DEFAULT_REGISTRY = V2_ROOT / "data_registry" / "acquisition_domain_feasibility.csv"
DEFAULT_DEFINITION = V2_ROOT / "data_registry" / "acquisition_domain_definition.json"
DEFAULT_ASSIGNMENTS = V2_ROOT / "outputs" / "private" / "domain_assignments.csv"
DEFAULT_LOG = V2_ROOT / "logs" / "fw0_domain_freeze_validation.json"

PRIMARY_MIN_AD = 5
PRIMARY_MIN_CN = 10
STABLE_MIN_AD = 10
STABLE_MIN_CN = 20

PHASE_ORDER = ["ADNI1", "ADNI3", "ADNI4"]
MANUFACTURER_ORDER = ["Siemens", "GE", "Philips"]
FIELD_ORDER = ["1.5T", "3.0T"]
PHASE_MANUFACTURER_ORDER = [
    "ADNI3_Siemens",
    "ADNI4_Siemens",
    "ADNI1_GE",
    "ADNI1_Siemens",
    "ADNI3_GE",
    "ADNI3_Philips",
    "ADNI1_Philips",
    "ADNI4_GE",
    "ADNI4_Philips",
]

REGISTRY_COLUMNS = [
    "domain_type",
    "domain",
    "n",
    "ad",
    "cn",
    "primary_auroc_eligible",
    "stable_sensitivity_eligible",
    "planned_role",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_manufacturer(value: object) -> str:
    text = str(value).strip().upper()
    if "SIEMENS" in text:
        return "Siemens"
    if "PHILIPS" in text:
        return "Philips"
    if text == "GE" or text.startswith("GE ") or "GE MEDICAL" in text:
        return "GE"
    raise ValueError(f"Unrecognized scanner manufacturer label: {value!r}")


def canonicalize_phase(value: object) -> str:
    phase = str(value).strip().upper()
    if phase not in PHASE_ORDER:
        raise ValueError(f"Unrecognized ADNI phase: {value!r}")
    return phase


def canonicalize_field_strength(value: object) -> str:
    try:
        strength = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unrecognized field strength: {value!r}") from exc
    if abs(strength - 1.5) < 1e-8:
        return "1.5T"
    if abs(strength - 3.0) < 1e-8:
        return "3.0T"
    raise ValueError(f"Unrecognized field strength: {value!r}")


def is_primary_eligible(ad: int, cn: int) -> bool:
    return ad >= PRIMARY_MIN_AD and cn >= PRIMARY_MIN_CN


def is_stable_eligible(ad: int, cn: int) -> bool:
    return ad >= STABLE_MIN_AD and cn >= STABLE_MIN_CN


def build_assignments(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "RID",
        "PHASE",
        "ScannerManufacturer",
        "FieldSgth",
        "AD_label",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Source cohort is missing required columns: {missing}")
    if source["RID"].isna().any() or source["RID"].duplicated().any():
        raise ValueError("RID must be complete and unique in the baseline cohort")

    labels = pd.to_numeric(source["AD_label"], errors="raise").astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError("AD_label must contain exactly the binary values 0 and 1")

    out = pd.DataFrame(
        {
            "RID": source["RID"].astype(str),
            "phase": source["PHASE"].map(canonicalize_phase),
            "manufacturer": source["ScannerManufacturer"].map(canonicalize_manufacturer),
            "field_strength": source["FieldSgth"].map(canonicalize_field_strength),
            "label": labels,
        }
    )
    out["acquisition_domain"] = out["phase"] + "_" + out["manufacturer"]

    observed_domains = set(out["acquisition_domain"].unique())
    expected_domains = set(PHASE_MANUFACTURER_ORDER)
    if observed_domains != expected_domains:
        raise ValueError(
            "Observed phase-manufacturer domains differ from the frozen domain set: "
            f"missing={sorted(expected_domains - observed_domains)}, "
            f"unexpected={sorted(observed_domains - expected_domains)}"
        )

    counts = out.groupby("acquisition_domain")["label"].agg(
        n="size", ad="sum"
    )
    counts["cn"] = counts["n"] - counts["ad"]
    primary = {
        domain: is_primary_eligible(int(row.ad), int(row.cn))
        for domain, row in counts.iterrows()
    }
    stable = {
        domain: is_stable_eligible(int(row.ad), int(row.cn))
        for domain, row in counts.iterrows()
    }
    out["primary_auroc_eligible"] = out["acquisition_domain"].map(primary)
    out["stable_sensitivity_eligible"] = out["acquisition_domain"].map(stable)
    return out.sort_values("RID", kind="stable").reset_index(drop=True)


def _aggregate(assignments: pd.DataFrame, column: str) -> dict[str, tuple[int, int, int]]:
    grouped = assignments.groupby(column, sort=False)["label"].agg(n="size", ad="sum")
    grouped["cn"] = grouped["n"] - grouped["ad"]
    return {
        str(domain): (int(row.n), int(row.ad), int(row.cn))
        for domain, row in grouped.iterrows()
    }


def _planned_role(domain_type: str, domain: str, ad: int, cn: int) -> str:
    if domain_type == "phase":
        return {
            "ADNI1": "phase shift and prevalence shift",
            "ADNI3": "phase shift and development domain",
            "ADNI4": "latest-phase stress test",
        }[domain]
    if domain_type == "manufacturer":
        return "leave-manufacturer-out"
    if domain_type == "field_strength":
        return "field-strength stress test"
    if domain == "ADNI4_GE":
        return "specificity and calibration stress set only"
    if domain == "ADNI4_Philips":
        return "descriptive stress set only"
    if domain in {"ADNI3_Philips", "ADNI1_Philips"}:
        return "primary acquisition-domain CV with wide uncertainty"
    return "primary acquisition-domain CV"


def build_registry_rows(assignments: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs: Iterable[tuple[str, str, list[str]]] = [
        ("phase", "phase", PHASE_ORDER),
        ("manufacturer", "manufacturer", MANUFACTURER_ORDER),
        ("field_strength", "field_strength", FIELD_ORDER),
        ("phase_manufacturer", "acquisition_domain", PHASE_MANUFACTURER_ORDER),
    ]
    for domain_type, column, order in specs:
        aggregates = _aggregate(assignments, column)
        for domain in order:
            n, ad, cn = aggregates[domain]
            rows.append(
                {
                    "domain_type": domain_type,
                    "domain": domain,
                    "n": n,
                    "ad": ad,
                    "cn": cn,
                    "primary_auroc_eligible": "yes" if is_primary_eligible(ad, cn) else "no",
                    "stable_sensitivity_eligible": "yes" if is_stable_eligible(ad, cn) else "no",
                    "planned_role": _planned_role(domain_type, domain, ad, cn),
                }
            )
    return rows


def csv_bytes(rows: list[dict[str, object]], columns: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def assignment_csv_bytes(assignments: pd.DataFrame) -> bytes:
    return assignments.to_csv(index=False, lineterminator="\n").encode("utf-8")


def json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def relative_source_label(source_path: Path) -> str:
    try:
        return source_path.resolve().relative_to(HISS_ROOT.resolve()).as_posix()
    except ValueError:
        return source_path.name


def freeze_payloads(source_path: Path) -> dict[str, bytes]:
    source = pd.read_csv(source_path, low_memory=False)
    assignments = build_assignments(source)
    registry_rows = build_registry_rows(assignments)
    registry_payload = csv_bytes(registry_rows, REGISTRY_COLUMNS)
    assignments_payload = assignment_csv_bytes(assignments)

    domain_rows = [
        row for row in registry_rows if row["domain_type"] == "phase_manufacturer"
    ]
    definition = {
        "freeze_id": "CRUA-AD-FW0-ACQUISITION-DOMAINS-v2.1",
        "frozen_date": "2026-07-18",
        "source": {
            "path": relative_source_label(source_path),
            "sha256": sha256_file(source_path),
            "rows": int(len(assignments)),
            "ad": int(assignments["label"].sum()),
            "cn": int((1 - assignments["label"]).sum()),
        },
        "normalization": {
            "phase": {"allowed": PHASE_ORDER, "operation": "strip and uppercase"},
            "manufacturer": {
                "SIEMENS substring": "Siemens",
                "PHILIPS substring": "Philips",
                "GE/GE MEDICAL prefix": "GE",
                "unknown": "fail closed",
            },
            "field_strength": {"1.5": "1.5T", "3.0": "3.0T", "unknown": "fail closed"},
            "acquisition_domain": "canonical phase + '_' + canonical manufacturer",
        },
        "eligibility": {
            "primary_outer_auroc": {"minimum_ad": PRIMARY_MIN_AD, "minimum_cn": PRIMARY_MIN_CN},
            "stable_domain_sensitivity": {"minimum_ad": STABLE_MIN_AD, "minimum_cn": STABLE_MIN_CN},
        },
        "eligible_primary_domains": [
            row["domain"] for row in domain_rows if row["primary_auroc_eligible"] == "yes"
        ],
        "eligible_stable_domains": [
            row["domain"] for row in domain_rows if row["stable_sensitivity_eligible"] == "yes"
        ],
        "stress_only_domains": [
            row["domain"] for row in domain_rows if row["primary_auroc_eligible"] == "no"
        ],
        "registry": {
            "path": "data_registry/acquisition_domain_feasibility.csv",
            "sha256": sha256_bytes(registry_payload),
        },
        "private_assignment": {
            "path": "outputs/private/domain_assignments.csv",
            "rows": int(len(assignments)),
            "sha256": sha256_bytes(assignments_payload),
            "git_policy": "ignored; contains RID-level controlled data",
        },
    }
    definition_payload = json_bytes(definition)
    validation_log = {
        "check_id": "FW0_ACQUISITION_DOMAIN_FREEZE",
        "date": "2026-07-18",
        "status": "PASS",
        "checks": {
            "unique_rid": True,
            "binary_label": True,
            "known_phase": True,
            "known_manufacturer": True,
            "known_field_strength": True,
            "exactly_one_domain_per_subject": True,
            "expected_primary_domain_count": len(definition["eligible_primary_domains"]) == 7,
            "aggregate_total": sum(int(row["n"]) for row in domain_rows) == len(assignments),
        },
        "artifacts": {
            "source_sha256": definition["source"]["sha256"],
            "registry_sha256": sha256_bytes(registry_payload),
            "definition_sha256": sha256_bytes(definition_payload),
            "private_assignment_sha256": sha256_bytes(assignments_payload),
        },
        "counts": {
            "subjects": int(len(assignments)),
            "ad": int(assignments["label"].sum()),
            "cn": int((1 - assignments["label"]).sum()),
            "primary_domains": len(definition["eligible_primary_domains"]),
            "stable_domains": len(definition["eligible_stable_domains"]),
            "stress_only_domains": len(definition["stress_only_domains"]),
        },
        "privacy": "No participant identifiers are present in this log.",
    }
    return {
        "registry": registry_payload,
        "definition": definition_payload,
        "assignments": assignments_payload,
        "log": json_bytes(validation_log),
    }


def write_or_verify(path: Path, payload: bytes, verify: bool) -> None:
    if verify:
        if not path.exists():
            raise FileNotFoundError(f"Frozen artifact is missing: {path}")
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen artifact differs from regenerated content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--definition", type=Path, default=DEFAULT_DEFINITION)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = freeze_payloads(args.source)
    write_or_verify(args.registry, payloads["registry"], args.verify)
    write_or_verify(args.definition, payloads["definition"], args.verify)
    write_or_verify(args.assignments, payloads["assignments"], args.verify)
    write_or_verify(args.log, payloads["log"], args.verify)
    verb = "Verified" if args.verify else "Wrote"
    print(f"{verb} acquisition-domain freeze: {args.definition}")


if __name__ == "__main__":
    main()
