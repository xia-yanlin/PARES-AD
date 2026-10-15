#!/usr/bin/env python3
"""Build deterministic nested acquisition-domain splits for V2.1.

RID-level memberships are controlled data and are written only below the
Git-ignored ``outputs/private`` directory.  The tracked registry is aggregate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd


V2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSIGNMENTS = V2_ROOT / "outputs" / "private" / "domain_assignments.csv"
DEFAULT_DOMAIN_DEFINITION = V2_ROOT / "data_registry" / "acquisition_domain_definition.json"
DEFAULT_MEMBERSHIP = V2_ROOT / "outputs" / "private" / "nested_domain_splits.csv"
DEFAULT_REGISTRY = V2_ROOT / "data_registry" / "nested_domain_split_registry.csv"
DEFAULT_DEFINITION = V2_ROOT / "data_registry" / "nested_domain_split_definition.json"
DEFAULT_LOG = V2_ROOT / "logs" / "fw1_nested_split_validation.json"

REGISTRY_COLUMNS = [
    "outer_domain",
    "outer_test_n",
    "outer_test_ad",
    "outer_test_cn",
    "outer_train_n",
    "outer_train_ad",
    "outer_train_cn",
    "inner_validation_domain",
    "inner_validation_n",
    "inner_validation_ad",
    "inner_validation_cn",
    "inner_train_n",
    "inner_train_ad",
    "inner_train_cn",
    "stress_only_in_inner_train_n",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_nested_membership(
    assignments: pd.DataFrame,
    eligible_domains: list[str],
    stress_domains: list[str],
) -> pd.DataFrame:
    required = {"RID", "label", "acquisition_domain"}
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise ValueError(f"Assignment table is missing required columns: {missing}")
    if assignments["RID"].isna().any() or assignments["RID"].duplicated().any():
        raise ValueError("RID must be complete and unique")
    if set(assignments["acquisition_domain"].unique()) != set(eligible_domains + stress_domains):
        raise ValueError("Assignment domains do not match the frozen eligible and stress sets")

    frames: list[pd.DataFrame] = []
    for outer_domain in eligible_domains:
        frame = assignments[["RID", "label", "acquisition_domain"]].copy()
        frame.insert(0, "outer_domain", outer_domain)
        frame["outer_role"] = "train_pool"
        outer_mask = frame["acquisition_domain"].eq(outer_domain)
        frame.loc[outer_mask, "outer_role"] = "test"

        for inner_domain in eligible_domains:
            column = f"inner_role__{inner_domain}"
            if inner_domain == outer_domain:
                frame[column] = "not_applicable"
                continue
            frame[column] = "training"
            frame.loc[outer_mask, column] = "excluded_outer"
            frame.loc[frame["acquisition_domain"].eq(inner_domain), column] = "validation"
        frames.append(frame)

    membership = pd.concat(frames, ignore_index=True)
    domain_rank = {domain: index for index, domain in enumerate(eligible_domains)}
    membership["_outer_rank"] = membership["outer_domain"].map(domain_rank)
    membership = membership.sort_values(["_outer_rank", "RID"], kind="stable").drop(columns="_outer_rank")
    return membership.reset_index(drop=True)


def _counts(frame: pd.DataFrame) -> tuple[int, int, int]:
    n = int(len(frame))
    ad = int(frame["label"].sum())
    return n, ad, n - ad


def build_split_registry(
    membership: pd.DataFrame,
    eligible_domains: list[str],
    stress_domains: list[str],
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for outer_domain in eligible_domains:
        outer = membership[membership["outer_domain"].eq(outer_domain)]
        outer_test = outer[outer["outer_role"].eq("test")]
        outer_train = outer[outer["outer_role"].eq("train_pool")]
        test_n, test_ad, test_cn = _counts(outer_test)
        train_n, train_ad, train_cn = _counts(outer_train)
        for inner_domain in eligible_domains:
            if inner_domain == outer_domain:
                continue
            role_column = f"inner_role__{inner_domain}"
            inner_val = outer[outer[role_column].eq("validation")]
            inner_train = outer[outer[role_column].eq("training")]
            val_n, val_ad, val_cn = _counts(inner_val)
            inner_n, inner_ad, inner_cn = _counts(inner_train)
            stress_n = int(inner_train["acquisition_domain"].isin(stress_domains).sum())
            rows.append(
                {
                    "outer_domain": outer_domain,
                    "outer_test_n": test_n,
                    "outer_test_ad": test_ad,
                    "outer_test_cn": test_cn,
                    "outer_train_n": train_n,
                    "outer_train_ad": train_ad,
                    "outer_train_cn": train_cn,
                    "inner_validation_domain": inner_domain,
                    "inner_validation_n": val_n,
                    "inner_validation_ad": val_ad,
                    "inner_validation_cn": val_cn,
                    "inner_train_n": inner_n,
                    "inner_train_ad": inner_ad,
                    "inner_train_cn": inner_cn,
                    "stress_only_in_inner_train_n": stress_n,
                }
            )
    return rows


def validate_membership(
    membership: pd.DataFrame,
    eligible_domains: list[str],
    stress_domains: list[str],
) -> dict[str, bool]:
    subjects_per_outer = membership.groupby("outer_domain")["RID"].nunique()
    total_subjects = membership["RID"].nunique()
    one_row_per_subject_outer = bool((subjects_per_outer == total_subjects).all())

    eligible_rows = membership[membership["acquisition_domain"].isin(eligible_domains)]
    test_counts = eligible_rows.groupby("RID")["outer_role"].apply(lambda values: int((values == "test").sum()))
    stress_rows = membership[membership["acquisition_domain"].isin(stress_domains)]
    stress_test_count = int((stress_rows["outer_role"] == "test").sum())

    inner_disjoint = True
    outer_excluded = True
    six_inner_folds = True
    stress_always_training = True
    for outer_domain in eligible_domains:
        outer = membership[membership["outer_domain"].eq(outer_domain)]
        active_inner = [domain for domain in eligible_domains if domain != outer_domain]
        six_inner_folds &= len(active_inner) == 6
        for inner_domain in active_inner:
            column = f"inner_role__{inner_domain}"
            val_ids = set(outer.loc[outer[column].eq("validation"), "RID"])
            train_ids = set(outer.loc[outer[column].eq("training"), "RID"])
            excluded_ids = set(outer.loc[outer[column].eq("excluded_outer"), "RID"])
            inner_disjoint &= not (val_ids & train_ids)
            outer_test_ids = set(outer.loc[outer["outer_role"].eq("test"), "RID"])
            outer_excluded &= excluded_ids == outer_test_ids
            stress_subset = outer[outer["acquisition_domain"].isin(stress_domains)]
            stress_always_training &= bool((stress_subset[column] == "training").all())

    checks = {
        "one_row_per_subject_per_outer_loop": one_row_per_subject_outer,
        "each_eligible_subject_tested_once": bool((test_counts == 1).all()),
        "stress_subjects_never_outer_test": stress_test_count == 0,
        "six_inner_folds_per_outer": bool(six_inner_folds),
        "inner_training_validation_disjoint": bool(inner_disjoint),
        "outer_test_excluded_from_every_inner_fold": bool(outer_excluded),
        "stress_subjects_remain_training_only": bool(stress_always_training),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Nested split validation failed: {failed}")
    return checks


def csv_bytes(rows: list[dict[str, int | str]], columns: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_payloads(assignments_path: Path, domain_definition_path: Path) -> dict[str, bytes]:
    with domain_definition_path.open("r", encoding="utf-8") as handle:
        domain_definition = json.load(handle)
    eligible_domains = list(domain_definition["eligible_primary_domains"])
    stress_domains = list(domain_definition["stress_only_domains"])
    assignments = pd.read_csv(assignments_path, dtype={"RID": str})
    membership = build_nested_membership(assignments, eligible_domains, stress_domains)
    checks = validate_membership(membership, eligible_domains, stress_domains)
    registry_rows = build_split_registry(membership, eligible_domains, stress_domains)
    membership_payload = membership.to_csv(index=False, lineterminator="\n").encode("utf-8")
    registry_payload = csv_bytes(registry_rows, REGISTRY_COLUMNS)

    definition = {
        "freeze_id": "CRUA-AD-FW1-NESTED-DOMAIN-SPLITS-v2.1",
        "frozen_date": "2026-07-18",
        "status": "FROZEN",
        "eligible_outer_domains": eligible_domains,
        "stress_only_domains": stress_domains,
        "outer_rule": "one eligible acquisition domain is test; every other subject is training pool",
        "inner_rule": "each remaining eligible acquisition domain is validation once; all other non-outer subjects are training",
        "stress_rule": "stress-only domains never serve as outer or inner validation and remain in training partitions",
        "early_stopping_rule": "created downstream only from the current inner-training partition; no inner-validation or outer-test rows",
        "expected_outer_loops": 7,
        "expected_inner_folds_per_outer": 6,
        "source": {
            "path": "outputs/private/domain_assignments.csv",
            "sha256": sha256_file(assignments_path),
            "domain_definition_sha256": sha256_file(domain_definition_path),
        },
        "aggregate_registry": {
            "path": "data_registry/nested_domain_split_registry.csv",
            "rows": len(registry_rows),
            "sha256": sha256_bytes(registry_payload),
        },
        "private_membership": {
            "path": "outputs/private/nested_domain_splits.csv",
            "rows": len(membership),
            "sha256": sha256_bytes(membership_payload),
            "git_policy": "ignored; contains RID-level controlled data",
        },
    }
    definition_payload = json_bytes(definition)
    validation_log = {
        "check_id": "FW1_NESTED_DOMAIN_SPLITS",
        "date": "2026-07-18",
        "status": "PASS",
        "checks": checks,
        "counts": {
            "subjects": int(membership["RID"].nunique()),
            "outer_loops": len(eligible_domains),
            "inner_folds": len(registry_rows),
            "private_membership_rows": len(membership),
            "stress_only_domains": len(stress_domains),
        },
        "artifacts": {
            "registry_sha256": sha256_bytes(registry_payload),
            "definition_sha256": sha256_bytes(definition_payload),
            "private_membership_sha256": sha256_bytes(membership_payload),
        },
        "privacy": "No participant identifiers are present in this log.",
    }
    return {
        "membership": membership_payload,
        "registry": registry_payload,
        "definition": definition_payload,
        "log": json_bytes(validation_log),
    }


def write_or_verify(path: Path, payload: bytes, verify: bool) -> None:
    if verify:
        if not path.exists() or path.read_bytes() != payload:
            raise RuntimeError(f"Frozen artifact missing or changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--domain-definition", type=Path, default=DEFAULT_DOMAIN_DEFINITION)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--definition", type=Path, default=DEFAULT_DEFINITION)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = build_payloads(args.assignments, args.domain_definition)
    write_or_verify(args.membership, payloads["membership"], args.verify)
    write_or_verify(args.registry, payloads["registry"], args.verify)
    write_or_verify(args.definition, payloads["definition"], args.verify)
    write_or_verify(args.log, payloads["log"], args.verify)
    print(("Verified" if args.verify else "Wrote") + f" nested domain splits: {args.definition}")


if __name__ == "__main__":
    main()
