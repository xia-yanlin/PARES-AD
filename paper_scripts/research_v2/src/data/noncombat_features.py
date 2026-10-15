#!/usr/bin/env python3
"""Training-local non-ComBat ROI preprocessing for the V2.1 core study."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


V2_ROOT = Path(__file__).resolve().parents[2]
HISS_ROOT = V2_ROOT.parent
DEFAULT_COHORT = HISS_ROOT / "data" / "processed" / "baseline_cohort_with_splits.csv"
DEFAULT_V1_NODE_LIST = HISS_ROOT / "data" / "processed" / "node_list.json"
DEFAULT_MAPPING = V2_ROOT / "data_registry" / "noncombat_roi_mapping_v2_1.json"
DEFAULT_LOG = V2_ROOT / "logs" / "fw1_noncombat_input_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_noncombat_mapping(v1_nodes: list[dict], columns: Sequence[str]) -> list[dict]:
    available = set(columns)
    mapping: list[dict] = []
    for expected_index, node in enumerate(v1_nodes):
        if int(node["index"]) != expected_index:
            raise ValueError("V1 node list indices are not contiguous and ordered")
        thickness = node["thickness_col"]
        volume = node["volume_col"]
        thickness_noncombat = None if thickness is None else str(thickness).removesuffix("_combat")
        volume_noncombat = str(volume).removesuffix("_combat")
        required = [volume_noncombat] + ([] if thickness_noncombat is None else [thickness_noncombat])
        missing = [column for column in required if column not in available]
        if missing:
            raise ValueError(f"Non-ComBat ROI columns are missing for node {node['name']}: {missing}")
        mapping.append(
            {
                "index": expected_index,
                "name": node["name"],
                "type": node["type"],
                "thickness_col": thickness_noncombat,
                "volume_col": volume_noncombat,
            }
        )
    if len(mapping) != 94:
        raise ValueError(f"Expected 94 frozen ROI nodes, found {len(mapping)}")
    return mapping


def raw_roi_arrays(
    frame: pd.DataFrame, mapping: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_subjects = len(frame)
    n_nodes = len(mapping)
    thickness = np.full((n_subjects, n_nodes), np.nan, dtype=np.float64)
    volume = np.full((n_subjects, n_nodes), np.nan, dtype=np.float64)
    node_type = np.zeros(n_nodes, dtype=np.float64)
    for node in mapping:
        index = int(node["index"])
        if node["thickness_col"] is not None:
            thickness[:, index] = pd.to_numeric(frame[node["thickness_col"]], errors="coerce")
        volume[:, index] = pd.to_numeric(frame[node["volume_col"]], errors="coerce")
        node_type[index] = 1.0 if node["type"] == "subcortical" else 0.0
    icv = pd.to_numeric(frame["EstimatedTotalIntraCranialVol"], errors="coerce").to_numpy(float)
    return thickness, volume, node_type, icv


@dataclass(frozen=True)
class FrozenROITransformer:
    mapping: tuple[dict, ...]
    cortical_mask: np.ndarray
    thickness_median: np.ndarray
    thickness_mean: np.ndarray
    thickness_scale: np.ndarray
    volume_median: np.ndarray
    icv_median: float
    volume_beta: np.ndarray
    volume_residual_mean: np.ndarray
    volume_residual_scale: np.ndarray

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        mapping: list[dict],
        train_indices: Sequence[int],
    ) -> "FrozenROITransformer":
        train = np.asarray(train_indices, dtype=int)
        if train.ndim != 1 or len(train) < 2 or len(np.unique(train)) != len(train):
            raise ValueError("train_indices must contain at least two unique rows")
        thickness, volume, node_type, icv = raw_roi_arrays(frame, mapping)
        cortical = node_type == 0

        thickness_median = np.zeros(len(mapping), dtype=float)
        thickness_mean = np.zeros(len(mapping), dtype=float)
        thickness_scale = np.ones(len(mapping), dtype=float)
        for node in np.flatnonzero(cortical):
            median = float(np.nanmedian(thickness[train, node]))
            filled = np.where(np.isnan(thickness[:, node]), median, thickness[:, node])
            thickness_median[node] = median
            thickness_mean[node] = float(filled[train].mean())
            scale = float(filled[train].std(ddof=0))
            thickness_scale[node] = scale if scale > 0 else 1.0

        volume_median = np.nanmedian(volume[train], axis=0)
        if not np.isfinite(volume_median).all() or (volume_median <= 0).any():
            raise ValueError("Training volume medians must be finite and positive")
        volume_filled = np.where(np.isnan(volume), volume_median[None, :], volume)
        if (volume_filled <= 0).any():
            raise ValueError("ROI volumes must be positive before log transformation")
        icv_median = float(np.nanmedian(icv[train]))
        icv_filled = np.where(np.isnan(icv), icv_median, icv)
        if icv_median <= 0 or (icv_filled <= 0).any():
            raise ValueError("eTIV must be positive before log transformation")

        log_volume = np.log(volume_filled)
        log_icv = np.log(icv_filled)
        design = np.column_stack([np.ones(len(train)), log_icv[train]])
        volume_beta, *_ = np.linalg.lstsq(design, log_volume[train], rcond=None)
        residual = log_volume - np.column_stack([np.ones(len(frame)), log_icv]) @ volume_beta
        residual_mean = residual[train].mean(axis=0)
        residual_scale = residual[train].std(axis=0, ddof=0)
        residual_scale[residual_scale <= 0] = 1.0

        return cls(
            mapping=tuple(mapping),
            cortical_mask=cortical,
            thickness_median=thickness_median,
            thickness_mean=thickness_mean,
            thickness_scale=thickness_scale,
            volume_median=volume_median,
            icv_median=icv_median,
            volume_beta=volume_beta,
            volume_residual_mean=residual_mean,
            volume_residual_scale=residual_scale,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        thickness, volume, node_type, icv = raw_roi_arrays(frame, list(self.mapping))
        thickness_out = np.zeros_like(thickness)
        for node in np.flatnonzero(self.cortical_mask):
            filled = np.where(np.isnan(thickness[:, node]), self.thickness_median[node], thickness[:, node])
            thickness_out[:, node] = (filled - self.thickness_mean[node]) / self.thickness_scale[node]

        volume_filled = np.where(np.isnan(volume), self.volume_median[None, :], volume)
        icv_filled = np.where(np.isnan(icv), self.icv_median, icv)
        if (volume_filled <= 0).any() or (icv_filled <= 0).any():
            raise ValueError("ROI volumes and eTIV must be positive")
        design = np.column_stack([np.ones(len(frame)), np.log(icv_filled)])
        residual = np.log(volume_filled) - design @ self.volume_beta
        volume_out = (residual - self.volume_residual_mean) / self.volume_residual_scale
        type_out = np.broadcast_to(node_type[None, :], volume_out.shape)
        output = np.stack([thickness_out, volume_out, type_out], axis=-1).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("Transformed ROI tensor contains non-finite values")
        return output


def build_structural_covariance_adjacency(
    transformed_features: np.ndarray,
    train_indices: Sequence[int],
    *,
    top_k: int,
) -> np.ndarray:
    train = np.asarray(train_indices, dtype=int)
    if transformed_features.ndim != 3 or transformed_features.shape[2] < 2:
        raise ValueError("Expected features with shape (subjects, nodes, >=2)")
    n_nodes = transformed_features.shape[1]
    if not 1 <= top_k < n_nodes:
        raise ValueError("top_k must be between 1 and n_nodes - 1")
    correlation = np.corrcoef(transformed_features[train, :, 1], rowvar=False)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for node in range(n_nodes):
        neighbors = np.argsort(-correlation[node], kind="stable")[:top_k]
        adjacency[node, neighbors] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def build_mapping_artifact(cohort_path: Path, node_list_path: Path) -> tuple[dict, dict]:
    frame = pd.read_csv(cohort_path, nrows=1)
    with node_list_path.open("r", encoding="utf-8") as handle:
        v1_nodes = json.load(handle)
    mapping = build_noncombat_mapping(v1_nodes, frame.columns)
    artifact = {
        "freeze_id": "CRUA-AD-FW1-NONCOMBAT-ROI-MAPPING-v2.1",
        "frozen_date": "2026-07-18",
        "status": "FROZEN",
        "source": {
            "path": "data/processed/baseline_cohort_with_splits.csv",
            "sha256": sha256_file(cohort_path),
            "v1_node_list_sha256": sha256_file(node_list_path),
        },
        "input": {
            "nodes": len(mapping),
            "cortical_nodes": sum(node["type"] == "cortical" for node in mapping),
            "subcortical_nodes": sum(node["type"] == "subcortical" for node in mapping),
            "features": ["training-standardized thickness", "training-local log-volume residual", "is_subcortical"],
            "combat_columns_used": False,
        },
        "nodes": mapping,
    }
    log = {
        "check_id": "FW1_NONCOMBAT_INPUT_MAPPING",
        "date": "2026-07-18",
        "status": "PASS",
        "checks": {
            "exactly_94_nodes": len(mapping) == 94,
            "all_noncombat_columns_present": True,
            "combat_columns_excluded": all(
                not str(node.get(key) or "").endswith("_combat")
                for node in mapping
                for key in ["thickness_col", "volume_col"]
            ),
        },
        "counts": {
            "nodes": len(mapping),
            "cortical_nodes": artifact["input"]["cortical_nodes"],
            "subcortical_nodes": artifact["input"]["subcortical_nodes"],
        },
        "privacy": "No participant identifiers or rows are present in this log.",
    }
    return artifact, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--v1-node-list", type=Path, default=DEFAULT_V1_NODE_LIST)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact, log = build_mapping_artifact(args.cohort, args.v1_node_list)
    payloads = {
        args.mapping: (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        args.log: (json.dumps(log, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }
    for path, payload in payloads.items():
        if args.verify:
            if not path.exists() or path.read_bytes() != payload:
                raise RuntimeError(f"Frozen artifact missing or changed: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print(("Verified" if args.verify else "Wrote") + f" non-ComBat ROI mapping: {args.mapping}")


if __name__ == "__main__":
    main()
