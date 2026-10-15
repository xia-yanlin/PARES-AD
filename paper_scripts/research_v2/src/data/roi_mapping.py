"""Explicit ROI mapping helpers used only after a registry is frozen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConceptDefinition:
    concept_id: str
    roi_names: tuple[str, ...]
    aggregation: str
    unit: str
    expected_ad_direction: str

    def validate(self) -> None:
        if not self.concept_id or not self.roi_names:
            raise ValueError("concept_id and roi_names are required")
        if len(set(self.roi_names)) != len(self.roi_names):
            raise ValueError("ROI names must be unique within a concept")
        if self.aggregation not in {"mean", "sum"}:
            raise ValueError("aggregation must be mean or sum")
        if self.expected_ad_direction not in {"higher", "lower", "none"}:
            raise ValueError("unexpected AD direction")


def aggregate_concept(
    roi_columns: Mapping[str, Sequence[float]],
    definition: ConceptDefinition,
) -> np.ndarray:
    definition.validate()
    missing = [name for name in definition.roi_names if name not in roi_columns]
    if missing:
        raise KeyError(f"missing registered ROIs: {', '.join(missing)}")
    arrays = [np.asarray(roi_columns[name], dtype=float) for name in definition.roi_names]
    lengths = {array.shape for array in arrays}
    if len(lengths) != 1 or arrays[0].ndim != 1:
        raise ValueError("every ROI must be a one-dimensional array of equal length")
    matrix = np.column_stack(arrays)
    if not np.isfinite(matrix).all():
        raise ValueError("primary concept ROI values must be complete and finite")
    if definition.aggregation == "sum":
        return matrix.sum(axis=1)
    return matrix.mean(axis=1)


def train_only_standardize(train_values: Sequence[float], values: Sequence[float]) -> np.ndarray:
    train = np.asarray(train_values, dtype=float)
    target = np.asarray(values, dtype=float)
    if train.ndim != 1 or target.ndim != 1 or not np.isfinite(train).all() or not np.isfinite(target).all():
        raise ValueError("standardization inputs must be finite one-dimensional arrays")
    scale = train.std(ddof=1)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("training standard deviation must be positive")
    return (target - train.mean()) / scale
