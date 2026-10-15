"""Frozen V2.1 concept definitions with training-partition-only fitting.

The functions in this module deliberately accept separate probe-training and
evaluation frames.  Every fitted median, regression coefficient, mean, and
scale is derived only from ``train_frame``.  Missing primary imaging components
are never imputed; affected rows are marked invalid for that concept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from research_v2.src.data.build_acquisition_domains import (
    canonicalize_field_strength,
    canonicalize_manufacturer,
    canonicalize_phase,
)


CONCEPT_IDS = (
    "C001",
    "C002",
    "C003",
    "C004",
    "C005",
    "C006",
    "C007",
    "D001",
    "D002",
    "D003",
    "D004",
    "S001",
    "S002",
    "S003",
    "S004",
    "N003",
)

CONTINUOUS_CONCEPT_IDS = frozenset(
    {"C001", "C002", "C003", "C004", "C005", "C006", "C007", "D001", "D003", "D004", "N003"}
)
CATEGORICAL_CONCEPT_IDS = frozenset({"D002", "S001", "S002", "S003", "S004"})

CONCEPT_NAMES = {
    "C001": "hippocampal_atrophy",
    "C002": "entorhinal_thinning",
    "C003": "ventricular_enlargement",
    "C004": "temporoparietal_thinning",
    "C005": "whole_brain_atrophy",
    "C006": "four_core_joint",
    "C007": "hippocampal_atrophy_conditional",
    "D001": "age",
    "D002": "sex",
    "D003": "education",
    "D004": "ICV",
    "S001": "scanner_manufacturer",
    "S002": "field_strength",
    "S003": "adni_phase",
    "S004": "acquisition_domain",
    "N003": "low_prior_anatomical_control",
}

CONCEPT_FAMILIES = {
    "C001": "disease",
    "C002": "disease",
    "C003": "disease",
    "C004": "disease",
    "C005": "disease",
    "C006": "disease joint",
    "C007": "disease conditional",
    "D001": "demographic",
    "D002": "demographic",
    "D003": "demographic",
    "D004": "anatomical scale",
    "S001": "scanner/domain",
    "S002": "scanner/domain",
    "S003": "scanner/domain",
    "S004": "scanner/domain",
    "N003": "negative control",
}

CATEGORY_LEVELS = {
    "D002": ("F", "M"),
    "S001": ("Siemens", "GE", "Philips"),
    "S002": ("1.5T", "3.0T"),
    "S003": ("ADNI1", "ADNI3", "ADNI4"),
    "S004": (
        "ADNI3_Siemens",
        "ADNI4_Siemens",
        "ADNI1_GE",
        "ADNI1_Siemens",
        "ADNI3_GE",
        "ADNI3_Philips",
        "ADNI1_Philips",
        "ADNI4_GE",
        "ADNI4_Philips",
    ),
}

TEMPOROPARIETAL_COLUMNS = tuple(
    f"{hemisphere}_{region}_thickness"
    for hemisphere in ("lh", "rh")
    for region in ("inferiortemporal", "middletemporal", "inferiorparietal", "supramarginal", "precuneus")
)


@dataclass(frozen=True)
class ConceptTarget:
    concept_id: str
    concept_name: str
    concept_family: str
    data_type: str
    train_values: np.ndarray
    evaluation_values: np.ndarray
    train_valid: np.ndarray
    evaluation_valid: np.ndarray
    category_levels: tuple[str, ...] = ()

    def validate(self, train_n: int, evaluation_n: int) -> None:
        if self.concept_id not in CONCEPT_IDS:
            raise ValueError(f"Unregistered concept: {self.concept_id}")
        if self.data_type not in {"continuous", "categorical"}:
            raise ValueError("Concept data_type must be continuous or categorical")
        if len(self.train_values) != train_n or len(self.train_valid) != train_n:
            raise ValueError("Training concept arrays do not align with the training frame")
        if len(self.evaluation_values) != evaluation_n or len(self.evaluation_valid) != evaluation_n:
            raise ValueError("Evaluation concept arrays do not align with the evaluation frame")
        if self.data_type == "continuous":
            if not np.isfinite(np.asarray(self.train_values)[self.train_valid].astype(float)).all():
                raise ValueError("Valid continuous training targets must be finite")
            if not np.isfinite(np.asarray(self.evaluation_values)[self.evaluation_valid].astype(float)).all():
                raise ValueError("Valid continuous evaluation targets must be finite")
        elif not self.category_levels:
            raise ValueError("Categorical concepts require frozen category levels")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Required concept columns are missing: {missing}")
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _complete_sum(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = _numeric(frame, columns)
    valid = np.isfinite(values).all(axis=1)
    out = np.full(len(frame), np.nan, dtype=float)
    out[valid] = values[valid].sum(axis=1)
    return out


def _complete_mean(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = _numeric(frame, columns)
    valid = np.isfinite(values).all(axis=1)
    out = np.full(len(frame), np.nan, dtype=float)
    out[valid] = values[valid].mean(axis=1)
    return out


def _zscore_from_training(train_raw: np.ndarray, evaluation_raw: np.ndarray, *, impute: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_raw = np.asarray(train_raw, dtype=float)
    evaluation_raw = np.asarray(evaluation_raw, dtype=float)
    train_valid = np.isfinite(train_raw)
    evaluation_valid = np.isfinite(evaluation_raw)
    if not train_valid.any():
        raise ValueError("No finite training values are available for the concept")
    fitted = train_raw[train_valid]
    median = float(np.median(fitted))
    if impute:
        train_raw = np.where(train_valid, train_raw, median)
        evaluation_raw = np.where(evaluation_valid, evaluation_raw, median)
        train_valid = np.ones(len(train_raw), dtype=bool)
        evaluation_valid = np.ones(len(evaluation_raw), dtype=bool)
        fitted = train_raw
    mean = float(fitted.mean())
    scale = float(fitted.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training concept values have zero or invalid variance")
    return (
        (train_raw - mean) / scale,
        (evaluation_raw - mean) / scale,
        train_valid,
        evaluation_valid,
    )


def _log_icv_residual(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    value_builder: Callable[[pd.DataFrame], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_value = value_builder(train_frame)
    evaluation_value = value_builder(evaluation_frame)
    train_icv = _numeric(train_frame, ("EstimatedTotalIntraCranialVol",)).reshape(-1)
    evaluation_icv = _numeric(evaluation_frame, ("EstimatedTotalIntraCranialVol",)).reshape(-1)
    train_valid = np.isfinite(train_value) & np.isfinite(train_icv) & (train_value > 0) & (train_icv > 0)
    evaluation_valid = (
        np.isfinite(evaluation_value)
        & np.isfinite(evaluation_icv)
        & (evaluation_value > 0)
        & (evaluation_icv > 0)
    )
    if train_valid.sum() < 3:
        raise ValueError("Too few complete positive training rows for log-eTIV residualization")
    design = np.column_stack([np.ones(train_valid.sum()), np.log(train_icv[train_valid])])
    beta, *_ = np.linalg.lstsq(design, np.log(train_value[train_valid]), rcond=None)
    train_residual = np.full(len(train_frame), np.nan, dtype=float)
    evaluation_residual = np.full(len(evaluation_frame), np.nan, dtype=float)
    train_residual[train_valid] = np.log(train_value[train_valid]) - design @ beta
    evaluation_design = np.column_stack(
        [np.ones(evaluation_valid.sum()), np.log(evaluation_icv[evaluation_valid])]
    )
    evaluation_residual[evaluation_valid] = (
        np.log(evaluation_value[evaluation_valid]) - evaluation_design @ beta
    )
    train_scaled, evaluation_scaled, _, _ = _zscore_from_training(train_residual, evaluation_residual)
    return train_scaled, evaluation_scaled, train_valid, evaluation_valid


def _core_components(train_frame: pd.DataFrame, evaluation_frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "C001": _log_icv_residual(
            train_frame,
            evaluation_frame,
            lambda frame: _complete_sum(frame, ("Left_Hippocampus", "Right_Hippocampus")),
        ),
        "C002": _zscore_from_training(
            _complete_mean(train_frame, ("lh_entorhinal_thickness", "rh_entorhinal_thickness")),
            _complete_mean(evaluation_frame, ("lh_entorhinal_thickness", "rh_entorhinal_thickness")),
        ),
        "C003": _log_icv_residual(
            train_frame,
            evaluation_frame,
            lambda frame: _complete_sum(
                frame,
                (
                    "Left_Lateral_Ventricle",
                    "Right_Lateral_Ventricle",
                    "Left_Inf_Lat_Vent",
                    "Right_Inf_Lat_Vent",
                ),
            ),
        ),
        "C004": _zscore_from_training(
            _complete_mean(train_frame, TEMPOROPARIETAL_COLUMNS),
            _complete_mean(evaluation_frame, TEMPOROPARIETAL_COLUMNS),
        ),
    }


def _categorical_values(concept_id: str, frame: pd.DataFrame) -> np.ndarray:
    if concept_id == "D002":
        if "PTGENDER" not in frame:
            raise ValueError("PTGENDER is required for D002")
        values = frame["PTGENDER"].astype(str).str.strip().str.upper().to_numpy(object)
        canonical = np.asarray(
            [
                "F"
                if value in {"F", "FEMALE", "2", "2.0"}
                else "M"
                if value in {"M", "MALE", "1", "1.0"}
                else ""
                for value in values
            ],
            dtype=object,
        )
    elif concept_id == "S001":
        canonical = frame["ScannerManufacturer"].map(canonicalize_manufacturer).to_numpy(object)
    elif concept_id == "S002":
        canonical = frame["FieldSgth"].map(canonicalize_field_strength).to_numpy(object)
    elif concept_id == "S003":
        canonical = frame["PHASE"].map(canonicalize_phase).to_numpy(object)
    elif concept_id == "S004":
        phase = frame["PHASE"].map(canonicalize_phase)
        manufacturer = frame["ScannerManufacturer"].map(canonicalize_manufacturer)
        canonical = (phase + "_" + manufacturer).to_numpy(object)
    else:
        raise ValueError(f"Concept is not categorical: {concept_id}")
    levels = set(CATEGORY_LEVELS[concept_id])
    invalid = [str(value) for value in canonical if value not in levels]
    if invalid:
        raise ValueError(f"Unregistered categorical values for {concept_id}: {sorted(set(invalid))}")
    return canonical


def build_concept_target(
    concept_id: str,
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> ConceptTarget:
    """Fit one frozen concept on probe-training rows and transform evaluation rows."""

    if concept_id not in CONCEPT_IDS:
        raise ValueError(f"Unregistered concept: {concept_id}")
    if len(train_frame) < 3 or len(evaluation_frame) < 1:
        raise ValueError("Concept fitting requires at least three training rows and one evaluation row")

    if concept_id in CATEGORICAL_CONCEPT_IDS:
        train_values = _categorical_values(concept_id, train_frame)
        evaluation_values = _categorical_values(concept_id, evaluation_frame)
        target = ConceptTarget(
            concept_id,
            CONCEPT_NAMES[concept_id],
            CONCEPT_FAMILIES[concept_id],
            "categorical",
            train_values,
            evaluation_values,
            np.ones(len(train_frame), dtype=bool),
            np.ones(len(evaluation_frame), dtype=bool),
            CATEGORY_LEVELS[concept_id],
        )
        target.validate(len(train_frame), len(evaluation_frame))
        return target

    if concept_id in {"C001", "C002", "C003", "C004", "C006", "C007"}:
        core = _core_components(train_frame, evaluation_frame)
    else:
        core = {}
    if concept_id in core:
        train_values, evaluation_values, train_valid, evaluation_valid = core[concept_id]
    elif concept_id == "C005":
        train_values, evaluation_values, train_valid, evaluation_valid = _log_icv_residual(
            train_frame,
            evaluation_frame,
            lambda frame: _numeric(frame, ("BrainSegVolNotVent",)).reshape(-1),
        )
    elif concept_id == "D001":
        train_values, evaluation_values, train_valid, evaluation_valid = _zscore_from_training(
            _numeric(train_frame, ("PHC_Age_T1",)).reshape(-1),
            _numeric(evaluation_frame, ("PHC_Age_T1",)).reshape(-1),
        )
    elif concept_id == "D003":
        train_values, evaluation_values, train_valid, evaluation_valid = _zscore_from_training(
            _numeric(train_frame, ("PTEDUCAT",)).reshape(-1),
            _numeric(evaluation_frame, ("PTEDUCAT",)).reshape(-1),
            impute=True,
        )
    elif concept_id == "D004":
        train_raw = _numeric(train_frame, ("EstimatedTotalIntraCranialVol",)).reshape(-1)
        evaluation_raw = _numeric(evaluation_frame, ("EstimatedTotalIntraCranialVol",)).reshape(-1)
        train_raw = np.where(train_raw > 0, np.log(train_raw), np.nan)
        evaluation_raw = np.where(evaluation_raw > 0, np.log(evaluation_raw), np.nan)
        train_values, evaluation_values, train_valid, evaluation_valid = _zscore_from_training(train_raw, evaluation_raw)
    elif concept_id == "N003":
        train_values, evaluation_values, train_valid, evaluation_valid = _zscore_from_training(
            _complete_mean(train_frame, ("lh_pericalcarine_thickness", "rh_pericalcarine_thickness")),
            _complete_mean(evaluation_frame, ("lh_pericalcarine_thickness", "rh_pericalcarine_thickness")),
        )
    elif concept_id in {"C006", "C007"}:
        # C006 and C007 depend on the four fitted core components.
        oriented_train = np.column_stack([-core["C001"][0], -core["C002"][0], core["C003"][0], -core["C004"][0]])
        oriented_evaluation = np.column_stack([-core["C001"][1], -core["C002"][1], core["C003"][1], -core["C004"][1]])
        valid_train = np.isfinite(oriented_train).all(axis=1)
        valid_evaluation = np.isfinite(oriented_evaluation).all(axis=1)
        if concept_id == "C006":
            raw_train = np.full(len(train_frame), np.nan, dtype=float)
            raw_evaluation = np.full(len(evaluation_frame), np.nan, dtype=float)
            raw_train[valid_train] = oriented_train[valid_train].mean(axis=1)
            raw_evaluation[valid_evaluation] = oriented_evaluation[valid_evaluation].mean(axis=1)
            train_values, evaluation_values, _, _ = _zscore_from_training(raw_train, raw_evaluation)
            train_valid, evaluation_valid = valid_train, valid_evaluation
        elif concept_id == "C007":
            if valid_train.sum() < 5:
                raise ValueError("Too few complete rows for conditional hippocampal concept")
            design_train = np.column_stack([np.ones(valid_train.sum()), oriented_train[valid_train, 1:]])
            beta, *_ = np.linalg.lstsq(design_train, oriented_train[valid_train, 0], rcond=None)
            raw_train = np.full(len(train_frame), np.nan, dtype=float)
            raw_evaluation = np.full(len(evaluation_frame), np.nan, dtype=float)
            raw_train[valid_train] = oriented_train[valid_train, 0] - design_train @ beta
            eval_design = np.column_stack([np.ones(valid_evaluation.sum()), oriented_evaluation[valid_evaluation, 1:]])
            raw_evaluation[valid_evaluation] = oriented_evaluation[valid_evaluation, 0] - eval_design @ beta
            train_values, evaluation_values, _, _ = _zscore_from_training(raw_train, raw_evaluation)
            train_valid, evaluation_valid = valid_train, valid_evaluation
    else:
        raise AssertionError(f"Unhandled continuous concept: {concept_id}")

    target = ConceptTarget(
        concept_id,
        CONCEPT_NAMES[concept_id],
        CONCEPT_FAMILIES[concept_id],
        "continuous",
        np.asarray(train_values, dtype=float),
        np.asarray(evaluation_values, dtype=float),
        np.asarray(train_valid, dtype=bool),
        np.asarray(evaluation_valid, dtype=bool),
    )
    target.validate(len(train_frame), len(evaluation_frame))
    return target
