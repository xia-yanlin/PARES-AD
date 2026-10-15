"""Deterministic linear and capacity-limited nonlinear concept probes."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .concepts import ConceptTarget


PROBE_FAMILIES = ("linear", "bounded_nonlinear")
CONTROL_MODES = ("actual", "permuted_train_labels")


@dataclass(frozen=True)
class ProbeOutput:
    status: str
    metrics: dict[str, float | int | None]
    evaluation_prediction: np.ndarray
    evaluation_probabilities: np.ndarray | None
    learned_classes: tuple[str, ...]
    iterations: int | None


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def deterministic_learning_subset(frame: pd.DataFrame, fraction: float, *, seed: int) -> np.ndarray:
    """Select a deterministic acquisition-domain/label-stratified learning subset."""

    if not 0 < fraction <= 1:
        raise ValueError("Learning-curve fraction must be in (0, 1]")
    required = {"RID", "AD_label", "PHASE", "ScannerManufacturer"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Learning subset frame is missing columns: {missing}")
    if frame["RID"].astype(str).duplicated().any():
        raise ValueError("Learning subset RIDs must be unique")
    if fraction == 1:
        return np.arange(len(frame), dtype=int)

    phase = frame["PHASE"].astype(str).str.strip().str.upper()
    manufacturer = frame["ScannerManufacturer"].astype(str).str.strip().str.upper()
    strata = phase + "|" + manufacturer + "|" + frame["AD_label"].astype(int).astype(str)
    selected: list[int] = []
    for _, positions in pd.Series(np.arange(len(frame)), index=frame.index).groupby(strata, sort=True):
        candidates = positions.to_numpy(dtype=int)
        ranked = sorted(
            candidates,
            key=lambda position: hashlib.sha256(
                f"{seed}|{frame.iloc[position]['RID']}".encode("utf-8")
            ).digest(),
        )
        take = max(1, int(np.ceil(fraction * len(ranked))))
        selected.extend(ranked[:take])
    return np.asarray(sorted(selected), dtype=int)


def permute_training_labels(values: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(values).copy()
    if len(values) < 2:
        raise ValueError("At least two training labels are required for permutation")
    return values[np.random.default_rng(seed).permutation(len(values))]


def _continuous_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    if len(y_true) < 2 or np.std(y_true) <= 0:
        return {"pearson_r": None, "spearman_r": None, "r2": None, "mae": None, "evaluation_class_count": None}
    pearson = float(pearsonr(y_true, prediction).statistic) if np.std(prediction) > 0 else None
    spearman = float(spearmanr(y_true, prediction).statistic) if np.std(prediction) > 0 else None
    return {
        "pearson_r": pearson,
        "spearman_r": spearman,
        "r2": float(r2_score(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "evaluation_class_count": None,
    }


def _categorical_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> tuple[str, dict[str, float | int | None]]:
    observed = np.unique(y_true)
    if len(observed) < 2:
        return "not_estimable_single_evaluation_class", {
            "auroc": None,
            "auprc": None,
            "balanced_accuracy": None,
            "log_loss": None,
            "evaluation_class_count": int(len(observed)),
        }
    if not set(observed).issubset(set(classes)):
        return "not_estimable_unseen_evaluation_class", {
            "auroc": None,
            "auprc": None,
            "balanced_accuracy": None,
            "log_loss": None,
            "evaluation_class_count": int(len(observed)),
        }
    aurocs: list[float] = []
    auprcs: list[float] = []
    for class_index, class_name in enumerate(classes):
        binary = (y_true == class_name).astype(int)
        if len(np.unique(binary)) != 2:
            continue
        aurocs.append(float(roc_auc_score(binary, probabilities[:, class_index])))
        auprcs.append(float(average_precision_score(binary, probabilities[:, class_index])))
    status = "pass" if aurocs else "not_estimable_no_one_vs_rest_contrast"
    return status, {
        "auroc": float(np.mean(aurocs)) if aurocs else None,
        "auprc": float(np.mean(auprcs)) if auprcs else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)) if aurocs else None,
        "log_loss": float(log_loss(y_true, probabilities, labels=list(classes))) if aurocs else None,
        "evaluation_class_count": int(len(observed)),
    }


def _iteration_count(model: object) -> int | None:
    value = getattr(model, "n_iter_", None)
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    count = int(array.max())
    return count or None


def fit_probe(
    train_representation: np.ndarray,
    evaluation_representation: np.ndarray,
    target: ConceptTarget,
    *,
    probe_family: str,
    control_mode: str,
    seed: int,
) -> ProbeOutput:
    """Fit a matched probe without modifying evaluation labels or rows."""

    if probe_family not in PROBE_FAMILIES:
        raise ValueError(f"Unregistered probe family: {probe_family}")
    if control_mode not in CONTROL_MODES:
        raise ValueError(f"Unregistered control mode: {control_mode}")
    x_train = np.asarray(train_representation, dtype=float)
    x_evaluation = np.asarray(evaluation_representation, dtype=float)
    if x_train.ndim != 2 or x_evaluation.ndim != 2 or x_train.shape[1] != x_evaluation.shape[1]:
        raise ValueError("Probe representations must be aligned 2D arrays")
    if len(x_train) != len(target.train_values) or len(x_evaluation) != len(target.evaluation_values):
        raise ValueError("Probe representations and concept targets do not align")
    train_mask = np.asarray(target.train_valid, dtype=bool)
    evaluation_mask = np.asarray(target.evaluation_valid, dtype=bool)
    if train_mask.sum() < 8 or evaluation_mask.sum() < 2:
        return ProbeOutput("not_estimable_insufficient_rows", {}, np.asarray([]), None, (), None)
    scaler = StandardScaler().fit(x_train[train_mask])
    train_x = scaler.transform(x_train[train_mask])
    evaluation_x = scaler.transform(x_evaluation[evaluation_mask])
    train_y = np.asarray(target.train_values)[train_mask].copy()
    evaluation_y = np.asarray(target.evaluation_values)[evaluation_mask].copy()
    if control_mode == "permuted_train_labels":
        train_y = permute_training_labels(train_y, seed=seed)

    iterations: int | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if target.data_type == "continuous":
            if not np.isfinite(train_y.astype(float)).all() or np.std(train_y.astype(float)) <= 0:
                return ProbeOutput("not_estimable_constant_training_target", {}, np.asarray([]), None, (), None)
            if probe_family == "linear":
                model = Ridge(alpha=10.0)
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=(16,),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    batch_size=min(32, int(train_mask.sum())),
                    learning_rate_init=0.001,
                    max_iter=120,
                    shuffle=True,
                    random_state=seed,
                    tol=1e-4,
                    n_iter_no_change=12,
                    early_stopping=False,
                )
            model.fit(train_x, train_y.astype(float))
            prediction = np.asarray(model.predict(evaluation_x), dtype=float)
            iterations = _iteration_count(model)
            return ProbeOutput(
                "pass",
                _continuous_metrics(evaluation_y.astype(float), prediction),
                prediction,
                None,
                (),
                iterations,
            )

        train_y = train_y.astype(str)
        evaluation_y = evaluation_y.astype(str)
        if len(np.unique(train_y)) < 2:
            return ProbeOutput("not_estimable_constant_training_target", {}, np.asarray([]), None, (), None)
        if probe_family == "linear":
            model = LogisticRegression(
                C=1.0,
                l1_ratio=0.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            )
        else:
            model = MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=min(32, int(train_mask.sum())),
                learning_rate_init=0.001,
                max_iter=120,
                shuffle=True,
                random_state=seed,
                tol=1e-4,
                n_iter_no_change=12,
                early_stopping=False,
            )
        model.fit(train_x, train_y)
        probabilities = np.asarray(model.predict_proba(evaluation_x), dtype=float)
        prediction = np.asarray(model.predict(evaluation_x), dtype=object)
        classes = np.asarray(model.classes_, dtype=object)
        status, metrics = _categorical_metrics(evaluation_y, prediction, probabilities, classes)
        iterations = _iteration_count(model)
        return ProbeOutput(status, metrics, prediction, probabilities, tuple(str(value) for value in classes), iterations)
