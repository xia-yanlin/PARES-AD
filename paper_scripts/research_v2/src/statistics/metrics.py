"""Binary prediction, calibration, and metric helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def sigmoid(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    positive = scores >= 0
    output = np.empty_like(scores)
    output[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    exp_scores = np.exp(scores[~positive])
    output[~positive] = exp_scores / (1.0 + exp_scores)
    return output


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class PlattCalibration:
    coefficient: float
    intercept: float

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return sigmoid(self.intercept + self.coefficient * np.asarray(scores, dtype=float))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattCalibration:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(scores) != len(labels) or len(scores) < 4:
        raise ValueError("Platt calibration requires aligned scores and labels")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Platt calibration requires both outcome classes")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(scores[:, None], labels)
    return PlattCalibration(float(model.coef_[0, 0]), float(model.intercept_[0]))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    ece = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return float(ece)


def calibration_intercept_slope(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    predictor = logit(probabilities)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(predictor[:, None], labels)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(labels) != len(probabilities) or not len(labels):
        raise ValueError("Labels and probabilities must be non-empty and aligned")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Binary metrics require both outcome classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(labels, probabilities)
    return {
        "n": int(len(labels)),
        "ad": int(labels.sum()),
        "cn": int((1 - labels).sum()),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10": expected_calibration_error(labels, probabilities, bins=10),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "threshold": float(threshold),
    }
