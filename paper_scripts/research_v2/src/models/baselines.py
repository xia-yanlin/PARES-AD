"""Factories for the three frozen strong-reference model families."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_reference_model(family: str, configuration: dict[str, Any], *, seed: int):
    if family == "logistic_regression":
        return LogisticRegression(
            C=float(configuration["C"]),
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )
    if family == "rbf_svm":
        return SVC(
            C=float(configuration["C"]),
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_leaf_nodes=int(configuration["max_leaf_nodes"]),
            learning_rate=float(configuration["learning_rate"]),
            max_iter=int(configuration["max_iter"]),
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"Unknown reference family: {family}")
