"""Framework-neutral representation contract for CRUA-AD neural models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    family: str
    configuration_id: str
    seed: int
    fold: str
    checkpoint_sha256: str
    input_manifest_sha256: str
    parameter_count: int


@dataclass(frozen=True)
class RepresentationBatch:
    subject_keys: tuple[str, ...]
    layer_name: str
    values: np.ndarray
    metadata: ModelMetadata

    def validate(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("representation values must be a 2D array")
        if self.values.shape[0] != len(self.subject_keys):
            raise ValueError("subject key count does not match representation rows")
        if len(set(self.subject_keys)) != len(self.subject_keys):
            raise ValueError("subject keys must be unique within a batch")
        if not self.subject_keys:
            raise ValueError("representation batch must not be empty")
        if not self.layer_name:
            raise ValueError("layer_name must be frozen and non-empty")
        if not np.issubdtype(self.values.dtype, np.number):
            raise ValueError("representation values must be numeric")
        if not np.isfinite(self.values).all():
            raise ValueError("representation values contain non-finite entries")


class RepresentationAdapter(ABC):
    """Read-only adapter around a fitted model/checkpoint.

    Implementations may wrap PyTorch, TensorFlow, JAX, or NumPy. They must not fit
    preprocessing or mutate model weights inside extraction/prediction calls.
    """

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        raise NotImplementedError

    @property
    @abstractmethod
    def layer_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def extract(
        self,
        inputs: Any,
        subject_keys: Sequence[str],
        *,
        layer_name: str,
    ) -> RepresentationBatch:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, inputs: Any, subject_keys: Sequence[str]) -> np.ndarray:
        """Return an (n_subjects, 2) probability array in [CN, AD] order."""
        raise NotImplementedError

    def validate_contract(
        self,
        inputs: Any,
        subject_keys: Sequence[str],
        *,
        layer_name: str,
    ) -> Mapping[str, int | str]:
        if layer_name not in self.layer_names:
            raise ValueError(f"unregistered representation layer: {layer_name}")
        batch = self.extract(inputs, subject_keys, layer_name=layer_name)
        batch.validate()
        if batch.metadata != self.metadata:
            raise ValueError("batch metadata differs from adapter metadata")
        probabilities = np.asarray(self.predict_proba(inputs, subject_keys), dtype=float)
        if probabilities.shape != (len(subject_keys), 2):
            raise ValueError("predict_proba must return shape (n_subjects, 2)")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("prediction probabilities must be finite and within [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("prediction probability rows must sum to one")
        if tuple(batch.subject_keys) != tuple(str(key) for key in subject_keys):
            raise ValueError("adapter changed subject order")
        return {
            "status": "PASS",
            "n_subjects": len(subject_keys),
            "representation_dim": int(batch.values.shape[1]),
            "layer_name": layer_name,
        }
