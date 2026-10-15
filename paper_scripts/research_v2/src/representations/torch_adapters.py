"""Read-only representation adapters for the four V2.1 PyTorch families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .base import ModelMetadata, RepresentationAdapter, RepresentationBatch


@dataclass(frozen=True)
class TorchModelInputs:
    features: np.ndarray
    adjacency: np.ndarray | None = None


class TorchRepresentationAdapter(RepresentationAdapter):
    def __init__(self, model: torch.nn.Module, metadata: ModelMetadata, *, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self._metadata = metadata
        self.device = torch.device(device)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def layer_names(self) -> tuple[str, ...]:
        return ("penultimate", "logits")

    def _tensor_inputs(self, inputs: TorchModelInputs) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(inputs, TorchModelInputs):
            raise TypeError("PyTorch adapters require TorchModelInputs")
        features = torch.as_tensor(inputs.features, dtype=torch.float32, device=self.device)
        if features.ndim != 3:
            raise ValueError("features must have shape (subjects, nodes, features)")
        adjacency = None
        if inputs.adjacency is not None:
            adjacency = torch.as_tensor(inputs.adjacency, dtype=torch.float32, device=self.device)
        return features, adjacency

    def _forward(self, inputs: TorchModelInputs) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        features, adjacency = self._tensor_inputs(inputs)
        self.model.eval()
        with torch.no_grad():
            logits, representations = self.model.forward_with_representations(features, adjacency)
        logits_np = logits.detach().cpu().numpy()
        reps_np = {name: value.detach().cpu().numpy() for name, value in representations.items()}
        return logits_np, reps_np

    def extract(
        self,
        inputs: TorchModelInputs,
        subject_keys: Sequence[str],
        *,
        layer_name: str,
    ) -> RepresentationBatch:
        if layer_name not in self.layer_names:
            raise ValueError(f"unregistered representation layer: {layer_name}")
        _, representations = self._forward(inputs)
        batch = RepresentationBatch(
            tuple(str(key) for key in subject_keys),
            layer_name,
            np.asarray(representations[layer_name], dtype=np.float32),
            self.metadata,
        )
        batch.validate()
        return batch

    def predict_proba(self, inputs: TorchModelInputs, subject_keys: Sequence[str]) -> np.ndarray:
        del subject_keys
        logits, _ = self._forward(inputs)
        ad = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        return np.column_stack([1.0 - ad, ad])


class FlatMLPAdapter(TorchRepresentationAdapter):
    pass


class NoEdgeROIAdapter(TorchRepresentationAdapter):
    pass


class GCNAdapter(TorchRepresentationAdapter):
    pass


class GATAdapter(TorchRepresentationAdapter):
    pass
