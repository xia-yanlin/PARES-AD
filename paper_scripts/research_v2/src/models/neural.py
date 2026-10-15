"""Low-cost auditable neural families frozen for CRUA-AD V2.1.

The graph implementations use dense 94-node adjacency matrices. This keeps the
representation interface identical across models and avoids hidden batching
behavior during the audit.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)])
        current = hidden
    return nn.Sequential(*layers)


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square 2D matrix")
    adjacency = (adjacency > 0).to(dtype=torch.float32).clone()
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class FlatMLPClassifier(nn.Module):
    family = "flat_mlp"

    def __init__(self, n_nodes: int, n_features: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        if not hidden_dims:
            raise ValueError("Flat MLP requires at least one hidden layer")
        self.encoder = _mlp(n_nodes * n_features, hidden_dims, dropout)
        self.classifier = nn.Linear(hidden_dims[-1], 1)

    def forward_with_representations(
        self, features: torch.Tensor, adjacency: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del adjacency
        penultimate = self.encoder(features.flatten(start_dim=1))
        logits = self.classifier(penultimate)
        return logits, {"penultimate": penultimate, "logits": logits}

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_with_representations(features, adjacency)[0]


class NoEdgeROIClassifier(nn.Module):
    family = "no_edge_roi_encoder"

    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        node_hidden: int,
        node_embedding: int,
        head_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_embedding = nn.Embedding(n_nodes, node_embedding)
        self.node_encoder = nn.Sequential(
            nn.Linear(n_features + node_embedding, node_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(n_nodes * node_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(head_hidden, 1)

    def forward_with_representations(
        self, features: torch.Tensor, adjacency: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del adjacency
        node_ids = torch.arange(self.n_nodes, device=features.device)
        embedding = self.node_embedding(node_ids).unsqueeze(0).expand(features.shape[0], -1, -1)
        encoded = self.node_encoder(torch.cat([features, embedding], dim=-1))
        penultimate = self.head(encoded.flatten(start_dim=1))
        logits = self.classifier(penultimate)
        return logits, {"penultimate": penultimate, "logits": logits}

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_with_representations(features, adjacency)[0]


class DenseGCNClassifier(nn.Module):
    family = "gcn"

    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        hidden_dim: int,
        node_embedding: int,
        readout: str,
        dropout: float,
    ):
        super().__init__()
        if readout not in {"mean", "meanmax"}:
            raise ValueError("readout must be mean or meanmax")
        self.n_nodes = n_nodes
        self.readout = readout
        self.dropout = dropout
        self.node_embedding = nn.Embedding(n_nodes, node_embedding)
        self.linear1 = nn.Linear(n_features + node_embedding, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        readout_dim = hidden_dim * (2 if readout == "meanmax" else 1)
        self.head = nn.Sequential(nn.Linear(readout_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward_with_representations(
        self, features: torch.Tensor, adjacency: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if adjacency is None:
            raise ValueError("GCN requires a training-frozen adjacency matrix")
        norm = normalize_adjacency(adjacency).to(features.device)
        node_ids = torch.arange(self.n_nodes, device=features.device)
        embedding = self.node_embedding(node_ids).unsqueeze(0).expand(features.shape[0], -1, -1)
        x = torch.cat([features, embedding], dim=-1)
        x = F.relu(torch.matmul(norm, self.linear1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(torch.matmul(norm, self.linear2(x)))
        pooled = x.mean(dim=1)
        if self.readout == "meanmax":
            pooled = torch.cat([pooled, x.max(dim=1).values], dim=-1)
        penultimate = self.head(pooled)
        logits = self.classifier(penultimate)
        return logits, {"penultimate": penultimate, "logits": logits}

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_with_representations(features, adjacency)[0]


class DenseGATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int, dropout: float):
        super().__init__()
        if output_dim % heads:
            raise ValueError("GAT output_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.dropout = dropout
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_source = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_target = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_target)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = features.shape
        projected = self.projection(features).view(batch, nodes, self.heads, self.head_dim)
        source = (projected * self.attn_source).sum(dim=-1)
        target = (projected * self.attn_target).sum(dim=-1)
        scores = F.leaky_relu(source.unsqueeze(2) + target.unsqueeze(1), negative_slope=0.2)
        mask = (adjacency > 0).to(features.device).clone()
        mask.fill_diagonal_(True)
        scores = scores.masked_fill(~mask[None, :, :, None], torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=2)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        aggregated = torch.einsum("bijh,bjhq->bihq", attention, projected)
        return aggregated.reshape(batch, nodes, self.heads * self.head_dim) + self.bias


class DenseGATClassifier(nn.Module):
    family = "gat"

    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        hidden_dim: int,
        attention_heads: int,
        node_embedding: int,
        readout: str,
        dropout: float,
    ):
        super().__init__()
        if readout not in {"mean", "meanmax"}:
            raise ValueError("readout must be mean or meanmax")
        self.n_nodes = n_nodes
        self.readout = readout
        self.dropout = dropout
        self.node_embedding = nn.Embedding(n_nodes, node_embedding)
        self.gat1 = DenseGATLayer(n_features + node_embedding, hidden_dim, attention_heads, dropout)
        self.gat2 = DenseGATLayer(hidden_dim, hidden_dim, attention_heads, dropout)
        readout_dim = hidden_dim * (2 if readout == "meanmax" else 1)
        self.head = nn.Sequential(nn.Linear(readout_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward_with_representations(
        self, features: torch.Tensor, adjacency: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if adjacency is None:
            raise ValueError("GAT requires a training-frozen adjacency matrix")
        node_ids = torch.arange(self.n_nodes, device=features.device)
        embedding = self.node_embedding(node_ids).unsqueeze(0).expand(features.shape[0], -1, -1)
        x = torch.cat([features, embedding], dim=-1)
        x = F.relu(self.gat1(x, adjacency))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gat2(x, adjacency))
        pooled = x.mean(dim=1)
        if self.readout == "meanmax":
            pooled = torch.cat([pooled, x.max(dim=1).values], dim=-1)
        penultimate = self.head(pooled)
        logits = self.classifier(penultimate)
        return logits, {"penultimate": penultimate, "logits": logits}

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_with_representations(features, adjacency)[0]


def build_neural_model(
    family: str,
    configuration: dict[str, Any],
    *,
    n_nodes: int,
    n_features: int,
) -> nn.Module:
    if family == "flat_mlp":
        return FlatMLPClassifier(n_nodes, n_features, list(configuration["hidden_dims"]), float(configuration["dropout"]))
    if family == "no_edge_roi_encoder":
        return NoEdgeROIClassifier(n_nodes, n_features, int(configuration["node_hidden"]), int(configuration["node_embedding"]), int(configuration["head_hidden"]), float(configuration["dropout"]))
    if family == "gcn":
        return DenseGCNClassifier(n_nodes, n_features, int(configuration["hidden_dim"]), int(configuration["node_embedding"]), str(configuration["readout"]), float(configuration["dropout"]))
    if family == "gat":
        return DenseGATClassifier(n_nodes, n_features, int(configuration["hidden_dim"]), int(configuration["attention_heads"]), int(configuration["node_embedding"]), str(configuration["readout"]), float(configuration["dropout"]))
    raise ValueError(f"Unknown neural family: {family}")
