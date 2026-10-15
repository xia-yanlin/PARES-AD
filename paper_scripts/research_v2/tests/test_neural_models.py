from __future__ import annotations

import unittest

import numpy as np
import torch

from research_v2.src.models.neural import build_neural_model, count_parameters
from research_v2.src.representations import ModelMetadata, TorchModelInputs, TorchRepresentationAdapter


class NeuralModelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.random.default_rng(7).normal(size=(4, 6, 3)).astype(np.float32)
        adjacency = np.eye(6, dtype=np.float32)
        for index in range(5):
            adjacency[index, index + 1] = 1.0
            adjacency[index + 1, index] = 1.0
        self.inputs = TorchModelInputs(self.features, adjacency)
        self.keys = ["s4", "s2", "s3", "s1"]
        self.configurations = {
            "flat_mlp": {"id": "m", "hidden_dims": [8, 4], "dropout": 0.1},
            "no_edge_roi_encoder": {"id": "n", "node_hidden": 4, "node_embedding": 2, "head_hidden": 5, "dropout": 0.1},
            "gcn": {"id": "g", "hidden_dim": 8, "node_embedding": 2, "readout": "meanmax", "dropout": 0.1},
            "gat": {"id": "a", "hidden_dim": 8, "attention_heads": 2, "node_embedding": 2, "readout": "meanmax", "dropout": 0.1},
        }

    def test_all_frozen_families_satisfy_representation_contract(self) -> None:
        for family, config in self.configurations.items():
            with self.subTest(family=family):
                model = build_neural_model(family, config, n_nodes=6, n_features=3)
                metadata = ModelMetadata(
                    model_id=f"synthetic-{family}",
                    family=family,
                    configuration_id=config["id"],
                    seed=7,
                    fold="synthetic",
                    checkpoint_sha256="a" * 64,
                    input_manifest_sha256="b" * 64,
                    parameter_count=count_parameters(model),
                )
                adapter = TorchRepresentationAdapter(model, metadata)
                result = adapter.validate_contract(self.inputs, self.keys, layer_name="penultimate")
                self.assertEqual(result["status"], "PASS")
                self.assertGreater(result["representation_dim"], 0)

    def test_graph_models_require_adjacency(self) -> None:
        for family in ["gcn", "gat"]:
            model = build_neural_model(family, self.configurations[family], n_nodes=6, n_features=3)
            with self.assertRaisesRegex(ValueError, "adjacency"):
                model.forward_with_representations(torch.as_tensor(self.features), None)

    def test_unknown_family_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown neural family"):
            build_neural_model("transformer", {}, n_nodes=6, n_features=3)


if __name__ == "__main__":
    unittest.main()
