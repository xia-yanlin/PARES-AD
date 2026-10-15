from __future__ import annotations

import unittest

import numpy as np

from research_v2.src.representations.base import ModelMetadata, RepresentationAdapter, RepresentationBatch


class SyntheticAdapter(RepresentationAdapter):
    def __init__(self) -> None:
        self._metadata = ModelMetadata(
            model_id="synthetic-model",
            family="unit-test",
            configuration_id="v1",
            seed=1,
            fold="synthetic",
            checkpoint_sha256="a" * 64,
            input_manifest_sha256="b" * 64,
            parameter_count=6,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def layer_names(self) -> tuple[str, ...]:
        return ("hidden",)

    def extract(self, inputs, subject_keys, *, layer_name: str) -> RepresentationBatch:
        values = np.asarray(inputs, dtype=float)
        return RepresentationBatch(tuple(map(str, subject_keys)), layer_name, values, self.metadata)

    def predict_proba(self, inputs, subject_keys) -> np.ndarray:
        logits = np.asarray(inputs, dtype=float).sum(axis=1)
        ad = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - ad, ad])


class AdapterTests(unittest.TestCase):
    def test_contract_passes_and_preserves_order(self) -> None:
        adapter = SyntheticAdapter()
        result = adapter.validate_contract(
            np.array([[0.0, 1.0], [2.0, -1.0]]),
            ["s2", "s1"],
            layer_name="hidden",
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["representation_dim"], 2)

    def test_duplicate_subjects_fail(self) -> None:
        adapter = SyntheticAdapter()
        with self.assertRaisesRegex(ValueError, "unique"):
            adapter.validate_contract(
                np.array([[0.0], [1.0]]),
                ["s1", "s1"],
                layer_name="hidden",
            )

    def test_unknown_layer_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unregistered"):
            SyntheticAdapter().validate_contract(np.array([[0.0]]), ["s1"], layer_name="late")


if __name__ == "__main__":
    unittest.main()
