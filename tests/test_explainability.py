from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pdmsa.cli import _gradcam_parser
from pdmsa.explainability import resolve_vit_target_layer, vit_reshape_transform
from pdmsa.gradcam import _extract_state_dict, _public_model_metadata


class _FakeTensor:
    """Minimal NumPy-backed tensor API for dependency-light reshape tests."""

    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, item):
        return _FakeTensor(self.array[item])

    def size(self, dimension):
        return self.array.shape[dimension]

    def reshape(self, *shape):
        return _FakeTensor(self.array.reshape(*shape))

    def permute(self, *dimensions):
        return _FakeTensor(self.array.transpose(dimensions))


class ExplainabilityTests(unittest.TestCase):
    def test_vit_reshape_transform_drops_class_token_and_builds_grid(self):
        tokens = _FakeTensor(np.arange(2 * 5 * 3).reshape(2, 5, 3))
        transformed = vit_reshape_transform(tokens)

        self.assertEqual(transformed.shape, (2, 3, 2, 2))
        np.testing.assert_array_equal(transformed.array[:, :, 0, 0], tokens.array[:, 1, :])

    def test_vit_reshape_transform_rejects_non_square_patch_grid(self):
        with self.assertRaisesRegex(ValueError, "not a square grid"):
            vit_reshape_transform(_FakeTensor(np.zeros((1, 7, 4))))

    def test_resolve_target_layer_accepts_negative_index_and_checks_bounds(self):
        layer_norms = [object(), object(), object()]
        layers = [type("Layer", (), {"layernorm_before": norm})() for norm in layer_norms]
        encoder = type("Encoder", (), {"layer": layers})()
        vit = type("Vit", (), {"encoder": encoder})()
        model = type("Model", (), {"vit": vit})()

        self.assertIs(resolve_vit_target_layer(model, -1), layer_norms[-1])
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_vit_target_layer(model, 3)

    def test_checkpoint_extraction_handles_wrappers_and_dataparallel_prefix(self):
        state = _extract_state_dict({"state_dict": {"module.vit.weight": "value"}})
        self.assertEqual(state, {"vit.weight": "value"})

    def test_public_model_metadata_removes_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            private_root = Path(directory)
            metadata = _public_model_metadata(
                {
                    "name_or_path": str(private_root / "private-server" / "vit-model"),
                    "checkpoint": str(private_root / "subject-identifier" / "weights.pth"),
                    "cache_dir": str(private_root / "transformers-cache"),
                    "extra": {"local_artifact": str(private_root / "private" / "artifact.json")},
                    "num_labels": 2,
                }
            )

            serialized = json.dumps(metadata)
            self.assertNotIn(str(private_root), serialized)
            self.assertEqual(metadata["name_or_path"], "vit-model")
            self.assertTrue(metadata["name_or_path_is_local"])
            self.assertNotIn("checkpoint", metadata)
            self.assertNotIn("cache_dir", metadata)

    def test_gradcam_cli_parser_exposes_reproducibility_arguments(self):
        args = _gradcam_parser().parse_args(
            [
                "--config",
                "run.toml",
                "--checkpoint",
                "model.pth",
                "--input",
                "slice.png",
                "--output-dir",
                "cam",
                "--target-class",
                "0",
                "--target-layer",
                "8",
            ]
        )

        self.assertEqual(args.target_class, 0)
        self.assertEqual(args.target_layer, 8)


if __name__ == "__main__":
    unittest.main()
