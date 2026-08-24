import unittest
from pathlib import Path

from pdmsa.config import load_config, public_config
from pdmsa.manifest import DEFAULT_SLICE_OFFSETS
from pdmsa.voting import DEFAULT_MSV_WEIGHTS


class ClassificationConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "classification.toml"
        )

    def test_centered_slice_and_voting_contract(self):
        self.assertEqual(tuple(self.config["data"]["slice_offsets"]), DEFAULT_SLICE_OFFSETS)
        self.assertEqual(tuple(self.config["voting"]["weights"]), DEFAULT_MSV_WEIGHTS)

    def test_uses_384_pixel_pretrained_model_with_replaceable_head(self):
        self.assertEqual(self.config["data"]["image_size"], 384)
        self.assertEqual(self.config["model"]["name_or_path"], "google/vit-base-patch16-384")
        self.assertEqual(self.config["model"]["num_labels"], 2)
        self.assertTrue(self.config["model"]["ignore_mismatched_sizes"])

    def test_public_metadata_removes_local_paths_and_secrets(self):
        sanitized = public_config(
            {
                "data": {
                    "manifest": "/private/cohort/manifest.csv",
                    "root": r"C:\private\images",
                },
                "output_dir": "/private/results/classification",
                "model": {
                    "name_or_path": "google/vit-base-patch16-384",
                    "checkpoint": "/private/checkpoints/model.pth",
                },
                "api_token": "do-not-publish",
            }
        )
        serialized = str(sanitized)
        self.assertNotIn("/private", serialized)
        self.assertNotIn(r"C:\private", serialized)
        self.assertNotIn("do-not-publish", serialized)
        self.assertEqual(sanitized["data"]["manifest"], "manifest.csv")
        self.assertEqual(sanitized["model"]["name_or_path"], "google/vit-base-patch16-384")
        self.assertEqual(sanitized["api_token"], "<redacted>")


if __name__ == "__main__":
    unittest.main()
