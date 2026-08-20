import unittest
from pathlib import Path

from pdmsa.config import load_config
from pdmsa.manifest import DEFAULT_SLICE_OFFSETS
from pdmsa.voting import DEFAULT_MSV_WEIGHTS


class ClassificationConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "classification.toml"
        )

    def test_article_aligned_slice_and_voting_contract(self):
        self.assertEqual(tuple(self.config["data"]["slice_offsets"]), DEFAULT_SLICE_OFFSETS)
        self.assertEqual(tuple(self.config["voting"]["weights"]), DEFAULT_MSV_WEIGHTS)

    def test_uses_384_pixel_pretrained_model_with_replaceable_head(self):
        self.assertEqual(self.config["data"]["image_size"], 384)
        self.assertEqual(self.config["model"]["name_or_path"], "google/vit-base-patch16-384")
        self.assertEqual(self.config["model"]["num_labels"], 2)
        self.assertTrue(self.config["model"]["ignore_mismatched_sizes"])


if __name__ == "__main__":
    unittest.main()
