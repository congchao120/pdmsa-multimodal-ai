import json
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SegmentationConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (PROJECT_ROOT / "configs" / "segmentation_fourfold.toml").open("rb") as handle:
            cls.config = tomllib.load(handle)
        with (PROJECT_ROOT / "segmentation" / "templates" / "Dataset800_PD" / "dataset.json").open(
            encoding="utf-8"
        ) as handle:
            cls.dataset = json.load(handle)

    def test_dataset_case_count_matches_configuration(self):
        self.assertEqual(
            self.config["study"]["num_training_cases"],
            self.dataset["numTraining"],
        )

    def test_fourfold_counts_cover_every_case_once(self):
        total = self.config["study"]["num_training_cases"]
        split = self.config["split"]
        train_cases = split["expected_train_cases_per_fold"]
        validation_cases = split["expected_validation_cases_per_fold"]
        self.assertEqual(split["n_splits"], 4)
        self.assertEqual(split["folds"], [0, 1, 2, 3])
        self.assertEqual(train_cases + validation_cases, total)
        self.assertEqual(split["n_splits"] * validation_cases, total)

    def test_configured_trainer_has_explicit_training_length(self):
        self.assertEqual(self.config["model"]["trainer"], "nnUNetTrainer_150epochs")


if __name__ == "__main__":
    unittest.main()
