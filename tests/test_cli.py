import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from pdmsa.cli import evaluate_oof_main


def _fold_predictions(fold: int) -> pd.DataFrame:
    label = fold % 2
    positive_score = 0.8 if label == 1 else 0.2
    center = 10 + fold
    rows = []
    for offset in (-2, -1, 0, 1, 2):
        rows.append(
            {
                "subject_id": f"SYN_{fold:03d}",
                "fold": fold,
                "center_slice_index": center,
                "slice_offset": offset,
                "slice_index": center + offset,
                "y_true": label,
                "p_class_0": 1.0 - positive_score,
                "p_class_1": positive_score,
                "predicted_class_label": int(positive_score > 0.5),
            }
        )
    return pd.DataFrame(rows)


class OofCliTests(unittest.TestCase):
    def test_evaluate_oof_accepts_four_fold_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for fold in range(4):
                path = root / f"fold_{fold}.csv"
                _fold_predictions(fold).to_csv(path, index=False)
                inputs.append(str(path))
            output = root / "evaluation"
            argv = [
                "pdmsa-evaluate-oof",
                "--input",
                *inputs,
                "--output-dir",
                str(output),
                "--bootstrap-repetitions",
                "20",
            ]
            with mock.patch.object(sys, "argv", argv):
                evaluate_oof_main()

            subject_predictions = pd.read_csv(output / "oof_subject_predictions.csv")
            self.assertEqual(len(subject_predictions), 4)
            self.assertEqual(set(subject_predictions["fold"]), {0, 1, 2, 3})
            metrics = json.loads((output / "oof_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["point_estimates"]["n"], 4)

    def test_evaluate_oof_rejects_missing_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for fold in range(3):
                path = root / f"fold_{fold}.csv"
                _fold_predictions(fold).to_csv(path, index=False)
                inputs.append(str(path))
            argv = [
                "pdmsa-evaluate-oof",
                "--input",
                *inputs,
                "--output-dir",
                str(root / "evaluation"),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                self.assertRaisesRegex(ValueError, "Expected held-out folds"),
            ):
                evaluate_oof_main()


if __name__ == "__main__":
    unittest.main()
