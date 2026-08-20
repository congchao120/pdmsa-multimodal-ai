import ast
import inspect
import textwrap
import unittest

import pandas as pd

from pdmsa.training import SHARED_CHECKPOINT_NAME, _audit_shared_frames, _train_shared_model


def role_frame(subjects, center_start=6):
    rows = []
    for index, subject_id in enumerate(subjects):
        center = center_start + index
        for offset in (-2, -1, 0, 1, 2):
            rows.append(
                {
                    "subject_id": subject_id,
                    "label": index % 2,
                    "center_slice_index": center,
                    "slice_offset": offset,
                    "slice_index": center + offset,
                    "fused_path": f"{subject_id}_{center + offset}.png",
                }
            )
    return pd.DataFrame(rows)


class SharedTrainingContractTests(unittest.TestCase):
    def test_every_slice_row_enters_the_shared_fold_dataset(self):
        frames = {
            "train": role_frame(["T1", "T2", "T3"]),
            "validation": role_frame(["V1", "V2"], center_start=12),
        }
        summary = _audit_shared_frames(frames, (-2, -1, 0, 1, 2))
        self.assertEqual(summary["train"], {"subjects": 3, "slice_rows": 15})
        self.assertEqual(summary["validation"], {"subjects": 2, "slice_rows": 10})

    def test_missing_slice_prevents_shared_training(self):
        frames = {
            "train": role_frame(["T1"]).iloc[:-1].copy(),
            "validation": role_frame(["V1"]),
        }
        with self.assertRaisesRegex(ValueError, "complete centered window"):
            _audit_shared_frames(frames, (-2, -1, 0, 1, 2))

    def test_shared_training_constructs_exactly_one_vit_and_one_checkpoint(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(_train_shared_model)))
        create_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_vit"
        ]
        self.assertEqual(len(create_calls), 1)
        self.assertEqual(SHARED_CHECKPOINT_NAME, "best_model_weights.pth")


if __name__ == "__main__":
    unittest.main()
