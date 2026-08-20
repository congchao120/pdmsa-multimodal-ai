import unittest

import pandas as pd

from pdmsa.manifest import validate_manifest


def centered_subject(subject_id: str, label: int, center: int) -> pd.DataFrame:
    offsets = [-2, -1, 0, 1, 2]
    return pd.DataFrame(
        {
            "subject_id": [subject_id] * 5,
            "label": [label] * 5,
            "center_slice_index": [center] * 5,
            "slice_offset": offsets,
            "slice_index": [center + value for value in offsets],
            "FDG": [f"{subject_id}_{center + value}.png" for value in offsets],
        }
    )


class ManifestTests(unittest.TestCase):
    def test_dynamic_centers_are_valid(self):
        frame = pd.concat(
            [centered_subject("S1", 0, 6), centered_subject("S2", 1, 14)],
            ignore_index=True,
        )
        audit = validate_manifest(frame, channels=["FDG"])
        self.assertEqual(audit.subjects, 2)
        self.assertEqual(audit.slice_offsets, (-2, -1, 0, 1, 2))

    def test_fractional_label_is_rejected(self):
        frame = centered_subject("S1", 0, 6)
        frame["label"] = frame["label"].astype(float)
        frame.loc[0, "label"] = 0.5
        with self.assertRaisesRegex(ValueError, "must be integers"):
            validate_manifest(frame, channels=["FDG"])

    def test_path_reuse_across_rows_is_rejected(self):
        frame = pd.concat(
            [centered_subject("S1", 0, 6), centered_subject("S2", 1, 14)],
            ignore_index=True,
        )
        frame.loc[5, "FDG"] = frame.loc[0, "FDG"]
        with self.assertRaisesRegex(ValueError, "reuses a file path"):
            validate_manifest(frame, channels=["FDG"])

    def test_missing_relative_position_is_rejected(self):
        frame = centered_subject("S1", 0, 6).iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "Expected 5 unique slices"):
            validate_manifest(frame, channels=["FDG"])

    def test_absolute_index_must_match_center_plus_offset(self):
        frame = centered_subject("S1", 0, 6)
        frame.loc[0, "slice_index"] = 99
        with self.assertRaisesRegex(ValueError, r"center_slice_index \+ slice_offset"):
            validate_manifest(frame, channels=["FDG"])


if __name__ == "__main__":
    unittest.main()
