import unittest

import pandas as pd

from pdmsa.manifest import validate_manifest


class ManifestTests(unittest.TestCase):
    def test_fractional_label_is_rejected(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["S1"],
                "label": [0.5],
                "slice_index": [0],
                "FDG": ["s1.png"],
            }
        )
        with self.assertRaisesRegex(ValueError, "must be integers"):
            validate_manifest(frame, channels=["FDG"], expected_slices=1)

    def test_path_reuse_across_subjects_is_rejected(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["S1", "S2"],
                "label": [0, 1],
                "slice_index": [0, 0],
                "FDG": ["same.png", "same.png"],
            }
        )
        with self.assertRaisesRegex(ValueError, "reuses a file path"):
            validate_manifest(frame, channels=["FDG"], expected_slices=1)


if __name__ == "__main__":
    unittest.main()
