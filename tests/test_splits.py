import unittest

import pandas as pd

from pdmsa.splits import (
    audit_four_fold_assignments,
    audit_manifest_assignment_alignment,
    make_four_fold_assignments,
)


class SplitTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for label, count in ((0, 20), (1, 8)):
            for subject_index in range(count):
                subject_id = f"L{label}_{subject_index:03d}"
                center = 5 + (subject_index % 9)
                for slice_offset in (-2, -1, 0, 1, 2):
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "label": label,
                            "center_slice_index": center,
                            "slice_offset": slice_offset,
                            "slice_index": center + slice_offset,
                        }
                    )
        self.manifest = pd.DataFrame(rows)

    def test_every_subject_is_validated_once(self):
        assignments = make_four_fold_assignments(self.manifest, n_folds=4, seed=42)
        audit = audit_four_fold_assignments(
            assignments,
            expected_subjects=28,
            expected_folds=4,
        )
        self.assertEqual(audit.validation_appearances_min, 1)
        self.assertEqual(audit.validation_appearances_max, 1)

    def test_split_is_deterministic_and_stratified(self):
        left = make_four_fold_assignments(self.manifest, n_folds=4, seed=9)
        right = make_four_fold_assignments(self.manifest, n_folds=4, seed=9)
        pd.testing.assert_frame_equal(left, right)
        validation = left.loc[left["role"] == "validation"]
        for _, fold in validation.groupby("fold"):
            self.assertEqual(fold["label"].value_counts().to_dict(), {0: 5, 1: 2})

    def test_manifest_assignment_alignment_rejects_label_mismatch(self):
        assignments = make_four_fold_assignments(self.manifest, n_folds=4, seed=42)
        assignments.loc[assignments["subject_id"] == "L0_000", "label"] = 1
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            audit_manifest_assignment_alignment(self.manifest, assignments, expected_folds=4)


if __name__ == "__main__":
    unittest.main()
