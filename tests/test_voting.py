import unittest

import pandas as pd

from pdmsa.voting import aggregate_slice_predictions


class VotingTests(unittest.TestCase):
    def test_soft_voting_keeps_all_slice_scores(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["S1"] * 5,
                "slice_index": [6, 7, 8, 9, 10],
                "y_true": [1] * 5,
                "p_positive": [0.2, 0.6, 0.8, 0.7, 0.4],
                "is_positive_prediction": [0, 1, 1, 1, 0],
            }
        )
        result = aggregate_slice_predictions(frame, method="soft")
        self.assertAlmostEqual(result.loc[0, "patient_score"], 0.54)
        self.assertEqual(result.loc[0, "patient_prediction"], 1)
        self.assertEqual(result.loc[0, "n_positive_votes"], 3)

    def test_label_does_not_affect_score(self):
        base = pd.DataFrame(
            {
                "subject_id": ["S1"] * 5,
                "slice_index": [6, 7, 8, 9, 10],
                "y_true": [0] * 5,
                "p_positive": [0.1, 0.2, 0.3, 0.4, 0.5],
                "is_positive_prediction": [0, 0, 0, 0, 1],
            }
        )
        first = aggregate_slice_predictions(base).loc[0, "patient_score"]
        base["y_true"] = 1
        second = aggregate_slice_predictions(base).loc[0, "patient_score"]
        self.assertEqual(first, second)

    def test_positive_class_zero_keeps_class_label_semantics(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["S0"] * 5,
                "slice_index": [6, 7, 8, 9, 10],
                "y_true": [0] * 5,
                "p_positive": [0.8, 0.7, 0.6, 0.4, 0.9],
                "is_positive_prediction": [1, 1, 1, 0, 1],
                "predicted_class_label": [0, 0, 0, 1, 0],
            }
        )
        result = aggregate_slice_predictions(frame, positive_label=0)
        self.assertEqual(result.loc[0, "patient_prediction"], 0)
        self.assertEqual(result.loc[0, "is_positive_prediction"], 1)
        self.assertEqual(result.loc[0, "positive_label"], 0)

    def test_fixed_weight_msv_emphasizes_layer_8_and_uses_strict_threshold(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["S1"] * 5,
                "slice_index": [6, 7, 8, 9, 10],
                "y_true": [1] * 5,
                "p_positive": [0.1, 0.1, 0.6, 0.1, 0.1],
                "is_positive_prediction": [0, 0, 1, 0, 0],
            }
        )
        result = aggregate_slice_predictions(
            frame,
            method="weighted_soft",
            weights=[0.05, 0.05, 0.8, 0.05, 0.05],
            threshold_inclusive=False,
        )
        self.assertAlmostEqual(result.loc[0, "patient_score"], 0.5)
        self.assertEqual(result.loc[0, "patient_prediction"], 0)
        self.assertEqual(result.loc[0, "slice_indices"], "[6, 7, 8, 9, 10]")


if __name__ == "__main__":
    unittest.main()
