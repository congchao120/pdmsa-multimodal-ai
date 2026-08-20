import unittest

import pandas as pd

from pdmsa.voting import DEFAULT_MSV_WEIGHTS, aggregate_slice_predictions


def prediction_frame(
    subject_id="S1",
    center=8,
    label=1,
    scores=(0.2, 0.6, 0.8, 0.7, 0.4),
):
    offsets = [-2, -1, 0, 1, 2]
    votes = [int(value > 0.5) for value in scores]
    return pd.DataFrame(
        {
            "subject_id": [subject_id] * 5,
            "center_slice_index": [center] * 5,
            "slice_offset": offsets,
            "slice_index": [center + value for value in offsets],
            "y_true": [label] * 5,
            "p_positive": list(scores),
            "is_positive_prediction": votes,
        }
    )


class VotingTests(unittest.TestCase):
    def test_soft_voting_keeps_all_slice_scores(self):
        result = aggregate_slice_predictions(prediction_frame(), method="soft")
        self.assertAlmostEqual(result.loc[0, "patient_score"], 0.54)
        self.assertEqual(result.loc[0, "patient_prediction"], 1)
        self.assertEqual(result.loc[0, "n_positive_votes"], 3)

    def test_label_does_not_affect_score(self):
        frame = prediction_frame(scores=(0.1, 0.2, 0.3, 0.4, 0.5), label=0)
        first = aggregate_slice_predictions(frame).loc[0, "patient_score"]
        frame["y_true"] = 1
        second = aggregate_slice_predictions(frame).loc[0, "patient_score"]
        self.assertEqual(first, second)

    def test_positive_class_zero_keeps_class_label_semantics(self):
        frame = prediction_frame(
            subject_id="S0", label=0, scores=(0.8, 0.7, 0.6, 0.4, 0.9)
        )
        frame["predicted_class_label"] = [0, 0, 0, 1, 0]
        result = aggregate_slice_predictions(frame, positive_label=0)
        self.assertEqual(result.loc[0, "patient_prediction"], 0)
        self.assertEqual(result.loc[0, "is_positive_prediction"], 1)
        self.assertEqual(result.loc[0, "positive_label"], 0)

    def test_article_weights_and_strict_threshold(self):
        frame = prediction_frame(center=12, scores=(0.0, 0.0, 1.0, 0.5, 0.0))
        result = aggregate_slice_predictions(
            frame,
            method="weighted_soft",
            weights=DEFAULT_MSV_WEIGHTS,
            threshold_inclusive=False,
        )
        self.assertAlmostEqual(result.loc[0, "patient_score"], 0.5)
        self.assertEqual(result.loc[0, "patient_prediction"], 0)
        self.assertEqual(result.loc[0, "slice_offsets"], "[-2, -1, 0, 1, 2]")
        self.assertEqual(result.loc[0, "slice_indices"], "[10, 11, 12, 13, 14]")
        self.assertEqual(result.loc[0, "slice_weights"], "[0.1, 0.2, 0.4, 0.2, 0.1]")

    def test_weighting_is_relative_to_each_subject_center(self):
        first = prediction_frame(subject_id="S1", center=6, scores=(0.1, 0.2, 0.8, 0.4, 0.3))
        second = prediction_frame(subject_id="S2", center=14, scores=(0.1, 0.2, 0.8, 0.4, 0.3))
        result = aggregate_slice_predictions(
            pd.concat([first, second], ignore_index=True),
            method="weighted_soft",
            weights=DEFAULT_MSV_WEIGHTS,
        )
        self.assertAlmostEqual(result.loc[0, "patient_score"], result.loc[1, "patient_score"])
        self.assertNotEqual(
            result.loc[0, "center_slice_index"], result.loc[1, "center_slice_index"]
        )

    def test_fractional_offsets_are_not_silently_truncated(self):
        frame = prediction_frame()
        frame["slice_offset"] = frame["slice_offset"].astype(float)
        frame.loc[0, "slice_offset"] = -1.5
        with self.assertRaisesRegex(ValueError, "finite integer"):
            aggregate_slice_predictions(frame)

    def test_probabilities_must_be_finite_and_bounded(self):
        frame = prediction_frame()
        frame.loc[0, "p_positive"] = 1.1
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            aggregate_slice_predictions(frame)

    def test_recorded_positive_label_must_match_setting(self):
        frame = prediction_frame()
        frame["positive_label"] = 0
        with self.assertRaisesRegex(ValueError, "do not match"):
            aggregate_slice_predictions(frame, positive_label=1)


if __name__ == "__main__":
    unittest.main()
