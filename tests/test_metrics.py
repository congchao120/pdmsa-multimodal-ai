import importlib.util
import unittest

from pdmsa.metrics import classification_metrics


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class MetricsTests(unittest.TestCase):
    def test_explicit_positive_class_zero(self):
        metrics = classification_metrics(
            y_true=[0, 0, 1, 1],
            y_score=[0.9, 0.8, 0.2, 0.1],
            positive_label=0,
        )
        self.assertEqual(metrics["positive_label"], 0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)


if __name__ == "__main__":
    unittest.main()
