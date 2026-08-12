import unittest

import numpy as np
import pandas as pd

from pdmsa.imbalance import class_weight_vector, row_sampling_weights


class ImbalanceTests(unittest.TestCase):
    def test_minority_class_has_larger_ce_weight(self):
        weights = class_weight_vector([0] * 12 + [1] * 4)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_sampler_balances_class_mass_and_subject_slices(self):
        frame = pd.DataFrame(
            {
                "subject_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 3,
                "label": [0] * 10 + [1] * 3,
            }
        )
        weights = row_sampling_weights(frame)
        self.assertAlmostEqual(weights[:10].sum(), weights[10:].sum())
        self.assertAlmostEqual(weights[:5].sum(), weights[5:10].sum())
        self.assertTrue(np.all(weights > 0))


if __name__ == "__main__":
    unittest.main()
