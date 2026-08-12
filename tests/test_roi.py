import unittest

import numpy as np

from pdmsa.roi import bounding_box_2d, select_five_slices


class RoiTests(unittest.TestCase):
    def test_selects_centered_five_slice_window(self):
        mask = np.zeros((10, 10, 12), dtype=np.uint8)
        mask[2:8, 2:8, 6] = 1
        self.assertEqual(select_five_slices(mask), [4, 5, 6, 7, 8])

    def test_boundary_window_still_contains_five_slices(self):
        mask = np.zeros((10, 10, 7), dtype=np.uint8)
        mask[2:8, 2:8, 0] = 1
        self.assertEqual(select_five_slices(mask), [0, 1, 2, 3, 4])

    def test_bounding_box_clips_padding(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 8:10] = 1
        self.assertEqual(bounding_box_2d(mask, padding=5), (0, 7, 3, 10))


if __name__ == "__main__":
    unittest.main()
