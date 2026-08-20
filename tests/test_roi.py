import unittest

import numpy as np

from pdmsa.roi import bounding_box_2d, centered_slice_metadata, select_five_slices


class RoiTests(unittest.TestCase):
    def test_selects_centered_five_slice_window(self):
        mask = np.zeros((10, 10, 12), dtype=np.uint8)
        mask[2:8, 2:8, 6] = 1
        self.assertEqual(select_five_slices(mask), [4, 5, 6, 7, 8])

    def test_boundary_center_is_rejected_instead_of_shifting_window(self):
        mask = np.zeros((10, 10, 7), dtype=np.uint8)
        mask[2:8, 2:8, 0] = 1
        with self.assertRaisesRegex(ValueError, "two real neighbors"):
            select_five_slices(mask)

    def test_negative_axis_is_rejected(self):
        mask = np.zeros((10, 10, 7), dtype=np.uint8)
        mask[2:8, 2:8, 3] = 1
        with self.assertRaisesRegex(ValueError, "axis must be"):
            select_five_slices(mask, axis=-1)

    def test_tied_maxima_choose_lowest_center_index(self):
        mask = np.zeros((10, 10, 12), dtype=np.uint8)
        mask[2:8, 2:8, 5] = 1
        mask[2:8, 2:8, 7] = 1
        self.assertEqual(select_five_slices(mask), [3, 4, 5, 6, 7])

    def test_manifest_metadata_uses_relative_offsets(self):
        mask = np.zeros((8, 8, 12), dtype=np.uint8)
        mask[1:5, 2:6, 7] = 1
        self.assertEqual(
            centered_slice_metadata(mask),
            [
                {"center_slice_index": 7, "slice_offset": -2, "slice_index": 5},
                {"center_slice_index": 7, "slice_offset": -1, "slice_index": 6},
                {"center_slice_index": 7, "slice_offset": 0, "slice_index": 7},
                {"center_slice_index": 7, "slice_offset": 1, "slice_index": 8},
                {"center_slice_index": 7, "slice_offset": 2, "slice_index": 9},
            ],
        )

    def test_bounding_box_clips_padding(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 8:10] = 1
        self.assertEqual(bounding_box_2d(mask, padding=5), (0, 7, 3, 10))


if __name__ == "__main__":
    unittest.main()
