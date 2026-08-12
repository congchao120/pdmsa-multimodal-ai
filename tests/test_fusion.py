import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pdmsa.fusion import fuse_channels


class FusionTests(unittest.TestCase):
    def test_channel_order_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, value in enumerate((10, 100, 240)):
                path = root / f"channel_{index}.png"
                array = np.zeros((16, 16), dtype=np.uint8)
                array[4:12, 4:12] = value
                Image.fromarray(array).save(path)
                paths.append(path)
            fused = np.asarray(fuse_channels(paths))
            center = fused[8, 8, :]
            self.assertTrue(np.array_equal(center, np.asarray([10, 100, 240])))


if __name__ == "__main__":
    unittest.main()
