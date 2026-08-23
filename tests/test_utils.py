import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GOLDEN_RATIO, get_colors


class UtilsTest(unittest.TestCase):
    def test_get_colors_uses_lattice_proportions(self):
        cases = ((6, 2.0 / 3.0), (5, 1.0 / GOLDEN_RATIO))
        for symmetry, light_fraction in cases:
            with self.subTest(symmetry=symmetry):
                colors = get_colors(symmetry, 96)
                num_light = round(96 * light_fraction)
                self.assertEqual(colors.dtype, torch.uint8)
                self.assertEqual(colors.shape, (96,))
                self.assertEqual((colors == 0).sum().item(), num_light)
                self.assertEqual((colors == 1).sum().item(), 96 - num_light)

    def test_utils_reject_unsupported_symmetry(self):
        for symmetry in (4, 7):
            with self.subTest(symmetry=symmetry):
                with self.assertRaises(ValueError):
                    get_colors(symmetry, 10)


if __name__ == "__main__":
    unittest.main()
