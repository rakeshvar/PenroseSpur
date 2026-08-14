import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GOLDEN_RATIO, get_colors, lattice_loss


class UtilsTest(unittest.TestCase):
    def test_hex_lattice_loss_accepts_hex_spacing(self):
        spacing = math.sqrt(3.0)
        for algo in ("quadratic", "logarithmic"):
            with self.subTest(algo=algo):
                xy = torch.tensor(
                    [[[0.0, 0.0], [spacing, 0.0], [2.0 * spacing, 0.0]]],
                    requires_grad=True,
                )
                loss = lattice_loss(6, xy, unit_side=1.0, algo=algo)
                self.assertLess(loss.item(), 1e-10)
                loss.backward()
                self.assertTrue(torch.isfinite(xy.grad).all())

    def test_penrose_lattice_loss_accepts_documented_spacings(self):
        spacings = (
            math.sin(math.pi / 5.0),
            math.sin(3.0 * math.pi / 10.0),
            math.sin(2.0 * math.pi / 5.0),
        )
        for algo in ("quadratic", "logarithmic"):
            for spacing in spacings:
                with self.subTest(algo=algo, spacing=spacing):
                    xy = torch.tensor(
                        [[[0.0, 0.0], [spacing, 0.0], [2.0 * spacing, 0.0]]]
                    )
                    loss = lattice_loss(5, xy, unit_side=1.0, algo=algo)
                    self.assertLess(loss.item(), 1e-10)

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
                with self.assertRaises(ValueError):
                    lattice_loss(symmetry, torch.zeros(1, 2, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
