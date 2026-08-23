import math
import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svg import save_polygons
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

    def test_svg_uses_per_polygon_opacity(self):
        angles = np.arange(6) * math.pi / 3
        first = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        polygons = np.stack((first, first + np.array([3.0, 0.0])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opacity.svg"
            save_polygons(
                path,
                polygons,
                np.array([0, 1]),
                opacities=np.array([0.25, 0.75]),
            )
            svg = path.read_text(encoding="utf-8")
        self.assertIn('opacity="0.2500"', svg)
        self.assertIn('opacity="0.7500"', svg)

    def test_svg_supports_arc_only_alpha_rendering(self):
        first = np.array(
            [[-1.0, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, -0.5]]
        )
        polygons = np.stack((first, first + np.array([3.0, 0.0])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arcs.svg"
            save_polygons(
                path,
                polygons,
                np.array([0, 1]),
                show_arcs=True,
                show_polygons=False,
                alpha=0.7,
            )
            svg = path.read_text(encoding="utf-8")
        self.assertNotIn("<polygon", svg)
        self.assertIn('<path class="arc aarc"', svg)
        self.assertIn('<path class="arc carc"', svg)
        self.assertIn("stroke-opacity: 0.5600", svg)


if __name__ == "__main__":
    unittest.main()
