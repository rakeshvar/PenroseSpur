import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lattice_loss import lattice_loss


class LatticeLossTest(unittest.TestCase):
    def test_hex_target_spacing_is_zero_for_every_algorithm(self):
        spacing = math.sqrt(3.0)
        xya = torch.tensor(
            [[[0.0, 0.0, -7.0], [spacing, 0.0, 9.0]]],
            dtype=torch.float64,
        )
        for algo in ("quadratic", "logarithmic", "multiplicative"):
            with self.subTest(algo=algo):
                first = lattice_loss(
                    6, 1.0, xya, torch.tensor([[0, 0]]), algo=algo
                )
                second = lattice_loss(
                    6, 1.0, xya, torch.tensor([[1, 7]]), algo=algo
                )
                self.assertAlmostEqual(first.item(), 0.0, places=12)
                self.assertEqual(first.item(), second.item())

    def test_penrose_targets_follow_nearest_neighbour_color_pair(self):
        cases = (
            ((0, 0), math.sin(2.0 * math.pi / 5.0)),
            ((1, 1), math.sin(math.pi / 5.0)),
            ((0, 1), math.sin(3.0 * math.pi / 10.0)),
            ((1, 0), math.sin(3.0 * math.pi / 10.0)),
        )
        side = 2.5
        for pair, unit_spacing in cases:
            for algo in ("quadratic", "logarithmic", "multiplicative"):
                with self.subTest(pair=pair, algo=algo):
                    spacing = unit_spacing * side
                    xya = torch.tensor(
                        [[[0.0, 0.0], [spacing, 0.0]]],
                        dtype=torch.float64,
                    )
                    colors = torch.tensor([pair], dtype=torch.uint8)
                    loss = lattice_loss(5, side, xya, colors, algo=algo)
                    self.assertAlmostEqual(loss.item(), 0.0, places=12)

    def test_algorithms_match_hand_calculation(self):
        target = math.sqrt(3.0)
        xya = torch.tensor(
            [[[0.0, 0.0], [2.0 * target, 0.0]]],
            dtype=torch.float64,
        )
        colors = torch.zeros((1, 2), dtype=torch.uint8)

        quadratic = lattice_loss(6, 1.0, xya, colors, algo="quadratic")
        logarithmic = lattice_loss(6, 1.0, xya, colors, algo="logarithmic")
        multiplicative = lattice_loss(6, 1.0, xya, colors)

        self.assertAlmostEqual(quadratic.item(), 3.0, places=12)
        self.assertAlmostEqual(
            logarithmic.item(), 1.0 - math.log(2.0), places=12
        )
        self.assertAlmostEqual(multiplicative.item(), 1.0, places=12)

    def test_batch_reduction_is_mean_over_tiles_and_batches(self):
        target = math.sqrt(3.0)
        xya = torch.tensor(
            [
                [[0.0, 0.0], [target, 0.0]],
                [[0.0, 0.0], [2.0 * target, 0.0]],
            ],
            dtype=torch.float64,
        )
        colors = torch.zeros((2, 2), dtype=torch.uint8)
        loss = lattice_loss(6, 1.0, xya, colors, algo="quadratic")
        self.assertAlmostEqual(loss.item(), 1.5, places=12)

    def test_all_algorithms_have_finite_gradients(self):
        colors = torch.tensor([[0, 1]], dtype=torch.uint8)
        for algo in ("quadratic", "logarithmic", "multiplicative"):
            with self.subTest(algo=algo):
                xya = torch.tensor(
                    [[[0.0, 0.0, 1.0], [0.7, 0.2, -1.0]]],
                    requires_grad=True,
                )
                loss = lattice_loss(5, 1.0, xya, colors, algo=algo)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(xya.grad).all())
                self.assertEqual(xya.grad[..., 2].abs().max().item(), 0.0)

    def test_invalid_inputs_are_rejected(self):
        valid = torch.zeros((1, 2, 2))
        colors = torch.zeros((1, 2), dtype=torch.uint8)
        cases = (
            lambda: lattice_loss(4, 1.0, valid, colors),
            lambda: lattice_loss(6, 0.0, valid, colors),
            lambda: lattice_loss(6, 1.0, torch.zeros((2, 2)), colors),
            lambda: lattice_loss(6, 1.0, torch.zeros((1, 1, 2)), colors[:, :1]),
            lambda: lattice_loss(6, 1.0, valid, torch.zeros((2,))),
            lambda: lattice_loss(5, 1.0, valid, torch.tensor([[0, 2]])),
            lambda: lattice_loss(6, 1.0, valid, colors, algo="unknown"),
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises((TypeError, ValueError)):
                    case()


if __name__ == "__main__":
    unittest.main()
