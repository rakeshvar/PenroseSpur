"""Focused assertions for PenroseSpur noise sampling and matching."""

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from match import MATCH_METHODS, match, sinkhorn, sinkhorn_annealed_argmax
from sampler import SpurSampler, _scaled_angle


def make_noise_sampler(num_tiles=96):
    """Build the small sampler subset needed by `noise`, without loading masks."""
    sampler = SpurSampler.__new__(SpurSampler)
    sampler.device = torch.device("cpu")
    sampler.cvertices = torch.empty(0, dtype=torch.float32)
    sampler.num_ret_tiles = num_tiles
    return sampler


def test_noise():
    sampler = make_noise_sampler()

    large = sampler.sample_noise(
        1024, generator=torch.Generator().manual_seed(7)
    )
    assert large.shape == (1024, 96, 3)
    xy = large[..., :2]
    assert abs(float(xy.mean())) < 0.02
    assert abs(float(xy.var()) - 1.) < 0.03
    angle = large[..., 2]
    assert angle.min() >= -math.sqrt(3.)
    assert angle.max() <= math.sqrt(3.)
    assert abs(float(angle.mean())) < 0.02
    assert abs(float(angle.var()) - 1.) < 0.03


def test_scaled_sample_angles():
    angles = torch.linspace(-9. * math.pi, 9. * math.pi, 1001)
    scaled = _scaled_angle(angles)
    assert scaled.min() >= -math.sqrt(3.) - 1e-6
    assert scaled.max() < math.sqrt(3.) + 1e-6

    sampler = SpurSampler.__new__(SpurSampler)
    sampler.device = torch.device("cpu")
    sampler.masks = torch.empty(1, 1, 1)
    sampler.M = sampler.num_ret_tiles = 4
    sampler.V1 = 1
    sampler.rotation_mask = 0.
    sampler.angles = torch.tensor(
        [-4. * math.pi, -math.pi / 2., math.pi / 2., 4. * math.pi]
    )
    sampler.colors = torch.zeros(4, dtype=torch.uint8)
    sampler.indices = torch.arange(4)
    sampler.labels = torch.zeros(1, dtype=torch.long)

    centers = torch.zeros(1, 4, 2, 2)
    centers[0, :, 0, 0] = torch.arange(4)
    centers[0, :, 1, 0] = torch.arange(4)

    def transform_and_inness(mask_idx, generator=None):
        batch = len(mask_idx)
        return (
            centers.expand(batch, -1, -1, -1),
            torch.zeros(batch),
            torch.arange(4, dtype=torch.float32).expand(batch, -1),
        )

    sampler.transform_and_inness = transform_and_inness
    batch = sampler.sample_batch(
        2, mask_idx=torch.zeros(2, dtype=torch.long), return_vertices=True
    )
    assert batch["xya"][..., 2].abs().max() <= math.sqrt(3.) + 1e-6
    assert torch.allclose(
        batch["xya"][..., :2].mean(dim=1), torch.zeros(2, 2), atol=1e-7
    )
    assert torch.allclose(
        batch["vertices"].mean(dim=(1, 2)), torch.zeros(2, 2), atol=1e-7
    )


def test_inness_is_normalized():
    sampler = SpurSampler.__new__(SpurSampler)
    sampler.device = torch.device("cpu")
    sampler.translation_cu = 0.0
    sampler.rotation_canvas = 0.0
    sampler.scaling = 1.0
    sampler.H = sampler.W = 1
    sampler.M, sampler.V1 = 2, 3
    sampler.mask_flat = torch.ones(1, 1)
    sampler.cvertices = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        ]
    )

    _, _, inness = sampler.transform_and_inness(torch.tensor([0]))

    assert torch.allclose(inness, torch.tensor([[1.0, 1.0 / 3.0]]))
    assert inness.min() >= 0.0
    assert inness.max() <= 1.0


def assert_permutation(permutation):
    expected = torch.arange(permutation.shape[1], device=permutation.device)
    for row in permutation:
        assert torch.equal(torch.sort(row).values, expected)


def test_sinkhorn_argmax_annealing():
    scores = torch.randn((1, 5, 5), generator=torch.Generator().manual_seed(0))
    initial = sinkhorn(scores, iterations=100).argmax(dim=2)
    assert torch.unique(initial).numel() < scores.shape[1]

    _, annealed, converged = sinkhorn_annealed_argmax(
        scores, iterations=100, anneal_steps=8
    )
    assert converged
    assert_permutation(annealed)


def test_matching():
    generator = torch.Generator().manual_seed(23)
    noise = torch.randn((3, 12, 3), generator=generator)
    source_permutation = torch.stack(
        [torch.randperm(12, generator=generator) for _ in range(3)]
    )
    data = noise.gather(
        1, source_permutation.unsqueeze(-1).expand(-1, -1, noise.shape[-1])
    )

    exact = match(data, noise, method="lsa", return_details=True)
    assert torch.equal(exact.matched_noise, data)
    assert exact.soft_permutation is None
    assert_permutation(exact.permutation)

    argmax = match(data, noise, method="sinkhorn.argmax", return_details=True)
    assert argmax.matched_noise.shape == noise.shape
    assert argmax.permutation.shape == noise.shape[:2]
    assert argmax.soft_permutation is not None
    assert_permutation(argmax.permutation)

    barycenter = match(
        data, noise, method="sinkhorn.barycenter", return_details=True
    )
    assert barycenter.matched_noise.shape == noise.shape
    assert barycenter.permutation is None
    assert barycenter.soft_permutation is not None
    weights = barycenter.soft_permutation
    weights = weights / weights.sum(dim=2, keepdim=True)
    assert torch.allclose(weights.sum(dim=2), torch.ones_like(weights[..., 0]))
    assert torch.allclose(barycenter.matched_noise, torch.bmm(weights, noise))
    assert MATCH_METHODS == (
        "lsa",
        "sinkhorn.argmax",
        "sinkhorn.barycenter",
        "sinkhorn.lsa",
    )


def test_color_constrained_matching():
    noise = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    data = noise.flip(dims=(1,))
    colors = torch.tensor([[0, 1]], dtype=torch.uint8)

    unconstrained = match(data, noise, method="lsa", return_details=True)
    assert torch.equal(unconstrained.matched_noise, data)
    assert torch.equal(unconstrained.permutation, torch.tensor([[1, 0]]))

    for method in MATCH_METHODS:
        result = match(
            data,
            noise,
            method=method,
            colors=colors,
            iterations=20,
            return_details=True,
        )
        assert torch.allclose(result.matched_noise, noise)
        if result.permutation is not None:
            assert torch.equal(result.permutation, torch.tensor([[0, 1]]))


if __name__ == "__main__":
    test_noise()
    test_scaled_sample_angles()
    test_inness_is_normalized()
    test_sinkhorn_argmax_annealing()
    test_matching()
    test_color_constrained_matching()
    print("noise and matching checks passed")
