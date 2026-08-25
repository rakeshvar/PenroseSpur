"""Tests for shared scaled circular-angle flow geometry."""

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flow_geometry import (  # noqa: E402
    ANGLE_PERIOD,
    angle_delta,
    canonicalize_xya,
    flow,
    flow_state,
    pairwise_xya_distance,
    reconstruction_loss,
    weighted_xya_average,
    xya_delta,
)


def test_shortest_boundary_path_both_directions() -> None:
    forward = angle_delta(torch.tensor(-1.70), torch.tensor(1.70))
    backward = angle_delta(torch.tensor(1.70), torch.tensor(-1.70))
    expected = 2.0 * math.sqrt(3.0) - 3.40
    torch.testing.assert_close(forward, torch.tensor(expected))
    torch.testing.assert_close(backward, torch.tensor(-expected))

    source = torch.tensor([[[0.0, 0.0, 1.70]]])
    target = torch.tensor([[[2.0, 4.0, -1.70]]])
    midpoint = flow_state(source, target, torch.tensor([0.5]))
    assert abs(midpoint[..., 2].item()) > 1.69
    state, velocity = flow(source, target, torch.tensor([1.0]))
    torch.testing.assert_close(state, canonicalize_xya(target))
    torch.testing.assert_close(velocity[..., 2], forward.reshape(1, 1))


def test_paired_and_pairwise_circular_distance() -> None:
    source = torch.tensor([[[0.0, 0.0, 1.70]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0, -1.70]]])
    delta = xya_delta(target, source)
    cost = pairwise_xya_distance(target, source)
    torch.testing.assert_close(cost[0, 0, 0], delta[..., 2].square().squeeze())
    assert cost.item() < 0.01
    cost.sum().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_reconstruction_reductions_and_period_equivalence() -> None:
    target = torch.tensor([[[1.0, -1.0, -1.70]]])
    prediction = target.clone()
    prediction[..., 0] += 3.0
    prediction[..., 2] += ANGLE_PERIOD
    l2 = reconstruction_loss(prediction, target, loss="l2")
    l1 = reconstruction_loss(prediction, target, loss="l1")
    torch.testing.assert_close(l2.angle, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(l2.total, torch.tensor(3.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(l1.total, torch.tensor(1.0), atol=1e-6, rtol=0)


def test_weighted_average_is_circular() -> None:
    values = torch.tensor([[[0.0, 0.0, 1.70], [2.0, 4.0, -1.70]]])
    weights = torch.tensor([[[0.5, 0.5]]])
    average = weighted_xya_average(weights, values)
    torch.testing.assert_close(average[..., :2], torch.tensor([[[1.0, 2.0]]]))
    assert abs(average[..., 2].item()) > 1.69


if __name__ == "__main__":
    test_shortest_boundary_path_both_directions()
    test_paired_and_pairwise_circular_distance()
    test_reconstruction_reductions_and_period_equivalence()
    test_weighted_average_is_circular()
    print("flow geometry checks passed")
