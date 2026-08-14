"""Shared losses and tile-color helpers."""

import math

import torch
from torch.nn import functional as F


GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


def _squared_distances(xy):
    """Return pairwise squared XY distances with the diagonal excluded."""
    if xy.ndim != 3 or xy.shape[-1] < 2:
        raise ValueError(f"Expected (B, N, D>=2), got {tuple(xy.shape)}")
    if xy.shape[1] < 2:
        raise ValueError("Lattice loss requires at least two tiles")

    positions = xy[..., :2]
    squared = (
        (positions.unsqueeze(2) - positions.unsqueeze(1)).square().sum(dim=-1)
    )
    diagonal = torch.eye(
        xy.shape[1], dtype=torch.bool, device=xy.device
    ).unsqueeze(0)
    return squared.masked_fill(diagonal, torch.inf)


def _lattice_loss_quadratic(
    xy, min_neighbour_distance, max_neighbour_distance, eps=1e-6
):
    squared = _squared_distances(xy)
    nearest = torch.sqrt(squared.min(dim=-1).values + eps)

    too_close = F.relu(1.0 - nearest / min_neighbour_distance).square()
    too_far = F.relu(nearest / max_neighbour_distance - 1.0).square()
    return (too_close + too_far).mean()


def _lattice_loss_logarithmic(
    xy,
    min_neighbour_distance,
    max_neighbour_distance,
    eps=1e-6,
):
    distances = torch.sqrt(_squared_distances(xy) + eps)

    # Attract each tile when its nearest neighbour is beyond the valid range.
    nearest_ratio = distances.min(dim=-1).values / max_neighbour_distance
    gap = (nearest_ratio - 1.0 - torch.log(nearest_ratio)).clamp_min(0.0)
    gap = (gap * (nearest_ratio > 1.0)).mean()

    # Repel every pair that is closer than the valid range.
    pair_ratio = distances / min_neighbour_distance
    pair_ratio = torch.where(
        torch.isfinite(pair_ratio), pair_ratio, torch.ones_like(pair_ratio)
    )
    overlap = (pair_ratio - 1.0 - torch.log(pair_ratio)).clamp_min(0.0)
    overlap = (overlap * (pair_ratio < 1.0)).sum(dim=-1).mean()

    return (gap + overlap) / 2.0


def lattice_loss(symmetry, xy, unit_side, algo="logarithmic"):
    """Penalize nearest-neighbour distances outside a lattice's valid range.

    Hex centers have one target distance, ``sqrt(3) * unit_side``. Penrose
    centers may have thin-thin, mixed, or thick-thick neighbours, so their
    accepted interval is ``[sin(pi/5), sin(2*pi/5)] * unit_side``.
    """
    if symmetry == 6:
        min_distance = max_distance = math.sqrt(3.0) * unit_side
    elif symmetry == 5:
        min_distance = math.sin(math.pi / 5.0) * unit_side
        max_distance = math.sin(2.0 * math.pi / 5.0) * unit_side
    else:
        raise ValueError(f"Unsupported symmetry: {symmetry} (must be 5 or 6)")

    if algo == "quadratic":
        return _lattice_loss_quadratic(
            xy, min_distance, max_distance
        )
    if algo == "logarithmic":
        return _lattice_loss_logarithmic(
            xy, min_distance, max_distance
        )
    raise ValueError(
        f"Unsupported lattice-loss algorithm: {algo!r} "
        "(must be 'quadratic' or 'logarithmic')"
    )


def get_colors(symmetry, num_tiles, device=None):
    """Return canonical tile colors (0 = light/thick, 1 = dark/thin).

    Hex tilings contain 2/3 light tiles. Penrose tilings contain 1/phi thick
    ("light") tiles. The light count is rounded to the nearest integer.
    """
    if isinstance(num_tiles, bool) or not isinstance(num_tiles, int):
        raise TypeError("num_tiles must be an integer")
    if num_tiles < 0:
        raise ValueError("num_tiles must be non-negative")

    if symmetry == 6:
        light_fraction = 2.0 / 3.0
    elif symmetry == 5:
        light_fraction = 1.0 / GOLDEN_RATIO
    else:
        raise ValueError(f"Unsupported symmetry: {symmetry} (must be 5 or 6)")

    num_light = round(num_tiles * light_fraction)
    colors = torch.ones(num_tiles, dtype=torch.uint8, device=device)
    colors[:num_light] = 0
    return colors
