"""Shared tile-color helpers."""

import math

import torch


GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


def get_colors(symmetry, num_tiles, device=None):
    """Return canonical tile colors (0 = light/thick, 1 = dark/thin).

    Hex tilings contain 2/3 light tiles. Penrose tilings contain 1/phi thick
    ("light") tiles. The light count is rounded to the nearest integer. These
    canonical counts are mask-independent; sample_batch colors are selected
    from canvas tiles according to mask inness.
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
