"""Color-aware nearest-neighbour lattice losses."""

import math

import torch


_ALGORITHMS = ("quadratic", "logarithmic", "multiplicative")


def _validate_inputs(symmetry, side, xya, colors):
    if symmetry not in (5, 6):
        raise ValueError(f"Unsupported symmetry: {symmetry} (must be 5 or 6)")
    if not isinstance(xya, torch.Tensor):
        raise TypeError("xya must be a torch.Tensor")
    if not xya.is_floating_point():
        raise TypeError("xya must have a floating-point dtype")
    if xya.ndim != 3 or xya.shape[-1] < 2:
        raise ValueError(f"Expected xya with shape (B, N, D>=2), got {tuple(xya.shape)}")
    if xya.shape[1] < 2:
        raise ValueError("Lattice loss requires at least two tiles")

    side_tensor = torch.as_tensor(side, dtype=xya.dtype, device=xya.device)
    if side_tensor.ndim != 0:
        raise ValueError("side must be a scalar")
    if not bool(torch.isfinite(side_tensor)) or not bool(side_tensor > 0):
        raise ValueError("side must be finite and positive")

    if not isinstance(colors, torch.Tensor):
        raise TypeError("colors must be a torch.Tensor")
    if colors.shape != xya.shape[:2]:
        raise ValueError(
            f"Expected colors with shape {tuple(xya.shape[:2])}, got {tuple(colors.shape)}"
        )
    colors = colors.to(device=xya.device)
    if symmetry == 5 and not bool(((colors == 0) | (colors == 1)).all()):
        raise ValueError("Penrose colors must contain only 0 (thick) and 1 (thin)")

    return side_tensor, colors


def _nearest_neighbours(xya):
    positions = xya[..., :2]
    distances = torch.cdist(positions, positions)
    diagonal = torch.eye(
        positions.shape[1], dtype=torch.bool, device=positions.device
    ).unsqueeze(0)
    distances = distances.masked_fill(diagonal, torch.inf)
    return distances.min(dim=-1)


def _target_distances(symmetry, side, colors, nearest_indices):
    if symmetry == 6:
        return torch.full(
            nearest_indices.shape,
            math.sqrt(3.0),
            dtype=side.dtype,
            device=nearest_indices.device,
        ) * side

    nearest_colors = torch.gather(colors, dim=1, index=nearest_indices)
    same_color = colors == nearest_colors
    thin = side.new_tensor(math.sin(math.pi / 5.0))
    thick = side.new_tensor(math.sin(2.0 * math.pi / 5.0))
    mixed = side.new_tensor(math.sin(3.0 * math.pi / 10.0))
    target_factor = torch.where(
        same_color,
        torch.where(colors == 1, thin, thick),
        mixed,
    )
    return target_factor * side


def lattice_loss(symmetry, side, xya, colors, algo="multiplicative"):
    """Return mean error between actual and color-aware target NN distances.

    Args:
        symmetry: 6 for hexagons or 5 for Penrose rhombuses.
        side: Polygon side length.
        xya: Batched tile geometry shaped ``(B, N, D>=2)``. Only XY is used.
        colors: Tile colors shaped ``(B, N)``. Ignored for symmetry 6.
        algo: ``quadratic``, ``logarithmic``, or ``multiplicative``.
    """
    if algo not in _ALGORITHMS:
        raise ValueError(
            f"Unsupported lattice-loss algorithm: {algo!r} "
            f"(must be one of {', '.join(repr(name) for name in _ALGORITHMS)})"
        )

    side_tensor, colors = _validate_inputs(symmetry, side, xya, colors)
    nearest_distances, nearest_indices = _nearest_neighbours(xya)
    targets = _target_distances(
        symmetry, side_tensor, colors, nearest_indices
    )

    if algo == "quadratic":
        return (nearest_distances - targets).square().mean()

    ratio = nearest_distances / targets
    if algo == "logarithmic":
        safe_ratio = ratio.clamp_min(torch.finfo(ratio.dtype).tiny)
        return (ratio - torch.log(safe_ratio) - 1.0).mean()

    epsilon = 1.0 / side_tensor
    reciprocal_ratio = (1.0 + epsilon) / (ratio + epsilon)
    return (torch.maximum(ratio, reciprocal_ratio) - 1.0).mean()
