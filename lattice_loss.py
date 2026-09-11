"""Position and orientation losses for scaled ``(x, y, angle)`` lattices."""

import math

import torch


_ALGORITHMS = ("quadratic", "logarithmic", "multiplicative")
_ANGLE_LOSS_SCALE = 0.1


def _validate_geometry(symmetry, xya):
    if symmetry not in (5, 6):
        raise ValueError(f"Unsupported symmetry: {symmetry} (must be 5 or 6)")
    if not isinstance(xya, torch.Tensor):
        raise TypeError("xya must be a torch.Tensor")
    if not xya.is_floating_point():
        raise TypeError("xya must have a floating-point dtype")
    if xya.ndim != 3 or xya.shape[-1] < 3:
        raise ValueError(f"Expected xya with shape (B, N, D>=3), got {tuple(xya.shape)}")
    if xya.shape[1] < 2:
        raise ValueError("Lattice loss requires at least two tiles")


def _validate_inputs(symmetry, side, xya, colors):
    _validate_geometry(symmetry, xya)

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


def lattice_loss_xy(symmetry, side, xya, colors, algo="multiplicative"):
    """Return mean color-aware nearest-neighbour distance error.

    Args:
        symmetry: 6 for hexagons or 5 for Penrose rhombuses.
        side: Polygon side length.
        xya: Batched scaled geometry shaped ``(B, N, D>=3)``.
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


def _angle_losses(symmetry, xya):
    _validate_geometry(symmetry, xya)
    harmonic = 6 if symmetry == 6 else 10
    radians = xya[..., 2] * (math.pi / math.sqrt(3.0))
    phase = harmonic * radians
    mean_cosine = phase.cos().mean(dim=1)
    mean_sine = phase.sin().mean(dim=1)
    resultant_squared = mean_cosine.square() + mean_sine.square()
    concentration_loss = (1.0 - resultant_squared).clamp(0.0, 1.0)
    return _ANGLE_LOSS_SCALE * concentration_loss


def lattice_loss_angle(symmetry, xya):
    """Return the mean loss of the symmetry-fold orientation concentration.

    Angles are scaled from radians by ``sqrt(3) / pi``. Hexagons use their
    six-fold rotational symmetry, while Penrose rhombus orientations occupy
    ten directions separated by ``pi / 5``. The loss is zero for a perfect
    common orientation coset and approaches ``0.1`` as those harmonic phases
    disperse. This default scale makes its magnitude comparable to the
    multiplicative XY loss.
    """
    return _angle_losses(symmetry, xya).mean()


def _tile_vertices(symmetry, side, xya, colors):
    centers = xya[..., :2]
    radians = xya[..., 2] * (math.pi / math.sqrt(3.0))
    if symmetry == 6:
        vertex_angles = (
            radians[..., None]
            + torch.arange(6, dtype=xya.dtype, device=xya.device) * (math.pi / 3.0)
            - math.pi / 6.0
        )
        offsets = side * torch.stack(
            (vertex_angles.cos(), vertex_angles.sin()), dim=-1
        )
    else:
        top_angles = torch.where(
            colors.bool(),
            xya.new_tensor(math.pi / 5.0),
            xya.new_tensor(3.0 * math.pi / 5.0),
        )
        minor = side * torch.sin(top_angles / 2.0)
        major = side * torch.cos(top_angles / 2.0)
        cosine, sine = radians.cos(), radians.sin()
        minor_axis = minor[..., None] * torch.stack((cosine, sine), dim=-1)
        major_axis = major[..., None] * torch.stack((-sine, cosine), dim=-1)
        offsets = torch.stack(
            (minor_axis, major_axis, -minor_axis, -major_axis), dim=-2
        )
    return centers[..., None, :] + offsets


def lattice_loss_edge(symmetry, side, xya, colors):
    """Return nearest-neighbour shared-edge vertex misalignment.

    For each tile, this finds its nearest centre neighbour, computes every
    vertex-to-vertex distance between the two polygons, and averages the two
    smallest distances. Dividing by ``side`` makes the result dimensionless:
    zero means the two vertices of a shared edge coincide exactly, while
    ``0.5`` means an average endpoint mismatch of half a tile side.

    Neighbour selection and selection of the two closest vertex pairs are
    discrete, but gradients flow through the selected vertex distances.
    """
    side_tensor, colors = _validate_inputs(symmetry, side, xya, colors)
    _, nearest_indices = _nearest_neighbours(xya)
    vertices = _tile_vertices(symmetry, side_tensor, xya, colors)
    vertex_count = vertices.shape[-2]
    neighbours = torch.gather(
        vertices,
        1,
        nearest_indices[..., None, None].expand(
            -1, -1, vertex_count, vertices.shape[-1]
        ),
    )
    vertex_distances = torch.cdist(
        vertices.flatten(0, 1),
        neighbours.flatten(0, 1),
    ).reshape(*xya.shape[:2], vertex_count * vertex_count)
    endpoint_distances = vertex_distances.topk(2, dim=-1, largest=False).values
    return (endpoint_distances / side_tensor).mean()


def lattice_loss(symmetry, side, xya, colors, algo="multiplicative"):
    """Return the sum of position, orientation, and shared-edge lattice losses."""
    return (
        lattice_loss_xy(symmetry, side, xya, colors, algo)
        + lattice_loss_angle(symmetry, xya)
        + lattice_loss_edge(symmetry, side, xya, colors)
    )
