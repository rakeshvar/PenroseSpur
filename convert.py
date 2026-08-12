"""Conversions between tile parameters and polygon vertices."""

import math

import numpy as np

π = np.pi

def _wrap(θ):
    return np.remainder(θ + π, 2 * π) - π

#--------------------------------------------------------------------------

def hex_vertices(centers, angles, colors, side):
    # Vertices at angle pi/3 * i - pi/6, distance `side` from center
    # TODO: can assume all angles = angles[0]
    vertex_angles = angles[:, None] + π/3 * np.arange(6) - π/6
    offsets = side * np.stack([np.cos(vertex_angles), np.sin(vertex_angles)], axis=-1)
    return centers[:, None, :] + offsets


def pen_vertices(M, angles, colors, side):
    top_angles = np.where(colors, π/5, 3*π/5)           # (N,)
    dmb = side * np.cos(top_angles/2)
    dma = side * np.sin(top_angles/2)
    sθ, cθ = np.sin(angles), np.cos(angles)
    MA = dma[:, None] * np.stack([cθ, sθ], axis=-1)     # (N, 2)
    MB = dmb[:, None] * np.stack([-sθ, cθ], axis=-1)
    offsets = np.stack([MA, MB, -MA, -MB], axis=1)      # (N, 4, 2)
    return M[:, None, :] + offsets                      # (N, 4, 2)

def vertices(symmetry, centers, angles, colors, side):
    if symmetry == 6:
        return hex_vertices(centers, angles, colors, side)
    elif symmetry == 5:
        return pen_vertices(centers, angles, colors, side)
    raise ValueError(f"Unknown symmetry: {symmetry}")

#--------------------------------------------------------------------------

def hex_xya_from_vertices(vertices):
    centers = vertices.mean(axis=-2)
    first = vertices[..., 0, :] - centers
    angles = np.atan2(first[..., 1], first[..., 0]) + π / 6
    return np.concatenate([centers, _wrap(angles)[..., None]], axis=-1)

def pen_xya_from_vertices(vertices):
    centers = vertices.mean(axis=-2)
    MA = vertices[..., 0, :] - centers
    angle = np.atan2(MA[..., 1], MA[..., 0])
    return np.concatenate([centers, _wrap(angle)[..., None]], axis=-1)

def vertices_to_xya(vertices, symmetry):
    if symmetry == 6:
        return hex_xya_from_vertices(vertices)
    elif symmetry == 5:
        return pen_xya_from_vertices(vertices)
    raise ValueError(f"Unknown symmetry: {symmetry}")

#--------------------------------------------------------------------------

