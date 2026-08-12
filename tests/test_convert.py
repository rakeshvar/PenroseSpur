import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canvas import build_canvas_for_mask
from convert import vertices as to_vertices
from convert import vertices_to_xya


def _periodic_angle_error(actual, expected, period):
    delta = np.remainder(actual - expected + period / 2, period) - period / 2
    return np.abs(delta).max().item()


def _unordered_vertex_error(actual, expected):
    distances = np.linalg.norm(
        actual[..., :, None, :] - expected[..., None, :, :], axis=-1
    )
    return distances.min(axis=-1).max().item()


def check_canvas_roundtrip(symmetry, num_tiles=32, seed=0):
    """Build a canvas and assert both conversion directions are consistent."""
    canvas = build_canvas_for_mask(
        symmetry=symmetry,
        num_tiles=num_tiles,
        mask_hw=(448, 448),
        target_on=32000,
        translation=2.0,
        seed=seed,
    )
    centers = canvas["centers"].numpy()
    angles = canvas["angles"].numpy()
    colors = canvas["colors"].numpy()
    original_vertices = canvas["vertices"].numpy()
    xya = np.concatenate([centers, angles[..., None]], axis=-1)

    converted_vertices = to_vertices(
        symmetry, centers, angles, colors, side=canvas["side"]
    )
    vertex_error = _unordered_vertex_error(converted_vertices, original_vertices)

    recovered_xya = vertices_to_xya(converted_vertices, symmetry)
    center_error = np.abs(recovered_xya[..., :2] - xya[..., :2]).max().item()
    angle_period = math.pi / 3 if symmetry == 6 else math.pi
    angle_error = _periodic_angle_error(
        recovered_xya[..., 2], xya[..., 2], angle_period
    )

    tolerance = 1e-5
    assert vertex_error < tolerance, f"vertex error {vertex_error:.3e}"
    assert center_error < tolerance, f"center error {center_error:.3e}"
    assert angle_error < tolerance, f"angle error {angle_error:.3e}"
    print(
        f"symmetry={symmetry} tiles={len(xya)} "
        f"vertex_error={vertex_error:.3e} "
        f"center_error={center_error:.3e} "
        f"angle_error={angle_error:.3e}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check xya/color and vertex conversions for both tile grids."
    )
    parser.add_argument("-n", "--num-tiles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for symmetry in (6, 5):
        check_canvas_roundtrip(symmetry, args.num_tiles, args.seed)
    print("All conversion checks passed.")


if __name__ == "__main__":
    main()
