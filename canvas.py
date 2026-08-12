"""
Build an in-memory mother canvas: an M-tile patch large enough to cover any
mask under any rotation plus a translation jitter of T polygon-sides.

Symmetry 6: hexagonal grid center/color/vertex formulas from QRS
Symmetry 5: Penrose P3 rhombuses via the de Bruijn pentagrid construction
    (color 1 = thin, tilt = direction from rhombus center to the apex vertex).

Tensors produced (float32 unless noted):
    centers (M, 2), angles (M,), colors (M,) uint8, indices (M,) int64,
    vertices (M, V, 2)   with V = 6 (hex) or 4 (pen)
plus scalar metadata (side, density, scaling, translation, ...).
"""
import math
import torch
import numpy as np

from convert import hex_vertices, pen_vertices

#--------------------------------------------------------------------------
# Constants for Area 
#--------------------------------------------------------------------------
π = np.pi
ψ = (math.sqrt(5) - 1) / 2        # 1/φ
ψψ = 1 - ψ                        # 1/φ²

# Area of unit regular hexagon = 3√3 / 2 ≈ 2.598076211353316
AREA_OF_UNIT_HEXAGON = 3 * math.sqrt(3) / 2

# Average area of a Penrose rhombus = (sin(π/5) * 1/φ^2 + sin(2π/5) * 1/φ) ≈ 0.8506508083520399
AREA_OF_AVG_RHOMBUS = (
    math.sin(math.pi / 5) * ψψ +
    math.sin(2 * math.pi / 5) * ψ
)

def area_of_unit_polygon(symmetry):
    if symmetry == 6:        return AREA_OF_UNIT_HEXAGON
    elif symmetry == 5:      return AREA_OF_AVG_RHOMBUS
    else:                    raise ValueError(f"Invalid symmetry: {symmetry}")


def area_of_polygon(symmetry, side):
    return area_of_unit_polygon(symmetry) * side ** 2

#--------------------------------------------------------------------------
# Variance of tiles
#--------------------------------------------------------------------------
# Universal geometric constant:
#   Var(x) ≈ Var(y) ≈ VAR_PER_AREA × occupied_area
# Calibrated once against the normalized MPEG7 masks.
VAR_PER_AREA = 0.16


def target_side_for_unit_var(symmetry, num_tiles, TARGET_VAR=1.):
    """
    N tiles of side s have total area = N * s^2 * unit_area
    Variance (of x or y) = VAR_PER_AREA * total_area
    solve for s, so that: 
        var(x) ≈ var(y) ≈ 1
    """
    total_var_for_unit_side = VAR_PER_AREA * num_tiles * area_of_unit_polygon(symmetry)
    target_side = math.sqrt(TARGET_VAR / total_var_for_unit_side)
    return target_side


#--------------------------------------------------------------------------
# Hexagonal grid (symmetry 6)
#--------------------------------------------------------------------------
def hex_color(q, r, s):
    """Same coloring as PenroseDiffusion: no two 'dark' hexagons touch."""
    a = np.stack([np.abs(q), np.abs(r), np.abs(s)])
    return ((a.max(axis=0) + a.min(axis=0)) % 3 == 0).astype(np.uint8)


def hex_grid(radius, side=1):
    """
    All hexagons (pointy-top, side `side`) whose centers lie within `radius`
    of the origin plus one extra ring, so the disk of `radius` is fully tiled.
    `radius` and `side` are in the same coordinate units.
    Returns centers (M,2), angles (M,), colors (M,), vertices (M,6,2).
    """
    # Centers: x = sqrt(3) s (q + r/2), y = 1.5 s r
    qmax = int(math.ceil(radius / (math.sqrt(3) * side))) + 2
    rmax = int(math.ceil(radius / (1.5 * side))) + 2
    q, r = np.meshgrid(np.arange(-qmax - rmax, qmax + rmax + 1),
                       np.arange(-rmax, rmax + 1), indexing="ij")
    q, r = q.ravel(), r.ravel()
    x = math.sqrt(3) * side * (q + r / 2.)
    y = 1.5 * side * r

    keep = np.hypot(x, y) <= radius + math.sqrt(3) * side
    q, r, x, y = q[keep], r[keep], x[keep], y[keep]

    centers = np.stack([x, y], axis=1)
    colors = hex_color(q, r, -q - r)
    angles = np.zeros(len(q))
    vertices = hex_vertices(centers, angles, colors, side)

    return centers, angles, colors, vertices


#--------------------------------------------------------------------------
# Penrose P3 via de Bruijn pentagrid (symmetry 5)
#--------------------------------------------------------------------------
def pen_grid_debruijn(radius, side=1, rng=None):
    """
    Penrose rhombus patch covering a disk of `radius`, with edge length `side`.

    Five families of grid lines with directions uᵢ = (cos(2πi/5), sin(2πi/5)) and offsets γᵢ
        Each family has 2k+1 lines. 

    In total, we have 5C2 * (2k+1)² = 10 * (2k+1)² intersection points.
    Each of which maps to one Penrose Rhombus (via a dual map).
    """
    if rng is None:
        rng = np.random.default_rng()

    γ = rng.uniform(0, 1, size=5)
    unit_radius = radius / side + 1.

    θ = 2 * π * np.arange(5) / 5
    U = np.stack([np.cos(θ), np.sin(θ)], axis=1)            # (5, 2)

    # The dual map scales distances by about 5/2
    grid_radius = unit_radius * 2 / 5
    kmax = int(math.ceil(grid_radius))
    K = np.arange(-kmax, kmax + 1)                          # (L=2k+1,)

    # All 10 unordered family pairs (I < J) crossed with all (KI, KJ) line indices
    I, J = np.triu_indices(5, k=1)                          # (10,)
    KI, KJ = np.meshgrid(K, K, indexing="ij")
    KI = KI.ravel()[None, :]
    KJ = KJ.ravel()[None, :]                                # (1, L*L)

    # Solve for the intersection of the two lines:
    #   uᵢ . (x, y) = kᵢ - γᵢ
    #   uⱼ . (x, y) = kⱼ - γⱼ
    DetIJ = (U[I, 0] * U[J, 1] - U[I, 1] * U[J, 0])[:, None]        # (10, 1)
    CI = KI - γ[I][:, None]                                         # (10, L*L)
    CJ = KJ - γ[J][:, None]
    Xx = (CI * U[J, 1][:, None] - CJ * U[I, 1][:, None]) / DetIJ    # (10, L*L)
    Yx = (U[I, 0][:, None] * CJ - U[J, 0][:, None] * CI) / DetIJ

    keep = np.hypot(Xx, Yx) <= grid_radius                         # (10, L*L)
    print(f"Keeping {keep.sum()} of {keep.size}=(10*{2*kmax+1}²) intersection points ({keep.mean() :.1%})")

    XYx = np.stack([Xx[keep], Yx[keep]], axis=1)                   # (N, 2)
    I = np.broadcast_to(I[:, None], keep.shape)[keep]              # (10,) -> (10, 1) -> (10, L*L) -> (N,)
    J = np.broadcast_to(J[:, None], keep.shape)[keep]
    KI = np.broadcast_to(KI, keep.shape)[keep]                     # (1, L*L) -> (10, L*L) -> (N,)
    KJ = np.broadcast_to(KJ, keep.shape)[keep]
    Δ = J - I
    is_thin = (Δ == 2) | (Δ == 3)

    # Reparameterize intersections into regions (indexed by 5D ints)
    # Project to the 5D line family space
    A5D = XYx @ U.T + γ[None, :]                                   # (N, 5)
    A5D = np.ceil(A5D)                         

    # Override with line indices for the two families that intersect at this point
    # To avoid ceil rounding error
    I5 = np.arange(5)[None, :]
    A5D = np.where(I5 == I[:, None], KI[:, None], A5D)
    A5D = np.where(I5 == J[:, None], KJ[:, None], A5D)
    arc_parity = (A5D.sum(axis=1).astype(np.int64) % 2).astype(bool)[:, None] # (N, 1)

    # Project back to the 2D plane
    # F is either A or C, depending on the parity of the arc.
    F = A5D @ U                                                   # (N, 2)
    AC = U[I] + U[J]
    M = F + AC/2
    θ1 = np.arctan2(AC[:, 1], AC[:, 0])
    θ2 = np.arctan2(-AC[:, 1], -AC[:, 0])
    angles = np.where(arc_parity.ravel(), θ1, θ2)

    colors = (is_thin).astype(np.uint8)
    M -= np.mean(M, axis=0)
    M *= side
    vertices = pen_vertices(M, angles, colors, side)
    return M, angles, colors, vertices

#--------------------------------------------------------------------------
# Canvas assembly
#--------------------------------------------------------------------------
def polar_sort(centers):
    """Order tiles by (radius, clockwise angle), like pen/mother.py polar_key."""
    radius = np.round(np.hypot(centers[:, 0], centers[:, 1]), 5)
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    theta_cw = (-theta + π) % (2 * π)
    return np.lexsort((theta_cw, radius))


def radius_side(symmetry, num_tiles, mask_hw, target_on, translation):
    Hmax, Wmax = mask_hw
    side = target_side_for_unit_var(symmetry, num_tiles)
    density = 1. / area_of_polygon(symmetry, side)
    scaling = math.sqrt(num_tiles / (target_on * density))    # canvas units per pixel
    radius = math.hypot(Hmax / 2, Wmax / 2) * scaling + translation * side
    print(f"""
    Hmax={Hmax} Wmax={Wmax} 
    Hypot={math.hypot(Hmax / 2, Wmax / 2)} 
    scaling={scaling} 
    Hypot * scaling = {math.hypot(Hmax / 2, Wmax / 2) * scaling:.2f}
    side={side:.5f} 
    density={density} 
    translation={translation:.5f}
    translation * side = {translation * side:.5f}
    radius = hypot * scaling + translation * side = {radius:.2f}
    """)
    
    print(f"""
    num_tiles = {num_tiles:7d}
    side      = {side:7.5f} canvas units
    radius    = {radius:7.2f} canvas units
    density   = {density:7.3f} tiles per unit canvas area
    scaling   = {scaling:7.5f} canvas units per pixel ({1/scaling:.2f} pixels per canvas unit)
    """)
    return radius, side

def build_canvas_for_mask(symmetry, num_tiles, mask_hw, target_on, translation, 
                          seed=None, return_indices=False):
    radius, side = radius_side(symmetry, num_tiles, mask_hw, target_on, translation)
    data = build_canvas(symmetry, radius, side, seed, return_indices)
    density = 1. / area_of_polygon(symmetry, side)
    scaling = math.sqrt(num_tiles / (target_on * density))
    data.update({
        "num_tiles": num_tiles,
        "side": side,
        "density": density,
        "scaling": scaling,
        "translation": translation * side,
        "radius": radius,
        "mask_hw": tuple(mask_hw),
        "target_on": target_on,
    })
    return data

def build_canvas(symmetry, radius, side, seed=None, return_indices=False):
    if symmetry == 6:
        centers, angles, colors, vertices = hex_grid(radius, side)
    else:
        rng = np.random.default_rng(seed)
        centers, angles, colors, vertices = pen_grid_debruijn(radius, side, rng)

    if return_indices:
        order = polar_sort(centers)
        centers, angles, colors, vertices = centers[order], angles[order], colors[order], vertices[order]

    M = len(centers)
    num_colored = colors.sum()
    print(f"""
    symmetry  = {symmetry:7d}
    canvas    = {M:7d} tiles
    coloredd  = {num_colored:7d} ({num_colored / M :.1%})"""
    )

    data = {
        "symmetry": symmetry,
        "centers": torch.from_numpy(centers).float(),                        # (M, 2)
        "angles": torch.from_numpy(angles).float(),                          # (M,)
        "colors": torch.from_numpy(colors.astype(np.uint8)),                 # (M,)
        "vertices": torch.from_numpy(vertices).float(),                      # (M, V, 2)
    }
    if return_indices:
        data["indices"] = torch.arange(M, dtype=torch.int64)

    return data


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from svg import save_polygons

    parser = argparse.ArgumentParser(description="Build mother canvas.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("symmetry", type=int, choices=[5, 6])
    parser.add_argument("-m", "--from-mask", action="store_true", default=False)

    # non-mask-based parameters
    parser.add_argument("-r", "--radius", type=float, default=5.)
    parser.add_argument("-s", "--side", type=float, default=1.)

    # mask-based parameters
    parser.add_argument("-n", "--num_tiles", type=int, default=96)
    parser.add_argument("-H", "--mask_height", type=int, default=448)
    parser.add_argument("-W", "--mask_width", type=int, default=448)
    parser.add_argument("-o", "--target_on", type=float, default=32000)
    parser.add_argument("-t", "--translation", type=float, help="Jitter range in polygon sides", default=2.)
    parser.add_argument("-d", "--seed", type=int, default=None)
    parser.add_argument("-i", "--indices", action="store_true", default=False)
    a = parser.parse_args()

    if a.from_mask:
        print(f"Building canvas from mask:")
        print(f"N={a.num_tiles} H={a.mask_height}x{a.mask_width}, target_on={a.target_on}, translation={a.translation}")
        canvas = build_canvas_for_mask(a.symmetry, a.num_tiles, (a.mask_height, a.mask_width), a.target_on, a.translation)
        name = f"{a.symmetry}_{a.num_tiles}_{a.mask_height}x{a.mask_width}_{a.target_on}_{a.translation}"
    else:
        print(f"Building canvas from scratch:")
        print(f"radius={a.radius} side={a.side}, seed={a.seed}, indices={a.indices}")
        canvas = build_canvas(a.symmetry,a.radius,a.side,seed=a.seed, return_indices=a.indices)
        name = f"{a.symmetry}_{a.radius}_{a.side}_{a.seed}"

    polygons = canvas["vertices"]
    output = Path(__file__).resolve().parent / "tests" / "output" / "canvas" / f"canvas{name}.svg" 
    save_polygons(output, polygons, canvas["colors"], palette=a.symmetry, show_arcs=a.symmetry == 5, radius=a.radius)
    
    print(f"Saved {output}")