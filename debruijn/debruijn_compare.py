"""
Compare the vectorized real-valued pen_grid_debruijn (canvas.py) against the
original loop-based complex-number implementation (kept here as reference).

Numeric check: same tile count, and matching centers / angles / colors /
vertices after sorting both outputs into a canonical order.

Visual check: SVGs in tests/debruijn/ -- one per implementation, plus an
overlay where the old tiling's edges are stroked on top of the new tiling's
filled rhombi (any misalignment would show as doubled edges).

Run:  python tests/debruijn_compare.py [radius] [seed]
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canvas import pen_grid_debruijn

OUT_DIR = Path(__file__).resolve().parent / "debruijn"


def pen_grid_debruijn_old(radius, gamma=None, rng=None):
    """Original implementation: Python loop over the 10 grid-family pairs,
    points represented as complex numbers."""
    if rng is None:
        rng = np.random.default_rng()
    if gamma is None:
        gamma = rng.uniform(0, 1, size=5)
    gamma = np.asarray(gamma, dtype=float)
    v = np.exp(2j * np.pi * np.arange(5) / 5)

    grid_radius = (radius + 1) * (2 / 5)

    all_pts, all_type, all_a, all_b, all_ka, all_kb = [], [], [], [], [], []
    kmax = int(math.ceil(grid_radius)) + 1
    ks = np.arange(-kmax, kmax + 1)

    for a in range(4):
        for b in range(a + 1, 5):
            det = v[a].real * v[b].imag - v[a].imag * v[b].real
            ka, kb = np.meshgrid(ks, ks, indexing="ij")
            ka, kb = ka.ravel(), kb.ravel()
            ca = ka - gamma[a]
            cb = kb - gamma[b]
            x = (ca * v[b].imag - cb * v[a].imag) / det
            y = (v[a].real * cb - v[b].real * ca) / det
            pt = x + 1j * y

            keep = np.abs(pt) <= grid_radius
            all_pts.append(pt[keep])
            all_ka.append(ka[keep])
            all_kb.append(kb[keep])
            all_a.append(np.full(keep.sum(), a))
            all_b.append(np.full(keep.sum(), b))
            all_type.append(np.full(keep.sum(), abs(a - b) in (1, 4)))

    pt = np.concatenate(all_pts)
    ka = np.concatenate(all_ka)
    kb = np.concatenate(all_kb)
    aa = np.concatenate(all_a)
    bb = np.concatenate(all_b)
    is_thick = np.concatenate(all_type)

    proj = pt.real[:, None] * v.real[None, :] + pt.imag[:, None] * v.imag[None, :]
    S = np.ceil(proj + gamma[None, :])

    K = S.copy()
    rows = np.arange(len(pt))
    K[rows, aa] = ka
    K[rows, bb] = kb

    p1 = K @ v
    p2 = p1 + v[aa]
    p3 = p2 + v[bb]
    p4 = p1 + v[bb]

    centers = (p1 + p3) / 2

    swap = (K.sum(axis=1).astype(np.int64) % 2).astype(bool)
    p1, p3 = np.where(swap, p3, p1), np.where(swap, p1, p3)
    angles = np.angle(p1 - centers)

    colors = (~is_thick).astype(np.uint8)
    vertices = np.stack([p1, p2, p3, p4], axis=1)
    vertices = np.stack([vertices.real, vertices.imag], axis=2)
    centers = np.stack([centers.real, centers.imag], axis=1)
    return centers, angles, colors, vertices


#--------------------------------------------------------------------------
# Numeric comparison
#--------------------------------------------------------------------------
def canonical_order(centers):
    return np.lexsort((np.round(centers[:, 1], 6), np.round(centers[:, 0], 6)))


def compare_numeric(radius, side, seed):
    c_old, a_old, col_old, v_old = pen_grid_debruijn_old(
        radius / side, rng=np.random.default_rng(seed)
    )
    c_old *= side
    v_old *= side
    c_new, a_new, col_new, v_new = pen_grid_debruijn(
        radius, side=side, rng=np.random.default_rng(seed)
    )

    print(f"Radius={radius:7.2f} Side={side:.2f} Seed={seed}  #old={len(c_old):6d}  #new={len(c_new):6d}  ")
    if len(c_old) != len(c_new):
        print("\t"*4 + "Tile count mismatch")
        return

    io, im = canonical_order(c_old), canonical_order(c_new)
    c_old, a_old, col_old, v_old = c_old[io], a_old[io], col_old[io], v_old[io]
    c_new, a_new, col_new, v_new = c_new[im], a_new[im], col_new[im], v_new[im]

    dc = np.abs(c_old - c_new).max()
    vertex_distances = np.linalg.norm(
        v_old[:, :, None, :] - v_new[:, None, :, :], axis=-1
    )
    dv = max(
        vertex_distances.min(axis=2).max(),
        vertex_distances.min(axis=1).max(),
    )
    da = np.abs((a_old - a_new + np.pi) % (2 * np.pi) - np.pi).max()
    dcol = int((col_old != col_new).sum())
    print(f"max|dcenters|={dc:.2e}  max|dvertices|={dv:.2e}  "
          f"max|dangles|={da:.2e}  color mismatches={dcol}")

    if (dc < 1e-9 and dv < 1e-9 and da < 1e-9 and dcol == 0):
        print(f"Numeric comparison passed")
    else:
        print(f"Numeric comparison failed")

#--------------------------------------------------------------------------
# SVG rendering
#--------------------------------------------------------------------------
def svg_tiling(path, vertices, colors, radius, extra=""):
    """Filled rhombi: thin = orange, thick = teal."""
    scale = 600 / (2 * radius)
    polys = []
    for verts, thin in zip(vertices, colors):
        pts = " ".join(f"{(x + radius) * scale:.2f},{(radius - y) * scale:.2f}"
                       for x, y in verts)
        fill = "#e8a33d" if thin else "#3d8f8f"
        polys.append(f'<polygon points="{pts}" fill="{fill}" '
                     f'stroke="#222" stroke-width="0.5"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" '
           f'viewBox="0 0 600 600">\n<rect width="600" height="600" fill="white"/>\n'
           + "\n".join(polys) + "\n" + extra + "</svg>\n")
    Path(path).write_text(svg)
    print(f"  wrote {path}")


def svg_overlay(path, v_new, col_new, v_old, radius):
    """New tiling filled + old tiling's outlines stroked in red on top."""
    scale = 600 / (2 * radius)
    parts = []
    for verts, thin in zip(v_new, col_new):
        pts = " ".join(f"{(x + radius) * scale:.2f},{(radius - y) * scale:.2f}"
                       for x, y in verts)
        fill = "#f3cf9a" if thin else "#9ecccc"
        parts.append(f'<polygon points="{pts}" fill="{fill}" stroke="none"/>')
    for verts in v_old:
        pts = " ".join(f"{(x + radius) * scale:.2f},{(radius - y) * scale:.2f}"
                       for x, y in verts)
        parts.append(f'<polygon points="{pts}" fill="none" '
                     f'stroke="#c0392b" stroke-width="0.7"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" '
           f'viewBox="0 0 600 600">\n<rect width="600" height="600" fill="white"/>\n'
           + "\n".join(parts) + "\n</svg>\n")
    Path(path).write_text(svg)
    print(f"  wrote {path}")


if __name__ == "__main__":
    radius = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    OUT_DIR.mkdir(exist_ok=True)

    for s in (seed, seed + 1, seed + 2):
        for r in (radius, radius * 3):
            for side in (1.0, 0.7):
                compare_numeric(r, side, s)

            c_old, a_old, col_old, v_old = pen_grid_debruijn_old(radius, rng=np.random.default_rng(seed))
            c_new, a_new, col_new, v_new = pen_grid_debruijn(radius, rng=np.random.default_rng(seed))
            
            name = f"r{r:.0f}_s{side:.2f}_seed{s}"
            svg_tiling(OUT_DIR / (name+".old.svg"), v_old, col_old, radius)
            svg_tiling(OUT_DIR / (name+".new.svg"), v_new, col_new, radius)
            svg_overlay(OUT_DIR / (name+".overlay.svg"), v_new, col_new, v_old, radius)
