import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import sys


def visualize_debruijn_grid(kmax, gamma=None, rng=None, num_dirs_to_show=5):
    """
    Visualize a Penrose rhombus patch from line indices -kmax through kmax.

    Five families of grid lines with directions 
        vⱼ = e^(2πij/5),   j ∈ {0,1,2,3,4}
        and random offsets γⱼ; 
    Each pairwise line intersection maps to one Penrose rhombus of the dual tiling with vertices p1..p4.

    Returns centers (M,2), angles (M,), colors (M,) (1=thin), vertices (M,4,2).
    """
    if rng is None:
        rng = np.random.default_rng()
    if gamma is None:
        gamma = rng.uniform(0, 1, size=5)
    gamma = np.asarray(gamma, dtype=float)
    v = np.exp(2j * np.pi * np.arange(5) / 5)

    # The dual map z -> sum_j ceil(Re(z conj(v_j)) + gamma_j) v_j scales
    # distances by about 5/2, so intersections within grid_radius are enough.

    all_pts, all_type, all_a, all_b, all_ka, all_kb = [], [], [], [], [], []
    ks = np.arange(-kmax, kmax + 1)

    for a in range(4):
        for b in range(a + 1, 5):
            # Solve
            # Re(vₐ) x + Im(vₐ) y = kₐ − γₐ
            # Re(vᵇ) x + Im(vᵇ) y = kᵇ − γᵇ
            det = v[a].real * v[b].imag - v[a].imag * v[b].real
            ka, kb = np.meshgrid(ks, ks, indexing="ij")
            ka, kb = ka.ravel(), kb.ravel()
            ca = ka - gamma[a]
            cb = kb - gamma[b]
            x = (ca * v[b].imag - cb * v[a].imag) / det
            y = (v[a].real * cb - v[b].real * ca) / det
            pt = x + 1j * y

            all_pts.append(pt)
            all_ka.append(ka)
            all_kb.append(kb)
            all_a.append(np.full(len(ka), a))
            all_b.append(np.full(len(kb), b))
            all_type.append(np.full(len(ka), abs(a - b) in (1, 4)))

    # point is at the intersection of the two lines:
    # Re(vₐ) x + Im(vₐ) y = kₐ − γₐ
    # Re(vᵇ) x + Im(vᵇ) y = kᵇ − γᵇ
    pt = np.concatenate(all_pts)
    ka = np.concatenate(all_ka)
    kb = np.concatenate(all_kb)
    aa = np.concatenate(all_a)
    bb = np.concatenate(all_b)
    is_thick = np.concatenate(all_type)

    # Recover the sub-region by projecting the points onto the lines and rounding to the nearest integer.
    proj = pt.real[:, None] * v.real[None, :] + pt.imag[:, None] * v.imag[None, :] # (N, 5)
    S = np.ceil(proj + gamma[None, :])

    # Correct the line indices by overriding the values of aa and bb as they are integers and may be rounded away from the true values.
    K = S.copy()
    rows = np.arange(len(pt))
    K[rows, aa] = ka
    K[rows, bb] = kb


    p1 = K @ v
    p2 = p1 + v[aa]
    p3 = p2 + v[bb]
    p4 = p1 + v[bb]

    centers = (p1 + p3) / 2

    # Recenter the patch, then keep tiles within the requested radius.
    shift = centers.mean()
    p1, p2, p3, p4, centers = (z - shift for z in (p1, p2, p3, p4, centers))

    # Angle convention of pen/xya.py Rhombus: vertices (A, B, C, D) with B the
    # apex (interior angle 108 deg for fat, 36 deg for thin -- both are p2
    # here), M = (A+C)/2 the rhombus center, tilt = phase(B - M) flipped by pi
    # if cross(MB, C - A) < 0.
    MB = p2 - centers
    AC = p3 - p1
    angles = np.angle(MB)
    flip = (MB.real * AC.imag - MB.imag * AC.real) < 0
    angles[flip] += np.pi
    angles = (angles + np.pi) % (2 * np.pi) - np.pi

    colors = (~is_thick).astype(np.uint8)     # 1 = thin, matching Rhombus.color
    vertices = np.stack([p1, p2, p3, p4], axis=1)
    vertices = np.stack([vertices.real, vertices.imag], axis=2)
    centers = np.stack([centers.real, centers.imag], axis=1)

#------------------------------------------------------------------------------------------------
# PLot
#------------------------------------------------------------------------------------------------
    selected = (aa<num_dirs_to_show) & (bb<num_dirs_to_show)
    pts_selected = pt[selected]

    print(f"Number of points selected: {len(pts_selected)}/{len(pt)}")
    thickness_palette = np.array(['#4C78A8', '#F28E2B'])  # muted blue and warm orange
    thickness_colors = thickness_palette[is_thick[selected].astype(int)]
    plt.scatter(pts_selected.real, pts_selected.imag, c=thickness_colors, s=1, alpha=0.7)

    # Plot the lines based on v_j, k_j, gamma_j . i.e. Re(v_j) x + Im(v_j) y = k_j - gamma_j
    L = 2*kmax
    colors = ['m', 'r', 'g', 'b', 'y']
    for j in range(num_dirs_to_show):
        a, b = v[j].real, v[j].imag
        for k in range(-kmax, kmax + 1):
            c = k - gamma[j]
            # plot the line a*x + b*y = c
            if b != 0:
                plt.plot([-L, L], [(c+a*L)/b, (c-a*L)/b], color=colors[j], linestyle='-', alpha=0.1, linewidth=0.4)
            else:
                plt.plot([c/a, c/a], [-L, L], 'm-', alpha=0.1, linewidth=0.4)
    plt.xlim(-2*L, 2*L)
    plt.ylim(-L, L)
    plt.gca().set_aspect('equal', adjustable='box')


    SCALE = 2.5
    ps1 = p1[selected]/SCALE
    ps2 = p2[selected]/SCALE
    ps3 = p3[selected]/SCALE
    ps4 = p4[selected]/SCALE
    # Plot p1 --- p2 --- p3 --- p4 --- p1
    rhombus_edges = np.stack([
        np.column_stack((ps1.real, ps1.imag)),
        np.column_stack((ps2.real, ps2.imag)),
        np.column_stack((ps3.real, ps3.imag)),
        np.column_stack((ps4.real, ps4.imag)),
        np.column_stack((ps1.real, ps1.imag)),
    ], axis=1)
    plt.gca().add_collection(
        LineCollection(rhombus_edges, colors=thickness_colors, linewidths=0.4, alpha=0.7)
    )

    # Draw lines from pto to corresponding centers
    centers_selected = centers[selected] / SCALE
    plot_to_vertex = -1
    if plot_to_vertex == 0:
        for i in range(len(pts_selected)):
            plt.plot([pts_selected.real[i], centers_selected[i, 0]], 
                    [pts_selected.imag[i], centers_selected[i, 1]], 
                    '--', color='gray', alpha=0.5, linewidth=0.4)
    elif plot_to_vertex > 0:
        puse = (None, ps1, ps2, ps3, ps4)[plot_to_vertex]
        for i in range(len(pts_selected)):
            plt.plot([pts_selected.real[i], puse.real[i]], 
                    [pts_selected.imag[i], puse.imag[i]], 
                    '--', color='gray', alpha=0.5, linewidth=0.4)

    Ks = K[selected].astype(int)
    for i in range(len(Ks)):
        if abs(pts_selected[i]) > 2*kmax and abs(Ks[i][0]) > 2 and abs(Ks[i][3]) > 2:
            plt.text(pts_selected.real[i], pts_selected.imag[i], f'{Ks[i][0]}{Ks[i][1]}{Ks[i][2]}{Ks[i][3]}{Ks[i][4]}', fontsize=3)
    
    if True:
        plt.axis('off')

    output_path = Path('debruijn/visual.svg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0)
    # plt.show()
#


    return centers, angles, colors, vertices

def main():
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_dirs_to_show = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"Number of lines on each side: {kmax}")
    print(f"Number of directions to show: {num_dirs_to_show}")
    centers, angles, colors, vertices = visualize_debruijn_grid(
        kmax=kmax, num_dirs_to_show=num_dirs_to_show
    )

if __name__ == "__main__":
    main()