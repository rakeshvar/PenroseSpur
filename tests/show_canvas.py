"""
Sanity check: render the full mother canvas with masks overlaid in canvas
coordinates, for both symmetries.

For each symmetry, draws all M mother-canvas polygons (lightly filled by
color), the canvas boundary circle, and four masks converted to canvas
coordinates: the widest-spread mask (max sigma_row^2 + sigma_col^2 of ON
pixels about the tensor center), the mask with the largest x-range (row
extent), the mask with the largest y-range (column extent), and one random.

Usage: python tests/show_canvas.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masks import build_masks
from canvas import build_canvas

OUT_DIR = Path(__file__).resolve().parent / "output"

NUM_TILES = 96
TRANSLATION = 2.0
SEED = 0

FACE = {0: "#dbe5f5", 1: "#f5dede"}          # light tile fills by color value
MASK_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
MASK_ALPHAS = [0.55, 0.45, 0.45, 0.45]

def mask_spreads(masks):
    """Weighted spatial variance of mask mass about the tensor center."""
    K, H, W = masks.shape
    row_counts = masks.sum(dim=2).double()          # (K, H)
    col_counts = masks.sum(dim=1).double()          # (K, W)
    on = row_counts.sum(dim=1)                      # (K,)
    dr2 = (torch.arange(H, dtype=torch.float64) - (H - 1) / 2) ** 2
    dc2 = (torch.arange(W, dtype=torch.float64) - (W - 1) / 2) ** 2
    return ((row_counts @ dr2 + col_counts @ dc2) / on).numpy()


def mask_ranges(masks):
    """ON-pixel row extent (x-range) and column extent (y-range) in px, per mask."""
    def extent(any_on):                              # (K, L) bool
        L = any_on.shape[1]
        first = any_on.float().argmax(dim=1)
        last = L - 1 - any_on.flip(1).float().argmax(dim=1)
        return (last - first + 1).numpy()
    return extent(masks.any(dim=2)), extent(masks.any(dim=1))


def mask_rgba(mask, color, alpha):
    """RGBA image with opacity proportional to soft mask values."""
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgb = matplotlib.colors.to_rgb(color)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.clip(mask, 0.0, 1.0) * alpha
    return rgba


def render(canvas, picks, masks, names, labels, describe, path):
    """Plot horizontal = canvas y (column direction), vertical = canvas x
    (row direction, inverted), matching the image convention."""
    centers = canvas["centers"].numpy()
    verts = canvas["vertices"].numpy()
    colors = canvas["colors"].numpy()
    radius = canvas["radius"]
    s = canvas["scaling"]
    H, W = canvas["mask_hw"]

    verts_plot = verts[:, :, ::-1]                            # plot as (y, x)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(PolyCollection(
        verts_plot, facecolors=[FACE[int(c)] for c in colors],
        edgecolors="0.55", linewidths=0.25))

    # Masks: pixel (r, c) -> canvas x = (r - (H-1)/2) s, y = (c - (W-1)/2) s.
    # imshow horizontal axis = y (from c), vertical = x (from r); origin
    # 'upper' puts row 0 (x_min) at the top, consistent with inverted x axis.
    y_lo, y_hi = (-0.5 - (W - 1) / 2) * s, (W - 0.5 - (W - 1) / 2) * s
    x_lo, x_hi = (-0.5 - (H - 1) / 2) * s, (H - 0.5 - (H - 1) / 2) * s
    handles = []
    for k, (idx, tag) in enumerate(picks):
        ax.imshow(mask_rgba(masks[idx].numpy(), MASK_COLORS[k], MASK_ALPHAS[k]),
                  extent=(y_lo, y_hi, x_hi, x_lo), origin="upper",
                  interpolation="nearest", zorder=3)
        label = f"{names[labels[idx]]}-{idx}{tag} ({describe(idx)})"
        handles.append(Patch(facecolor=MASK_COLORS[k], alpha=MASK_ALPHAS[k], label=label))

    ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="black",
                        linewidth=1.0, linestyle="--", zorder=4))
    handles.append(Patch(facecolor="none", edgecolor="black", linestyle="--",
                         label=f"canvas radius {radius:.2f}"))

    lim = radius * 1.06
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)                                    # x axis inverted
    ax.set_aspect("equal")
    ax.set_xlabel("canvas y")
    ax.set_ylabel("canvas x")
    ax.set_title(f"symmetry {canvas['symmetry']}  M={len(centers)} tiles  "
                 f"N={canvas['num_tiles']}  side={canvas['side']}")
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    masks_data = build_masks()
    masks = masks_data["masks"]
    labels = masks_data["labels"].numpy()
    names = masks_data["class_names"]
    K = len(masks)

    print("Computing ON-pixel spreads and ranges ...")
    spreads = mask_spreads(masks)
    xrange_, yrange_ = mask_ranges(masks)
    describe = lambda i: f"$\\sigma^2$ {spreads[i]:.0f}, x {xrange_[i]}px, y {yrange_[i]}px"

    extrema = {}                                        # idx -> merged tags
    for idx, tag in [(int(spreads.argmax()), "max spread"),
                     (int(xrange_.argmax()), "max x-range"),
                     (int(yrange_.argmax()), "max y-range")]:
        extrema[idx] = f"{extrema[idx]}, {tag}" if idx in extrema else tag
    picks = [(idx, f" {tag}") for idx, tag in extrema.items()]
    rng = np.random.default_rng()
    chosen = {i for i, _ in picks}
    random_pick = int(rng.choice([i for i in range(K) if i not in chosen]))
    picks.append((random_pick, " random"))
    for idx, tag in picks:
        print(f"  mask {idx} ({names[labels[idx]]}-{idx}){tag}: "
              f"spread {spreads[idx]:.1f} px^2, x-range {xrange_[idx]}, y-range {yrange_[idx]}")

    for symmetry in (6, 5):
        print(f"\n=== Symmetry {symmetry} ===")
        canvas = build_canvas(
            symmetry,
            NUM_TILES,
            tuple(masks.shape[1:]),
            masks_data["target_on"],
            TRANSLATION,
            seed=SEED,
        )
        render(canvas, picks, masks, names, labels, describe,
               OUT_DIR / f"canvas_masks_s{symmetry}.png")


if __name__ == "__main__":
    main()
