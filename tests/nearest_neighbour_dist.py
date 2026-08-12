"""
Nearest-neighbour distance statistics for the mother canvases.

For each symmetry (6 = hexagonal, 5 = pentagonal / Penrose), loads the
mother canvas, computes the center-to-center distance from every polygon
to its nearest neighbour, normalizes by unit_side, and plots one
histogram per tile color (hex: light/dark; penrose: thick/thin).

For hexagonal symmetry (6), the nearest-neighbour distance is always √3 ≈ 1.732

For pentagonal symmetry (5), the nearest-neighbour distance is 
    thick-thick : sin(2π/5)    ≈ 0.9510565162951535
    thin-thin : sin(π/5)      ≈ 0.5877852522924731
    thick-thin : sin(3π/10)   ≈ 0.8090169943749475
Usage: python tests/nearest_neighbour_dist.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masks import build_masks
from canvas import build_canvas

OUT_DIR = Path(__file__).resolve().parent / "output"

NUM_TILES = 96
TRANSLATION = 2.0
SEED = 0

COLOR_NAMES = {6: {0: "light", 1: "dark"}, 5: {0: "thick", 1: "thin"}}
FACE = {0: "#4c72b0", 1: "#c44e52"}

def nn_distances(canvas):
    """Normalized (by unit_side) distance from each tile center to its
    nearest neighbour's center."""
    centers = canvas["centers"]
    d = torch.cdist(centers, centers)
    d.fill_diagonal_(float("inf"))
    return d.min(dim=1).values / canvas["side"]


def plot_histograms(canvas, nn, ax, bin_width=0.01):
    symmetry = canvas["symmetry"]
    colors = canvas["colors"]
    lo = np.floor(nn.min().item() / bin_width) * bin_width
    hi = np.ceil(nn.max().item() / bin_width) * bin_width
    bins = np.arange(lo, hi + bin_width, bin_width)
    for c in (0, 1):
        vals = nn[colors == c].numpy()
        name = COLOR_NAMES[symmetry][c]
        ax.hist(vals, bins=bins, color=FACE[c], alpha=0.6,
                label=f"{name} (n={len(vals)}, "
                      f"mean {vals.mean():.3f}, min {vals.min():.3f})")
    ax.set_xlabel("nearest-neighbour distance / unit_side")
    ax.set_ylabel("tile count")
    ax.set_title(f"symmetry {symmetry}  M={len(colors)} tiles  "
                 f"unit_side={canvas['side']:.4g}")
    ax.legend(fontsize=8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    masks_data = build_masks()
    mask_hw = tuple(masks_data["masks"].shape[1:])
    target_on = masks_data["target_on"]
    del masks_data

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    for ax, symmetry in zip(axes, (6, 5)):
        print(f"=== Symmetry {symmetry} ===")
        canvas = build_canvas(
            symmetry,
            NUM_TILES,
            mask_hw,
            target_on,
            TRANSLATION,
            seed=SEED,
        )
        nn = nn_distances(canvas)
        for c in (0, 1):
            vals = nn[canvas["colors"] == c]
            print(f"  color {c} ({COLOR_NAMES[symmetry][c]}): "
                  f"n={len(vals)}  mean={vals.mean():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")
        plot_histograms(canvas, nn, ax)

    fig.tight_layout()
    path = OUT_DIR / "nearest_neighbour_dist.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
