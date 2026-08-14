"""
Render one mask on the mother canvas with every tile's inness shown.

Asks for an MPEG7 class (or takes it as the first argument, by name or id) and
picks a random sample of that class from range(20) unless a sample id is given
as the second argument. The canvas gets one random rotation + translation draw
(as in real sample generation); every mother tile is filled with an alpha that
is an affine function of its inness, and tiles with inness > 0 show the value
at the polygon center. One image per symmetry, dpi 300.

Usage: python tests/show_inness.py [class_name_or_id] [sample_id 0..19]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampler import SpurSampler

OUT_DIR = Path(__file__).resolve().parent / "output"

NUM_TILES = 96
TRANSLATION = 2.0
SEED = 0

FACE = {0: "#4878cf", 1: "#d65f5f"}
ALPHA_LO, ALPHA_HI = 0.05, 0.95


def pick_class(names, arg=None):
    if arg is None:
        cols = 5
        for i in range(0, len(names), cols):
            print("   ".join(f"{j:2d} {names[j]:<16s}" for j in range(i, min(i + cols, len(names)))))
        arg = input("Class name or id [enter = random]: ").strip()
    if arg == "":
        return int(np.random.randint(len(names)))
    try:
        return int(arg)
    except ValueError:
        assert arg in names, f"Unknown class {arg!r}"
        return names.index(arg)


def render(sampler, points, inness, mask, title, path):
    """Draw all mother tiles with opacity and text determined by inness."""
    H, W, s = sampler.H, sampler.W, sampler.scaling
    centers = points[:, 0, :].cpu().numpy()              # (M, 2)
    verts = points[:, 1:, :].cpu().numpy()               # (M, V, 2)
    inness = inness.cpu().numpy()
    vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 10))

    # Mask underlay in canvas coordinates (plot horizontal = y, vertical = x)
    y_lo, y_hi = (-0.5 - (W - 1) / 2) * s, (W - 0.5 - (W - 1) / 2) * s
    x_lo, x_hi = (-0.5 - (H - 1) / 2) * s, (H - 0.5 - (H - 1) / 2) * s
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., 3] = np.clip(mask, 0.0, 1.0) * 0.30
    ax.imshow(rgba, extent=(y_lo, y_hi, x_hi, x_lo), origin="upper",
              interpolation="nearest", zorder=1)

    facecolors = np.array([matplotlib.colors.to_rgba(FACE[int(c)])
                           for c in sampler.colors.cpu().numpy()])
    facecolors[:, 3] = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * inness / vmax
    ax.add_collection(PolyCollection(verts[:, :, ::-1], facecolors=facecolors,
                                     edgecolors=(0, 0, 0, 0.15), linewidths=0.2,
                                     zorder=2))

    for m in np.flatnonzero(inness > 0):
        ax.text(centers[m, 1], centers[m, 0], f"{inness[m]:.2f}", fontsize=2.2,
                ha="center", va="center", zorder=3)

    lim = float(np.abs(centers).max()) * 1.04
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)
    ax.set_aspect("equal")
    ax.set_xlabel("canvas y")
    ax.set_ylabel("canvas x")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sampler = SpurSampler(6, NUM_TILES, TRANSLATION, seed=SEED)
    names = sampler.class_names
    labels = sampler.labels.cpu().numpy()

    class_id = pick_class(names, sys.argv[1] if len(sys.argv) > 1 else None)
    in_class = np.flatnonzero(labels == class_id)
    sample_id = int(sys.argv[2]) if len(sys.argv) > 2 else int(np.random.randint(len(in_class)))
    mask_idx = int(in_class[sample_id])
    print(f"Class {class_id} ({names[class_id]}), sample {sample_id} -> mask {mask_idx}")

    for symmetry in (6, 5):
        if symmetry == 5:
            del sampler
            sampler = SpurSampler(5, NUM_TILES, TRANSLATION, seed=SEED)
        points, theta, inness, _ = sampler.transform_and_inness(
            torch.tensor([mask_idx], device=sampler.device)
        )
        points, theta, inness = points[0], theta[0], inness[0]

        title = (f"{names[class_id]}-{sample_id}   symmetry {symmetry}   "
                 f"M={sampler.M}   inness 0..1   "
                 f"rot {np.degrees(float(theta)):+.0f}\u00b0")
        render(
            sampler,
            points,
            inness,
            sampler.masks[mask_idx].cpu().numpy(),
            title,
            OUT_DIR / f"inness_s{symmetry}_{names[class_id]}-{sample_id}.png",
        )


if __name__ == "__main__":
    main()
