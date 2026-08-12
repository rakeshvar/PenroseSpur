"""
Sanity check: display the normalized MPEG7 masks as a grid.

Renders every mask: 20 collages (one per in-class instance), each a 7x10
grid with quarter-size entries per class, saved to output/scaled_masks/.
Cells are separated by single gray grid lines, no labels. Also prints
size/ON-mass statistics.

Usage: python tests/show_masks.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masks import build_masks

OUT_DIR = Path(__file__).resolve().parent / "output" / "scaled_masks"

GRAY = 128


def resize_masks(masks, size):
    """Downscale float masks for compact visualization."""
    resized = []
    for mask in masks:
        image = Image.fromarray(mask.astype(np.float32), mode="F")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        resized.append(np.asarray(image, dtype=np.float32))
    return np.stack(resized)


def collage(cells, rows, cols, path):
    """Tile float masks with one-pixel gray grid lines."""
    _, H, W = cells.shape
    out = np.full((rows * H + rows - 1, cols * W + cols - 1), GRAY, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            cell = np.clip(1.0 - cells[r * cols + c], 0.0, 1.0)
            cell = (cell * 255).astype(np.uint8)  # white background, black mask
            out[r * (H + 1):r * (H + 1) + H, c * (W + 1):c * (W + 1) + W] = cell
    Image.fromarray(out).save(path)
    print(f"Saved {path}")


def main():
    data = build_masks()
    masks = data["masks"].numpy()
    masks = resize_masks(masks, masks.shape[-1] // 4)
    labels = data["labels"].numpy()
    inclass_ids = data["inclass_ids"].numpy()
    names = data["class_names"]
    ons = data["on_counts"].numpy()

    K, H, W = masks.shape
    n_classes = len(names)
    n_instances = K // n_classes
    print(f"masks: ({K}, {H}, {W})   classes: {n_classes}   target_on: {data['target_on']}")
    print(f"ON counts: min={ons.min()} mean={ons.mean():.0f} max={ons.max()}"
          f"  (spread {100 * (ons.max() - ons.min()) / ons.mean():.1f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # All masks: one collage per in-class instance, one entry per class
    for k in range(n_instances):
        idx = []
        for c in range(n_classes):
            members = np.where(labels == c)[0]
            idx.append(int(members[np.argsort(inclass_ids[members])][k]))
        collage(masks[idx], 7, 10, OUT_DIR / f"masks_by_class_{k + 1:02d}.png")


if __name__ == "__main__":
    main()
