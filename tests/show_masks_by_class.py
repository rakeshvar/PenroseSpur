"""Render every normalized mask, grouped into one 4x5 grid per class.

Usage: python tests/render_masks_by_class.py

Outputs ``tests/output/masks_by_class/<class_id>_<class_name>.png``. Each
image contains the class's 20 masks, ordered by their in-class ID and styled
like the grids in ``tests/output/scaled_masks``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masks import DEFAULT_ARCHIVE, DEFAULT_HEIGHT, DEFAULT_WIDTH, build_masks


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "masks_by_class"
ROWS = 4
COLS = 5
MASKS_PER_CLASS = ROWS * COLS
SCALE_DIVISOR = 4
GRAY = 128


def render_grid(masks, output_path):
    """Save 20 float masks as a four-row, five-column grayscale grid."""
    if len(masks) != MASKS_PER_CLASS:
        raise ValueError(f"Expected {MASKS_PER_CLASS} masks, got {len(masks)}")

    height = masks.shape[1] // SCALE_DIVISOR
    width = masks.shape[2] // SCALE_DIVISOR
    grid = np.full(
        (ROWS * height + ROWS - 1, COLS * width + COLS - 1),
        GRAY,
        dtype=np.uint8,
    )
    for index, mask in enumerate(masks):
        row, col = divmod(index, COLS)
        image = Image.fromarray(mask.astype(np.float32), mode="F")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        image = np.asarray(image, dtype=np.float32)
        image = (np.clip(1.0 - image, 0.0, 1.0) * 255).astype(np.uint8)
        top = row * (height + 1)
        left = col * (width + 1)
        grid[
            top : top + height,
            left : left + width,
        ] = image

    Image.fromarray(grid).save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-H", "--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("-W", "--width", type=int, default=DEFAULT_WIDTH)
    args = parser.parse_args()

    data = build_masks(
        archive=args.archive,
        target_height=args.height,
        target_width=args.width,
    )
    masks = data["masks"].numpy()
    labels = data["labels"].numpy()
    inclass_ids = data["inclass_ids"].numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for class_id, class_name in enumerate(data["class_names"]):
        indices = np.flatnonzero(labels == class_id)
        indices = indices[np.argsort(inclass_ids[indices])]
        output_path = args.output_dir / f"{class_id}_{class_name}.png"
        render_grid(masks[indices], output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
