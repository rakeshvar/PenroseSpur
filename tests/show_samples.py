"""
Sanity check: generate exactly one on-the-fly sample from each of the first 10
preferred cool classes and render the chosen polygons over the source mask, for
both symmetries. Also reports the empirical std of the tile x, y coordinates
(should be about 1).

Usage: python tests/show_samples.py [num_tiles]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.transforms import Affine2D
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampler import SpurSampler

OUT_DIR = Path(__file__).resolve().parent / "output"
FACE = {0: "#4878cf", 1: "#d65f5f"}
NUM_COOL_CLASSES = 10


def render(sampler, batch, path, num_show):
    """Draw chosen polygons (in mask pixel coords) over the source mask."""
    H, W, s = sampler.H, sampler.W, sampler.scaling
    verts = batch["vertices"].cpu().numpy()
    colors = batch["colors"].cpu().numpy()
    inness = batch["inness"].cpu().numpy()
    mask_idx = batch["mask_idx"].cpu().numpy()
    labels = batch["labels"].cpu().numpy()
    rotations = batch["rotation_mask"].cpu().numpy()

    rows = min(2, num_show)
    cols = (num_show + rows - 1) // rows
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.2, rows * 3.2 * H / W),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for b, ax in enumerate(flat_axes[:num_show]):
        # Canvas coordinates are (row, col), so their positive rotation is a
        # negative image-coordinate rotation around the mask center.
        mask_transform = (
            Affine2D().rotate_around((W - 1) / 2, (H - 1) / 2, -rotations[b])
            + ax.transData
        )
        ax.imshow(sampler.masks[mask_idx[b]].cpu().numpy(), cmap="gray_r",
                  alpha=0.25, interpolation="nearest", transform=mask_transform)
        for n in range(verts.shape[1]):
            # canvas -> pixel: row = x/s + (H-1)/2, col = y/s + (W-1)/2
            r = verts[b, n, :, 0] / s + (H - 1) / 2
            c = verts[b, n, :, 1] / s + (W - 1) / 2
            ax.add_patch(MplPolygon(list(zip(c, r)), closed=True,
                                    facecolor=FACE[int(colors[b, n])],
                                    edgecolor="black", linewidth=0.3, alpha=0.5))
        class_name = sampler.class_names[int(labels[b])]
        ax.set_title(f"{class_name} · min inness {inness[b].min():.2f}", fontsize=7)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
    for ax in flat_axes[num_show:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main(num_tiles=96):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for symmetry in (6, 5):
        print(
            f"\n=== Symmetry {symmetry}, num_tiles {num_tiles}, "
            f"one sample from each of {NUM_COOL_CLASSES} cool classes ==="
        )
        sampler = SpurSampler(
            symmetry,
            num_tiles,
            2.0,
            seed=0,
            num_cool_classes=NUM_COOL_CLASSES,
        )

        mask_idx = torch.stack([
            torch.nonzero(sampler.labels == class_id, as_tuple=True)[0][0]
            for class_id in sampler.cool_class_ids
        ])
        num_show = NUM_COOL_CLASSES
        assert len(mask_idx) == num_show
        batch = sampler.sample_batch(
            num_show,
            mask_idx=mask_idx,
            return_vertices=True,
        )
        render(sampler, batch, OUT_DIR / f"samples_s{symmetry}_t{num_tiles}.png", num_show)

        big = sampler.sample_batch(256)
        xy = big["xya"][..., :2]
        print(f"x std {xy[..., 0].std():.3f}   y std {xy[..., 1].std():.3f}   (target ~1)")
        print(f"inness: min {big['inness'].min().item():.2f}"
              f"  mean {big['inness'].mean():.2f}"
              f"  max {sampler.V1}")
        frac1 = (big["colors"] == 1).float().mean()
        print(f"color-1 fraction: {frac1:.3f}"
              + ("  (thin, golden ratio target 0.382)" if symmetry == 5 else ""))
        del sampler, batch, big


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 96)
