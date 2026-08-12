"""
Sanity check: generate on-the-fly samples and render the chosen polygons over
the source mask, for both symmetries. Also reports the empirical std of the
tile x, y coordinates (should be about 1).

Usage: python tests/show_samples.py [num_tiles]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampler import SpurSampler

OUT_DIR = Path(__file__).resolve().parent / "output"
FACE = {0: "#4878cf", 1: "#d65f5f"}


def render(sampler, batch, path, num_show):
    """Draw chosen polygons (in mask pixel coords) over the source mask."""
    H, W, s = sampler.H, sampler.W, sampler.scaling
    verts = batch["vertices"].cpu().numpy()
    colors = batch["colors"].cpu().numpy()
    inness = batch["inness"].cpu().numpy()
    mask_idx = batch["mask_idx"].cpu().numpy()

    rows = 4
    cols = num_show // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2 * H / W))
    for b, ax in enumerate(axes.ravel()):
        ax.imshow(sampler.masks[mask_idx[b]].cpu().numpy(), cmap="gray_r",
                  alpha=0.25, interpolation="nearest")
        for n in range(verts.shape[1]):
            # canvas -> pixel: row = x/s + (H-1)/2, col = y/s + (W-1)/2
            r = verts[b, n, :, 0] / s + (H - 1) / 2
            c = verts[b, n, :, 1] / s + (W - 1) / 2
            ax.add_patch(MplPolygon(list(zip(c, r)), closed=True,
                                    facecolor=FACE[int(colors[b, n])],
                                    edgecolor="black", linewidth=0.3, alpha=0.85))
        ax.set_title(f"min inness {inness[b].min():.2f}", fontsize=7)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main(num_tiles=96):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for symmetry in (6, 5):
        print(f"\n=== Symmetry {symmetry}, num_tiles {num_tiles} ===")
        sampler = SpurSampler(symmetry, num_tiles, 2.0, seed=0)

        num_show = 20
        batch = sampler.sample_batch(num_show, return_vertices=True)
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
