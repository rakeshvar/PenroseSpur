"""Plot returned-tile inness distributions as the return count increases.

For each symmetry, write one Plotly HTML file containing a subplot for every
``extra`` in ``range(0, 13) * 8``. Each subplot has a normalized histogram for
each of the 70 MPEG7 classes plus a thick overall histogram.

Usage: python tests/plot_inness_by_extra.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampler import SpurSampler


SYMMETRIES = (5, 6)
NUM_TILES = 96
EXTRAS = tuple(range(0, 13 * 8, 8))
TRANSLATION_CANVAS = 2.0
SEED = 0
NUM_MASKS = 1400
NUM_CLASSES = 70
BIN_WIDTH = 0.1
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "inness_by_extra"


def normalized_histogram(values, max_bin):
    """Return fractions in inness bins of width exactly 0.1."""
    edges = np.arange(max_bin + 2, dtype=np.float64) * BIN_WIDTH
    counts, _ = np.histogram(values.ravel(), bins=edges)
    return counts / counts.sum()


def returned_inness(sampler, mask_idx, generator):
    """Select returned-tile inness using the sampler's top-k rule."""
    _, _, inness = sampler.transform_and_inness(mask_idx, generator=generator)
    noisy = inness.float() + 1e-3 * torch.rand(
        inness.shape,
        device=sampler.device,
        generator=generator,
    )
    top = torch.topk(noisy, sampler.num_ret_tiles, dim=1).indices
    return torch.gather(inness, 1, top)


def add_histograms(fig, inness, labels, class_names, extra, max_inness, row, col):
    """Add the 70 class histograms and thick overall histogram to a subplot."""
    max_bin = round(max_inness / BIN_WIDTH)
    bin_values = np.arange(max_bin + 1) * BIN_WIDTH

    for class_id in range(NUM_CLASSES):
        class_values = inness[labels == class_id]
        fig.add_trace(
            go.Scattergl(
                x=bin_values,
                y=normalized_histogram(class_values, max_bin),
                mode="lines",
                name=class_names[class_id],
                legendgroup=f"class-{class_id}",
                showlegend=False,
                line={"color": "rgba(31, 119, 180, 0.22)", "width": 1},
                hovertemplate=(
                    f"{class_names[class_id]}<br>"
                    "inness=%{x:.1f}<br>fraction=%{y:.4f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scattergl(
            x=bin_values,
            y=normalized_histogram(inness, max_bin),
            mode="lines",
            name="overall",
            legendgroup="overall",
            showlegend=extra == EXTRAS[0],
            line={"color": "black", "width": 4},
            hovertemplate=(
                "overall<br>inness=%{x:.1f}<br>"
                "fraction=%{y:.4f}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def plot_symmetry(symmetry):
    """Sample one symmetry and write all extra-count plots to one HTML file."""
    cols = 2
    rows = (len(EXTRAS) + cols - 1) // cols
    subplot_titles = [
        f"extra={extra}, num_ret_tiles={NUM_TILES + extra}"
        for extra in EXTRAS
    ]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.035,
    )
    sampler = SpurSampler(
        symmetry=symmetry,
        num_tiles=NUM_TILES,
        translation_canvas=TRANSLATION_CANVAS,
        num_ret_tiles=NUM_TILES,
        seed=SEED,
    )
    if len(sampler) != NUM_MASKS or len(sampler.class_names) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_MASKS} masks in {NUM_CLASSES} classes, found "
            f"{len(sampler)} masks in {len(sampler.class_names)} classes"
        )
    mask_idx = torch.arange(NUM_MASKS, device=sampler.device)
    labels = sampler.labels[mask_idx].cpu().numpy()

    for plot_index, extra in enumerate(EXTRAS):
        num_ret_tiles = NUM_TILES + extra
        sampler.num_ret_tiles = num_ret_tiles
        print(
            f"[symmetry {symmetry}] Sampling masks 0:{NUM_MASKS} with "
            f"num_ret_tiles={num_ret_tiles} ...",
            flush=True,
        )
        generator = torch.Generator(device=sampler.device).manual_seed(SEED)
        with torch.no_grad():
            inness = returned_inness(sampler, mask_idx, generator)
        row, col = divmod(plot_index, cols)
        add_histograms(
            fig,
            inness.cpu().numpy(),
            labels,
            sampler.class_names,
            extra,
            1.0,
            row + 1,
            col + 1,
        )
        del inness

    fig.update_xaxes(title_text="inness", dtick=0.1)
    fig.update_yaxes(title_text="fraction of returned tiles")
    fig.update_layout(
        title=(
            f"Symmetry {symmetry}: inness by class "
            f"(num_tiles={NUM_TILES}, bin size={BIN_WIDTH})"
        ),
        height=350 * rows,
        width=1400,
        hovermode="closest",
        template="plotly_white",
    )
    output_path = OUTPUT_DIR / f"inness_symmetry_{symmetry}.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Saved {output_path}")
    del sampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for symmetry in SYMMETRIES:
        plot_symmetry(symmetry)


if __name__ == "__main__":
    main()
