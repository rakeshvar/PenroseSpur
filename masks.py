"""
Build normalized MPEG7 masks in memory from the cropped-GIF archive.

The masks are density-scaled with bilinear interpolation, kept as float32,
proportionally fit inside the target canvas when needed, and center-padded.
No tensor cache is written to disk.
"""

import argparse
import io
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "tests" / "output"
DEFAULT_ARCHIVE = PROJECT_DIR / "mpeg7_silhouttes_cropped.zip"
DEFAULT_HEIGHT = 448
DEFAULT_WIDTH = 448
EXPECTED_MASKS = 1400


def load_raw_masks(archive=DEFAULT_ARCHIVE):
    """Load archived GIFs as tightly cropped binary float arrays."""
    archive = Path(archive)
    with zipfile.ZipFile(archive) as gifs:
        names = sorted(name for name in gifs.namelist() if name.lower().endswith(".gif"))

        masks, class_names, inclass_ids = [], [], []
        for name in names:
            with gifs.open(name) as file:
                with Image.open(io.BytesIO(file.read())) as image:
                    mask = (np.asarray(image.convert("L")) > 0).astype(np.float32)
            masks.append(mask)
            stem = Path(name).stem
            class_name, inclass_id = stem.rsplit("-", 1)
            class_names.append(class_name)
            inclass_ids.append(int(inclass_id))

    return masks, class_names, inclass_ids


def resize_float_mask(mask, height, width):
    """Resize a float mask without thresholding it back to binary."""
    image = Image.fromarray(mask.astype(np.float32), mode="F")
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def density_scale(mask, target_on):
    """Scale mask area toward target_on while retaining soft edge values."""
    factor = math.sqrt(target_on / float(mask.sum()))
    height = max(1, round(mask.shape[0] * factor))
    width = max(1, round(mask.shape[1] * factor))
    return resize_float_mask(mask, height, width)


def fit_to_canvas(mask, target_height, target_width):
    """Proportionally shrink a mask only when it exceeds the canvas."""
    height, width = mask.shape
    if height <= target_height and width <= target_width:
        return mask

    factor = min(target_height / height, target_width / width)
    new_height = min(target_height, max(1, round(height * factor)))
    new_width = min(target_width, max(1, round(width * factor)))
    return resize_float_mask(mask, new_height, new_width)


def center_pad(mask, target_height, target_width):
    """Center a float mask in a fixed-size zero canvas."""
    height, width = mask.shape
    out = np.zeros((target_height, target_width), dtype=np.float32)
    top = (target_height - height) // 2
    left = (target_width - width) // 2
    out[top : top + height, left : left + width] = mask
    return out


def plot_stats(scaled_heights, scaled_widths, class_names, inclass_ids, target_height, target_width, output_name, stage):
    """Print size statistics and save the interactive PDF/CDF HTML plot."""
    print("--------------------------------")
    for values, label in ((scaled_heights, "Scaled Height"), (scaled_widths, "Scaled Width")):
        print(f"{label}:")
        for func, name in ((np.argmin, "Min"), (np.argmax, "Max")):
            idx = func(values)
            print(f"  {name:<6}: {values[idx]:6d} {class_names[idx]}-{inclass_ids[idx]}")

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        specs=[[{"secondary_y": True}]] * 2,
                        subplot_titles=(f"Heights {stage}", f"Widths {stage}"))

    percentiles = (90, 95, 96, 97, 98, 99, 100)
    for row, (values, label, target) in enumerate(((scaled_heights, "Heights", target_height), (scaled_widths, "Widths", target_width))):
        fig.add_histogram(x=values, xbins=dict(start=0, size=1), name=f"{label} PDF",
                          marker_color="steelblue", opacity=0.7, row=row, col=1, secondary_y=False)
        sorted_values = np.sort(values)
        fig.add_scatter(x=sorted_values, y=np.arange(1, len(sorted_values) + 1) / len(sorted_values),
                        name=f"{label} CDF", line=dict(color="darkred"), row=row, col=1, secondary_y=True)
        fig.add_vline(x=target, line_color="black", line_width=2, row=row, col=1)
        for pct in percentiles:
            fig.add_vline(x=np.percentile(values, pct), line_dash="dash", line_color="gray", line_width=1, row=row, col=1)
        fig.update_yaxes(title_text="PDF", color="steelblue", row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="CDF", color="darkred", range=[0, 1], row=row, col=1, secondary_y=True)

    fig.update_xaxes(title_text="pixels", row=2, col=1)
    fig.update_layout(height=1000, hovermode="x unified", bargap=0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / output_name
    fig.write_html(output, include_plotlyjs="cdn")
    print(f"Saved {output}")


def build_masks(
    archive=DEFAULT_ARCHIVE,
    target_height=DEFAULT_HEIGHT,
    target_width=DEFAULT_WIDTH,
    show=lambda _: None,
    save_plot=False,
):
    """Build the fixed-size float mask tensor and its metadata."""
    show(f"Loading GIFs from {archive} ...")
    raw_masks, class_names, inclass_ids = load_raw_masks(archive)
    raw_on_counts = np.asarray([mask.sum() for mask in raw_masks])
    target_on = float(np.median(raw_on_counts))
    show(f"Raw ON counts: min={raw_on_counts.min():.0f}, median={target_on:.0f}, max={raw_on_counts.max():.0f}")

    density_scaled = [density_scale(mask, target_on) for mask in raw_masks]
    scaled_heights = np.asarray([mask.shape[0] for mask in density_scaled])
    scaled_widths = np.asarray([mask.shape[1] for mask in density_scaled])
    
    if save_plot:
        plot_stats(scaled_heights, scaled_widths, class_names, inclass_ids, target_height, target_width, 
                   "masks_stats.html", f"before {target_height}x{target_width} restriction")

    fitted = [fit_to_canvas(mask, target_height, target_width) for mask in density_scaled]
    masks = np.empty((len(fitted), target_height, target_width), dtype=np.float32)
    for index, mask in enumerate(fitted):
        masks[index] = center_pad(mask, target_height, target_width)

    unique_names = sorted(set(class_names))
    name_to_id = {name: index for index, name in enumerate(unique_names)}
    labels = np.asarray([name_to_id[name] for name in class_names], dtype=np.int64)
    on_counts = np.asarray([mask.sum() for mask in fitted], dtype=np.float32)

    data = {
        "masks": torch.from_numpy(masks),
        "labels": torch.from_numpy(labels),
        "inclass_ids": torch.tensor(inclass_ids),
        "class_names": unique_names,
        "target_on": target_on,
        "on_counts": torch.from_numpy(on_counts),
    }
    print(f"Masks: {tuple(data['masks'].shape)}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("-H", "--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("-W", "--width", type=int, default=DEFAULT_WIDTH)
    args = parser.parse_args()
    build_masks(
        args.archive,
        args.height,
        args.width,
        show=True,
    )


if __name__ == "__main__":
    main()
