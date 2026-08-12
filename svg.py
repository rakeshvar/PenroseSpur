"""Small SVG helpers for visualizing polygon collections."""

from html import escape
from pathlib import Path
from typing import NamedTuple

import numpy as np


class Palette(NamedTuple):
    color0: str
    color1: str
    color2: str | None = None
    color3: str | None = None
    aarccolor: str | None = None
    carccolor: str | None = None


HEX_PALETTE = Palette("#D8D388", "#4ee055", "#ddb4a2", "#38b2c2")
PENROSE_PALETTE = Palette(
    "#ffcccc",
    "#ff99aa",
    aarccolor="#80cc80",
    carccolor="#808040",
)
PALETTES = {
    5: PENROSE_PALETTE,
    6: HEX_PALETTE,
}
DISPLAY_HEIGHT = 1080


def _as_numpy(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _svg_arc(center, first, second):
    """Return the minor SVG arc between two edge midpoints around center."""
    start = (center + first) / 2
    end = (center + second) / 2
    radius = np.linalg.norm(first - center) / 2
    start_vector = start - center
    end_vector = end - center
    cross = (
        start_vector[0] * end_vector[1]
        - start_vector[1] * end_vector[0]
    )
    if cross < 0:
        start, end = end, start
    return (
        f"M {start[0]:.3f} {-start[1]:.3f} "
        f"A {radius:.3f} {radius:.3f} 0 0 0 "
        f"{end[0]:.3f} {-end[1]:.3f}"
    )


def save_polygons(
    path,
    polygons,
    colors,
    palette=None,
    background="white",
    stroke="#222",
    show_arcs=False,
    radius=None,
):
    """Save polygon vertices and categorical colors as a standalone SVG."""
    polygons = _as_numpy(polygons)
    colors = _as_numpy(colors)
    symmetry = {4: 5, 6: 6}[polygons.shape[1]]

    if palette is None:
        palette = PALETTES[symmetry]
    elif isinstance(palette, (int, np.integer)):
        palette = PALETTES[int(palette)]

    show_arcs = show_arcs and symmetry == 5
    minimum = polygons.min(axis=(0, 1))
    maximum = polygons.max(axis=(0, 1))
    span = np.maximum(maximum - minimum, np.finfo(float).eps)
    padding = 0.05 * span
    view_min_x = minimum[0] - padding[0]
    view_min_y = -maximum[1] - padding[1]
    view_width, view_height = span + 2 * padding
    display_width = round(DISPLAY_HEIGHT * view_width / view_height)
    side = np.linalg.norm(polygons[0, 1] - polygons[0, 0])
    stroke_width = side / 100 * DISPLAY_HEIGHT / view_height

    color_styles = "\n".join(
        f".color{i} {{ fill: {escape(color)}; }}"
        for i, color in enumerate(palette[:4])
        if color is not None
    )

    arc_styles = ""
    arc_elements = []

    if show_arcs:
        arc_stroke_width = stroke_width * 3
        arc_styles = (
            f".arc {{ fill: none; stroke-width: {arc_stroke_width:.4f}; "
            "stroke-opacity: 0.8; vector-effect: non-scaling-stroke; }"
            f"\n.aarc {{ stroke: {escape(palette.aarccolor)}; }}"
            f"\n.carc {{ stroke: {escape(palette.carccolor)}; }}"
        )

    elements = [
        "<style>",
        (
            f".tile {{ stroke: {escape(stroke)}; stroke-width: {stroke_width:.4f}; "
            "stroke-linejoin: round; vector-effect: non-scaling-stroke; }"
        ),
        color_styles,
        arc_styles,
        "</style>",
        f'<rect x="{view_min_x:.3f}" y="{view_min_y:.3f}" '
        f'width="{view_width:.3f}" height="{view_height:.3f}" '
        f'fill="{escape(background)}"/>',
    ]
    for vertices, color in zip(polygons, colors):
        x = vertices[:, 0]
        y = -vertices[:, 1]
        points = " ".join(f"{px:.2f},{py:.2f}" for px, py in zip(x, y))
        elements.append(
            f'<polygon class="tile color{int(color)}" points="{points}"/>'
        )
        if show_arcs:
            A, B, C, D = vertices
            for arc, arc_class in (
                (_svg_arc(A, B, D), "aarc"),
                (_svg_arc(C, B, D), "carc"),
            ):
                arc_elements.append(
                    f'<path class="arc {arc_class}" d="{arc}"/>'
                )

    if radius is not None:
        elements.append(f'<circle cx="0" cy="0" r="{radius}" fill="none" stroke="black" stroke-width="{3*stroke_width:.4f}" vector-effect="non-scaling-stroke"/>')

    svg = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        'preserveAspectRatio="xMidYMid meet" '
        f'width="{display_width}" height="{DISPLAY_HEIGHT}" '
        f'viewBox="{view_min_x:.3f} {view_min_y:.3f} '
        f'{view_width:.3f} {view_height:.3f}">\n'
        + "\n".join(elements)
        + "\n"
        + "\n".join(arc_elements)
        + "\n</svg>\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path
