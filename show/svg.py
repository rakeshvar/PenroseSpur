"""Dependency-light SVG rendering for Penrose and hexagonal tilings."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from convert import vertices as tile_vertices

from .schemes import get_scheme
from .types import ColorScheme, Palette, ViewBox


DISPLAY_HEIGHT = 1080


def as_numpy(values) -> np.ndarray:
    """Convert NumPy-compatible values, including CPU/CUDA Torch tensors."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def svg_coordinates(values) -> np.ndarray:
    """Map canvas ``(x=row, y=column)`` coordinates to SVG ``(x, y)``."""
    coordinates = as_numpy(values)
    if coordinates.shape[-1:] != (2,):
        raise ValueError(
            f"Expected coordinates with final dimension 2, got {coordinates.shape}"
        )
    return coordinates[..., ::-1]


def _validate_polygons(polygons) -> tuple[np.ndarray, int]:
    values = as_numpy(polygons)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"Expected polygons with shape (N,V,2), got {values.shape}")
    if len(values) == 0:
        raise ValueError("At least one polygon is required")
    try:
        symmetry = {4: 5, 6: 6}[values.shape[1]]
    except KeyError as error:
        raise ValueError("Polygons must have 4 or 6 vertices") from error
    if not np.isfinite(values).all():
        raise ValueError("Polygon coordinates must be finite")
    return values.astype(float, copy=False), symmetry


def _validate_colors(colors, count: int) -> np.ndarray:
    values = as_numpy(colors)
    if values.shape != (count,):
        raise ValueError(f"Expected {count} colors, got {values.shape}")
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Color indices must be finite integers")
    values = values.astype(np.int64, copy=False)
    if not ((0 <= values) & (values <= 3)).all():
        raise ValueError("Color indices must be in [0, 3]")
    return values


def _validate_opacities(opacities, count: int) -> np.ndarray:
    if opacities is None:
        return np.ones(count, dtype=float)
    values = as_numpy(opacities).astype(float)
    if values.shape != (count,):
        raise ValueError(f"Expected {count} opacities, got {values.shape}")
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError("Opacities must be finite values in [0, 1]")
    return values


def polygon_viewbox(
    polygons,
    *,
    padding_fraction: float = 0.05,
    padding_minimum: float = 0.0,
) -> ViewBox:
    """Compute an SVG viewBox from mask-aligned canvas polygons."""
    values, _ = _validate_polygons(polygons)
    if padding_fraction < 0 or padding_minimum < 0:
        raise ValueError("ViewBox padding must be non-negative")
    display_values = svg_coordinates(values)
    minimum = display_values.min(axis=(0, 1))
    maximum = display_values.max(axis=(0, 1))
    span = np.maximum(maximum - minimum, np.finfo(float).eps)
    padding = np.maximum(padding_fraction * span, padding_minimum)
    return ViewBox(
        float(minimum[0] - padding[0]),
        float(minimum[1] - padding[1]),
        float(span[0] + 2 * padding[0]),
        float(span[1] + 2 * padding[1]),
    )


def shared_viewbox(
    polygon_sets: Sequence,
    *,
    padding_fraction: float = 0.04,
    padding_minimum: float = 0.0,
) -> ViewBox:
    """Compute one stable viewBox spanning several polygon collections."""
    if not polygon_sets:
        raise ValueError("At least one polygon set is required")
    validated = [_validate_polygons(item)[0] for item in polygon_sets]
    return polygon_viewbox(
        np.concatenate(validated, axis=0),
        padding_fraction=padding_fraction,
        padding_minimum=padding_minimum,
    )


def polygons_from_xya(
    xya,
    colors,
    symmetry: int,
    side: float,
    *,
    angle_scale: float = 1.0,
    batch_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw or Spur-scaled XYA values to polygons."""
    if symmetry not in (5, 6):
        raise ValueError(f"symmetry must be 5 or 6, got {symmetry}")
    if not np.isfinite(side) or side <= 0:
        raise ValueError("side must be a positive finite value")
    if not np.isfinite(angle_scale) or angle_scale == 0:
        raise ValueError("angle_scale must be a non-zero finite value")
    values = as_numpy(xya)
    color_values = as_numpy(colors)
    if values.ndim == 3:
        if not 0 <= batch_index < values.shape[0]:
            raise IndexError(f"batch_index {batch_index} is out of range")
        values = values[batch_index]
        if color_values.ndim == 2:
            color_values = color_values[batch_index]
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Expected XYA with shape (N,3+), got {values.shape}")
    if not np.isfinite(values[:, :3]).all():
        raise ValueError("XYA coordinates must be finite")
    color_values = _validate_colors(color_values, len(values))
    polygons = tile_vertices(
        symmetry,
        values[:, :2],
        values[:, 2] / angle_scale,
        color_values,
        float(side),
    )
    return polygons, color_values


def xya_shared_viewbox(
    geometries: Sequence,
    colors,
    symmetry: int,
    side: float,
    *,
    angle_scale: float = 1.0,
    padding_fraction: float = 0.04,
    padding_minimum: float | None = None,
) -> ViewBox:
    """Compute a shared viewBox directly from a sequence of XYA states."""
    minimum = side * 0.5 if padding_minimum is None else padding_minimum
    polygons = [
        polygons_from_xya(item, colors, symmetry, side, angle_scale=angle_scale)[0]
        for item in geometries
    ]
    return shared_viewbox(
        polygons,
        padding_fraction=padding_fraction,
        padding_minimum=minimum,
    )


def _svg_arc(center: np.ndarray, first: np.ndarray, second: np.ndarray, precision: int) -> str:
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
    start_svg, end_svg = svg_coordinates(np.stack((start, end)))
    return (
        f"M {start_svg[0]:.{precision}f} {start_svg[1]:.{precision}f} "
        f"A {radius:.{precision}f} {radius:.{precision}f} 0 0 0 "
        f"{end_svg[0]:.{precision}f} {end_svg[1]:.{precision}f}"
    )


def build_svg_document(
    viewbox: ViewBox,
    body: Iterable[str],
    *,
    styles: Iterable[str] = (),
    display_height: int = DISPLAY_HEIGHT,
    background: str | None = None,
    header: Iterable[str] = (),
) -> str:
    """Build a standalone SVG document from escaped caller-created elements."""
    width, height = viewbox.display_size(display_height)
    xmin, ymin, world_width, world_height = viewbox.as_tuple()
    elements: list[str] = []
    if styles:
        elements.extend(("<style>", *styles, "</style>"))
    if background is not None:
        elements.append(
            f'<rect class="background" x="{xmin:.5f}" y="{ymin:.5f}" '
            f'width="{world_width:.5f}" height="{world_height:.5f}" '
            f'fill="{escape(background)}"/>'
        )
    elements.extend(header)
    elements.extend(body)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        'preserveAspectRatio="xMidYMid meet" '
        f'width="{width}" height="{height}" '
        f'viewBox="{xmin:.5f} {ymin:.5f} {world_width:.5f} {world_height:.5f}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )


def render_polygons_svg(
    polygons,
    colors,
    *,
    scheme: str | int | ColorScheme | Palette | None = None,
    palette: str | int | ColorScheme | Palette | None = None,
    background: str | None = None,
    stroke: str | None = None,
    stroke_width: float | None = None,
    show_arcs: bool = False,
    radius: float | None = None,
    opacities=None,
    show_polygons: bool = True,
    alpha: float = 0.7,
    viewbox: ViewBox | tuple[float, float, float, float] | None = None,
    display_height: int = DISPLAY_HEIGHT,
    precision: int = 3,
    padding: float = 0.05,
    mark_duplicates: bool = False,
    batch_index: int = 0,
) -> str:
    """Return a standalone SVG for polygon vertices and categorical colors."""
    raw_polygons = as_numpy(polygons)
    raw_colors = as_numpy(colors)
    if raw_polygons.ndim == 4:
        if not 0 <= batch_index < raw_polygons.shape[0]:
            raise IndexError(f"batch_index {batch_index} is out of range")
        polygons = raw_polygons[batch_index]
        if raw_colors.ndim == 2:
            colors = raw_colors[batch_index]
    values, symmetry = _validate_polygons(polygons)
    color_values = _validate_colors(colors, len(values))
    opacity_values = _validate_opacities(opacities, len(values))
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if precision < 0:
        raise ValueError("precision must be non-negative")
    if stroke_width is not None and (
        not np.isfinite(stroke_width) or stroke_width <= 0
    ):
        raise ValueError("stroke_width must be a positive finite display-pixel width")
    selected = palette if palette is not None else scheme
    resolved = get_scheme(symmetry, selected)
    background = resolved.background if background is None else background
    stroke = resolved.stroke if stroke is None else stroke
    if viewbox is None:
        box = polygon_viewbox(values, padding_fraction=padding)
    elif isinstance(viewbox, ViewBox):
        box = viewbox
    else:
        box = ViewBox(*map(float, viewbox))
    side = float(np.linalg.norm(values[0, 1] - values[0, 0]))
    resolved_stroke_width = (
        side / 100 * display_height / box.height
        if stroke_width is None
        else float(stroke_width)
    )
    styles = [
        (
            f".tile {{ stroke: {escape(stroke)}; "
            f"stroke-width: {resolved_stroke_width:.5f}; "
            "stroke-linejoin: round; vector-effect: non-scaling-stroke; }"
        )
    ]
    for index, color in enumerate(resolved.tile_colors()):
        if color is not None:
            styles.append(f".color{index} {{ fill: {escape(color)}; }}")
    draw_arcs = show_arcs and symmetry == 5
    if draw_arcs:
        styles.append(
            f".arc {{ fill: none; stroke-width: {3 * resolved_stroke_width:.5f}; "
            f"stroke-opacity: {0.8 * alpha:.4f}; "
            "vector-effect: non-scaling-stroke; }"
        )
        if resolved.a_arc is not None:
            styles.append(f".aarc {{ stroke: {escape(resolved.a_arc)}; }}")
        if resolved.c_arc is not None:
            styles.append(f".carc {{ stroke: {escape(resolved.c_arc)}; }}")
    body: list[str] = []
    for polygon, color, opacity in zip(values, color_values, opacity_values):
        points = " ".join(
            f"{x:.{precision}f},{y:.{precision}f}"
            for x, y in svg_coordinates(polygon)
        )
        if show_polygons:
            body.append(
                f'<polygon class="tile color{color}" '
                f'opacity="{opacity * alpha:.4f}" points="{points}"/>'
            )
        if draw_arcs:
            A, B, C, D = polygon
            if resolved.a_arc is not None:
                body.append(
                    f'<path class="arc aarc" d="{_svg_arc(A, B, D, precision)}"/>'
                )
            if resolved.c_arc is not None:
                body.append(
                    f'<path class="arc carc" d="{_svg_arc(C, B, D, precision)}"/>'
                )
    if radius is not None:
        if radius <= 0:
            raise ValueError("radius must be positive")
        body.append(
            f'<circle class="radius" cx="0" cy="0" r="{radius:.{precision}f}" '
            f'fill="none" stroke="{escape(stroke)}" '
            f'stroke-width="{3 * resolved_stroke_width:.5f}" '
            'vector-effect="non-scaling-stroke"/>'
        )
    if mark_duplicates:
        centers = values.mean(axis=1)
        _, inverse, counts = np.unique(
            np.round(centers, decimals=precision + 1),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        for center in centers[counts[inverse] > 1]:
            svg_center = svg_coordinates(center)
            body.append(
                f'<circle class="duplicate" cx="{svg_center[0]:.{precision}f}" '
                f'cy="{svg_center[1]:.{precision}f}" r="{side / 3:.{precision}f}" '
                'fill="#0000c0"/>'
            )
    return build_svg_document(
        box,
        body,
        styles=styles,
        display_height=display_height,
        background=background,
    )


def save_polygons(
    path,
    polygons,
    colors,
    palette=None,
    background=None,
    stroke=None,
    show_arcs=False,
    radius=None,
    opacities=None,
    show_polygons=True,
    alpha=0.7,
    *,
    stroke_width=None,
    scheme=None,
    viewbox=None,
    display_height=DISPLAY_HEIGHT,
    precision=3,
    padding=0.05,
    mark_duplicates=False,
    batch_index=0,
) -> Path:
    """Save polygons, preserving the historical PenroseSpur call signature."""
    svg = render_polygons_svg(
        polygons,
        colors,
        palette=palette,
        scheme=scheme,
        background=background,
        stroke=stroke,
        stroke_width=stroke_width,
        show_arcs=show_arcs,
        radius=radius,
        opacities=opacities,
        show_polygons=show_polygons,
        alpha=alpha,
        viewbox=viewbox,
        display_height=display_height,
        precision=precision,
        padding=padding,
        mark_duplicates=mark_duplicates,
        batch_index=batch_index,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def render_tiles_svg(
    xya,
    colors,
    *,
    symmetry: int,
    side: float,
    angle_scale: float = 1.0,
    batch_index: int = 0,
    **options,
) -> str:
    """Render raw-radian or scaled XYA values."""
    polygons, color_values = polygons_from_xya(
        xya,
        colors,
        symmetry,
        side,
        angle_scale=angle_scale,
        batch_index=batch_index,
    )
    return render_polygons_svg(polygons, color_values, **options)


def save_tiles_svg(
    path,
    xya,
    colors,
    *,
    symmetry: int,
    side: float,
    angle_scale: float = 1.0,
    batch_index: int = 0,
    **options,
) -> Path:
    """Save raw-radian or scaled XYA values as SVG."""
    svg = render_tiles_svg(
        xya,
        colors,
        symmetry=symmetry,
        side=side,
        angle_scale=angle_scale,
        batch_index=batch_index,
        **options,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output
