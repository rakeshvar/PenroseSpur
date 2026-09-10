"""Composable multi-tiling comparison and assignment SVG scenes."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .schemes import get_scheme
from .svg import (
    DISPLAY_HEIGHT,
    as_numpy,
    build_svg_document,
    polygons_from_xya,
    shared_viewbox,
    svg_coordinates,
)
from .types import AssignmentStyle, ComparisonLayer, LayerStyle, ViewBox


def _layer_polygons(
    layer: ComparisonLayer,
    symmetry: int,
    side: float,
) -> tuple[np.ndarray, np.ndarray]:
    if layer.kind == "polygons":
        polygons = as_numpy(layer.values)
        colors = as_numpy(layer.colors).astype(np.int64)
        if polygons.ndim != 3 or polygons.shape[-1] != 2:
            raise ValueError(f"Expected polygon layer (N,V,2), got {polygons.shape}")
        if colors.shape != (len(polygons),):
            raise ValueError("Layer colors must match the polygon count")
        return polygons, colors
    return polygons_from_xya(
        layer.values,
        layer.colors,
        symmetry,
        side,
        angle_scale=layer.angle_scale,
    )


def _points(polygon: np.ndarray, precision: int = 5) -> str:
    return " ".join(
        f"{x:.{precision}f},{y:.{precision}f}"
        for x, y in svg_coordinates(polygon)
    )


def render_comparison_svg(
    layers: Sequence[ComparisonLayer],
    *,
    symmetry: int,
    side: float,
    scheme=None,
    viewbox: ViewBox | tuple[float, float, float, float] | None = None,
    assignments: Sequence[
        tuple[int, int, object | None, AssignmentStyle]
    ] = (),
    header: Sequence[str] = (),
    metrics: Mapping[str, object] | None = None,
    display_height: int = DISPLAY_HEIGHT,
    background: str | None = None,
    header_fraction: float = 0.12,
) -> str:
    """Render named fill/outline layers and optional correspondence methods."""
    if not layers:
        raise ValueError("A scene requires at least one layer")
    if symmetry not in (5, 6):
        raise ValueError("symmetry must be 5 or 6")
    rendered = [_layer_polygons(layer, symmetry, side) for layer in layers]
    if viewbox is None:
        box = shared_viewbox(
            [item[0] for item in rendered],
            padding_fraction=0.04,
            padding_minimum=side * 0.5,
        )
    elif isinstance(viewbox, ViewBox):
        box = viewbox
    else:
        box = ViewBox(*map(float, viewbox))
    if (header or metrics) and header_fraction:
        box = box.expand_header(header_fraction)
    resolved = get_scheme(symmetry, scheme)
    background = resolved.background if background is None else background
    line_width = max(side / 65.0, box.width / 5000.0)
    styles = [
        ".scene-layer { stroke-linejoin: round; }",
        ".scene-text { font-family: sans-serif; }",
    ]
    body: list[str] = []
    for layer_index, (layer, (polygons, colors)) in enumerate(zip(layers, rendered)):
        style = layer.style
        if not 0 <= style.opacity <= 1:
            raise ValueError("Layer opacity must be in [0, 1]")
        if style.color_role == "aux":
            mapped = np.where(colors < 2, colors + 2, colors)
        else:
            mapped = colors
        opacities = (
            np.ones(len(polygons), dtype=float)
            if layer.opacities is None
            else as_numpy(layer.opacities).astype(float)
        )
        if opacities.shape != (len(polygons),):
            raise ValueError("Layer opacities must match its polygon count")
        default_stroke = resolved.stroke if style.mode == "fill_outline" else "none"
        stroke = style.stroke if style.stroke is not None else default_stroke
        stroke_width = (
            style.stroke_width
            if style.stroke_width is not None
            else max(side / 70.0, box.width / 6000.0)
        )
        fill = "none" if style.mode == "outline" else None
        css_class = escape(style.css_class or f"layer-{layer_index}")
        for polygon, color, opacity in zip(polygons, mapped, opacities):
            scheme_color = resolved.tile_colors()[int(color)]
            if scheme_color is None:
                raise ValueError(f"Scheme has no color for index {color}")
            fill_value = fill or scheme_color
            outline = scheme_color if style.mode == "outline" else stroke
            body.append(
                f'<polygon class="scene-layer {css_class}" '
                f'fill="{escape(fill_value)}" stroke="{escape(outline)}" '
                f'stroke-width="{stroke_width:.5f}" '
                f'opacity="{style.opacity * opacity:.4f}" '
                f'points="{_points(polygon)}"/>'
            )
    for assignment_index, item in enumerate(assignments):
        source_index, target_index, permutation, style = item
        if not (0 <= source_index < len(rendered) and 0 <= target_index < len(rendered)):
            raise IndexError("Assignment layer index is out of range")
        source = rendered[source_index][0].mean(axis=1)
        target = rendered[target_index][0].mean(axis=1)
        if permutation is None:
            if len(source) != len(target):
                raise ValueError("Identity correspondence requires equal layer sizes")
            indices = np.arange(len(source))
        else:
            indices = as_numpy(permutation).astype(np.int64)
            if indices.shape != (len(source),):
                raise ValueError("Permutation must contain one target per source")
            if not ((0 <= indices) & (indices < len(target))).all():
                raise ValueError("Permutation contains an invalid target index")
        marker_id = f"arrow-{assignment_index}"
        if style.arrows:
            body.append(
                f'<defs><marker id="{marker_id}" markerWidth="8" markerHeight="8" '
                'refX="7" refY="3" orient="auto" markerUnits="strokeWidth">'
                f'<path d="M0,0 L0,6 L8,3 z" fill="{escape(style.color)}"/>'
                "</marker></defs>"
            )
        width = line_width if style.width is None else style.width
        dash = ' stroke-dasharray="6 4"' if style.dashed else ""
        marker = f' marker-end="url(#{marker_id})"' if style.arrows else ""
        method_class = f"assignment-{assignment_index}"
        for start, end in zip(source, target[indices]):
            start_svg, end_svg = svg_coordinates(np.stack((start, end)))
            body.append(
                f'<line class="correspondence {method_class}" '
                f'x1="{start_svg[0]:.5f}" y1="{start_svg[1]:.5f}" '
                f'x2="{end_svg[0]:.5f}" y2="{end_svg[1]:.5f}" '
                f'stroke="{escape(style.color)}" stroke-opacity="{style.opacity:.4f}" '
                f'stroke-width="{width:.5f}"{dash}{marker}/>'
            )
    text_lines = [*header]
    if metrics:
        text_lines.extend(f"{key}: {value}" for key, value in metrics.items())
    if text_lines:
        font_size = box.height * 0.028
        x = box.xmin + box.width * 0.02
        y = box.ymin + font_size * 1.25
        for index, text in enumerate(text_lines):
            body.append(
                f'<text class="scene-text" x="{x:.5f}" '
                f'y="{y + index * font_size * 1.25:.5f}" '
                f'font-size="{font_size:.5f}" fill="{escape(resolved.stroke)}">'
                f"{escape(str(text))}</text>"
            )
    legend_entries = [
        style for _, _, _, style in assignments if style.label
    ]
    if legend_entries:
        font_size = box.height * 0.024
        x = box.xmin + box.width * 0.76
        y = box.ymin + font_size * 1.25
        for index, style in enumerate(legend_entries):
            line_y = y + index * font_size * 1.5
            body.append(
                f'<line x1="{x:.5f}" y1="{line_y:.5f}" '
                f'x2="{x + box.width * 0.05:.5f}" y2="{line_y:.5f}" '
                f'stroke="{escape(style.color)}" stroke-width="{line_width:.5f}"/>'
            )
            body.append(
                f'<text class="scene-text" x="{x + box.width * 0.065:.5f}" '
                f'y="{line_y + font_size * 0.3:.5f}" font-size="{font_size:.5f}" '
                f'fill="{escape(resolved.stroke)}">{escape(style.label)}</text>'
            )
    return build_svg_document(
        box,
        body,
        styles=styles,
        display_height=display_height,
        background=background,
    )


def save_scene_svg(path, layers: Sequence[ComparisonLayer], **options) -> Path:
    """Save a generic comparison scene."""
    svg = render_comparison_svg(layers, **options)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def render_flow_comparison_svg(
    initial,
    produced,
    colors,
    symmetry: int,
    side: float,
    viewbox: ViewBox | tuple[float, float, float, float] | None = None,
    *,
    angle_scale: float | None = None,
    scheme=None,
    mode: str = "overlaid",
) -> str:
    """Render a predicted/target-style comparison compatible with Bream."""
    if mode not in {"initial", "noised", "overlaid", "produced"}:
        raise ValueError(f"Unknown comparison mode: {mode}")
    scale = (np.sqrt(3.0) / np.pi) if angle_scale is None else angle_scale
    if mode in {"initial", "noised"}:
        layers = [
            ComparisonLayer(
                initial,
                colors,
                "initial",
                LayerStyle("fill_outline", css_class="noised-filled"),
                angle_scale=scale,
            )
        ]
        assignments = ()
    elif mode == "produced":
        layers = [
            ComparisonLayer(
                produced,
                colors,
                "produced",
                LayerStyle("fill_outline", css_class="reconstructed"),
                angle_scale=scale,
            ),
        ]
        assignments = ()
    else:
        layers = [
            ComparisonLayer(
                produced,
                colors,
                "produced",
                LayerStyle("fill_outline", css_class="reconstructed"),
                angle_scale=scale,
            ),
            ComparisonLayer(
                initial,
                colors,
                "initial",
                LayerStyle("outline", color_role="aux", css_class="noised"),
                angle_scale=scale,
            ),
        ]
        assignments = ((1, 0, None, AssignmentStyle()),)
    return render_comparison_svg(
        layers,
        symmetry=symmetry,
        side=side,
        scheme=scheme,
        viewbox=viewbox,
        assignments=assignments,
    )


def save_comparison_svg(
    path,
    initial,
    produced,
    colors,
    symmetry: int,
    side: float,
    viewbox: ViewBox | tuple[float, float, float, float] | None = None,
    *,
    angle_scale: float | None = None,
    scheme=None,
    mode: str = "overlaid",
) -> Path:
    """Save a predicted/target-style comparison compatible with Bream."""
    svg = render_flow_comparison_svg(
        initial,
        produced,
        colors,
        symmetry,
        side,
        viewbox,
        angle_scale=angle_scale,
        scheme=scheme,
        mode=mode,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def save_evaluation_set(
    output,
    stem: str,
    initial,
    produced: Sequence,
    colors,
    symmetry: int,
    side: float,
    *,
    angle_scale: float | None = None,
    scheme=None,
) -> tuple[Path, ...]:
    """Save one initial and paired overlaid/produced SVG per result."""
    if not produced:
        raise ValueError("At least one produced state is required")
    scale = (np.sqrt(3.0) / np.pi) if angle_scale is None else angle_scale
    box = _shared_xya_viewbox(
        [initial, *produced], colors, symmetry, side, scale
    )
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [
        save_comparison_svg(
            directory / f"{stem}_noised.svg",
            initial,
            produced[0],
            colors,
            symmetry,
            side,
            box,
            angle_scale=scale,
            scheme=scheme,
            mode="noised",
        )
    ]
    for index, result in enumerate(produced, start=1):
        for mode in ("overlaid", "produced"):
            paths.append(
                save_comparison_svg(
                    directory / f"{stem}_{mode}_i{index}.svg",
                    initial,
                    result,
                    colors,
                    symmetry,
                    side,
                    box,
                    angle_scale=scale,
                    scheme=scheme,
                    mode=mode,
                )
            )
    return tuple(paths)


def _shared_xya_viewbox(states, colors, symmetry, side, angle_scale) -> ViewBox:
    polygons = [
        polygons_from_xya(
            state, colors, symmetry, side, angle_scale=angle_scale
        )[0]
        for state in states
    ]
    return shared_viewbox(
        polygons, padding_fraction=0.04, padding_minimum=side * 0.5
    )


def save_interpolation_svg(
    path,
    fixed_layer: ComparisonLayer,
    ghost_layers: Sequence[ComparisonLayer],
    *,
    symmetry: int,
    side: float,
    scheme=None,
) -> Path:
    """Save fixed geometry plus caller-styled interpolation ghost layers."""
    if not ghost_layers:
        raise ValueError("At least one ghost layer is required")
    return save_scene_svg(
        path,
        [fixed_layer, *ghost_layers],
        symmetry=symmetry,
        side=side,
        scheme=scheme,
    )
