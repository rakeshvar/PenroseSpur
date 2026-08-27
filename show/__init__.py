"""Public PenroseSpur rendering facade.

The root PenroseSpur modules own tiling mathematics and sampling.  This package
owns only presentation: schemes, SVG scenes, view boxes, and MP4 encoding.
"""

from .schemes import (
    HEX_PALETTE,
    PALETTES,
    PENROSE_PALETTE,
    get_scheme,
    hex_schemes,
    pen_schemes,
    register_scheme,
    scheme_names,
)
from .scenes import (
    render_comparison_svg,
    render_flow_comparison_svg,
    save_comparison_svg,
    save_evaluation_set,
    save_interpolation_svg,
    save_scene_svg,
)
from .svg import (
    DISPLAY_HEIGHT,
    as_numpy,
    build_svg_document,
    polygon_viewbox,
    polygons_from_xya,
    render_polygons_svg,
    render_tiles_svg,
    save_polygons,
    save_tiles_svg,
    shared_viewbox,
    xya_shared_viewbox,
)
from .trajectory import (
    TrajectoryKind,
    resample_trajectory,
    trajectory_movements,
    wrapped_angle_delta,
)
from .types import (
    AssignmentStyle,
    ColorScheme,
    ComparisonLayer,
    LayerStyle,
    Palette,
    VideoOptions,
    ViewBox,
)
from .video import (
    normalize_svg_canvas,
    rasterize_svg_frame,
    read_viewbox,
    save_mp4,
    save_trajectory_mp4,
)


__all__ = [
    "AssignmentStyle",
    "ColorScheme",
    "ComparisonLayer",
    "DISPLAY_HEIGHT",
    "HEX_PALETTE",
    "LayerStyle",
    "PALETTES",
    "PENROSE_PALETTE",
    "Palette",
    "TrajectoryKind",
    "VideoOptions",
    "ViewBox",
    "as_numpy",
    "build_svg_document",
    "get_scheme",
    "hex_schemes",
    "normalize_svg_canvas",
    "pen_schemes",
    "polygon_viewbox",
    "polygons_from_xya",
    "rasterize_svg_frame",
    "read_viewbox",
    "register_scheme",
    "resample_trajectory",
    "render_comparison_svg",
    "render_flow_comparison_svg",
    "render_polygons_svg",
    "render_tiles_svg",
    "save_comparison_svg",
    "save_evaluation_set",
    "save_interpolation_svg",
    "save_mp4",
    "save_polygons",
    "save_scene_svg",
    "save_tiles_svg",
    "save_trajectory_mp4",
    "scheme_names",
    "shared_viewbox",
    "trajectory_movements",
    "wrapped_angle_delta",
    "xya_shared_viewbox",
]
