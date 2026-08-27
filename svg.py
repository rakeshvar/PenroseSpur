"""Compatibility facade for the rendering package.

New code should import from ``show``. Existing callers can continue importing
``save_polygons`` and palette constants from this root module.
"""

from show import (  # noqa: F401
    DISPLAY_HEIGHT,
    HEX_PALETTE,
    PALETTES,
    PENROSE_PALETTE,
    Palette,
    render_polygons_svg,
    save_polygons,
)

__all__ = [
    "DISPLAY_HEIGHT",
    "HEX_PALETTE",
    "PALETTES",
    "PENROSE_PALETTE",
    "Palette",
    "render_polygons_svg",
    "save_polygons",
]
