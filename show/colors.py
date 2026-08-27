"""Color validation and compositing helpers."""

from __future__ import annotations

import re

from .types import ColorScheme


_HEX = re.compile(r"^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$")
_SAFE_NAMED = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def validate_color(value: str | None, *, allow_none: bool = False) -> str | None:
    """Validate a CSS color accepted by the dependency-free SVG renderer."""
    if value is None:
        if allow_none:
            return None
        raise ValueError("Color cannot be None")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected a non-empty color string, got {value!r}")
    if not (_HEX.fullmatch(value) or _SAFE_NAMED.fullmatch(value)):
        raise ValueError(
            f"Unsupported color {value!r}; use #RRGGBB, #RRGGBBAA, or a CSS name"
        )
    return value


def validate_scheme(scheme: ColorScheme) -> ColorScheme:
    """Return a scheme after validating all of its colors."""
    for color in scheme.tile_colors():
        validate_color(color, allow_none=color is None)
    validate_color(scheme.a_arc, allow_none=True)
    validate_color(scheme.c_arc, allow_none=True)
    validate_color(scheme.background)
    validate_color(scheme.stroke)
    return scheme


def rgba(color: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    """Convert a hex color to 8-bit RGBA."""
    validate_color(color)
    if not _HEX.fullmatch(color):
        raise ValueError("RGBA conversion requires a hexadecimal color")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be in [0, 1]")
    raw = color[1:]
    alpha = int(raw[6:8], 16) / 255 if len(raw) == 8 else 1.0
    return (
        int(raw[0:2], 16),
        int(raw[2:4], 16),
        int(raw[4:6], 16),
        round(255 * alpha * opacity),
    )


def composite(color: str, background: str = "#ffffff", opacity: float = 1.0) -> str:
    """Composite one hexadecimal color over another and return #RRGGBB."""
    foreground = rgba(color, opacity)
    backdrop = rgba(background)
    alpha = foreground[3] / 255
    channels = [
        round(foreground[index] * alpha + backdrop[index] * (1 - alpha))
        for index in range(3)
    ]
    return "#" + "".join(f"{value:02x}" for value in channels)
