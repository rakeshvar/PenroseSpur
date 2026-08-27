"""Built-in and user-registered tiling color schemes."""

from __future__ import annotations

from .colors import validate_scheme
from .types import ColorScheme, Palette


hex_schemes: dict[str, ColorScheme] = {
    "spur": ColorScheme(
        "#D8D388", "#4ee055", "#ddb4a2", "#38b2c2"
    ),
    "honeycomb": ColorScheme(
        "#F6C453", "#2D2926", "#FFF1B8", "#A66321", background="#FFF9E8"
    ),
    "ocean": ColorScheme(
        "#0077B6", "#48CAE4", "#FF7F51", "#F4D35E", background="#F2FBFD"
    ),
    "forest": ColorScheme(
        "#1B4332", "#74C69D", "#DDA15E", "#FEFAE0", background="#F6F8EE"
    ),
    "twilight": ColorScheme(
        "#3A0CA3", "#F72585", "#4CC9F0", "#FFD166", background="#110B2E",
        stroke="#F7F2FF",
    ),
    "okabe_ito": ColorScheme(
        "#0072B2", "#E69F00", "#56B4E9", "#F0E442", background="#FFFFFF"
    ),
}

pen_schemes: dict[str, ColorScheme] = {
    "spur": ColorScheme(
        "#ffcccc", "#ff99aa", "#ddb4a2", "#38b2c2",
        a_arc="#80cc80", c_arc="#808040",
    ),
    "classic": ColorScheme(
        "#0035F3", "#0080F0", "#8EC5FF", "#1D2F6F",
        a_arc="#FF8000", c_arc="#F0C030", background="#05070D",
        stroke="#FFFFFF",
    ),
    "sun": ColorScheme(
        "#FFD60A", "#FF3B30", "#FFF3B0", "#9D0208",
        a_arc="#7F5539", c_arc="#2A9D8F", background="#FFF8E7",
    ),
    "star": ColorScheme(
        "#99FF33", "#009933", "#E8FFD0", "#005522",
        a_arc="#FF5E25", c_arc=None, background="#071A0D",
        stroke="#E9FFE8",
    ),
    "amethyst": ColorScheme(
        "#5A189A", "#C77DFF", "#72EFDD", "#FFD166",
        a_arc="#00B4D8", c_arc="#FFB703", background="#14001F",
        stroke="#F8EFFF",
    ),
    "ink": ColorScheme(
        "#F4F1DE", "#2B2D42", "#E07A5F", "#81B29A",
        a_arc="#D62828", c_arc="#1D4ED8", background="#FAF9F6",
        stroke="#111111",
    ),
}


def _registry(symmetry: int) -> dict[str, ColorScheme]:
    if symmetry == 5:
        return pen_schemes
    if symmetry == 6:
        return hex_schemes
    raise ValueError(f"symmetry must be 5 or 6, got {symmetry}")


def scheme_names(symmetry: int) -> tuple[str, ...]:
    """Return sorted available scheme names for one symmetry."""
    return tuple(sorted(_registry(symmetry)))


def get_scheme(
    symmetry: int,
    scheme: str | int | ColorScheme | Palette | None = None,
) -> ColorScheme:
    """Resolve a scheme name, old symmetry number, explicit scheme, or default."""
    if isinstance(scheme, ColorScheme):
        return validate_scheme(scheme)
    if isinstance(scheme, Palette):
        return validate_scheme(scheme.as_scheme())
    if isinstance(scheme, int):
        if scheme not in (5, 6):
            raise ValueError(f"Unknown palette symmetry: {scheme}")
        symmetry = scheme
        scheme = None
    name = "spur" if scheme is None else str(scheme).strip().lower().replace("-", "_")
    registry = _registry(symmetry)
    try:
        return registry[name]
    except KeyError as error:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown symmetry-{symmetry} scheme {scheme!r}; available: {available}"
        ) from error


def register_scheme(
    symmetry: int,
    name: str,
    scheme: ColorScheme,
    *,
    replace: bool = False,
) -> ColorScheme:
    """Register a custom in-process scheme."""
    normalized = name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Scheme name cannot be empty")
    registry = _registry(symmetry)
    if normalized in registry and not replace:
        raise ValueError(f"Scheme {normalized!r} is already registered")
    registry[normalized] = validate_scheme(scheme)
    return scheme


HEX_PALETTE = hex_schemes["spur"]
PENROSE_PALETTE = pen_schemes["spur"]
PALETTES = {5: PENROSE_PALETTE, 6: HEX_PALETTE}
