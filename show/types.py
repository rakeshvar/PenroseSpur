"""Rendering-only value types for PenroseSpur show features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, NamedTuple


@dataclass(frozen=True)
class ViewBox:
    """An SVG world-coordinate view box."""

    xmin: float
    ymin: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ViewBox width and height must be positive")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.xmin, self.ymin, self.width, self.height

    def display_size(self, height: int = 1080, *, even: bool = False) -> tuple[int, int]:
        if height <= 0:
            raise ValueError("Display height must be positive")
        width = max(1, round(height * self.width / self.height))
        if even:
            width += width % 2
            height += height % 2
        return width, height

    def expand_header(self, fraction: float = 0.12) -> "ViewBox":
        if fraction < 0:
            raise ValueError("Header fraction must be non-negative")
        extra = self.height * fraction
        return ViewBox(self.xmin, self.ymin - extra, self.width, self.height + extra)


@dataclass(frozen=True)
class ColorScheme:
    """Four tile colors plus optional Penrose matching-rule arc colors."""

    color0: str
    color1: str
    aux0: str | None = None
    aux1: str | None = None
    a_arc: str | None = None
    c_arc: str | None = None
    background: str = "white"
    stroke: str = "#222222"

    @property
    def color2(self) -> str | None:
        return self.aux0

    @property
    def color3(self) -> str | None:
        return self.aux1

    @property
    def aarccolor(self) -> str | None:
        return self.a_arc

    @property
    def carccolor(self) -> str | None:
        return self.c_arc

    def tile_colors(self) -> tuple[str | None, ...]:
        return self.color0, self.color1, self.aux0, self.aux1

    def legacy_values(self) -> tuple[str | None, ...]:
        return (*self.tile_colors(), self.a_arc, self.c_arc)

    def __iter__(self) -> Iterator[str | None]:
        return iter(self.legacy_values())

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index):
        return self.legacy_values()[index]


class Palette(NamedTuple):
    """Historical six-field palette accepted by the root ``svg`` module."""

    color0: str
    color1: str
    color2: str | None = None
    color3: str | None = None
    aarccolor: str | None = None
    carccolor: str | None = None

    def as_scheme(self) -> ColorScheme:
        return ColorScheme(
            self.color0,
            self.color1,
            self.color2,
            self.color3,
            self.aarccolor,
            self.carccolor,
        )


@dataclass(frozen=True)
class LayerStyle:
    """Visual treatment for one tiling layer."""

    mode: Literal["fill", "outline", "fill_outline"] = "fill"
    opacity: float = 0.7
    stroke: str | None = None
    stroke_width: float | None = None
    color_role: Literal["main", "aux"] = "main"
    css_class: str = ""


@dataclass(frozen=True)
class ComparisonLayer:
    """One named polygon or XYA layer in a comparison scene."""

    values: Any
    colors: Any
    label: str = ""
    style: LayerStyle = field(default_factory=LayerStyle)
    kind: Literal["xya", "polygons"] = "xya"
    angle_scale: float = 1.0
    opacities: Any | None = None


@dataclass(frozen=True)
class AssignmentStyle:
    """Line or arrow style for one correspondence method."""

    label: str = ""
    color: str = "#777777"
    opacity: float = 0.55
    width: float | None = None
    dashed: bool = False
    arrows: bool = False


@dataclass(frozen=True)
class VideoOptions:
    """H.264 MP4 encoding options."""

    fps: int = 30
    background: str = "black"
    crf: int = 18
    codec: str = "libx264"
    overwrite: bool = True
    display_height: int = 1080

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be in [0, 51]")
        if self.display_height <= 0:
            raise ValueError("display_height must be positive")
