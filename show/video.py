"""SVG-frame normalization and H.264 MP4 encoding."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .svg import (
    as_numpy,
    save_polygons,
    save_tiles_svg,
    shared_viewbox,
    xya_shared_viewbox,
)
from .trajectory import TrajectoryKind, _resample_trajectory
from .types import VideoOptions, ViewBox


_ROOT = re.compile(r"<svg\b[^>]*>")
_VIEWBOX = re.compile(
    r'\bviewBox="([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+'
    r'([0-9.eE+-]+)\s+([0-9.eE+-]+)"'
)


def _opacity_array(opacities) -> np.ndarray:
    if isinstance(opacities, (list, tuple)):
        return np.asarray([as_numpy(item) for item in opacities])
    return as_numpy(opacities)


def read_viewbox(path: Path | str) -> ViewBox:
    """Read the root viewBox from an SVG file."""
    text = Path(path).read_text(encoding="utf-8")
    root = _ROOT.search(text)
    if root is None:
        raise ValueError(f"Missing SVG root element: {path}")
    match = _VIEWBOX.search(root.group(0))
    if match is None:
        raise ValueError(f"Missing or malformed SVG viewBox: {path}")
    return ViewBox(*(float(value) for value in match.groups()))


def normalize_svg_canvas(
    path: Path | str,
    viewbox: ViewBox | tuple[float, float, float, float],
    *,
    background: str = "white",
    display_height: int = 1080,
    output_path: Path | str | None = None,
) -> tuple[int, int]:
    """Rewrite one SVG onto a fixed canvas and background."""
    source = Path(path)
    destination = source if output_path is None else Path(output_path)
    box = viewbox if isinstance(viewbox, ViewBox) else ViewBox(*map(float, viewbox))
    text = source.read_text(encoding="utf-8")
    root_match = _ROOT.search(text)
    if root_match is None:
        raise ValueError(f"Missing SVG root element: {source}")
    width, height = box.display_size(display_height, even=True)
    root = root_match.group(0)
    if re.search(r'\bwidth="[^"]*"', root):
        root = re.sub(r'\bwidth="[^"]*"', f'width="{width}"', root)
    else:
        root = root[:-1] + f' width="{width}">'
    if re.search(r'\bheight="[^"]*"', root):
        root = re.sub(r'\bheight="[^"]*"', f'height="{height}"', root)
    else:
        root = root[:-1] + f' height="{height}">'
    replacement = " ".join(f"{value:.8f}" for value in box.as_tuple())
    if _VIEWBOX.search(root):
        root = _VIEWBOX.sub(f'viewBox="{replacement}"', root)
    else:
        root = root[:-1] + f' viewBox="{replacement}">'
    text = text[: root_match.start()] + root + text[root_match.end() :]
    xmin, ymin, world_width, world_height = box.as_tuple()
    background_rect = (
        f'<rect class="background" x="{xmin:.8f}" y="{ymin:.8f}" '
        f'width="{world_width:.8f}" height="{world_height:.8f}" '
        f'fill="{background}"/>'
    )
    text, replacements = re.subn(
        r'<rect\b[^>]*/>',
        background_rect,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise ValueError(f"Missing SVG background rectangle: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return width, height


def _raster_safe_svg(source: Path, destination: Path) -> None:
    """Convert non-scaling stroke declarations to world-coordinate widths."""
    text = source.read_text(encoding="utf-8")
    box = read_viewbox(source)
    root_match = _ROOT.search(text)
    if root_match is None:
        raise ValueError(f"Missing SVG root element: {source}")
    height_match = re.search(r'\bheight="([0-9.]+)"', root_match.group(0))
    if height_match is None:
        raise ValueError(f"Missing SVG display height: {source}")
    world_units_per_pixel = box.height / float(height_match.group(1))

    def replacement(match: re.Match[str]) -> str:
        width = float(match.group(1)) * world_units_per_pixel
        return f"stroke-width: {width:.8f};{match.group(2)}"

    text = re.sub(
        r"stroke-width:\s*([0-9.eE+-]+);([^}]*)"
        r"vector-effect:\s*non-scaling-stroke;",
        replacement,
        text,
    )
    destination.write_text(text, encoding="utf-8")


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            "ffmpeg is required to create MP4 files; install it and ensure it is on PATH"
        )
    return executable


def rasterize_svg_frame(
    svg_path: Path | str,
    png_path: Path | str,
    *,
    background: str = "white",
    ffmpeg: str | None = None,
) -> Path:
    """Rasterize one SVG frame with stroke behavior preserved."""
    executable = ffmpeg or _ffmpeg()
    source = Path(svg_path)
    destination = Path(png_path)
    if not source.is_file():
        raise FileNotFoundError(f"SVG frame does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_svg = destination.with_suffix(".svg")
    _raster_safe_svg(source, safe_svg)
    try:
        subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-threads",
                "1",
                "-y",
                "-i",
                str(safe_svg),
                "-vf",
                (
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2:"
                    f"(ow-iw)/2:(oh-ih)/2:color={background},format=rgb24"
                ),
                "-frames:v",
                "1",
                str(destination),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ffmpeg failed to rasterize {source} with exit code {error.returncode}"
        ) from error
    finally:
        safe_svg.unlink(missing_ok=True)
    return destination


def _rasterize_svg_sequence(
    svg_paths: Sequence[Path],
    output_directory: Path,
    *,
    background: str,
    ffmpeg: str,
) -> list[Path]:
    """Rasterize a numbered SVG sequence in one ffmpeg process."""
    if not svg_paths:
        return []
    output_pattern = output_directory / "unique_%05d.png"
    try:
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-threads",
                "1",
                "-y",
                "-framerate",
                "1",
                "-i",
                str(output_directory / "raster_%05d.svg"),
                "-vf",
                (
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2:"
                    f"(ow-iw)/2:(oh-ih)/2:color={background},format=rgb24"
                ),
                "-vsync",
                "0",
                "-start_number",
                "0",
                str(output_pattern),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "ffmpeg failed to batch-rasterize "
            f"{len(svg_paths)} SVG frames with exit code {error.returncode}"
        ) from error
    outputs = [
        output_directory / f"unique_{index:05d}.png"
        for index in range(len(svg_paths))
    ]
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(
            "ffmpeg batch rasterization did not produce the expected frame: "
            f"{missing[0]}"
        )
    return outputs


def _link_or_copy(source: Path, destination: Path) -> None:
    """Materialize one numbered frame, preferring a zero-copy hard link."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def save_mp4(
    svg_paths: Sequence[Path | str],
    output_path: Path | str,
    *,
    options: VideoOptions | None = None,
    viewbox: ViewBox | tuple[float, float, float, float] | None = None,
) -> Path:
    """Normalize an ordered SVG sequence and encode an H.264 MP4."""
    if not svg_paths:
        raise ValueError("At least one SVG frame is required")
    settings = options or VideoOptions()
    frames = [Path(path) for path in svg_paths]
    missing = [path for path in frames if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SVG frame does not exist: {missing[0]}")
    box = read_viewbox(frames[0]) if viewbox is None else (
        viewbox if isinstance(viewbox, ViewBox) else ViewBox(*map(float, viewbox))
    )
    executable = _ffmpeg()
    output = Path(output_path)
    if output.exists() and not settings.overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="penrose-spur-video-") as temporary:
        directory = Path(temporary)
        unique_by_digest: dict[bytes, int] = {}
        unique_svgs: list[Path] = []
        frame_to_unique: list[int] = []
        for index, frame in enumerate(frames):
            digest = hashlib.sha256(frame.read_bytes()).digest()
            unique_index = unique_by_digest.get(digest)
            if unique_index is not None:
                frame_to_unique.append(unique_index)
                continue
            unique_index = len(unique_svgs)
            unique_by_digest[digest] = unique_index
            frame_to_unique.append(unique_index)
            normalized = directory / f"normalized_{unique_index:05d}.svg"
            normalize_svg_canvas(
                frame,
                box,
                background=settings.background,
                display_height=settings.display_height,
                output_path=normalized,
            )
            raster_safe = directory / f"raster_{unique_index:05d}.svg"
            _raster_safe_svg(normalized, raster_safe)
            unique_svgs.append(raster_safe)
        unique_pngs = _rasterize_svg_sequence(
            unique_svgs,
            directory,
            background=settings.background,
            ffmpeg=executable,
        )
        for index, unique_index in enumerate(frame_to_unique):
            _link_or_copy(
                unique_pngs[unique_index],
                directory / f"frame_{index:05d}.png",
            )
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}-",
            suffix=".mp4",
            dir=output.parent,
            delete=False,
        ) as staging_file:
            temporary_output = Path(staging_file.name)
        try:
            subprocess.run(
                [
                    executable,
                    "-v",
                    "error",
                    "-y",
                    "-framerate",
                    str(settings.fps),
                    "-i",
                    str(directory / "frame_%05d.png"),
                    "-c:v",
                    settings.codec,
                    "-crf",
                    str(settings.crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary_output),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as error:
            temporary_output.unlink(missing_ok=True)
            raise RuntimeError(
                f"ffmpeg failed to encode MP4 with exit code {error.returncode}"
            ) from error
        temporary_output.replace(output)
    return output


def save_trajectory_mp4(
    trajectory,
    colors,
    output_path: Path | str,
    *,
    symmetry: int | None = None,
    side: float | None = None,
    kind: TrajectoryKind = "xya",
    target_duration: float | None = None,
    angle_scale: float = 1.0,
    scheme=None,
    opacities=None,
    alpha: float = 0.7,
    show_arcs: bool | None = None,
    options: VideoOptions | None = None,
) -> Path:
    """Render an XYA or polygon trajectory through SVG and encode it as MP4."""
    states = list(trajectory)
    if not states:
        raise ValueError("Trajectory must contain at least one state")
    if kind not in ("xya", "polygons"):
        raise ValueError(f"kind must be 'xya' or 'polygons', got {kind!r}")
    settings = options or VideoOptions()
    render_opacities = opacities
    if target_duration is not None:
        render_states, source_indices, fractions = _resample_trajectory(
            states,
            kind=kind,
            target_duration=target_duration,
            fps=settings.fps,
            angle_scale=angle_scale,
        )
        if opacities is not None:
            opacity_array = _opacity_array(opacities)
            if opacity_array.ndim == 2 and opacity_array.shape[0] == len(states):
                if len(states) == 1:
                    render_opacities = np.repeat(
                        opacity_array[:1], len(render_states), axis=0
                    )
                else:
                    weights = fractions[:, None]
                    render_opacities = (
                        opacity_array[source_indices] * (1.0 - weights)
                        + opacity_array[source_indices + 1] * weights
                    )
    else:
        render_states = states
    render_states = list(render_states)

    if kind == "xya":
        if symmetry not in (5, 6):
            raise ValueError("symmetry must be 5 or 6 for an XYA trajectory")
        if side is None:
            raise ValueError("side is required for an XYA trajectory")
        box = xya_shared_viewbox(
            render_states,
            colors,
            symmetry,
            side,
            angle_scale=angle_scale,
        )
        render_symmetry = symmetry
    else:
        box = shared_viewbox(render_states)
        first_state = as_numpy(render_states[0])
        vertex_count = first_state.shape[-2] if first_state.ndim >= 2 else 0
        render_symmetry = 5 if vertex_count == 4 else 6 if vertex_count == 6 else None
        if render_symmetry is None:
            raise ValueError("Polygon states must contain 4- or 6-vertex tiles")
        if symmetry is not None and symmetry != render_symmetry:
            raise ValueError("symmetry does not match polygon vertex count")

    with tempfile.TemporaryDirectory(prefix="penrose-spur-svg-frames-") as temporary:
        directory = Path(temporary)
        frame_paths = []
        per_frame_opacity = False
        if render_opacities is not None:
            opacity_array = _opacity_array(render_opacities)
            per_frame_opacity = (
                opacity_array.ndim == 2
                and opacity_array.shape[0] == len(render_states)
            )
        for index, state in enumerate(render_states):
            frame_opacity = (
                render_opacities[index]
                if per_frame_opacity
                else render_opacities
            )
            path = directory / f"frame_{index:04d}.svg"
            arcs = (render_symmetry == 5) if show_arcs is None else show_arcs
            if kind == "xya":
                save_tiles_svg(
                    path,
                    state,
                    colors,
                    symmetry=render_symmetry,
                    side=side,
                    angle_scale=angle_scale,
                    scheme=scheme,
                    opacities=frame_opacity,
                    alpha=alpha,
                    show_arcs=arcs,
                    viewbox=box,
                )
            else:
                save_polygons(
                    path,
                    state,
                    colors,
                    scheme=scheme,
                    opacities=frame_opacity,
                    alpha=alpha,
                    show_arcs=arcs,
                    viewbox=box,
                )
            frame_paths.append(path)
        return save_mp4(frame_paths, output_path, options=settings, viewbox=box)
