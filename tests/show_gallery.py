"""Render the complete built-in scheme gallery to tests/output/show_gallery.

Usage:
    ~/.aivenv/bin/python tests/show_gallery.py
    ~/.aivenv/bin/python tests/show_gallery.py -o tests/output/my_gallery
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from show import pen_schemes, hex_schemes, save_tiles_svg  # noqa: E402


def hex_sample() -> tuple[np.ndarray, np.ndarray]:
    centers = []
    colors = []
    for row in range(-3, 4):
        for column in range(-3, 4):
            centers.append(
                (1.5 * column, math.sqrt(3) * (row + 0.5 * (column & 1)), 0.0)
            )
            colors.append((row - column) % 4)
    return np.asarray(centers), np.asarray(colors)


def pen_sample() -> tuple[np.ndarray, np.ndarray]:
    values = []
    colors = []
    for ring in (1.0, 2.3, 3.6):
        for index in range(10):
            angle = index * math.pi / 5
            values.append((ring * math.cos(angle), ring * math.sin(angle), angle))
            colors.append(index % 2)
    return np.asarray(values), np.asarray(colors)


def render_gallery(output: Path) -> tuple[Path, ...]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    hex_xya, hex_colors = hex_sample()
    for name in hex_schemes:
        paths.append(
            save_tiles_svg(
                output / f"hex_{name}.svg",
                hex_xya,
                hex_colors,
                symmetry=6,
                side=0.9,
                scheme=name,
            )
        )
    pen_xya, pen_colors = pen_sample()
    for name in pen_schemes:
        paths.append(
            save_tiles_svg(
                output / f"pen_{name}.svg",
                pen_xya,
                pen_colors,
                symmetry=5,
                side=1.0,
                scheme=name,
                show_arcs=True,
            )
        )
    cards = "\n".join(
        (
            "<figure>"
            f'<img src="{path.name}" alt="{path.stem}">'
            f"<figcaption>{path.stem}</figcaption>"
            "</figure>"
        )
        for path in paths
    )
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>PenroseSpur schemes</title>"
        "<style>body{font-family:sans-serif;background:#eee}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));"
        "gap:1rem}figure{margin:0;padding:1rem;background:white}"
        "img{width:100%;height:260px;object-fit:contain}"
        "figcaption{text-align:center;font-weight:bold}</style>"
        f"<h1>PenroseSpur scheme gallery</h1><main>{cards}</main>",
        encoding="utf-8",
    )
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render every PenroseSpur scheme")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "tests" / "output" / "show_gallery",
    )
    args = parser.parse_args()
    paths = render_gallery(args.output)
    print(f"Rendered {len(paths)} schemes to {args.output}")


if __name__ == "__main__":
    main()
