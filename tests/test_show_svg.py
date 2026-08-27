import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from show import (  # noqa: E402
    AssignmentStyle,
    ComparisonLayer,
    LayerStyle,
    Palette,
    Palette,
    ViewBox,
    hex_schemes,
    pen_schemes,
    polygons_from_xya,
    read_viewbox,
    render_comparison_svg,
    render_polygons_svg,
    render_tiles_svg,
    save_polygons,
    save_interpolation_svg,
    save_tiles_svg,
    scheme_names,
    xya_shared_viewbox,
)


class SvgTest(unittest.TestCase):
    def setUp(self):
        self.hex_xya = np.array(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, math.pi / 6]], dtype=float
        )
        self.colors = np.array([0, 1], dtype=np.uint8)

    def test_catalog_has_six_schemes_per_symmetry(self):
        self.assertEqual(len(hex_schemes), 6)
        self.assertEqual(len(pen_schemes), 6)
        self.assertEqual(len(scheme_names(5)), 6)
        self.assertEqual(len(scheme_names(6)), 6)
        self.assertTrue(
            all(
                scheme.a_arc is not None
                for scheme in pen_schemes.values()
                if scheme.c_arc is not None
            )
        )

    def test_raw_and_spur_scaled_angles_produce_same_polygons(self):
        raw, _ = polygons_from_xya(self.hex_xya, self.colors, 6, 0.5)
        scale = math.sqrt(3) / math.pi
        scaled = self.hex_xya.copy()
        scaled[:, 2] *= scale
        recovered, _ = polygons_from_xya(
            scaled, self.colors, 6, 0.5, angle_scale=scale
        )
        np.testing.assert_allclose(raw, recovered)

    def test_svg_uses_mask_aligned_yx_coordinates_and_viewbox(self):
        polygons = np.array(
            [[[1.0, 10.0], [2.0, 10.0], [2.0, 30.0], [1.0, 30.0]]]
        )
        svg = render_polygons_svg(
            polygons,
            np.array([0], dtype=np.uint8),
            padding=0.0,
        )
        root = ET.fromstring(svg)
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        polygon = root.find(".//svg:polygon", namespace)
        self.assertIsNotNone(polygon)
        self.assertEqual(
            polygon.attrib["points"],
            "10.000,1.000 10.000,2.000 30.000,2.000 30.000,1.000",
        )
        np.testing.assert_allclose(
            [float(value) for value in root.attrib["viewBox"].split()],
            [10.0, 1.0, 20.0, 1.0],
        )

    def test_batched_xya_selects_requested_sample(self):
        values = np.stack((self.hex_xya, self.hex_xya + [3.0, 0.0, 0.0]))
        colors = np.stack((self.colors, self.colors))
        svg = render_tiles_svg(
            values,
            colors,
            symmetry=6,
            side=0.5,
            batch_index=1,
            scheme="ocean",
        )
        self.assertEqual(svg.count("<polygon"), 2)
        ET.fromstring(svg)

    def test_polygon_and_xya_routes_share_viewbox(self):
        polygons, colors = polygons_from_xya(self.hex_xya, self.colors, 6, 0.5)
        box = xya_shared_viewbox([self.hex_xya], self.colors, 6, 0.5)
        polygon_svg = render_polygons_svg(polygons, colors, viewbox=box)
        xya_svg = render_tiles_svg(
            self.hex_xya,
            self.colors,
            symmetry=6,
            side=0.5,
            viewbox=box,
        )
        root_a = ET.fromstring(polygon_svg)
        root_b = ET.fromstring(xya_svg)
        self.assertEqual(root_a.attrib["viewBox"], root_b.attrib["viewBox"])

    def test_penrose_arcs_and_duplicate_markers(self):
        xya = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
        polygons, colors = polygons_from_xya(xya, self.colors, 5, 1.0)
        svg = render_polygons_svg(
            polygons,
            colors,
            scheme="amethyst",
            show_arcs=True,
            mark_duplicates=True,
        )
        self.assertEqual(svg.count('class="arc aarc"'), 2)
        self.assertEqual(svg.count('class="arc carc"'), 2)
        self.assertEqual(svg.count('class="duplicate"'), 2)
        first_polygon = svg.index("<polygon")
        first_a_arc = svg.index('class="arc aarc"')
        first_c_arc = svg.index('class="arc carc"')
        second_polygon = svg.index("<polygon", first_polygon + 1)
        self.assertLess(first_polygon, first_a_arc)
        self.assertLess(first_a_arc, first_c_arc)
        self.assertLess(first_c_arc, second_polygon)
        expected_arc_start = (polygons[0, 0] + polygons[0, 1]) / 2
        self.assertIn(
            f"M {expected_arc_start[1]:.3f} {expected_arc_start[0]:.3f}",
            svg,
        )
        self.assertIn('cx="2.000" cy="1.000"', svg)

    def test_comparison_supports_three_layers_and_assignments(self):
        moved = self.hex_xya + [0.1, 0.2, 0.0]
        farther = self.hex_xya + [0.3, -0.1, 0.0]
        layers = [
            ComparisonLayer(
                self.hex_xya,
                self.colors,
                "target",
                LayerStyle("fill_outline"),
            ),
            ComparisonLayer(
                moved,
                self.colors,
                "current",
                LayerStyle("outline", color_role="aux"),
            ),
            ComparisonLayer(
                farther,
                self.colors,
                "predicted",
                LayerStyle("outline"),
            ),
        ]
        svg = render_comparison_svg(
            layers,
            symmetry=6,
            side=0.5,
            assignments=[
                (1, 0, None, AssignmentStyle("current", dashed=True)),
                (
                    1,
                    2,
                    None,
                    AssignmentStyle("prediction", "#cc0000", arrows=True),
                ),
            ],
            header=["t < 1 & prediction"],
            metrics={"cost": "0.25 < 1"},
        )
        root = ET.fromstring(svg)
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        self.assertEqual(len(root.findall(".//svg:polygon", namespace)), 6)
        self.assertEqual(
            len(
                root.findall(
                    ".//svg:line[@class='correspondence assignment-0']",
                    namespace,
                )
            ),
            2,
        )
        self.assertIn("stroke-dasharray", svg)
        self.assertIn("marker-end", svg)
        self.assertIn("&lt;", svg)
        self.assertIn("&amp;", svg)
        first_line = root.find(
            ".//svg:line[@class='correspondence assignment-0']",
            namespace,
        )
        self.assertIsNotNone(first_line)
        np.testing.assert_allclose(
            [
                float(first_line.attrib[name])
                for name in ("x1", "y1", "x2", "y2")
            ],
            [0.2, -0.9, 0.0, -1.0],
            atol=1e-5,
        )

    def test_save_creates_parent_and_viewbox_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "sample.svg"
            save_tiles_svg(
                path,
                self.hex_xya,
                self.colors,
                symmetry=6,
                side=0.5,
            )
            box = read_viewbox(path)
            self.assertIsInstance(box, ViewBox)
            self.assertTrue(path.is_file())

    def test_interpolation_helper_saves_fixed_and_ghost_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghosts.svg"
            save_interpolation_svg(
                path,
                ComparisonLayer(
                    self.hex_xya,
                    self.colors,
                    "fixed",
                    LayerStyle("fill", opacity=0.6),
                ),
                [
                    ComparisonLayer(
                        self.hex_xya + [0.2, 0.1, 0.0],
                        self.colors,
                        "ghost",
                        LayerStyle("outline", opacity=0.4, color_role="aux"),
                    )
                ],
                symmetry=6,
                side=0.5,
            )
            root = ET.parse(path).getroot()
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        self.assertEqual(len(root.findall(".//svg:polygon", namespace)), 4)

    def test_legacy_save_polygons_signature(self):
        polygons, colors = polygons_from_xya(self.hex_xya, self.colors, 6, 0.5)
        palette = Palette(
            "#111111",
            "#eeeeee",
            color2="#cc0000",
            color3="#00cc00",
            aarccolor="#0000cc",
            carccolor="#cccc00",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.svg"
            save_polygons(
                path,
                polygons,
                colors,
                palette,
                "black",
                "white",
                False,
                None,
                np.array([0.25, 0.75]),
                True,
                0.8,
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn('opacity="0.2000"', text)
        self.assertIn('opacity="0.6000"', text)

    def test_legacy_palette_keyword_names_remain_supported(self):
        palette = Palette(
            "#112233",
            "#445566",
            aarccolor="#778899",
            carccolor="#aabbcc",
        )
        xya = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        polygons, colors = polygons_from_xya(xya, self.colors, 5, 0.5)
        svg = render_polygons_svg(
            polygons, colors, palette=palette, show_arcs=True
        )
        self.assertIn("#778899", svg)
        self.assertIn("#aabbcc", svg)


if __name__ == "__main__":
    unittest.main()
