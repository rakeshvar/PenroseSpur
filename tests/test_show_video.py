import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from show import (  # noqa: E402
    VideoOptions,
    ViewBox,
    normalize_svg_canvas,
    read_viewbox,
    resample_trajectory,
    save_mp4,
    save_tiles_svg,
    save_trajectory_mp4,
    trajectory_movements,
)


class VideoTest(unittest.TestCase):
    def _frame(self, directory: Path, name: str, shift: float = 0.0) -> Path:
        path = directory / name
        xya = np.array([[shift, 0.0, 0.0], [shift + 1.0, 0.0, 0.0]])
        save_tiles_svg(
            path,
            xya,
            np.array([0, 1]),
            symmetry=6,
            side=0.4,
        )
        return path

    def test_normalize_svg_canvas_uses_fixed_even_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._frame(root, "source.svg")
            output = root / "normalized.svg"
            box = ViewBox(-2.0, -1.0, 5.0, 2.0)
            size = normalize_svg_canvas(
                source,
                box,
                background="black",
                display_height=721,
                output_path=output,
            )
            self.assertEqual(size[0] % 2, 0)
            self.assertEqual(size[1] % 2, 0)
            self.assertEqual(read_viewbox(output), box)
            self.assertIn('fill="black"', output.read_text(encoding="utf-8"))

    def test_missing_ffmpeg_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = self._frame(Path(directory), "frame.svg")
            with mock.patch("show.video.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg is required"):
                    save_mp4([frame], Path(directory) / "out.mp4")

    def test_save_mp4_preserves_frame_order_and_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = [
                self._frame(root, "b.svg", 0.0),
                self._frame(root, "a.svg", 0.2),
            ]
            commands = []

            def fake_run(command, check):
                commands.append(command)
                Path(command[-1]).write_bytes(b"fake")

            output = root / "movie.mp4"
            with (
                mock.patch("show.video.shutil.which", return_value="/fake/ffmpeg"),
                mock.patch("show.video.subprocess.run", side_effect=fake_run),
            ):
                result = save_mp4(
                    frames,
                    output,
                    options=VideoOptions(fps=7, crf=22),
                )
            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            encode = commands[-1]
            self.assertEqual(encode[encode.index("-framerate") + 1], "7")
            self.assertEqual(encode[encode.index("-crf") + 1], "22")
            self.assertIn("+faststart", encode)
            self.assertEqual(len(commands), 3)

    def test_static_tile_opacities_are_not_mistaken_for_frame_opacities(self):
        states = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]),
        ]
        captured = []

        def inspect_frames(paths, output, **options):
            captured.extend(Path(path).read_text(encoding="utf-8") for path in paths)
            return Path(output)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("show.video.save_mp4", side_effect=inspect_frames),
        ):
            save_trajectory_mp4(
                states,
                np.array([0, 1]),
                Path(directory) / "out.mp4",
                symmetry=6,
                side=0.4,
                opacities=np.array([0.2, 0.8]),
            )
        self.assertEqual(len(captured), 2)
        for svg in captured:
            self.assertIn('opacity="0.1400"', svg)
            self.assertIn('opacity="0.5600"', svg)

    def test_default_video_rate_is_thirty_fps(self):
        self.assertEqual(VideoOptions().fps, 30)

    def test_xya_resampling_uses_cumulative_movement(self):
        states = np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[10.0, 0.0, 0.0]],
                [[60.0, 0.0, 0.0]],
            ]
        )
        movements = trajectory_movements(states, kind="xya")
        np.testing.assert_allclose(movements, [10.0, 50.0])
        frames = resample_trajectory(
            states,
            kind="xya",
            target_duration=1.0,
            fps=7,
        )
        np.testing.assert_allclose(frames[:, 0, 0], np.arange(0.0, 61.0, 10.0))

    def test_xya_resampling_wraps_across_angle_boundary(self):
        states = np.array(
            [
                [[0.0, 0.0, np.pi - 0.1]],
                [[0.0, 0.0, -np.pi + 0.1]],
            ]
        )
        frames = resample_trajectory(
            states,
            kind="xya",
            target_duration=1.0,
            fps=3,
        )
        self.assertAlmostEqual(abs(frames[1, 0, 2]), np.pi, places=7)
        self.assertAlmostEqual(
            trajectory_movements(states, kind="xya")[0],
            0.2,
            places=7,
        )

    def test_polygon_resampling_interpolates_vertices_directly(self):
        polygon = np.array(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]
        )
        states = np.stack((polygon, polygon + np.array([2.0, 0.0])))
        movements = trajectory_movements(states, kind="polygons")
        np.testing.assert_allclose(movements, [2.0])
        frames = resample_trajectory(
            states,
            kind="polygons",
            target_duration=1.0,
            fps=3,
        )
        np.testing.assert_allclose(frames[1], polygon + np.array([1.0, 0.0]))

    def test_duration_resampling_interpolates_frame_opacities(self):
        states = [
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[3.0, 0.0, 0.0]]),
        ]
        captured = []

        def inspect_frames(paths, output, **options):
            captured.extend(Path(path).read_text(encoding="utf-8") for path in paths)
            return Path(output)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("show.video.save_mp4", side_effect=inspect_frames),
        ):
            save_trajectory_mp4(
                states,
                np.array([0]),
                Path(directory) / "out.mp4",
                symmetry=6,
                side=0.4,
                target_duration=1.0,
                opacities=np.array([[0.0], [1.0]]),
                options=VideoOptions(fps=4),
            )
        self.assertEqual(len(captured), 4)
        expected = (0.0, 0.2333, 0.4667, 0.7)
        for svg, opacity in zip(captured, expected):
            self.assertIn(f'opacity="{opacity:.4f}"', svg)

    def test_polygon_trajectory_uses_polygon_renderer(self):
        polygon = np.array(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]
        )
        rendered = []

        def inspect_polygon(path, values, colors, **options):
            rendered.append(np.asarray(values))
            return Path(path)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("show.video.save_polygons", side_effect=inspect_polygon),
            mock.patch(
                "show.video.save_mp4",
                side_effect=lambda paths, output, **options: Path(output),
            ),
        ):
            save_trajectory_mp4(
                [polygon, polygon + np.array([2.0, 0.0])],
                np.array([0]),
                Path(directory) / "out.mp4",
                kind="polygons",
                target_duration=1.0,
                options=VideoOptions(fps=3),
            )
        self.assertEqual(len(rendered), 3)
        np.testing.assert_allclose(rendered[1], polygon + np.array([1.0, 0.0]))

    def test_stationary_trajectory_repeats_for_target_duration(self):
        state = np.array([[1.0, 2.0, 0.25]])
        frames = resample_trajectory(
            [state, state],
            target_duration=2.0,
            fps=3,
        )
        self.assertEqual(frames.shape[0], 6)
        np.testing.assert_allclose(frames, np.repeat(state[None], 6, axis=0))

    def test_trajectory_validation_rejects_mismatched_states(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            resample_trajectory(
                [np.zeros((1, 3)), np.zeros((2, 3))],
                target_duration=1.0,
            )


if __name__ == "__main__":
    unittest.main()
