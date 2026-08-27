"""Constant-movement resampling for XYA and polygon trajectories."""

from __future__ import annotations

from typing import Literal

import numpy as np


TrajectoryKind = Literal["xya", "polygons"]


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validate_trajectory(trajectory, kind: TrajectoryKind) -> np.ndarray:
    if kind not in ("xya", "polygons"):
        raise ValueError(f"kind must be 'xya' or 'polygons', got {kind!r}")
    states = [_as_numpy(state) for state in trajectory]
    if not states:
        raise ValueError("Trajectory must contain at least one state")
    shape = states[0].shape
    if any(state.shape != shape for state in states[1:]):
        raise ValueError("All trajectory states must have the same shape")
    values = np.asarray(states, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Trajectory values must be finite")
    if kind == "xya":
        if values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError("XYA trajectory states must have shape (N, 3)")
    elif values.ndim != 4 or values.shape[-1] != 2:
        raise ValueError("Polygon trajectory states must have shape (N, V, 2)")
    return values


def wrapped_angle_delta(source, target, *, angle_scale: float = 1.0):
    """Return shortest signed angle deltas in the supplied angle coordinates."""
    if not np.isfinite(angle_scale) or angle_scale == 0:
        raise ValueError("angle_scale must be a non-zero finite value")
    half_period = np.pi * abs(float(angle_scale))
    period = 2.0 * half_period
    return np.remainder(_as_numpy(target) - _as_numpy(source) + half_period, period) - half_period


def trajectory_movements(
    trajectory,
    *,
    kind: TrajectoryKind = "xya",
    angle_scale: float = 1.0,
) -> np.ndarray:
    """Measure movement between consecutive keyframes in their native space."""
    values = _validate_trajectory(trajectory, kind)
    if len(values) == 1:
        return np.empty(0, dtype=float)
    delta = np.diff(values, axis=0)
    if kind == "xya":
        xy_distance = np.linalg.norm(delta[..., :2], axis=-1)
        angle_distance = np.abs(
            wrapped_angle_delta(
                values[:-1, ..., 2],
                values[1:, ..., 2],
                angle_scale=angle_scale,
            )
        )
        return np.mean(xy_distance + angle_distance, axis=1)
    return np.sqrt(np.mean(np.sum(delta * delta, axis=-1), axis=(1, 2)))


def _resample_trajectory(
    trajectory,
    *,
    kind: TrajectoryKind,
    target_duration: float,
    fps: int,
    angle_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return resampled values plus left-keyframe indices and local fractions."""
    if not np.isfinite(target_duration) or target_duration <= 0:
        raise ValueError("target_duration must be a positive finite value")
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    values = _validate_trajectory(trajectory, kind)
    frame_count = max(1, round(float(target_duration) * fps))
    movements = trajectory_movements(
        values,
        kind=kind,
        angle_scale=angle_scale,
    )
    cumulative = np.concatenate(([0.0], np.cumsum(movements)))
    total = float(cumulative[-1])
    if len(values) == 1 or total <= np.finfo(float).eps:
        frames = np.repeat(values[:1], frame_count, axis=0)
        return frames, np.zeros(frame_count, dtype=int), np.zeros(frame_count)

    render_velocity = total / float(target_duration)
    timestamps = np.linspace(0.0, float(target_duration), frame_count)
    positions = render_velocity * timestamps
    left = np.searchsorted(cumulative, positions, side="right") - 1
    left = np.clip(left, 0, len(values) - 2)
    widths = movements[left]
    fractions = np.divide(
        positions - cumulative[left],
        widths,
        out=np.zeros_like(positions),
        where=widths > np.finfo(float).eps,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    weight_shape = (frame_count,) + (1,) * (values.ndim - 1)
    weight = fractions.reshape(weight_shape)
    frames = values[left] + weight * (values[left + 1] - values[left])
    if kind == "xya":
        angle_delta = wrapped_angle_delta(
            values[left, ..., 2],
            values[left + 1, ..., 2],
            angle_scale=angle_scale,
        )
        frames[..., 2] = values[left, ..., 2] + fractions[:, None] * angle_delta
        half_period = np.pi * abs(float(angle_scale))
        frames[..., 2] = np.remainder(frames[..., 2] + half_period, 2 * half_period) - half_period
    return frames, left, fractions


def resample_trajectory(
    trajectory,
    *,
    kind: TrajectoryKind = "xya",
    target_duration: float,
    fps: int = 30,
    angle_scale: float = 1.0,
) -> np.ndarray:
    """Resample uneven keyframes to constant movement over a target duration."""
    frames, _, _ = _resample_trajectory(
        trajectory,
        kind=kind,
        target_duration=target_duration,
        fps=fps,
        angle_scale=angle_scale,
    )
    return frames
