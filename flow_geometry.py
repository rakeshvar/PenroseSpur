"""Circular geometry operations for scaled ``(x, y, angle)`` tensors."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


ANGLE_HALF_PERIOD = math.sqrt(3.0)
ANGLE_PERIOD = 2.0 * ANGLE_HALF_PERIOD


@dataclass(frozen=True)
class GeometryLoss:
    total: torch.Tensor
    xy: torch.Tensor
    angle: torch.Tensor


def _validate_xya(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError(
            f"{name} must have final dimension 3, got {tuple(value.shape)}"
        )


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap a scaled angle to ``[-sqrt(3), sqrt(3))``."""
    return torch.remainder(angle + ANGLE_HALF_PERIOD, ANGLE_PERIOD) - ANGLE_HALF_PERIOD


def angle_delta(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Return the shortest signed scaled-angle displacement source -> target."""
    return wrap_angle(target - source)


def canonicalize_xya(xya: torch.Tensor) -> torch.Tensor:
    """Return XYA with its scaled-angle channel in the canonical period."""
    _validate_xya(xya, "xya")
    return torch.cat((xya[..., :2], wrap_angle(xya[..., 2:3])), dim=-1)


def xya_delta(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Return paired source -> target displacement with a wrapped angle."""
    _validate_xya(target, "target")
    _validate_xya(source, "source")
    try:
        xy = target[..., :2] - source[..., :2]
        angle = angle_delta(target[..., 2], source[..., 2])
    except RuntimeError as error:
        raise ValueError(
            f"target and source are not broadcast-compatible: "
            f"{tuple(target.shape)} and {tuple(source.shape)}"
        ) from error
    return torch.cat((xy, angle[..., None]), dim=-1)


def pairwise_xya_distance(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    squared: bool = True,
) -> torch.Tensor:
    """Return pairwise circular XYA distances with shape ``(B, N, M)``."""
    _validate_xya(target, "target")
    _validate_xya(source, "source")
    if target.ndim != 3 or source.ndim != 3:
        raise ValueError("target and source must have shape (B,N,3) and (B,M,3)")
    if target.shape[0] != source.shape[0]:
        raise ValueError(
            f"target and source batch sizes differ: {target.shape[0]} and "
            f"{source.shape[0]}"
        )
    if target.device != source.device:
        raise ValueError("target and source must be on the same device")
    if target.dtype != source.dtype:
        raise ValueError("target and source must have the same dtype")

    xy_sq = torch.cdist(target[..., :2], source[..., :2]).square()
    angular = angle_delta(target[..., 2, None], source[:, None, :, 2])
    distance_sq = xy_sq + angular.square()
    return distance_sq if squared else distance_sq.sqrt()


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss: str = "l2",
) -> GeometryLoss:
    """Return circular paired XYA reconstruction loss and components."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    error = xya_delta(prediction, target)
    if loss == "l1":
        component = error.abs()
    elif loss == "l2":
        component = error.square()
    else:
        raise ValueError("loss must be 'l1' or 'l2'")
    return GeometryLoss(
        total=component.mean(),
        xy=component[..., :2].mean(),
        angle=component[..., 2].mean(),
    )


def weighted_xya_average(
    weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Average XYA values with row weights and a circular angle mean."""
    _validate_xya(values, "values")
    if weights.ndim != 3 or values.ndim != 3:
        raise ValueError("weights and values must have shapes (B,N,M) and (B,M,3)")
    if weights.shape[0] != values.shape[0] or weights.shape[2] != values.shape[1]:
        raise ValueError(
            f"weights and values have incompatible shapes: "
            f"{tuple(weights.shape)} and {tuple(values.shape)}"
        )
    mass = weights.sum(dim=2, keepdim=True)
    tiny = torch.finfo(weights.dtype).tiny
    normalized = weights / mass.clamp_min(tiny)
    xy = torch.bmm(normalized, values[..., :2])
    phase = values[..., 2] * (math.pi / ANGLE_HALF_PERIOD)
    sine = torch.bmm(normalized, phase.sin().unsqueeze(-1)).squeeze(-1)
    cosine = torch.bmm(normalized, phase.cos().unsqueeze(-1)).squeeze(-1)
    angle = torch.atan2(sine, cosine) * (ANGLE_HALF_PERIOD / math.pi)
    return torch.cat((xy, angle[..., None]), dim=-1)


def flow_velocity(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return constant shortest-path velocity from source to target."""
    if source.shape != target.shape:
        raise ValueError(
            f"source and target shapes differ: "
            f"{tuple(source.shape)} and {tuple(target.shape)}"
        )
    return xya_delta(target, source)


def flow_step(
    source: torch.Tensor,
    velocity: torch.Tensor,
    amount: float | torch.Tensor,
) -> torch.Tensor:
    """Advance an XYA state by a tangent velocity and wrap its angle."""
    if source.shape != velocity.shape:
        raise ValueError(
            f"source and velocity shapes differ: "
            f"{tuple(source.shape)} and {tuple(velocity.shape)}"
        )
    _validate_xya(source, "source")
    _validate_xya(velocity, "velocity")
    step = torch.as_tensor(amount, device=source.device, dtype=source.dtype)
    if step.ndim == 1:
        if step.shape != (source.shape[0],):
            raise ValueError(
                f"one-dimensional amount must have shape ({source.shape[0]},), "
                f"got {tuple(step.shape)}"
            )
        step = step[:, None, None]
    elif step.ndim != 0:
        raise ValueError("amount must be a scalar or have shape (B,)")
    result = source + step * velocity
    return canonicalize_xya(result)


def flow_state(
    source: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Interpolate XY linearly and scaled angle along its shortest arc."""
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError(
            f"source and target must share shape (B,N,3), got "
            f"{tuple(source.shape)} and {tuple(target.shape)}"
        )
    _validate_xya(source, "source")
    _validate_xya(target, "target")
    if time.shape != (source.shape[0],):
        raise ValueError(
            f"time must have shape ({source.shape[0]},), got {tuple(time.shape)}"
        )
    if not time.is_floating_point():
        raise TypeError("time must have a floating-point dtype")
    if bool(((time < 0) | (time > 1)).any()):
        raise ValueError("time must lie in [0,1]")

    amount = time.to(device=source.device, dtype=source.dtype)
    return flow_step(source, flow_velocity(source, target), amount)


def flow(
    source: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shortest-path flow state and its constant target velocity."""
    return flow_state(source, target, time), flow_velocity(source, target)
