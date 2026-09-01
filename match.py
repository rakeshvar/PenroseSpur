"""Optimal-transport matching for `(x, y, scaled_angle)` tile tensors."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from flow_geometry import ANGLE_HALF_PERIOD, ANGLE_PERIOD, pairwise_xya_distance

MATCH_METHODS = ("lsa", "sinkhorn.argmax", "sinkhorn.barycenter", "sinkhorn.lsa")


@dataclass
class MatchResult:
    matched_noise: torch.Tensor
    permutation: torch.Tensor | None
    soft_permutation: torch.Tensor | None
    converged: bool


def sinkhorn_non_log(P, iterations):
    for _ in range(iterations):
        P = P / P.sum(dim=2, keepdim=True)
        P = P / P.sum(dim=1, keepdim=True)
    return P

def sinkhorn(logP, iterations):
    logU = -torch.logsumexp(logP, dim=2)
    for _ in range(iterations):
        logV = -torch.logsumexp(logP + logU[:, :, None], dim=1)
        logU = -torch.logsumexp(logP + logV[:, None, :], dim=2)
    return logP + logU[:, :, None] + logV[:, None, :]


def sinkhorn_annealed_argmax(logP, iterations=5, anneal_rate=0.5, anneal_steps=100):
    permutation = None
    converged = False

    for _ in range(anneal_steps):
        logP = sinkhorn(logP, iterations)

        # Check if the permutation is a bijection
        permutation = logP.argmax(dim=2)
        column_counts = torch.zeros_like(permutation)
        column_counts.scatter_add_(1, permutation, torch.ones_like(permutation))
        if torch.all(column_counts == 1):
            converged = True
            break

        logP = logP / anneal_rate

    return logP.exp(), permutation, converged


def gather_noise(noise, permutation):
    """Gather `(B, N, D)` noise using row-to-column assignment indices."""
    indices = permutation.unsqueeze(-1).expand(-1, -1, noise.shape[-1])
    return noise.gather(1, indices)


def balanced_group_sizes(count, target_size=64, max_size=80):
    """Balance ``count`` near target, avoiding sizes <= 40 when possible."""
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (count, target_size, max_size)):
        raise TypeError("count, target_size, and max_size must be integers")
    if count <= 0 or target_size <= 0 or max_size <= 0:
        raise ValueError("count, target_size, and max_size must be positive")
    if target_size > max_size:
        raise ValueError("target_size cannot exceed max_size")

    min_groups = math.ceil(count / max_size)
    max_groups_over_40 = count // 41
    if min_groups <= max_groups_over_40:
        groups = min(
            range(min_groups, max_groups_over_40 + 1),
            key=lambda candidate: (
                abs(count / candidate - target_size),
                candidate,
            ),
        )
    else:
        groups = min_groups

    small, remainder = divmod(count, groups)
    return tuple([small + 1] * remainder + [small] * (groups - remainder))


def _group_assignment(batch, indices, data, noise, squared):
    target = data[indices]
    source = noise[indices]
    xy = target[:, None, :2] - source[None, :, :2]
    angle = np.remainder(
        target[:, None, 2] - source[None, :, 2] + ANGLE_HALF_PERIOD,
        ANGLE_PERIOD,
    ) - ANGLE_HALF_PERIOD
    cost = np.square(xy).sum(axis=-1) + np.square(angle)
    if not squared:
        cost = np.sqrt(cost)
    rows, columns = linear_sum_assignment(cost)
    return batch, indices[rows], indices[columns]


def grouped_lsa(
    data,
    noise,
    colors=None,
    *,
    target_size=64,
    max_size=80,
    workers=None,
    generator=None,
    squared=True,
):
    """Return exact assignments within shuffled, balanced same-color groups."""
    if colors is not None and colors.shape != data.shape[:2]:
        raise ValueError(
            f"colors must have shape {tuple(data.shape[:2])}, got {tuple(colors.shape)}"
        )

    data_numpy = data.detach().cpu().numpy()
    noise_numpy = noise.detach().cpu().numpy()
    colors_numpy = None if colors is None else colors.detach().cpu().numpy()
    batch_size, num_tiles = data.shape[:2]
    permutation = np.broadcast_to(
        np.arange(num_tiles, dtype=np.int64), (batch_size, num_tiles)
    ).copy()
    tasks = []
    for batch in range(batch_size):
        batch_colors = (
            np.zeros(num_tiles, dtype=np.uint8)
            if colors_numpy is None else colors_numpy[batch]
        )
        for color in np.unique(batch_colors):
            indices = np.flatnonzero(batch_colors == color)
            shuffle = torch.randperm(
                len(indices), device=data.device, generator=generator
            ).cpu().numpy()
            indices = indices[shuffle]
            offset = 0
            for size in balanced_group_sizes(
                len(indices), target_size, max_size
            ):
                group = indices[offset:offset + size]
                tasks.append(
                    (batch, group, data_numpy[batch], noise_numpy[batch], squared)
                )
                offset += size

    available = os.cpu_count() or 1
    worker_count = max(1, min(int(workers or available), available, len(tasks)))
    if worker_count == 1:
        assignments = map(lambda args: _group_assignment(*args), tasks)
        for batch, rows, columns in assignments:
            permutation[batch, rows] = columns
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            assignments = executor.map(
                lambda args: _group_assignment(*args), tasks
            )
            for batch, rows, columns in assignments:
                permutation[batch, rows] = columns
    return torch.from_numpy(permutation).to(data.device)



def lsa(cost, colors=None, workers=None):            # CPU-based
    """
    Solve each square cost matrix with SciPy's exact LSA solver.

    Assignments are solved independently within each color. When `colors` is
    None, all rows and columns belong to one color.
    """
    if colors is not None and colors.shape != cost.shape[:2]:
        raise ValueError(
            f"colors must have shape {tuple(cost.shape[:2])}, got {tuple(colors.shape)}"
        )

    cost_numpy = cost.detach().cpu().numpy()
    colors_numpy = None if colors is None else colors.detach().cpu().numpy()
    available = os.cpu_count() or 1
    worker_count = max(1, min(int(workers or available), available, len(cost_numpy)))

    def _solve_one(item):
        cost, item_colors = item
        permutation = np.empty(cost.shape[0], dtype=np.int64)
        if item_colors is None:
            rows, columns = linear_sum_assignment(cost)
            permutation[rows] = columns
            return permutation

        for color in np.unique(item_colors):
            indices = np.flatnonzero(item_colors == color)
            rows, columns = linear_sum_assignment(cost[np.ix_(indices, indices)])
            permutation[indices[rows]] = indices[columns]
        return permutation

    color_rows = [None] * len(cost_numpy) if colors_numpy is None else colors_numpy
    items = list(zip(cost_numpy, color_rows))
    if worker_count == 1:
        permutations = [_solve_one(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            permutations = list(executor.map(_solve_one, items))

    return torch.from_numpy(np.stack(permutations)).to(cost.device)


def match(
    data,
    noise,
    method,
    colors=None,
    *,
    epsilon=0.03,
    iterations=10,
    anneal_rate=0.5,
    anneal_steps=100,
    lsa_workers=None,
    lsa_target_size=64,
    lsa_max_size=80,
    generator=None,
    return_details=False,
    squared=True,
):
    """Match noise to data and return it in data-row order.

    Inputs use the sampler's unit-variance `(x, y, scaled_angle)` coordinates.
    Matching is restricted to tiles of the same color. If `colors` is None,
    all tiles are treated as one color. By default, `lsa` shuffles each color
    and solves balanced groups targeting 64 tiles with a maximum of 80; it
    produces a true permutation. Pass `generator` for reproducible grouping.
    `sinkhorn.argmax` anneals its Sinkhorn temperature until row argmaxes cover
    every column, when possible. `sinkhorn.barycenter` returns the
    row-normalized soft Sinkhorn average.
    """
    assert data.shape == noise.shape, f"data and noise must have the same shape, got {data.shape} and {noise.shape}"
    if method not in MATCH_METHODS:
        raise NotImplementedError(f"Unknown match method {method!r}; expected one of {MATCH_METHODS}")
    if colors is not None and colors.shape != data.shape[:2]:
        raise ValueError(
            f"colors must have shape {tuple(data.shape[:2])}, got {tuple(colors.shape)}"
        )
    if colors is not None:
        colors = colors.to(data.device)
    
    soft_permutation = None
    permutation = None
    converged = True

    if method == "lsa":
        permutation = grouped_lsa(
            data,
            noise,
            colors,
            target_size=lsa_target_size,
            max_size=lsa_max_size,
            workers=lsa_workers,
            generator=generator,
            squared=squared,
        )
        matched_noise = gather_noise(noise, permutation)
    else:
        cost = pairwise_xya_distance(data, noise, squared=squared)
        logP = -cost / epsilon
        if colors is not None:
            same_color = colors.unsqueeze(2) == colors.unsqueeze(1)
            logP = logP.masked_fill(~same_color, -torch.inf)

    if method == "sinkhorn.barycenter":
        soft_permutation = sinkhorn(logP, iterations).exp()
        row_mass = soft_permutation.sum(dim=2, keepdim=True)
        barycentric_weights = soft_permutation / row_mass
        matched_noise = torch.bmm(barycentric_weights, noise)

    if method == "sinkhorn.argmax":
        soft_permutation, permutation, converged = sinkhorn_annealed_argmax(logP, iterations, anneal_rate, anneal_steps)
        matched_noise = gather_noise(noise, permutation)
    
    if method == "sinkhorn.lsa":
        soft_permutation = sinkhorn(logP, iterations).exp()
        permutation = lsa(-soft_permutation, colors=colors, workers=lsa_workers)
        matched_noise = gather_noise(noise, permutation)

    if return_details:
        return MatchResult(matched_noise, permutation, soft_permutation, converged)
    else:
        return matched_noise