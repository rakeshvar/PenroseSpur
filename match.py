"""Optimal-transport matching for `(x, y, scaled_angle)` tile tensors."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


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



def lsa(cost, workers=None):            # CPU-based 
    """
    Solve each square cost matrix with SciPy's exact LSA solver.
        Loops over each batch index.
    """
    cost_numpy = cost.detach().cpu().numpy()
    available = os.cpu_count() or 1
    worker_count = max(1, min(int(workers or available), available, len(cost_numpy)))

    def _solve_one(cost):
        rows, columns = linear_sum_assignment(cost)
        permutation = np.empty(cost.shape[0], dtype=np.int64)
        permutation[rows] = columns
        return permutation

    if worker_count == 1:
        permutations = [_solve_one(item) for item in cost_numpy]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            permutations = list(executor.map(_solve_one, cost_numpy))

    return torch.from_numpy(np.stack(permutations)).to(cost.device)


def match(
    data,
    noise,
    method,
    *,
    epsilon=0.03,
    iterations=10,
    anneal_rate=0.5,
    anneal_steps=100,
    lsa_workers=None,
    return_details=False,
    squared=True,
):
    """Match noise to data and return it in data-row order.

    Inputs use the sampler's unit-variance `(x, y, scaled_angle)` coordinates.
    `lsa` produces a true permutation. `argmax` anneals its Sinkhorn
    temperature until row argmaxes cover every column, when possible.
    `barycenter` returns the row-normalized soft Sinkhorn average.
    """
    assert data.shape == noise.shape, f"data and noise must have the same shape, got {data.shape} and {noise.shape}"
    if method not in MATCH_METHODS:
        raise NotImplementedError(f"Unknown match method {method!r}; expected one of {MATCH_METHODS}")
    
    soft_permutation = None
    permutation = None
    converged = True

    if squared: 
        cost = torch.cdist(data, noise).square()
    else:
        cost = torch.cdist(data, noise)

    if method == "lsa":
        permutation = lsa(cost, workers=lsa_workers)
        matched_noise = gather_noise(noise, permutation)
    else:
        logP = -cost / epsilon

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
        permutation = lsa(-soft_permutation, workers=lsa_workers)
        matched_noise = gather_noise(noise, permutation)

    if return_details:
        return MatchResult(matched_noise, permutation, soft_permutation, converged)
    else:
        return matched_noise