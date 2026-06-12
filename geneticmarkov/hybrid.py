"""Hybrid GA + DEMC move utilities."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .operators import fitness_value, invalidate_fitness, reflect_scalar


def apply_demc_hybrid_moves(
    population: Sequence[Any],
    *,
    evaluate: Callable[[Any], tuple[tuple[float], dict[str, Any]]],
    record_result: Callable[[dict[str, Any]], None],
    continuous_indices: Sequence[int],
    get_bounds: Callable[[int], tuple[float, float]],
    repair: Callable[[Any], None] | None = None,
    clone: Callable[[Any], Any],
    fraction: float = 0.40,
    generation: int = 0,
    big_step_every: int = 6,
    jitter: float = 1e-6,
    rng: Any = random,
) -> tuple[int, int]:
    """Apply DEMC-style local refinement moves to a population.

    This is intended for hybrid GA workflows where a fraction of the
    population is updated by differential-evolution proposals after normal
    GA selection/crossover/mutation.

    Returns
    -------
    accepted, attempted
        Number of accepted proposals and number attempted.
    """

    n_walkers = len(population)
    if n_walkers < 3:
        return 0, 0

    n_update = max(1, int(n_walkers * fraction))
    n_update = min(n_update, n_walkers)

    ranked = sorted(
        range(n_walkers),
        key=lambda i: fitness_value(population[i]),
        reverse=True,
    )
    update_indices = ranked[:n_update]

    ndim = len(continuous_indices)
    if ndim == 0:
        return 0, 0

    gamma = 2.38 / np.sqrt(2.0 * ndim)

    if big_step_every and generation % big_step_every == 0:
        gamma = 1.0

    accepted = 0
    attempted = 0

    for idx in update_indices:
        others = [i for i in range(n_walkers) if i != idx]
        if len(others) < 2:
            continue

        r1, r2 = rng.sample(others, 2)

        current = population[idx]
        proposal = clone(current)

        for i in continuous_indices:
            diff = float(population[r1][i]) - float(population[r2][i])
            lo, hi = get_bounds(i)
            proposal[i] = reflect_scalar(
                float(current[i]) + gamma * diff + rng.gauss(0.0, jitter),
                lo,
                hi,
            )

        if repair is not None:
            repair(proposal)

        invalidate_fitness(proposal)

        fit, result = evaluate(proposal)
        proposal.fitness.values = fit
        record_result(result)

        attempted += 1

        current_loss = fitness_value(current)
        proposal_loss = fitness_value(proposal)

        log_alpha = -(proposal_loss - current_loss)

        if np.log(rng.random()) < log_alpha:
            current[:] = proposal[:]
            current.fitness.values = proposal.fitness.values
            accepted += 1

    return accepted, attempted
