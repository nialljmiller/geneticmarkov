"""Generic GA operator helpers used by GeneticMarkov examples and engines."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def fitness_value(individual: Any, default: float = float("inf")) -> float:
    """Return a scalar DEAP-style fitness value, or default if unavailable."""

    fitness = getattr(individual, "fitness", None)
    if fitness is None:
        return default

    values = getattr(fitness, "values", ())
    valid = getattr(fitness, "valid", False)

    if not valid or len(values) == 0:
        return default

    return float(values[0])


def reflect_scalar(value: float, lo: float, hi: float) -> float:
    """Reflect a scalar value into [lo, hi]."""

    value = float(value)
    lo = float(lo)
    hi = float(hi)

    if lo >= hi:
        return lo

    width = hi - lo
    t = (value - lo) % (2.0 * width)

    if t <= width:
        return lo + t

    return lo + 2.0 * width - t


def tournament_select(
    individuals: Sequence[Any],
    k: int | None = None,
    *,
    tournsize: int = 3,
    minimize: bool = True,
    rng: Any = random,
) -> list[Any]:
    """Tournament selection for DEAP-like individuals."""

    if k is None:
        k = len(individuals)

    if len(individuals) == 0:
        return []

    selected: list[Any] = []

    for _ in range(k):
        aspirants = rng.sample(list(individuals), min(tournsize, len(individuals)))
        key = fitness_value
        selected.append(min(aspirants, key=key) if minimize else max(aspirants, key=key))

    return selected


def fitness_scale(
    fitness: float,
    recent_fitnesses: Sequence[float],
    *,
    minimize: bool = True,
    good_scale: float = 0.5,
    mid_scale: float = 1.0,
    bad_scale: float = 1.5,
    min_history: int = 20,
) -> float:
    """Return an adaptive mutation scale based on recent fitness rank."""

    finite = np.asarray([x for x in recent_fitnesses if np.isfinite(x)], dtype=float)

    if len(finite) < min_history or not np.isfinite(fitness):
        return mid_scale

    q25, q75 = np.percentile(finite, [25, 75])

    if minimize:
        if fitness <= q25:
            return good_scale
        if fitness <= q75:
            return mid_scale
        return bad_scale

    if fitness >= q75:
        return good_scale
    if fitness >= q25:
        return mid_scale
    return bad_scale


def invalidate_fitness(individual: Any) -> None:
    """Invalidate a DEAP-style individual fitness in-place."""

    fitness = getattr(individual, "fitness", None)
    if fitness is not None and hasattr(fitness, "values"):
        del fitness.values


def deduplicate_population(
    population: Sequence[Any],
    *,
    continuous_indices: Sequence[int],
    get_bounds: Callable[[int], tuple[float, float]],
    clone: Callable[[Any], Any],
    invalidate: Callable[[Any], None] = invalidate_fitness,
    jitter_fraction: float = 0.002,
    ndigits: int = 6,
    rng: Any = random,
) -> list[Any]:
    """Jitter duplicate individuals in continuous dimensions."""

    seen: set[tuple[float, ...]] = set()
    unique: list[Any] = []

    for ind in population:
        key = tuple(round(float(x), ndigits) for x in ind)

        if key not in seen:
            seen.add(key)
            unique.append(ind)
            continue

        new_ind = clone(ind)

        for i in continuous_indices:
            lo, hi = get_bounds(i)
            width = hi - lo
            new_ind[i] = reflect_scalar(
                float(new_ind[i]) + rng.gauss(0.0, width * jitter_fraction),
                lo,
                hi,
            )

        invalidate(new_ind)
        unique.append(new_ind)

    return unique
