"""Parameter schema objects for GeneticMarkov."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


def log_uniform(lo: float, hi: float, rng: Any = random) -> float:
    """Sample uniformly in log10 space."""
    if lo <= 0 or hi <= 0:
        raise ValueError("log_uniform requires positive bounds")
    return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi))


def should_use_log(lo: float, hi: float, threshold: float = 2.0) -> bool:
    """Return True when a positive range spans threshold dex or more."""
    if lo <= 0 or hi <= 0:
        return False
    return np.log10(hi / lo) >= threshold


@dataclass(frozen=True)
class CategoricalParameter:
    """A discrete model-choice parameter."""

    name: str
    options: Sequence[Any]

    def sample(self, rng: Any = random) -> int:
        if len(self.options) == 0:
            raise ValueError(f"Categorical parameter {self.name!r} has no options")
        return rng.randint(0, len(self.options) - 1)

    def repair(self, value: Any) -> int:
        if len(self.options) == 0:
            raise ValueError(f"Categorical parameter {self.name!r} has no options")
        return int(np.clip(round(float(value)), 0, len(self.options) - 1))


@dataclass(frozen=True)
class ContinuousParameter:
    """A bounded continuous parameter."""

    name: str
    lo: float
    hi: float
    log: bool = False

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"{self.name}: lo > hi")
        if self.log and (self.lo <= 0 or self.hi <= 0):
            raise ValueError(f"{self.name}: log sampling requires positive bounds")

    def sample(self, rng: Any = random) -> float:
        if self.log:
            return log_uniform(self.lo, self.hi, rng=rng)
        return rng.uniform(self.lo, self.hi)

    def repair(self, value: Any) -> float:
        return float(np.clip(float(value), self.lo, self.hi))


@dataclass(frozen=True)
class ParameterSchema:
    """Ordered mixed categorical/continuous parameter schema."""

    categorical: Sequence[CategoricalParameter]
    continuous: Sequence[ContinuousParameter]

    @property
    def parameters(self) -> list[CategoricalParameter | ContinuousParameter]:
        return list(self.categorical) + list(self.continuous)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def categorical_indices(self) -> list[int]:
        return list(range(len(self.categorical)))

    @property
    def continuous_indices(self) -> list[int]:
        start = len(self.categorical)
        return list(range(start, start + len(self.continuous)))

    def get_bounds(self, index: int) -> tuple[float, float]:
        if index not in self.continuous_indices:
            return (0.0, 1.0)
        p = self.parameters[index]
        assert isinstance(p, ContinuousParameter)
        return (p.lo, p.hi)

    def sample(self, rng: Any = random) -> list[float]:
        values: list[float] = []
        for p in self.categorical:
            values.append(float(p.sample(rng=rng)))
        for p in self.continuous:
            values.append(float(p.sample(rng=rng)))
        return values

    def repair(self, values: Sequence[Any]) -> list[float]:
        if len(values) != len(self.parameters):
            raise ValueError(
                f"Expected {len(self.parameters)} parameters, got {len(values)}"
            )

        repaired: list[float] = []
        for value, param in zip(values, self.parameters):
            repaired.append(float(param.repair(value)))
        return repaired

    def as_dict(self, values: Sequence[Any]) -> dict[str, Any]:
        repaired = self.repair(values)
        out: dict[str, Any] = {}
        for value, param in zip(repaired, self.parameters):
            if isinstance(param, CategoricalParameter):
                idx = int(value)
                out[param.name] = param.options[idx]
                out[f"{param.name}_idx"] = idx
            else:
                out[param.name] = float(value)
        return out
