"""Search-space primitives: the parameters an optimizer proposes values for.

A :class:`SearchSpace` is an ordered collection of named parameters. Each
parameter maps a point in the unit interval ``[0, 1)`` to a concrete value, so
every sampler (random, quasi-random, model-based) can work in a single normalized
cube and let the space decode proposals. This is the standard trick that lets one
optimizer handle continuous, log-scaled, integer, and categorical parameters
uniformly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Value = float | int | str


@runtime_checkable
class Parameter(Protocol):
    """A named hyperparameter that decodes a unit value into a concrete value."""

    name: str

    def decode(self, unit: float) -> Value:
        """Map ``unit`` in ``[0, 1)`` to a concrete parameter value."""

    def encode(self, value: Value) -> float:
        """Map a concrete value back to its unit position (inverse of decode)."""


def _clamp_unit(unit: float) -> float:
    return min(1.0 - 1e-12, max(0.0, unit))


@dataclass(frozen=True)
class Uniform:
    """A continuous parameter uniform on ``[low, high]``."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"{self.name}: high must exceed low")

    def decode(self, unit: float) -> Value:
        return self.low + _clamp_unit(unit) * (self.high - self.low)

    def encode(self, value: Value) -> float:
        return (float(value) - self.low) / (self.high - self.low)


@dataclass(frozen=True)
class LogUniform:
    """A continuous parameter uniform on a logarithmic scale (for example a learning rate)."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low <= 0.0:
            raise ValueError(f"{self.name}: low must be positive for a log scale")
        if self.high <= self.low:
            raise ValueError(f"{self.name}: high must exceed low")

    def decode(self, unit: float) -> Value:
        log_low, log_high = math.log(self.low), math.log(self.high)
        return math.exp(log_low + _clamp_unit(unit) * (log_high - log_low))

    def encode(self, value: Value) -> float:
        log_low, log_high = math.log(self.low), math.log(self.high)
        return (math.log(float(value)) - log_low) / (log_high - log_low)


@dataclass(frozen=True)
class IntUniform:
    """An integer parameter uniform on the inclusive range ``[low, high]``."""

    name: str
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"{self.name}: high must be at least low")

    def decode(self, unit: float) -> Value:
        span = self.high - self.low + 1
        return self.low + min(span - 1, int(_clamp_unit(unit) * span))

    def encode(self, value: Value) -> float:
        span = self.high - self.low + 1
        return (int(value) - self.low + 0.5) / span


@dataclass(frozen=True)
class Categorical:
    """A categorical parameter over a fixed list of choices."""

    name: str
    choices: Sequence[Value]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError(f"{self.name}: choices must be non-empty")

    def decode(self, unit: float) -> Value:
        index = min(len(self.choices) - 1, int(_clamp_unit(unit) * len(self.choices)))
        return self.choices[index]

    def encode(self, value: Value) -> float:
        index = list(self.choices).index(value)
        return (index + 0.5) / len(self.choices)


class SearchSpace:
    """An ordered set of named parameters forming a normalized unit cube."""

    def __init__(self, parameters: Sequence[Parameter]) -> None:
        if not parameters:
            raise ValueError("a search space needs at least one parameter")
        names = [parameter.name for parameter in parameters]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        self._parameters = tuple(parameters)

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return self._parameters

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self._parameters)

    @property
    def dimension(self) -> int:
        return len(self._parameters)

    def decode(self, unit_point: Sequence[float]) -> dict[str, Value]:
        """Decode a point in the unit cube into a concrete configuration."""
        if len(unit_point) != self.dimension:
            raise ValueError("unit point has the wrong dimension")
        return {
            parameter.name: parameter.decode(unit)
            for parameter, unit in zip(self._parameters, unit_point, strict=True)
        }

    def encode(self, configuration: Mapping[str, Value]) -> list[float]:
        """Encode a concrete configuration back into the unit cube."""
        return [parameter.encode(configuration[parameter.name]) for parameter in self._parameters]
