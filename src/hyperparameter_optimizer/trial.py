"""Trials: a proposed configuration and, once evaluated, its objective value."""

from __future__ import annotations

from dataclasses import dataclass, field

from hyperparameter_optimizer.space import Value


@dataclass(frozen=True)
class Trial:
    """One proposed configuration, identified by number, with its unit point."""

    number: int
    unit_point: tuple[float, ...]
    configuration: dict[str, Value]


@dataclass(frozen=True)
class CompletedTrial:
    """A trial paired with the objective value it achieved (lower is better)."""

    trial: Trial
    value: float
    intermediate: tuple[float, ...] = field(default_factory=tuple)

    @property
    def number(self) -> int:
        return self.trial.number

    @property
    def configuration(self) -> dict[str, Value]:
        return self.trial.configuration
