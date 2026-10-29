"""The sampler protocol: propose a unit point given the trials seen so far."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from hyperparameter_optimizer.trial import CompletedTrial


@runtime_checkable
class Sampler(Protocol):
    """Proposes the next point in the unit cube to evaluate."""

    def sample(self, dimension: int, history: Sequence[CompletedTrial]) -> list[float]:
        """Return the next unit-cube point given completed trials so far."""
