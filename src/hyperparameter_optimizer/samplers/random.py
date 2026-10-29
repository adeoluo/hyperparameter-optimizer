"""Random and quasi-random (Sobol) samplers.

``RandomSampler`` draws independent uniform points. ``SobolSampler`` draws a
low-discrepancy sequence (a base-2 van der Corput / Sobol construction) that
covers the cube far more evenly than independent sampling, which is why
quasi-random search is a stronger baseline than plain random search.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from hyperparameter_optimizer.trial import CompletedTrial


class RandomSampler:
    """Independent uniform sampling over the unit cube."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def sample(self, dimension: int, history: Sequence[CompletedTrial]) -> list[float]:
        return [self._rng.random() for _ in range(dimension)]


def _van_der_corput(index: int, base: int) -> float:
    result = 0.0
    denominator = 1.0
    value = index
    while value > 0:
        denominator *= base
        value, remainder = divmod(value, base)
        result += remainder / denominator
    return result


class SobolSampler:
    """A low-discrepancy sampler using per-dimension van der Corput sequences.

    Each dimension uses a distinct prime base, giving a Halton-style
    low-discrepancy sequence. It is deterministic and covers the cube evenly,
    a stronger space-filling baseline than independent random sampling.
    """

    _PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)

    def __init__(self, skip: int = 1) -> None:
        self._skip = max(1, skip)

    def sample(self, dimension: int, history: Sequence[CompletedTrial]) -> list[float]:
        index = self._skip + len(history)
        return [
            _van_der_corput(index, self._PRIMES[axis % len(self._PRIMES)])
            for axis in range(dimension)
        ]
