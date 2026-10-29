"""The Tree-structured Parzen Estimator (TPE).

TPE is the model-based sampler behind Hyperopt and Optuna. Instead of modeling
``p(value | config)`` directly, it splits completed trials into a *good* group
(the best quantile of objective values) and a *bad* group, fits a density to each
group per dimension, and proposes the candidate that maximizes the ratio
``l(x) / g(x)`` of good-density to bad-density. That ratio is monotonic in
Expected Improvement, so TPE performs Bayesian optimization without a surrogate
regression model.

The densities here are Parzen-window (kernel density) estimators over the unit
cube with reflective boundaries, built from scratch.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from hyperparameter_optimizer.trial import CompletedTrial


def _gaussian(x: float, mean: float, sigma: float) -> float:
    z = (x - mean) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


class _ParzenDensity:
    """A one-dimensional Parzen-window density over ``[0, 1]`` with a uniform prior."""

    def __init__(self, observations: Sequence[float]) -> None:
        self._means = list(observations)
        count = len(observations)
        # Scott-style bandwidth with a floor, widened when there are few points.
        self._sigma = max(0.05, 1.0 / math.sqrt(count + 1))
        self._prior_weight = 1.0 / (count + 1)

    def pdf(self, x: float) -> float:
        if not self._means:
            return 1.0
        kernel = sum(_gaussian(x, mean, self._sigma) for mean in self._means)
        kernel /= len(self._means)
        # Mix with a uniform prior so no region has zero density.
        return (1.0 - self._prior_weight) * kernel + self._prior_weight * 1.0


class TPESampler:
    """Tree-structured Parzen Estimator over the unit cube.

    Until ``n_startup`` trials have completed it samples at random to seed the
    densities, then proposes the candidate (out of ``n_candidates`` random draws)
    that maximizes the good/bad density ratio in every dimension.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.25,
        n_startup: int = 8,
        n_candidates: int = 48,
        seed: int = 0,
    ) -> None:
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if n_startup < 1:
            raise ValueError("n_startup must be at least 1")
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1")
        self._gamma = gamma
        self._n_startup = n_startup
        self._n_candidates = n_candidates
        self._rng = random.Random(seed)

    def sample(self, dimension: int, history: Sequence[CompletedTrial]) -> list[float]:
        if len(history) < self._n_startup:
            return [self._rng.random() for _ in range(dimension)]

        ordered = sorted(history, key=lambda completed: completed.value)
        split = max(1, math.ceil(self._gamma * len(ordered)))
        good = ordered[:split]
        bad = ordered[split:] or ordered[split - 1 :]

        best_point: list[float] = []
        best_score = float("-inf")
        for _ in range(self._n_candidates):
            candidate = [self._rng.random() for _ in range(dimension)]
            score = 0.0
            for axis in range(dimension):
                good_density = _ParzenDensity([t.trial.unit_point[axis] for t in good])
                bad_density = _ParzenDensity([t.trial.unit_point[axis] for t in bad])
                good_pdf = good_density.pdf(candidate[axis])
                bad_pdf = bad_density.pdf(candidate[axis])
                score += math.log(good_pdf + 1e-12) - math.log(bad_pdf + 1e-12)
            if score > best_score:
                best_score = score
                best_point = candidate
        return best_point
