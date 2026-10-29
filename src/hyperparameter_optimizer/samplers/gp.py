"""Gaussian-Process Bayesian optimization with Expected Improvement.

A from-scratch Gaussian process regressor with a squared-exponential (RBF) kernel
provides a posterior mean and variance at any point in the unit cube. The sampler
proposes the candidate that maximizes **Expected Improvement** over the best value
seen, the canonical Bayesian-optimization acquisition that balances exploiting
low-mean regions against exploring high-variance ones.

Everything is pure Python: the kernel, the Cholesky solve, and the acquisition.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from hyperparameter_optimizer.linalg import cholesky, cholesky_solve
from hyperparameter_optimizer.trial import CompletedTrial


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class GaussianProcessSampler:
    """Bayesian optimization with a Gaussian-process surrogate and Expected Improvement."""

    def __init__(
        self,
        *,
        length_scale: float = 0.2,
        signal_variance: float = 1.0,
        noise: float = 1e-6,
        n_startup: int = 5,
        n_candidates: int = 64,
        seed: int = 0,
    ) -> None:
        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive")
        if n_startup < 1:
            raise ValueError("n_startup must be at least 1")
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1")
        self._length_scale = length_scale
        self._signal_variance = signal_variance
        self._noise = noise
        self._n_startup = n_startup
        self._n_candidates = n_candidates
        self._rng = random.Random(seed)

    def _kernel(self, a: Sequence[float], b: Sequence[float]) -> float:
        squared = sum((ai - bi) ** 2 for ai, bi in zip(a, b, strict=True))
        return self._signal_variance * math.exp(-0.5 * squared / self._length_scale**2)

    def sample(self, dimension: int, history: Sequence[CompletedTrial]) -> list[float]:
        if len(history) < self._n_startup:
            return [self._rng.random() for _ in range(dimension)]

        points = [list(trial.trial.unit_point) for trial in history]
        raw_values = [trial.value for trial in history]
        mean_value = sum(raw_values) / len(raw_values)
        targets = [value - mean_value for value in raw_values]

        size = len(points)
        gram = [
            [
                self._kernel(points[i], points[j]) + (self._noise if i == j else 0.0)
                for j in range(size)
            ]
            for i in range(size)
        ]
        lower = cholesky(gram)
        alpha = cholesky_solve(lower, targets)
        best = min(raw_values)

        best_point: list[float] = []
        best_acquisition = float("-inf")
        for _ in range(self._n_candidates):
            candidate = [self._rng.random() for _ in range(dimension)]
            acquisition = self._expected_improvement(
                candidate, points, lower, alpha, mean_value, best
            )
            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_point = candidate
        return best_point

    def _expected_improvement(
        self,
        candidate: Sequence[float],
        points: Sequence[Sequence[float]],
        lower: list[list[float]],
        alpha: Sequence[float],
        mean_value: float,
        best: float,
    ) -> float:
        k_star = [self._kernel(candidate, point) for point in points]
        mean = mean_value + sum(k_star[i] * alpha[i] for i in range(len(points)))
        v = cholesky_solve(lower, k_star)
        variance = self._signal_variance - sum(k_star[i] * v[i] for i in range(len(points)))
        sigma = math.sqrt(max(variance, 1e-12))
        # Expected improvement for minimization over the incumbent ``best``.
        improvement = best - mean
        z = improvement / sigma
        return improvement * _standard_normal_cdf(z) + sigma * _standard_normal_pdf(z)
