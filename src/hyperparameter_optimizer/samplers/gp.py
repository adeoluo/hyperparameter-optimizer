"""Gaussian-Process Bayesian optimization with Expected Improvement.

A from-scratch Gaussian process regressor with a squared-exponential (RBF) kernel
provides a posterior mean and variance at any point in the unit cube. The sampler
proposes the candidate that maximizes **Expected Improvement** over the best value
seen, the canonical Bayesian-optimization acquisition that balances exploiting
low-mean regions against exploring high-variance ones.

The kernel matrix, the posterior and the acquisition are written out here and evaluated
with numpy: the Gram matrix is factored once per proposal (LAPACK Cholesky, with the same
jitter fallback as :func:`hyperparameter_optimizer.linalg.cholesky`), and every candidate is
scored in one batch. The posterior variance needs only the forward solve ``v = L^-1 k*``, since
``k*^T K^-1 k* = |v|^2``. Candidates are drawn in the same order from the same seeded
generator as before, so proposals match the earlier pure-Python sampler up to floating-point
rounding.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from hyperparameter_optimizer.linalg import cholesky
from hyperparameter_optimizer.trial import CompletedTrial

Array = npt.NDArray[np.float64]


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

    def _kernel(self, a: Array, b: Array) -> Array:
        """RBF kernel between every row of ``a`` and every row of ``b``."""
        squared = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        result: Array = self._signal_variance * np.exp(-0.5 * squared / self._length_scale**2)
        return result

    def _factor(self, gram: Array) -> Array:
        try:
            lower: Array = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError:
            # Not numerically positive definite: fall back to the factorization that
            # clamps non-positive pivots.
            lower = np.asarray(cholesky(gram.tolist()), dtype=np.float64)
        return lower

    def sample(self, dimension: int, history: Sequence[CompletedTrial]) -> list[float]:
        if len(history) < self._n_startup:
            return [self._rng.random() for _ in range(dimension)]

        points = np.asarray([trial.trial.unit_point for trial in history], dtype=np.float64)
        raw_values = [trial.value for trial in history]
        mean_value = sum(raw_values) / len(raw_values)
        targets = np.asarray([value - mean_value for value in raw_values], dtype=np.float64)

        gram = self._kernel(points, points) + self._noise * np.eye(len(points))
        lower = self._factor(gram)
        alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, targets))
        best = min(raw_values)

        candidates = [
            [self._rng.random() for _ in range(dimension)] for _ in range(self._n_candidates)
        ]
        k_star = self._kernel(np.asarray(candidates, dtype=np.float64), points)
        means = mean_value + k_star @ alpha
        v = np.linalg.solve(lower, k_star.T)
        variances = self._signal_variance - (v * v).sum(axis=0)
        acquisition = [
            self._expected_improvement(float(mean), float(variance), best)
            for mean, variance in zip(means, variances, strict=True)
        ]
        # First maximum wins, as in a strict ``>`` scan.
        return candidates[int(np.argmax(acquisition))]

    @staticmethod
    def _expected_improvement(mean: float, variance: float, best: float) -> float:
        sigma = math.sqrt(max(variance, 1e-12))
        # Expected improvement for minimization over the incumbent ``best``.
        improvement = best - mean
        z = improvement / sigma
        return improvement * _standard_normal_cdf(z) + sigma * _standard_normal_pdf(z)
