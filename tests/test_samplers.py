"""Tests for the TPE and Gaussian-process samplers and the linear-algebra kernel."""

from __future__ import annotations

import math

import pytest

from hyperparameter_optimizer import (
    GaussianProcessSampler,
    RandomSampler,
    SearchSpace,
    Study,
    TPESampler,
    Uniform,
)
from hyperparameter_optimizer.linalg import cholesky, cholesky_solve


def _branin_space() -> SearchSpace:
    return SearchSpace([Uniform("x", -5.0, 10.0), Uniform("y", 0.0, 15.0)])


def _branin(config: dict[str, object]) -> float:
    x, y = float(config["x"]), float(config["y"])
    a, b, c, r, s, t = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi, 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (y - b * x * x + c * x - r) ** 2 + s * (1 - t) * math.cos(x) + s


def test_cholesky_solves_a_system() -> None:
    matrix = [[4.0, 2.0], [2.0, 3.0]]
    lower = cholesky(matrix)
    # Reconstruct the matrix from its factor.
    reconstructed = [
        [sum(lower[i][k] * lower[j][k] for k in range(2)) for j in range(2)] for i in range(2)
    ]
    assert reconstructed[0][0] == pytest.approx(4.0)
    assert reconstructed[1][1] == pytest.approx(3.0)
    solution = cholesky_solve(lower, [1.0, 1.0])
    assert sum(matrix[0][k] * solution[k] for k in range(2)) == pytest.approx(1.0)


def test_tpe_beats_random_on_branin() -> None:
    random_best = Study(_branin_space(), RandomSampler(seed=0)).optimize(_branin, 40).best_value
    tpe_best = Study(_branin_space(), TPESampler(seed=0)).optimize(_branin, 40).best_value
    assert tpe_best < random_best
    assert tpe_best < 1.0  # near the global optimum of about 0.397


def test_gp_beats_random_on_branin() -> None:
    random_best = Study(_branin_space(), RandomSampler(seed=0)).optimize(_branin, 40).best_value
    gp_best = (
        Study(_branin_space(), GaussianProcessSampler(seed=0)).optimize(_branin, 40).best_value
    )
    assert gp_best < random_best
    assert gp_best < 1.0


def test_tpe_is_deterministic() -> None:
    first = Study(_branin_space(), TPESampler(seed=7)).optimize(_branin, 20).best_value
    second = Study(_branin_space(), TPESampler(seed=7)).optimize(_branin, 20).best_value
    assert first == second


def test_gp_is_deterministic() -> None:
    first = Study(_branin_space(), GaussianProcessSampler(seed=3)).optimize(_branin, 20).best_value
    second = Study(_branin_space(), GaussianProcessSampler(seed=3)).optimize(_branin, 20).best_value
    assert first == second


def test_samplers_validate_parameters() -> None:
    with pytest.raises(ValueError):
        TPESampler(gamma=0.0)
    with pytest.raises(ValueError):
        TPESampler(n_startup=0)
    with pytest.raises(ValueError):
        GaussianProcessSampler(length_scale=0.0)
    with pytest.raises(ValueError):
        GaussianProcessSampler(n_candidates=0)


def test_samplers_fall_back_to_random_before_startup() -> None:
    sampler = GaussianProcessSampler(n_startup=5)
    point = sampler.sample(2, [])
    assert len(point) == 2
    assert all(0.0 <= value < 1.0 for value in point)
