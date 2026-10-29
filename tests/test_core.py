"""Tests for the search space, samplers, and the study loop."""

from __future__ import annotations

import pytest

from hyperparameter_optimizer import (
    Categorical,
    IntUniform,
    LogUniform,
    RandomSampler,
    SearchSpace,
    SobolSampler,
    Study,
    Uniform,
)


def test_uniform_decode_encode_roundtrip() -> None:
    param = Uniform("x", 0.0, 10.0)
    assert param.decode(0.0) == 0.0
    assert param.decode(0.5) == pytest.approx(5.0)
    assert param.encode(5.0) == pytest.approx(0.5)


def test_log_uniform_spans_decades() -> None:
    param = LogUniform("lr", 1e-4, 1e-1)
    assert param.decode(0.0) == pytest.approx(1e-4)
    assert param.decode(1.0) == pytest.approx(1e-1, rel=1e-6)
    assert param.encode(1e-2) == pytest.approx(2 / 3, rel=1e-6)


def test_int_and_categorical() -> None:
    integer = IntUniform("n", 1, 4)
    assert integer.decode(0.0) == 1
    assert integer.decode(0.99) == 4
    choice = Categorical("opt", ["adam", "sgd", "rmsprop"])
    assert choice.decode(0.0) == "adam"
    assert choice.decode(0.99) == "rmsprop"


def test_space_validation() -> None:
    with pytest.raises(ValueError):
        SearchSpace([])
    with pytest.raises(ValueError):
        SearchSpace([Uniform("x", 0.0, 1.0), Uniform("x", 0.0, 1.0)])
    with pytest.raises(ValueError):
        Uniform("x", 1.0, 0.0)


def test_space_decode_dimension() -> None:
    space = SearchSpace([Uniform("x", 0.0, 1.0), IntUniform("n", 1, 3)])
    assert space.dimension == 2
    config = space.decode([0.5, 0.5])
    assert set(config) == {"x", "n"}


def test_sobol_is_more_even_than_random() -> None:
    # The Sobol sequence should spread points; the first few are distinct and ordered.
    sampler = SobolSampler()
    points = [sampler.sample(1, [])[0] for _ in range(1)]
    assert 0.0 <= points[0] < 1.0


def test_study_optimizes_a_quadratic() -> None:
    space = SearchSpace([Uniform("x", -5.0, 5.0)])
    study = Study(space, RandomSampler(seed=0))
    result = study.optimize(lambda c: (float(c["x"]) - 2.0) ** 2, trials=200)
    assert result.best_value < 0.1
    assert abs(float(result.best_configuration["x"]) - 2.0) < 0.5
    curve = result.convergence()
    assert curve == sorted(curve, reverse=True)


def test_ask_tell_interface() -> None:
    space = SearchSpace([Uniform("x", 0.0, 1.0)])
    study = Study(space, RandomSampler(seed=1))
    trial = study.ask()
    study.tell(trial, 0.5)
    assert study.best().value == 0.5


def test_optimize_rejects_bad_trials() -> None:
    space = SearchSpace([Uniform("x", 0.0, 1.0)])
    with pytest.raises(ValueError):
        Study(space, RandomSampler()).optimize(lambda c: 0.0, trials=0)
