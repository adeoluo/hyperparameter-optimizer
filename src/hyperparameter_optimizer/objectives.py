"""Standard black-box objective functions used to benchmark samplers.

These are the classic optimization test functions (Branin, Rastrigin, Ackley,
sphere), each wrapped with its search space and known global minimum, so a
sampler's sample efficiency can be measured against ground truth. All are framed
as minimization.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from hyperparameter_optimizer.space import SearchSpace, Uniform, Value

ObjectiveFn = Callable[[Mapping[str, Value]], float]


@dataclass(frozen=True)
class Benchmark:
    """A named objective with its search space and known global minimum."""

    name: str
    space: SearchSpace
    function: ObjectiveFn
    optimum: float


def _sphere(config: Mapping[str, Value]) -> float:
    return float(config["x"]) ** 2 + float(config["y"]) ** 2


def _branin(config: Mapping[str, Value]) -> float:
    x, y = float(config["x"]), float(config["y"])
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (y - b * x * x + c * x - r) ** 2 + s * (1 - t) * math.cos(x) + s


def _rastrigin(config: Mapping[str, Value]) -> float:
    values = [float(config["x"]), float(config["y"])]
    return 20.0 + sum(v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in values)


def _ackley(config: Mapping[str, Value]) -> float:
    x, y = float(config["x"]), float(config["y"])
    term1 = -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (x * x + y * y)))
    term2 = -math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
    return term1 + term2 + math.e + 20.0


_BENCHMARKS: dict[str, Benchmark] = {
    "sphere": Benchmark(
        "sphere",
        SearchSpace([Uniform("x", -5.0, 5.0), Uniform("y", -5.0, 5.0)]),
        _sphere,
        optimum=0.0,
    ),
    "branin": Benchmark(
        "branin",
        SearchSpace([Uniform("x", -5.0, 10.0), Uniform("y", 0.0, 15.0)]),
        _branin,
        optimum=0.397887,
    ),
    "rastrigin": Benchmark(
        "rastrigin",
        SearchSpace([Uniform("x", -5.12, 5.12), Uniform("y", -5.12, 5.12)]),
        _rastrigin,
        optimum=0.0,
    ),
    "ackley": Benchmark(
        "ackley",
        SearchSpace([Uniform("x", -5.0, 5.0), Uniform("y", -5.0, 5.0)]),
        _ackley,
        optimum=0.0,
    ),
}


def available_benchmarks() -> list[str]:
    """Return the names of the built-in benchmark objectives."""
    return list(_BENCHMARKS)


def load_benchmark(name: str) -> Benchmark:
    """Return a benchmark by name, raising ``KeyError`` when unknown."""
    if name not in _BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; choose from {', '.join(_BENCHMARKS)}")
    return _BENCHMARKS[name]
