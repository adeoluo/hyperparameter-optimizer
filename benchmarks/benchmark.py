"""A deterministic micro-benchmark for the samplers.

Measures each sampler's wall-clock time and the simple regret it reaches on the
Branin function in a fixed trial budget, so the README can quote concrete
numbers. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from hyperparameter_optimizer.evaluation import SAMPLER_NAMES
from hyperparameter_optimizer.objectives import load_benchmark
from hyperparameter_optimizer.samplers import (
    GaussianProcessSampler,
    RandomSampler,
    SobolSampler,
    TPESampler,
)
from hyperparameter_optimizer.samplers.base import Sampler
from hyperparameter_optimizer.study import Study


def _sampler(name: str) -> Sampler:
    if name == "random":
        return RandomSampler(seed=0)
    if name == "sobol":
        return SobolSampler()
    if name == "tpe":
        return TPESampler(seed=0)
    return GaussianProcessSampler(seed=0)


def main() -> None:
    benchmark = load_benchmark("branin")
    trials = 40
    print(f"Benchmark: branin (optimum {benchmark.optimum:.4f}) | Trials: {trials}\n")
    print(f"{'sampler':<10}{'time (ms)':>12}{'regret':>12}")
    print("-" * 34)
    for name in SAMPLER_NAMES:
        start = time.perf_counter()
        result = Study(benchmark.space, _sampler(name)).optimize(benchmark.function, trials)
        elapsed_ms = (time.perf_counter() - start) * 1000
        regret = result.best_value - benchmark.optimum
        print(f"{name:<10}{elapsed_ms:>12.1f}{regret:>12.4f}")


if __name__ == "__main__":
    main()
