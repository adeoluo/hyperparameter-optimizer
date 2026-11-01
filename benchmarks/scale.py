"""A large-workload scale benchmark for the TPE sampler.

Runs the Tree-structured Parzen Estimator through progressively larger trial
budgets on the multimodal Rastrigin objective and reports, per workload, the
wall-clock time, throughput in trials per second, the peak traced memory, and the
simple regret reached (best value found minus the known global optimum). It
complements ``benchmark.py`` (a fixed 40-trial comparison) by exposing the central
scaling tension of model-based search: more trials lower the regret, but the
per-trial cost of a pure-Python TPE grows with the accumulated history, so
throughput falls as the search deepens.

Timing and memory are captured in separate phases on purpose. ``tracemalloc``
instruments every allocation and dominates the wall-clock of these numeric loops,
so it would distort the throughput numbers; the peak footprint tracks the growing
trial history, so the traced phase runs the full budget to measure it faithfully.
Deterministic and offline; run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from hyperparameter_optimizer.objectives import Benchmark, load_benchmark
from hyperparameter_optimizer.samplers import TPESampler
from hyperparameter_optimizer.study import Study

WORKLOADS: tuple[int, ...] = (50, 100, 200)
_BENCHMARK = "rastrigin"
_SEED = 0


@dataclass(frozen=True)
class ScalePoint:
    """One row of the scale table for a single trial budget."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    regret: float


def measure(benchmark: Benchmark, trials: int, *, seed: int = _SEED) -> ScalePoint:
    """Run ``trials`` TPE iterations on ``benchmark`` and measure the cost."""

    tracemalloc.start()
    tracemalloc.reset_peak()
    Study(benchmark.space, TPESampler(seed=seed)).optimize(benchmark.function, trials)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    start = time.perf_counter()
    result = Study(benchmark.space, TPESampler(seed=seed)).optimize(benchmark.function, trials)
    seconds = time.perf_counter() - start

    return ScalePoint(
        size=trials,
        seconds=seconds,
        throughput=trials / seconds if seconds > 0 else 0.0,
        peak_mib=peak / 1048576,
        regret=result.best_value - benchmark.optimum,
    )


def main() -> None:
    print("Synthetic Rastrigin objective; a pure-Python, from-scratch TPE sampler.")
    print("Wall-clock is not competitive with compiled numerical libraries.")
    print("Regret is the best value found minus the known global optimum.\n")

    benchmark = load_benchmark(_BENCHMARK)

    print(f"{'trials':>8}  {'seconds':>8}  {'trials/sec':>10}  {'peak MiB':>9}  {'regret':>8}")
    for size in WORKLOADS:
        point = measure(benchmark, size)
        print(
            f"{point.size:>8}  {point.seconds:>8.3f}  {point.throughput:>10.0f}  "
            f"{point.peak_mib:>9.2f}  {point.regret:>8.4f}"
        )


if __name__ == "__main__":
    main()
