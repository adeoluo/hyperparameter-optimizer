"""Evaluate sampler sample-efficiency across the benchmark objectives.

For each sampler and benchmark, run a fixed trial budget and record the simple
regret (best value found minus the known global minimum). Averaged over several
seeds, this is the standard way to show that model-based optimization reaches a
good solution in fewer trials than random search.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hyperparameter_optimizer.objectives import Benchmark, available_benchmarks, load_benchmark
from hyperparameter_optimizer.samplers import (
    GaussianProcessSampler,
    RandomSampler,
    SobolSampler,
    TPESampler,
)
from hyperparameter_optimizer.samplers.base import Sampler
from hyperparameter_optimizer.study import Study


def _make_sampler(name: str, seed: int) -> Sampler:
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "sobol":
        return SobolSampler(skip=seed + 1)
    if name == "tpe":
        return TPESampler(seed=seed)
    if name == "gp":
        return GaussianProcessSampler(seed=seed)
    raise KeyError(f"unknown sampler {name!r}")


SAMPLER_NAMES: tuple[str, ...] = ("random", "sobol", "tpe", "gp")


@dataclass(frozen=True)
class Cell:
    """The mean simple regret of one sampler on one benchmark."""

    sampler: str
    benchmark: str
    regret: float


def _regret(sampler_name: str, benchmark: Benchmark, trials: int, seeds: Sequence[int]) -> float:
    total = 0.0
    for seed in seeds:
        study = Study(benchmark.space, _make_sampler(sampler_name, seed))
        result = study.optimize(benchmark.function, trials)
        total += result.best_value - benchmark.optimum
    return total / len(seeds)


def run_evaluation(
    *,
    trials: int = 40,
    seeds: Sequence[int] = (0, 1, 2),
    benchmarks: Sequence[str] | None = None,
    samplers: Sequence[str] = SAMPLER_NAMES,
) -> list[Cell]:
    """Compute mean simple regret for each sampler on each benchmark."""
    names = list(benchmarks) if benchmarks is not None else available_benchmarks()
    cells: list[Cell] = []
    for benchmark_name in names:
        benchmark = load_benchmark(benchmark_name)
        for sampler_name in samplers:
            regret = _regret(sampler_name, benchmark, trials, seeds)
            cells.append(Cell(sampler_name, benchmark_name, regret))
    return cells


def format_table(cells: Sequence[Cell]) -> str:
    """Render the evaluation as a sampler-by-benchmark regret table."""
    benchmarks = sorted({cell.benchmark for cell in cells})
    samplers = sorted({cell.sampler for cell in cells})
    lookup = {(cell.sampler, cell.benchmark): cell.regret for cell in cells}
    header = f"{'sampler':<10}" + "".join(f"{name:>12}" for name in benchmarks)
    lines = [header, "-" * len(header)]
    for sampler in samplers:
        row = f"{sampler:<10}" + "".join(
            f"{lookup[(sampler, benchmark)]:>12.4f}" for benchmark in benchmarks
        )
        lines.append(row)
    lines.append("")
    lines.append(
        "Values are mean simple regret (best found minus global optimum); lower is better."
    )
    return "\n".join(lines)
