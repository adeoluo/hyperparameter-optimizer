"""Command line interface for the hyperparameter optimizer."""

from __future__ import annotations

import argparse

from hyperparameter_optimizer.evaluation import SAMPLER_NAMES, format_table, run_evaluation
from hyperparameter_optimizer.objectives import available_benchmarks, load_benchmark
from hyperparameter_optimizer.samplers import (
    GaussianProcessSampler,
    RandomSampler,
    SobolSampler,
    TPESampler,
)
from hyperparameter_optimizer.samplers.base import Sampler
from hyperparameter_optimizer.study import Study


def _sampler(name: str, seed: int) -> Sampler:
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "sobol":
        return SobolSampler(skip=seed + 1)
    if name == "tpe":
        return TPESampler(seed=seed)
    return GaussianProcessSampler(seed=seed)


def _cmd_benchmarks(_: argparse.Namespace) -> None:
    print("Benchmarks:")
    for name in available_benchmarks():
        print(f"  - {name} (optimum {load_benchmark(name).optimum:.4f})")
    print("Samplers:")
    for name in SAMPLER_NAMES:
        print(f"  - {name}")


def _cmd_optimize(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    study = Study(benchmark.space, _sampler(args.sampler, args.seed))
    result = study.optimize(benchmark.function, args.trials)
    print(f"Benchmark: {benchmark.name} (optimum {benchmark.optimum:.4f})")
    print(f"Sampler: {args.sampler}  Trials: {args.trials}")
    print(f"Best value: {result.best_value:.6f}")
    print(f"Simple regret: {result.best_value - benchmark.optimum:.6f}")
    print("Best configuration:")
    for key, value in result.best_configuration.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key} = {formatted}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    cells = run_evaluation(trials=args.trials)
    print(format_table(cells))


def _cmd_serve(args: argparse.Namespace) -> None:
    from hyperparameter_optimizer.service import serve

    serve(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hyperparameter-optimizer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("benchmarks", help="list built-in benchmarks and samplers")

    optimize = sub.add_parser("optimize", help="optimize one benchmark with one sampler")
    optimize.add_argument("benchmark", choices=available_benchmarks())
    optimize.add_argument("--sampler", choices=SAMPLER_NAMES, default="gp")
    optimize.add_argument("--trials", type=int, default=40)
    optimize.add_argument("--seed", type=int, default=0)

    evaluate = sub.add_parser("evaluate", help="compare all samplers across benchmarks")
    evaluate.add_argument("--trials", type=int, default=40)

    serve = sub.add_parser("serve", help="start the HTTP optimization service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {
        "benchmarks": _cmd_benchmarks,
        "optimize": _cmd_optimize,
        "evaluate": _cmd_evaluate,
        "serve": _cmd_serve,
    }
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
