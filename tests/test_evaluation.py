"""Tests for the benchmark objectives, evaluation, and command line interface."""

from __future__ import annotations

import math

import pytest

from hyperparameter_optimizer.cli import main
from hyperparameter_optimizer.evaluation import format_table, run_evaluation
from hyperparameter_optimizer.objectives import available_benchmarks, load_benchmark


def test_benchmarks_known_optima() -> None:
    assert "branin" in available_benchmarks()
    branin = load_benchmark("branin")
    # Evaluate near the known global minimizer.
    value = branin.function({"x": -math.pi, "y": 12.275})
    assert value == pytest.approx(branin.optimum, abs=1e-3)
    with pytest.raises(KeyError):
        load_benchmark("nope")


def test_evaluation_model_based_beats_random_on_average() -> None:
    cells = run_evaluation(trials=40, seeds=(0, 1, 2))
    by = {(cell.sampler, cell.benchmark): cell.regret for cell in cells}
    # The Gaussian process should clearly beat random on the smooth benchmarks.
    for benchmark in ("sphere", "branin", "ackley"):
        assert by[("gp", benchmark)] < by[("random", benchmark)]


def test_format_table_contains_samplers() -> None:
    table = format_table(run_evaluation(trials=20, seeds=(0,)))
    assert "gp" in table
    assert "regret" in table.lower()


def test_cli_benchmarks(capsys) -> None:
    main(["benchmarks"])
    out = capsys.readouterr().out
    assert "branin" in out


def test_cli_optimize(capsys) -> None:
    main(["optimize", "branin", "--sampler", "gp", "--trials", "30"])
    out = capsys.readouterr().out
    assert "Simple regret" in out
    assert "Best configuration" in out


def test_cli_evaluate(capsys) -> None:
    main(["evaluate", "--trials", "20"])
    out = capsys.readouterr().out
    assert "sampler" in out
