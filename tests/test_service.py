"""Tests for the dependency-free HTTP optimization service (pure handler layer)."""

from __future__ import annotations

from hyperparameter_optimizer.service import OptimizerService, handle

_SERVICE = OptimizerService.build()


def test_health() -> None:
    status, payload = handle("GET", "/health", {}, _SERVICE)
    assert status == 200
    assert payload["status"] == "ok"
    assert "branin" in payload["benchmarks"]
    assert "gp" in payload["samplers"]


def test_optimize_returns_regret_and_convergence() -> None:
    status, payload = handle(
        "POST", "/optimize", {"benchmark": "branin", "sampler": "gp", "trials": 30}, _SERVICE
    )
    assert status == 200
    assert payload["simple_regret"] >= 0.0
    assert len(payload["convergence"]) == 30
    assert payload["convergence"] == sorted(payload["convergence"], reverse=True)


def test_evaluate_table() -> None:
    status, payload = handle("GET", "/evaluate", {}, _SERVICE)
    assert status == 200
    assert any(cell["sampler"] == "gp" for cell in payload["cells"])


def test_optimize_rejects_unknown_benchmark() -> None:
    status, payload = handle("POST", "/optimize", {"benchmark": "nope"}, _SERVICE)
    assert status == 400
    assert "benchmark" in payload["error"]


def test_optimize_rejects_bad_trials() -> None:
    status, _ = handle("POST", "/optimize", {"benchmark": "branin", "trials": 0}, _SERVICE)
    assert status == 400


def test_unknown_route() -> None:
    status, payload = handle("GET", "/nope", {}, _SERVICE)
    assert status == 404
    assert "error" in payload
