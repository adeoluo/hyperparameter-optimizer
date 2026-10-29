"""A dependency-free HTTP service exposing the optimizer.

The service optimizes a chosen benchmark with a chosen sampler and reports the
best configuration and simple regret, or runs the full sampler-versus-benchmark
evaluation. A pure handler is separated from the socket binding for testing.

Endpoints
---------
``GET  /health``    liveness, benchmark names, and sampler names.
``POST /optimize``  body ``{"benchmark", "sampler"?, "trials"?, "seed"?}``;
                    returns the best value, regret, configuration, and convergence.
``GET  /evaluate``  the sampler-by-benchmark simple-regret table.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from hyperparameter_optimizer.evaluation import SAMPLER_NAMES, run_evaluation
from hyperparameter_optimizer.objectives import available_benchmarks, load_benchmark
from hyperparameter_optimizer.observability import ServiceMetrics
from hyperparameter_optimizer.samplers import (
    GaussianProcessSampler,
    RandomSampler,
    SobolSampler,
    TPESampler,
)
from hyperparameter_optimizer.samplers.base import Sampler
from hyperparameter_optimizer.study import Study

JsonObject = dict[str, Any]
_MAX_TRIALS = 500


def _sampler(name: str, seed: int) -> Sampler:
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "sobol":
        return SobolSampler(skip=seed + 1)
    if name == "tpe":
        return TPESampler(seed=seed)
    return GaussianProcessSampler(seed=seed)


@dataclass(frozen=True)
class OptimizerService:
    """Optimizes benchmarks and runs the sampler evaluation."""

    benchmarks: tuple[str, ...]
    samplers: tuple[str, ...]
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def build(cls) -> OptimizerService:
        return cls(benchmarks=tuple(available_benchmarks()), samplers=SAMPLER_NAMES)

    def optimize(self, benchmark_name: str, sampler: str, trials: int, seed: int) -> JsonObject:
        benchmark = load_benchmark(benchmark_name)
        study = Study(benchmark.space, _sampler(sampler, seed))
        result = study.optimize(benchmark.function, trials)
        return {
            "benchmark": benchmark.name,
            "sampler": sampler,
            "trials": trials,
            "optimum": benchmark.optimum,
            "best_value": round(result.best_value, 6),
            "simple_regret": round(result.best_value - benchmark.optimum, 6),
            "best_configuration": {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in result.best_configuration.items()
            },
            "convergence": [round(value, 6) for value in result.convergence()],
        }

    def evaluate(self) -> JsonObject:
        cells = run_evaluation()
        return {
            "cells": [
                {
                    "sampler": cell.sampler,
                    "benchmark": cell.benchmark,
                    "regret": round(cell.regret, 6),
                }
                for cell in cells
            ]
        }


def _validate_optimize(
    body: Mapping[str, Any], service: OptimizerService
) -> tuple[str, str, int, int]:
    benchmark = body.get("benchmark")
    if benchmark not in service.benchmarks:
        raise ValueError(f"'benchmark' must be one of: {', '.join(service.benchmarks)}")
    sampler = body.get("sampler", "gp")
    if sampler not in service.samplers:
        raise ValueError(f"'sampler' must be one of: {', '.join(service.samplers)}")
    trials = body.get("trials", 40)
    if not isinstance(trials, int) or not 1 <= trials <= _MAX_TRIALS:
        raise ValueError(f"'trials' must be an integer in [1, {_MAX_TRIALS}]")
    seed = body.get("seed", 0)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("'seed' must be a non-negative integer")
    return str(benchmark), str(sampler), trials, seed


def _route(
    method: str, path: str, body: Mapping[str, Any], service: OptimizerService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {
            "status": "ok",
            "benchmarks": list(service.benchmarks),
            "samplers": list(service.samplers),
        }
    if method == "GET" and path == "/evaluate":
        return 200, service.evaluate()
    if method == "POST" and path == "/optimize":
        try:
            benchmark, sampler, trials, seed = _validate_optimize(body, service)
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, service.optimize(benchmark, sampler, trials, seed)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: OptimizerService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready"}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: OptimizerService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": OptimizerService.build()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"hyperparameter-optimizer service listening on http://{host}:{port}")
    server.serve_forever()
