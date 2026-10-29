# Operations Guide - Hyperparameter Optimizer

The operational contract of the optimization service.

## Running the service

```bash
uv run python -m hyperparameter_optimizer serve --host 0.0.0.0 --port 8000
```

A standard-library HTTP server with **zero runtime dependencies**.

## Endpoints

| Method | Path | Body | Success |
|---|---|---|---|
| GET | `/health` | none | status, benchmark names, sampler names |
| GET | `/ready` | none | readiness of the optimizer |
| GET | `/metrics` | none | per-route request, error, and latency metrics (Prometheus text) |
| POST | `/optimize` | `{"benchmark", "sampler"?, "trials"?, "seed"?}` | best value, regret, configuration, convergence |
| GET | `/evaluate` | none | the sampler-by-benchmark regret table |

Errors return JSON `{"error": ...}` with a 4xx status. Benchmark and sampler names must be
known; `trials` is bounded.

## Service level objectives

Measured by `benchmarks/stress.py` (each request runs a 25-trial study) on a developer laptop:

| Objective | Target | Observed (single worker) |
|---|---|---|
| Optimize latency p50 (TPE, 25 trials) | < 10 ms | about 6 ms |
| Optimize latency p99 (1 worker) | < 10 ms | about 6 ms |
| Throughput (1 worker) | > 100 studies/second | about 170/second |

Latency depends heavily on the sampler and trial count: the Gaussian process is cubic in the
trial count, so a 40-trial GP study is far slower than a 25-trial TPE study.

## Operational constraints

- **Cost is dominated by the sampler and the trial count, not request overhead.** The
  Gaussian process is cubic in trials (the Cholesky solve); TPE is roughly quadratic in its
  candidate evaluations; random and Sobol are constant. Choose the sampler for the budget.
- **The optimizer is CPU bound and holds the global interpreter lock.** Throughput is flat
  across threads and the p99 tail grows with concurrency; scale with **processes**.
- **Studies are deterministic for a fixed seed**, so results are reproducible and cacheable by
  `(benchmark, sampler, trials, seed)`.
- **Trials are bounded** (the service caps at 500), so a single request cannot run unbounded
  cubic work.
- **No persistence, no authentication.** Front the service with a gateway.

## Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| Slow optimize | Gaussian process with many trials | Use TPE, or cap trials; the GP is cubic |
| High p99 under load | Concurrency above one worker per core | Add replicas; cap concurrency |
| Poor regret on a goal | Multimodal objective | Expected on Rastrigin-like surfaces; add restarts or budget |
| `400` on optimize | Unknown benchmark/sampler or bad trials | Inspect the `error` field |

## Capacity planning

Throughput depends on the sampler. Size replicas for the heaviest sampler and budget a client
may request, or route Gaussian-process requests to a dedicated pool.
