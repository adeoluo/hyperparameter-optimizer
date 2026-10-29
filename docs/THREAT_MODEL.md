# Threat Model - Hyperparameter Optimizer

## Assets

- Integrity of the proposed configurations and the reported regret and convergence.

## Trust boundaries

- **Benchmark names, sampler names, and numeric bounds are untrusted input to the service.**
  They are validated against allowed values; the objective is a bundled function, never
  caller-supplied code over the wire.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Unbounded compute | Huge trial count | Resource exhaustion | Trials bounded (service caps at 500) |
| Irreproducible results | Unseeded sampling | Flaky numbers | Deterministic, seeded samplers |
| Numerical failure | Ill-conditioned kernel matrix | Crash or wrong posterior | Jitter on the Cholesky diagonal |
| Supply-chain compromise | Malicious dependency | Code execution | Zero runtime dependencies; Dependabot and CodeQL in continuous integration |

## Residual risk

The Gaussian process is cubic in the trial count, so very large budgets are slow; the service
caps trials and the limit is documented rather than hidden.
