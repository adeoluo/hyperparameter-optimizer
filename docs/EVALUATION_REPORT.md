# Evaluation Report - Hyperparameter Optimizer

> Reproduce with `python -m hyperparameter_optimizer evaluate`. Deterministic and offline.

## Setup

- **Objectives:** sphere, Branin, Rastrigin, Ackley, each with a known global minimum.
- **Metric:** mean simple regret (best value found minus the optimum) over three seeds.
- **Budget:** 40 trials per run.
- **Samplers:** random, Sobol, TPE, Gaussian process.

## Results

| Sampler | sphere | branin | ackley | rastrigin |
|---|---:|---:|---:|---:|
| random | 0.347 | 1.601 | 2.279 | 3.992 |
| sobol | 0.913 | 1.110 | 3.312 | 7.569 |
| tpe | 0.140 | 0.426 | 3.144 | 5.259 |
| gp | 0.011 | 0.045 | 1.127 | 4.630 |

## Interpretation

On the smooth and moderately deceptive objectives the Gaussian process is the clear winner,
reaching near-optimal regret in the same budget where random search is still far away (Branin
0.045 vs 1.601, a 35x improvement). TPE is a strong second and much cheaper per trial. On the
pathologically multimodal Rastrigin function every sampler struggles and random search is
competitive, which is the honest and expected outcome: a smooth surrogate cannot model a
surface with a minimum in every lattice cell at this budget. The sample-efficiency versus
compute tradeoff (see the README) is the practical takeaway: pay the surrogate's cost when
objective evaluations are expensive.

## Reproducibility

All samplers are seeded; the numbers regenerate exactly.
