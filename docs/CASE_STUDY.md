# End-to-End Case Study - Tuning Under a Fixed Trial Budget

How to spend a small, expensive evaluation budget well, and why the sampler choice is the
whole decision.

## Scenario

A team can afford only **40 evaluations** of an expensive objective (each is a model training
run). The question is which optimizer reaches the best configuration within that budget.

## Running each sampler

```bash
python -m hyperparameter_optimizer optimize branin --sampler random --trials 40
python -m hyperparameter_optimizer optimize branin --sampler gp --trials 40
```

Using the Branin function as a stand-in for the expensive objective (global minimum 0.3979):

| Sampler | best value | simple regret |
|---|---:|---:|
| random | about 4.42 | about 4.02 |
| sobol | about 1.51 | about 1.11 |
| tpe | about 0.57 | about 0.17 |
| gp | about 0.41 | about 0.016 |

With the **same 40 evaluations**, the Gaussian process finds a configuration **two orders of
magnitude closer to optimal** than random search. When each evaluation is a training run,
that is the difference between a good model and a wasted budget.

## The cost side of the decision

From the sample-efficiency table, the Gaussian process spends about **139 ms** of optimizer
time versus **0.1 ms** for random search on this problem. That overhead is negligible next to
a real training run, so the surrogate's cost is free in practice. If the objective were itself
near-instant, the cheaper samplers would win.

## What an operator sees

- **A convergence curve.** The `/optimize` response includes best-so-far after each trial, so
  a caller can watch progress and stop early.
- **A reported regret.** Because the benchmarks have known optima, the service reports how far
  from optimal the run ended, not just the raw value.
- **Reproducibility.** A fixed seed makes the whole study regenerate exactly, so a reported
  result can be checked.

## Takeaway

Under an expensive, limited budget, model-based optimization is not a nicety - it is the
decision that determines whether the budget produces a good configuration at all.
