# ADR-0003: A from-scratch Gaussian process

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Bayesian optimization needs a surrogate with calibrated uncertainty. A Gaussian process is
the standard choice, but importing one would add a heavy dependency and hide the machinery
that is the point of the project.

## Decision

Implement the Gaussian process directly: a squared-exponential kernel, a Cholesky
factorization and triangular solves for the posterior, and the Expected Improvement
acquisition, all in pure Python.

## Alternatives considered

- **A library Gaussian process (scikit-learn, GPyTorch).** Faster and more featureful, but a
  large dependency and a black box for a learning-focused reference.
- **A random-forest surrogate (SMAC-style).** Cheaper on high dimensions but a less exact probabilistic model
  about uncertainty; the Parzen-estimator sampler already covers the non-GP approach.

## Consequences

- The project stays dependency-free and the Bayesian-optimization math is fully inspectable.
- The naive Cholesky is cubic in the number of trials, which is fine for hyperparameter
  budgets (tens to low hundreds of trials) and documented as a limit.
- A fixed length scale is used; marginal-likelihood fitting is left as a roadmap item.
