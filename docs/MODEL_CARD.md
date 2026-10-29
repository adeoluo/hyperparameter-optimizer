# System Card - Hyperparameter Optimizer

## Overview

- **What it is:** a black-box hyperparameter optimization library - random, quasi-random,
  Tree-structured Parzen Estimator, and Gaussian-process Bayesian optimization.
- **What it is not:** a training framework. It proposes configurations and consumes scalar
  objective values; the caller evaluates them.
- **Version:** 0.1.0.

## Intended use

- **Primary use cases:** tuning hyperparameters of any system with a scalar objective;
  learning how model-based optimization works from a transparent reference.
- **Primary users:** engineers and researchers tuning models or systems.

## Out-of-scope / prohibited use

- Not intended for very high-dimensional spaces without added structure.
- The naive Gaussian process is cubic in the trial count; use modest budgets.

## Performance

See the README: on smooth benchmarks the Gaussian process reaches near-optimal regret where
random search is far away (Branin 0.045 vs 1.601), at a compute cost documented in the
sample-efficiency tradeoff table.

## Risks and limitations

- A fixed Gaussian-process length scale, not fitted by marginal likelihood.
- Highly multimodal objectives defeat all samplers at small budgets.

## Safety mitigations

- Deterministic under a fixed seed; offline; bounded trial budgets in the service.

## Environmental / cost footprint

Negligible; all in-memory with no model calls.
