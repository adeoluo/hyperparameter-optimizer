# ADR-0002: A normalized unit-cube search space

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Hyperparameters are heterogeneous: continuous, log-scaled, integer, and categorical. A
sampler should not need to know about each type.

## Decision

Represent every parameter as a map from the unit interval ``[0, 1)`` to a concrete value.
A search space is then a normalized unit cube; samplers propose points in the cube and the
space decodes them. Encoding is the inverse, so completed trials can be mapped back into the
cube for model-based samplers.

## Alternatives considered

- **Type-aware samplers.** Each sampler handles each parameter type directly. More direct but
  multiplies the code and couples samplers to parameter types.
- **One-hot categorical embedding.** General, but inflates dimensionality and complicates the
  density and kernel models.

## Consequences

- One sampler implementation handles all parameter types.
- The Gaussian process and TPE operate in a clean continuous space.
- Integer and categorical decoding is piecewise constant, which the samplers tolerate.
