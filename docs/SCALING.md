# Scaling and Design Tradeoffs - Hyperparameter Optimizer

How the reference design grows, and the tradeoffs taken.

## Reference architecture (today)

A single process runs an ask/tell study with a pluggable sampler over a search space, served
over a standard-library HTTP service. Deterministic for a fixed seed.

## Scaling path

| Stage | Bottleneck | Change |
|---|---|---|
| One process | Sampler CPU, global interpreter lock | One replica per core |
| Many studies | Independent studies | Run studies across replicas; cache by study key |
| Expensive objective | Objective evaluation, not the sampler | Parallelize trials with constant-liar batching |
| Large budgets | Cubic Gaussian process | Switch to TPE, or add a sparse/approximate GP |
| High dimensions | Surrogate quality | Add structure (groups, priors) or fall back to TPE |

Because studies are independent and deterministic, they cache cleanly and replicas share
nothing.

## Substantial tradeoffs

- **Sample efficiency vs compute per trial.** The Gaussian process reaches far lower regret
  per trial but costs about a thousand times more optimizer time than random search (measured).
  The right choice depends entirely on how expensive an objective evaluation is - the project
  makes that tradeoff explicit rather than hiding it behind one default.
- **Gaussian process vs TPE.** The GP has calibrated uncertainty and the strongest sample
  efficiency on smooth problems but is cubic in trials and sensitive to the length scale. TPE
  is cheaper, scales better, and handles conditional spaces, at some loss of sample efficiency.
  Both are offered behind one protocol.
- **Fixed length scale vs marginal-likelihood fitting.** A fixed length scale keeps the GP
  simple and dependency-free at some loss of adaptivity; fitting it would help but adds an
  inner optimization. Documented as a roadmap item.
- **Sequential vs parallel proposals.** Sequential proposals use every observation but serialize
  evaluations. Constant-liar batching trades a little information for parallelism when the
  objective can be evaluated concurrently.

## What would change under real load

Run studies across process-per-core replicas, cache by study key, parallelize expensive
objective evaluations with constant-liar batching, and switch to TPE or an approximate GP for
large budgets. The study loop and the sampler protocol are unchanged.
