# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## Unreleased

### Features

- Add search space, trials, random/quasi-random samplers, and the study
- Add TPE and Gaussian-process Bayesian optimization samplers
- Add benchmark objectives, sampler evaluation, and command line interface
- Add HTTP service, benchmarks, and full documentation
- Add production observability (metrics and readiness) and reword footer

### Documentation

- Use plain, factual wording (#12)
- Standard README structure and language framing (#14)
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Recorded demos of more commands, with their vhs tapes (#1)
- Recorded demos of the HTTP service and more commands (#2)

### Build and CI

- Run CodeQL on main and on a weekly schedule only

### Maintenance

- Scaffold hyperparameter-optimizer project
- Bump continuous integration action versions and dev dependencies
- Remove Dependabot configuration
