"""The Study: the optimization loop and its ask/tell interface.

A :class:`Study` drives a sampler over a search space, minimizing a scalar
objective. It supports two styles:

* ``optimize(objective, trials)`` - the batteries-included loop.
* ``ask()`` / ``tell(trial, value)`` - a define-by-run interface for callers
  that evaluate trials themselves (across processes, with early stopping, etc.).

Objectives are minimized; maximize by negating. Everything is deterministic given
a seeded sampler.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from hyperparameter_optimizer.samplers.base import Sampler
from hyperparameter_optimizer.space import SearchSpace, Value
from hyperparameter_optimizer.trial import CompletedTrial, Trial

Objective = Callable[[Mapping[str, Value]], float]


@dataclass(frozen=True)
class StudyResult:
    """The outcome of a study: the best trial and the full history."""

    best: CompletedTrial
    history: tuple[CompletedTrial, ...]

    @property
    def best_value(self) -> float:
        return self.best.value

    @property
    def best_configuration(self) -> dict[str, Value]:
        return self.best.configuration

    def convergence(self) -> list[float]:
        """The best-so-far value after each trial (a monotonically decreasing curve)."""
        curve: list[float] = []
        best = float("inf")
        for completed in self.history:
            best = min(best, completed.value)
            curve.append(best)
        return curve


class Study:
    """Minimizes a scalar objective over a search space using a sampler."""

    def __init__(self, space: SearchSpace, sampler: Sampler) -> None:
        self._space = space
        self._sampler = sampler
        self._history: list[CompletedTrial] = []
        self._counter = 0

    @property
    def history(self) -> tuple[CompletedTrial, ...]:
        return tuple(self._history)

    def ask(self) -> Trial:
        """Propose the next trial to evaluate."""
        unit_point = self._sampler.sample(self._space.dimension, tuple(self._history))
        configuration = self._space.decode(unit_point)
        trial = Trial(
            number=self._counter,
            unit_point=tuple(unit_point),
            configuration=configuration,
        )
        self._counter += 1
        return trial

    def tell(self, trial: Trial, value: float, intermediate: tuple[float, ...] = ()) -> None:
        """Report the objective value of a trial."""
        self._history.append(CompletedTrial(trial=trial, value=value, intermediate=intermediate))

    def best(self) -> CompletedTrial:
        """The best completed trial so far."""
        if not self._history:
            raise ValueError("no trials have been completed")
        return min(self._history, key=lambda completed: completed.value)

    def optimize(self, objective: Objective, trials: int) -> StudyResult:
        """Run ``trials`` ask/evaluate/tell iterations and return the result."""
        if trials < 1:
            raise ValueError("trials must be at least 1")
        for _ in range(trials):
            trial = self.ask()
            value = objective(trial.configuration)
            self.tell(trial, value)
        return StudyResult(best=self.best(), history=self.history)
