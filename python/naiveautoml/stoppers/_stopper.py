import abc
import copy

from numbers import Number
import numpy as np


class Stopper(abc.ABC):
    """An abstract class describing the interface of a Stopper.

    Args:
        max_steps (int): the maximum number of calls to ``observe(budget, objective)``.
    """

    def __init__(self, max_steps: int) -> None:
        assert max_steps > 0
        self.max_steps = max_steps
        self.context = []

        # Initialize list to collect observations
        self.observed_budgets = []
        self.observed_objectives = []

        self._stop_was_called = False

        self._best_objective = -np.inf
        self._best_budget = None

    @property
    def best_objective(self):
        return self._best_objective

    def to_json(self):
        """Returns a dict version of the stopper which can be saved as JSON."""
        json_format = type(self).__name__
        return json_format

    def transform_objective(self, objective: float):
        """Replaces the currently observed objective by the maximum objective observed from the
        start. Identity transformation by default."""
        # prev_objective = (
        #     self.observed_objectives[-1] if len(self.observed_objectives) > 0 else None
        # )
        # if prev_objective is not None:
        #     objective = max(prev_objective, objective)
        return objective

    @property
    def step(self):
        """Last observed step."""
        return self.observed_budgets[-1] if len(self.observed_budgets) > 0 else 0

    def observe(self, budget: float, objective: float) -> None:
        """Observe a new objective value.

        Args:
            budget (float): the budget used to obtain the objective (e.g., the number of epochs).
            objective (float): the objective value to observe (e.g, the accuracy).
        """
        objective = self.transform_objective(objective)

        self.observed_budgets.append(budget)
        self.observed_objectives.append(objective)

        # memorize best objective
        if objective > self._best_objective:
            self._best_objective = objective
            self._best_budget = budget

    def stop(self) -> bool:
        """Returns ``True`` if the evaluation should be stopped and ``False`` otherwise.

        Returns:
            bool: ``(step >= max_steps)``.
        """
        if not self._stop_was_called:
            self._stop_was_called = True

        if self.step >= self.max_steps:
            return True

        return False

    @property
    def observations(self) -> list:
        """Returns a copy of the list of observations with 0-index the budgets and 1-index the objectives."""
        obs = [self.observed_budgets, self.observed_objectives]
        return copy.deepcopy(obs)

    @property
    def objective(self):
        """Last observed objective."""
        return self.observations[-1][-1] if len(self.observations) > 0 and len(self.observations[0]) > 0 else None
