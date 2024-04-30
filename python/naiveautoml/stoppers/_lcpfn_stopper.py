# Source code: https://github.com/automl/lcpfn
#! LCPFN is not directly compatible with torch>=2.0.0
#! to make it work I commented out the `is_causal` in the transformer forward
#! and I commented out the `approximate` in the GELU forward
import sys
from numbers import Number

import lcpfn
import numpy as np
import torch
from ._stopper import Stopper


def area_learning_curve(z, f, z_max) -> float:
    assert len(z) == len(f)
    assert z[-1] <= z_max
    area = 0
    for i in range(1, len(z)):
        # z: is always monotinic increasing but not f!
        area += (z[i] - z[i - 1]) * f[i - 1]
    if z[-1] < z_max:
        area += (z_max - z[-1]) * f[-1]
    return area


class LCPFNStopper(Stopper):
    """Stopper based on learning curve extrapolation (LCE) to evaluate if the iterations of the learning algorithm
    should be stopped.

    .. list-table::
        :widths: 25 25 25
        :header-rows: 1

        * - Single-Objective
          - Multi-Objectives
          - Failures
        * - ✅
          - ❌
          - ❌

    The LCE is based on a parametric learning curve model (LCM) which is modeling the score as a function of the number of training steps. Training steps can correspond to the number of training epochs, the number of training batches, the number of observed samples or any other quantity that is iterated through during the training process. The LCE is based on the following steps:

    1. An early stopping condition is always checked first. If the early stopping condition is met, the LCE is not applied.
    2. Then, some safeguard conditions are checked to ensure that the LCE can be applied (number of observed steps must be greater or equal to the number of parameters of the LCM).
    3. If the LCM cannot be fitted (number of observed steps is less than number of parameters of the model), then the last observed step is compared to hitorical performance of others at the same step to check if it is a low-performing outlier (outlier in the direction of performing worse!) using the IQR criterion.
    4. If the LCM can be fitted, a least square fit is performed to estimate the parameters of the LCM.
    5. The probability of the current LC to perform worse than the best observed score at the maximum iteration is computed using Monte-Carlo Markov Chain (MCMC).

    To use this stopper, you need to install the following dependencies:

    .. code-block:: bash

        $ jax>=0.3.25
        $ numpyro

    Args:
        max_steps (int): The maximum number of training steps which can be performed.
        min_steps (int, optional): The minimum number of training steps which can be performed. Defaults to ``4``. It is better to have at least as many steps as the number of parameters of the fitted learning curve model. For example, if ``lc_model="mmf4"`` then ``min_steps`` should be at least ``4``.
        min_done_for_outlier_detection (int, optional): The minimum number of observed scores at the same step to check for if it is a lower-bound outlier. Defaults to ``10``.
        iqr_factor_for_outlier_detection (float, optional): The IQR factor for outlier detection. The higher it is the more inclusive the condition will be (i.e. if set very large it is likely not going to detect any outliers). Defaults to ``1.5``.
        prob_promotion (float, optional): The threshold probabily to stop the iterations. If the current learning curve has a probability greater than ``prob_promotion`` to be worse that the best observed score accross all evaluations then the current iterations are stopped. Defaults to ``0.9`` (i.e. probability of 0.9 of being worse).
        early_stopping_patience (float, optional): The patience of the early stopping condition. If it is an ``int`` it is directly corresponding to a number of iterations. If it is a ``float`` then it is corresponding to a proportion between [0,1] w.r.t. ``max_steps``. Defaults to ``0.25`` (i.e. 25% of ``max_steps``).
        objective_returned (str, optional): The returned objective. It can be a value in ``["last", "max", "alc"]`` where ``"last"`` corresponds to the last observed score, ``"max"`` corresponds to the maximum observed score and ``"alc"`` corresponds to the area under the learning curve. Defaults to "last".

    Raises:
        ValueError: parameters are not valid.
    """

    def __init__(
        self,
        max_steps: int,
        min_steps: int = 1,
        min_obs_to_fit_lc_model=1,
        min_done_for_outlier_detection=10,
        iqr_factor_for_outlier_detection=1.5,
        prob_promotion=0.9,
        early_stopping_patience=0.25,
        reduction_factor=1,
        objective_returned="last",
        random_state=None,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.min_steps = min_steps

        self._min_obs_to_fit_lc_model = min_obs_to_fit_lc_model
        self._reduction_factor = reduction_factor

        self.min_done_for_outlier_detection = min_done_for_outlier_detection
        self.iqr_factor_for_outlier_detection = iqr_factor_for_outlier_detection

        self.prob_promotion = prob_promotion
        if type(early_stopping_patience) is int:
            self.early_stopping_patience = early_stopping_patience
        elif type(early_stopping_patience) is float:
            self.early_stopping_patience = int(early_stopping_patience * self.max_steps)
        else:
            raise ValueError("early_stopping_patience must be int or float")
        self.objective_returned = objective_returned

        self._rung = 0

        self._random_state = random_state
        self.lc_model = lcpfn.LCPFN()

        self._lc_objectives = []

    def _compute_halting_step(self):
        return (self.min_steps - 1) * self._reduction_factor**self._rung

    def _get_competiting_objectives(self, rung) -> list:
        values = []
        for subject in self.observed_objectives:
            if len(self.observed_objectives[subject]) > rung:
                values.append(self.observed_objectives[subject][rung])
        return values

    def observe(self, subject, budget: float, objective: float):
        super().observe(
            subject=subject,
            budget=budget,
            objective=objective
        )
        self._budget = self.observed_budgets[subject][-1]
        self._lc_objectives.append(self.get_objective(subject))
        self._objective = self._lc_objectives[-1]

    def stop(self, subject) -> bool:
        # Enforce Pre-conditions Before Learning-Curve based Early Discarding
        if super().stop(subject):
            self.infos_stopped = "max steps reached"
            return True

        if self.get_current_step(subject) - self._best_step >= self.early_stopping_patience:
            self.infos_stopped = "early stopping"
            return True

        # This condition will enforce the stopper to stop the evaluation at the first step
        # for the first evaluation (The FABOLAS method does the same, bias the first samples with
        # small budgets)

        halting_step = self._compute_halting_step()

        if self.get_current_step(subject) < self.min_steps:
            if self.get_current_step(subject) >= halting_step:
                self._rung += 1
            return False

        if self.get_current_step(subject) < self._min_obs_to_fit_lc_model:
            if self.get_current_step(subject) >= halting_step:
                competing_objectives = self._get_competiting_objectives(self._rung)
                if len(competing_objectives) > self.min_done_for_outlier_detection:
                    q1 = np.quantile(
                        competing_objectives,
                        q=0.25,
                    )
                    q3 = np.quantile(
                        competing_objectives,
                        q=0.75,
                    )
                    iqr = q3 - q1
                    # lower than the minimum of a box plot
                    if (
                        self._objective
                        < q1 - self.iqr_factor_for_outlier_detection * iqr
                    ):
                        self.infos_stopped = "outlier"
                        self.observed_budgets = []
                        self.observed_objectives = []
                        self._stop_was_called = False
                        return True
                self._rung += 1

            return False

        # Check if the halting budget condition is met
        if self.get_current_step(subject) < halting_step:
            return False

        # Check if the evaluation should be stopped based on LC-Model

        # Fit and predict the performance of the learning curve model
        z_train = self.observed_budgets[subject]
        y_train = self.observed_objectives[subject]
        z_train, y_train = np.asarray(z_train), np.asarray(y_train)

        z_train = torch.Tensor(z_train.reshape(-1, 1))
        y_train = torch.Tensor(y_train.reshape(-1, 1))
        y_pred = self.lc_model.predict_quantiles(
            x_train=z_train,
            y_train=y_train,
            x_test=torch.Tensor([[self.max_steps]]),
            qs=[self.prob_promotion],
        )
        y_pred = y_pred[0][0].numpy()

        # Return whether the configuration should be stopped
        if self.get_objective(subject=subject) <= y_pred:
            self._rung += 1
        else:
            self.infos_stopped = f"objective={y_pred:.3f}"

            return True
        
    def get_objective(self, subject):
        observations = self.get_observations(subject=subject)
        if self.objective_returned == "last":
            return observations[-1][-1]
        elif self.objective_returned == "max":
            return max(observations[-1])
        elif self.objective_returned == "alc":
            z, y = observations
            return area_learning_curve(z, y, z_max=self.max_steps)
        else:
            raise ValueError("objective_returned must be one of 'last', 'best', 'alc'")
