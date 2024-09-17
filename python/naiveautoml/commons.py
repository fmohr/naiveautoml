from ._interfaces import SupervisedTask
import numpy as np
import pandas as pd
import logging
import warnings
import os
import psutil
import scipy.sparse
import time
import pynisher
from .evaluators import LccvEvaluator, KFoldEvaluator, MccvEvaluator

import ConfigSpace
import traceback


class EvaluationPool:

    def __init__(self,
                 task: SupervisedTask,
                 evaluation_fun=None,
                 tolerance_tuning=0.05,
                 tolerance_estimation_error=0.01,
                 logger_name=None,
                 use_caching=True,
                 error_treatment="info",
                 kwargs_evaluation_fun={},
                 random_state=None
                 ):

        self.random_state = random_state
        domains_task_type = ["classification", "multilabel-indicator", "regression"]
        if task.inferred_task_type not in domains_task_type:
            raise ValueError(f"task_type must be in {domains_task_type} but is {task.inferred_task_type}.")
        self.task_type = task.inferred_task_type

        error_treatment_domain = ["debug", "info", "warning", "error", "raise"]
        if error_treatment not in error_treatment_domain:
            raise ValueError(f"error_treatment must be in {error_treatment_domain} but is {error_treatment}.")
        self.error_treatment = error_treatment

        self.logger = logging.getLogger('naiveautoml.evalpool' if logger_name is None else logger_name)

        # disable warnings by default
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')

        self.task = task

        if not isinstance(self.task.X, (
                pd.DataFrame,
                np.ndarray,
                scipy.sparse.csr_matrix,
                scipy.sparse.csc_matrix,
                scipy.sparse.lil_matrix
        )):
            raise TypeError(f"X must be a numpy array but is {type(self.task.X)}")
        if self.task.y is None:
            raise TypeError("Parameter y must not be None")

        self.X = self.task.X
        self.y = self.task.y
        self.scoring = task.scoring
        if task.passive_scorings is None:
            self.side_scores = []
        elif isinstance(task.passive_scorings, list):
            self.side_scores = task.passive_scorings
        else:
            self.logger.warning("side scores was not given as list, casting it to a list of size 1 implicitly.")
            self.side_scores = [task.passive_scorings]
        self.evaluation_fun = self.get_evaluation_fun(evaluation_fun, kwargs_evaluation_fun)
        self.bestScore = -np.inf
        self.tolerance_tuning = tolerance_tuning
        self.tolerance_estimation_error = tolerance_estimation_error
        self.cache = {}
        self.use_caching = use_caching

    def get_evaluation_fun(self, evaluation_fun, kwargs_evaluation_fun):

        task = self.task

        if evaluation_fun is None:
            self.logger.info("Choosing mccv as default evaluation function.")
            evaluation_fun = "mccv"

        if evaluation_fun in ["lccv", "mccv"]:
            is_small_dataset = task.X.shape[0] < 2000
            is_medium_dataset = not is_small_dataset and task.X.shape[0] < 20000
            is_large_dataset = not (is_small_dataset or is_medium_dataset)

            if not kwargs_evaluation_fun:
                if is_small_dataset:
                    self.logger.info("This is a small dataset, choosing 5 splits for evaluation")
                    kwargs_evaluation_fun["n_splits"] = 5
                elif is_medium_dataset:
                    self.logger.info("This is a medium dataset, choosing 3 splits for evaluation")
                    kwargs_evaluation_fun["n_splits"] = 3
                elif is_large_dataset:
                    self.logger.info("This is a large dataset, choosing 1 split for evaluation")
                    kwargs_evaluation_fun["n_splits"] = 1
                else:
                    raise ValueError(
                        "Invalid case for dataset size!! This should never happen. Please report this as a bug.")

            if evaluation_fun == "mccv":
                return MccvEvaluator(task.inferred_task_type, random_state=self.random_state, **kwargs_evaluation_fun)
            elif evaluation_fun == "kfold":
                return KFoldEvaluator(task.inferred_task_type, random_state=self.random_state, **kwargs_evaluation_fun)

        elif evaluation_fun == "lccv":
            return LccvEvaluator(task.inferred_task_type, random_state=self.random_state, **kwargs_evaluation_fun)
        else:
            return evaluation_fun

    def tellEvaluation(self, pl, scores, evaluation_report, timestamp):
        spl = str(pl)
        self.cache[spl] = (spl, scores, evaluation_report, timestamp)
        score = np.mean(scores)
        if score > self.bestScore:
            self.bestScore = score
            self.best_spl = spl

    def evaluate(self, pl, timeout=None):

        # disable warnings by default
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')

        process = psutil.Process(os.getpid())
        mem = int(process.memory_info().rss / 1024 / 1024)
        self.logger.info(
            f"Initializing evaluation of {pl}. Current memory consumption {mem}MB. Now awaiting results."
        )

        start_outer = time.time()
        spl = str(pl)
        if self.use_caching and spl in self.cache:
            out = {scoring["name"]: np.nan for scoring in [self.scoring] + self.side_scores}
            out[self.scoring["name"]] = np.round(np.mean(self.cache[spl][1]), 4)
            return "cache", out, self.cache[spl][2], None

        timestamp = time.time()
        scores = None
        evaluation_report = None
        exception = None
        try:
            if timeout is not None:
                if timeout > 1:
                    with pynisher.limit(self.evaluation_fun, wall_time=timeout) as limited_evaluation:
                        if hasattr(self.evaluation_fun, "error_treatment"):
                            scores, evaluation_report = limited_evaluation(
                                pl,
                                self.X,
                                self.y,
                                [self.scoring] + self.side_scores,
                                error_treatment=self.error_treatment
                            )
                        else:
                            scores, evaluation_report = limited_evaluation(
                                pl,
                                self.task.X,
                                self.task.y,
                                [self.task.scoring] + self.task.passive_scorings
                            )
                    status = "ok"

                else:  # no time left
                    status = "timeout"
            else:
                scores, evaluation_report = self.evaluation_fun(pl, self.X, self.y, [self.scoring] + self.side_scores)
                status = "ok"
                if not isinstance(scores, dict):
                    raise TypeError(f"""
                    scores is of type {type(scores)} but must be a dictionary
                    with entries for {self.scoring["name"]}. Probably you inserted an
                    evaluation_fun argument that does not return a proper dictionary."""
                                    )

        except pynisher.WallTimeoutException:
            status = "timeout"
        except Exception:
            status = "exception"
            exception = traceback.format_exc()
            if self.error_treatment == "raise":
                raise
            log_txt = f"Observed Exception during evaluation: {exception}"
            if self.error_treatment == "debug":
                self.logger.debug(log_txt)
            elif self.error_treatment == "info":
                self.logger.info(log_txt)
            elif self.error_treatment == "warning":
                self.logger.warning(log_txt)
            elif self.error_treatment == "error":
                self.logger.error(log_txt)

        if scores is None:
            scores = {s: np.nan for s in [self.task.scoring["name"]] + [s["name"] for s in self.task.passive_scorings]}

        # here we give the evaluator the chance to update itself
        # this looks funny, but it is done because the evaluation could have been done with a copy of the evaluator
        if hasattr(self.evaluation_fun, "update"):
            self.evaluation_fun.update(pl, scores)

        runtime = time.time() - start_outer

        self.logger.info(f"Completed evaluation of {spl} after {runtime}s. Scores are {scores}")
        self.tellEvaluation(pl, scores[self.scoring["name"]], evaluation_report, timestamp)
        return status, scores, evaluation_report, exception


def get_hyperparameter_space_size(config_space):
    hps = config_space.get_hyperparameters()
    if not hps:
        return 0
    size = 1
    for hp in hps:
        if isinstance(hp, (
                ConfigSpace.hyperparameters.UnParametrizedHyperparameter,
                ConfigSpace.hyperparameters.Constant
        )):
            continue

        if isinstance(hp, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            size *= len(list(hp.choices))
        elif isinstance(hp, ConfigSpace.hyperparameters.IntegerHyperparameter):
            size *= (hp.upper - hp.lower + 1)
        else:
            return np.inf
    return size
