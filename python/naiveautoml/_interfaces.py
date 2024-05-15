from abc import ABC
from typing import Callable

import numpy as np
from scipy.sparse import issparse, spmatrix
from ConfigSpace import ConfigurationSpace
from sklearn.metrics import get_scorer, make_scorer
import pandas as pd
import time
from tqdm import tqdm

from sklearn.utils.multiclass import type_of_target


class SupervisedTask:

    def __init__(self,
                 X,
                 y,
                 scoring,
                 passive_scorings,
                 task_type=None,
                 categorical_attributes=None,
                 timeout_overall=None,
                 timeout_candidate=300,
                 max_hpo_iterations=100,
                 max_hpo_iterations_without_imp=100,
                 max_hpo_time_without_imp=3600
                 ):
        self._X = X
        self._y = y
        self.specified_task_type = task_type
        self.categorical_attributes = categorical_attributes
        self._sparse_X = issparse(X)
        self._sparse_y = issparse(y)
        self._labels = list(np.unique(y))
        self._inferred_task_type = None
        self.timeout_overall = timeout_overall
        self.timeout_candidate = timeout_candidate
        self.max_hpo_iterations = max_hpo_iterations
        self.max_hpo_iterations_without_imp = max_hpo_iterations_without_imp
        self.max_hpo_time_without_imp = max_hpo_time_without_imp

        # configure scorings
        def prepare_scoring(scoring):
            out = {
                "name": scoring if isinstance(scoring, str) else scoring["name"]
            }
            if type_of_target(self._y) == "multilabel-indicator":
                out["fun"] = None
            else:
                if isinstance(scoring, str):
                    out["fun"] = get_scorer(scoring)
                else:
                    out["fun"] = make_scorer(**{key: val for key, val in scoring.items() if key != "name"})
            return out

        if scoring is None:
            self.scoring = None
            if self.inferred_task_type == "classification":
                self.scoring = prepare_scoring("roc_auc" if len(self._labels) == 2 else "neg_log_loss")
            elif self.inferred_task_type == "multilabel-indicator":
                self.scoring = {
                    "name": "f1_macro",
                    "fun": None
                }
            else:
                self.scoring = prepare_scoring("neg_mean_squared_error")
        else:
            self.scoring = prepare_scoring(scoring)

        self.passive_scorings = []
        if passive_scorings is not None:
            for scoring in passive_scorings:
                self.passive_scorings.append(prepare_scoring(scoring))

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def sparse_X(self):
        return self._sparse_X

    @property
    def sparse_y(self):
        return self._sparse_y

    @property
    def labels(self):
        return self._labels

    @property
    def num_labels(self):
        return len(self._labels)

    @property
    def inferred_task_type(self):
        if self._inferred_task_type is None:
            if self.specified_task_type is None or self.specified_task_type == "auto":
                self._inferred_task_type = self.infer_task_type()
            else:
                self._inferred_task_type = self.specified_task_type
        return self._inferred_task_type

    @property
    def description(self, indent=1):
        return f"""
        Input type: {type(self._X)} (sparse: {self._sparse_X})
    Input shape: {self._X.shape}
    Target type: {type(self._y)} (sparse: {self._sparse_y})
    Target shape: {self._y.shape}.
    Scoring: {self.scoring}
    Other scorings computed: {self.passive_scorings}
    Timeout Overall: {self.timeout_overall}
    Timeout per Candidate: {self.timeout_candidate}
    Max HPO iterations: {self.max_hpo_iterations}
    Max HPO iterations w/o improvement: {self.max_hpo_iterations_without_imp}
    Max HPO time (s) w/o improvement: {self.max_hpo_time_without_imp}
    """

    def infer_task_type(self):
        """
        :param X: the descriptions of the instances
        :param y: the labels of the instances
        :return:
        """
        if type_of_target(self._y) == "multilabel-indicator":
            return "multilabel-indicator"

        # infer task type
        if isinstance(self.scoring, str):
            return "regression" if self.scoring in [
                "explained_variance",
                "max_error",
                "neg_mean_absolute_error",
                "neg_mean_squared_error",
                "neg_root_mean_squared_error",
                "neg_mean_squared_log_error",
                "neg_median_absolute_error",
                "r2",
                "neg_mean_poisson_deviance",
                "neg_mean_gamma_deviance",
                "neg_mean_absolute_percentage_error",
                "d2_absolute_error_score",
                "d2_pinball_score",
                "d2_tweedie_score"
            ] else "classification"
        elif isinstance(self._y, spmatrix):
            return "regression" if np.issubdtype(self._y.dtype, np.number) else "classification"
        else:
            return "regression" if self.num_labels > 100 else "classification"
        raise Exception("Could not infer task type!")


class AlgorithmSelector(ABC):

    def __init__(self):
        super().__init__()

    def run(self, task):
        """

        :param task:
        :return: a dataframe that contains one row for every recommended algorithm (ordered by recommendation)
            """
        raise NotImplementedError

    @property
    def description(self):
        raise NotImplementedError

    def get_config_space(self, as_report):
        raise NotImplementedError

    def get_history_descriptor(self, base_pl_descriptor, hp_config):
        raise NotImplementedError


class HPOptimizer(ABC):

    def __init__(self, show_progress=False, logger=None):
        super().__init__()

        self.show_progress = show_progress
        self.logger = logger

        self.task = None
        self.runtime_of_default_config = None
        self.config_space = None
        self.create_history_descriptor = None
        self.evaluator = None
        self.is_pipeline_forbidden = None
        self.is_timeout_required = None
        self._history = None
        self.best_score = None
        self.active = False
        self.pbar = None

    def reset(self,
              task: SupervisedTask,
              runtime_of_default_config,
              config_space: ConfigurationSpace,
              history_descriptor_creation_fun: Callable,
              evaluator: Callable,
              is_pipeline_forbidden: Callable,
              is_timeout_required: Callable
              ):
        self.task = task
        self.runtime_of_default_config = runtime_of_default_config
        self.config_space = config_space
        self.create_history_descriptor = history_descriptor_creation_fun
        self.evaluator = evaluator
        self.is_pipeline_forbidden = is_pipeline_forbidden
        self.is_timeout_required = is_timeout_required
        self._history = None
        self.best_score = -np.inf
        self.active = True
        if self.show_progress:
            print("Progress for hyperparameter optimization:")
            self.pbar = tqdm(total=self.task.max_hpo_iterations)

    def step(self):
        """
            Evaluates one or more candidates in the configuration space
        :return: pandas dataframe with the evaluated candidates
        """
        raise NotImplementedError

    def optimize(self, deadline=None):

        # now conduct HPO until there is no local improvement or the deadline is hit
        self.logger.info("--------------------------------------------------")
        self.logger.info(
            f"Entering HPO phase."
            f"{('Remaining time: ' + str(round(deadline - time.time(), 2)) + 's') if deadline is not None else ''}"
        )
        self.logger.info("--------------------------------------------------")

        df_result = None
        time_for_last_step = self.runtime_of_default_config
        for _ in range(self.task.max_hpo_iterations):
            if deadline is not None and deadline - time.time() < time_for_last_step:
                self.logger.info("Next iteration would probably take more time than the deadline allows. Stopping HPO.")
                break

            start = time.time()
            df_step = self.step()
            time_for_last_step = time.time() - start
            df_result = df_step if df_result is None else pd.concat([df_result, df_step])
        return df_result

    def do_exhaustive_search(self):

        remaining_time = 0
        # check whether we do a quick exhaustive search and then disable this module
        if len(self.eval_runtimes) >= 10:
            total_expected_runtime = self.space_size * np.mean(self.eval_runtimes)
            if self.allow_exhaustive_search and total_expected_runtime < np.inf and (
                    remaining_time is None or total_expected_runtime < remaining_time
            ):
                self.active = False
                self.logger.info(
                    f"Expected time to evaluate all configurations is only {total_expected_runtime}."
                    "Doing exhaustive search."
                )
                configs = None  # get_all_configurations(self.config_spaces)
                self.logger.info(f"Now evaluation all {len(configs)} possible configurations.")
                for configs_by_comps in configs:
                    status, scores, evaluation_report, exception = self.evalComp(configs_by_comps)
                    score = scores[self.scoring]
                    self.logger.info(f"Observed score of {score} for params {configs_by_comps}")
                    if score > self.best_score:
                        self.logger.info("This is a NEW BEST SCORE!")
                        self.best_score = score
                        self.best_configs = configs_by_comps
                self.logger.info("Configuration space completely exhausted.")
        return self.get_parametrized_pipeline(
            configs_by_comps), status, scores, evaluation_report, None, exception

    @property
    def description(self):
        raise NotImplementedError
