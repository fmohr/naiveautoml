import logging
import traceback

import pandas as pd
import time

# naiveautoml commons
from ._interfaces import SupervisedTask, HPOptimizer
from .algorithm_selection.sklearn import SKLearnAlgorithmSelector
from .hpo.random_search import RandomHPO
from .commons import EvaluationPool

import numpy as np


class NaiveAutoML:

    def __init__(self,
                 task_type: str = "auto",
                 algorithm_selector="sklearn",
                 hp_optimizer="random",
                 scoring=None,
                 passive_scorings=None,
                 evaluation_fun=None,
                 num_cpus=8,
                 timeout_candidate=300,
                 timeout_overall=None,
                 show_progress=False,
                 max_hpo_iterations=100,
                 max_hpo_iterations_without_imp=100,
                 max_hpo_time_without_imp=1800,
                 kwargs_as={},
                 kwargs_hpo={},
                 logger_name=None,
                 strictly_naive: bool = False,
                 raise_errors: bool = False):
        """

        :param task_type:
        :param algorithm_selector
        :param hp_optimizer
        :param scoring:
        :param passive_scorings:
        :param evaluation_fun:
        :param num_cpus:
        :param timeout_candidate:
        :param timeout_overall:
        :param max_hpo_iterations:
        :param max_hpo_iterations_without_imp:
        :param max_hpo_time_without_imp:
        :param logger_name:
        :param strictly_naive:
        :param raise_errors:
        """

        # init logger
        self.logger_name = logger_name
        self.logger = logging.getLogger('naiveautoml' if logger_name is None else logger_name)

        # configure algorithm selector
        if isinstance(algorithm_selector, str):
            accepted_selectors = ["sklearn"]
            if "logger" not in kwargs_as:
                kwargs_as["logger"] = self.logger
            if "strictly_naive" not in kwargs_as:
                kwargs_as["strictly_naive"] = strictly_naive
            if algorithm_selector == "sklearn":
                self.algorithm_selector = SKLearnAlgorithmSelector(
                    show_progress=show_progress,
                    raise_errors=raise_errors,
                    **kwargs_as
                )
            else:
                raise ValueError(
                    f"algorithm_selector was specified as string '{algorithm_selector}' "
                    f"but is not in {accepted_selectors}."
                )
        else:
            self.algorithm_selector = algorithm_selector

        # configure hyperparameter optimizer
        if isinstance(hp_optimizer, str):
            accepted_optimizers = ["random"]
            if hp_optimizer not in accepted_optimizers:
                raise ValueError(f"hp_optimizer if string must be in {accepted_optimizers} but is {hp_optimizer}")
            self.hp_optimizer = RandomHPO(
                show_progress=show_progress,
                logger=self.logger,
                **kwargs_hpo
            )
        else:
            if not isinstance(hp_optimizer, HPOptimizer):
                raise ValueError(
                    f"hp_optimizer must be a string or an instance of HPOptimizer but is {type(hp_optimizer)}"
                )
            self.hp_optimizer = hp_optimizer

        # configure evaluation function
        self.evaluation_fun = evaluation_fun

        # memorize scorings
        self.scoring = None
        self.passive_scorings = None
        self._configured_scoring = scoring
        self._configured_passive_scorings = passive_scorings

        # memorize other behavioral configurations
        self.num_cpus = num_cpus
        self.timeout_candidate = timeout_candidate
        self.timeout_overall = timeout_overall
        self.raise_errors = raise_errors
        self.max_hpo_iterations = max_hpo_iterations
        self.max_hpo_iterations_without_imp = max_hpo_iterations_without_imp
        self.max_hpo_time_without_imp = max_hpo_time_without_imp

        # state variables
        self.start_time = None
        self.deadline = None
        self.best_score_overall = None
        self._chosen_model = None
        self._history = None
        self.chosen_attributes = None
        self.hpo_process = None

        # mandatory pre-processing steps
        self.mandatory_pre_processing = None

        # state variables
        self.evaluator = None
        self.task_type = task_type
        self.task = None

    @property
    def history(self):
        """

        :return: a dataframe with all considered candidates
        """
        return self._history.copy() if self._history is not None else None  # avoid actual history to be changed

    @property
    def leaderboard(self):
        """

        :return: a dataframe with all successful evaluations, sorted by performance
        """
        df_successful = self._history[self._history["status"] == "ok"]
        df_sorted = df_successful.sort_values([self.task.scoring["name"], "new_best"], ascending=False)
        return df_sorted.reset_index().rename(columns={"index": "order"}).copy()

    @property
    def chosen_model(self):
        return self._chosen_model.clone()

    def get_evaluation_pool(self, task):
        return EvaluationPool(
            task=task,
            evaluation_fun=self.evaluation_fun,
            logger_name=None if self.logger_name is None else self.logger_name + ".pool"
        )

    def get_task_from_data(self, X, y, categorical_attributes=None):

        # initialize taskis_pipeline_forbidden
        return SupervisedTask(
            X=X,
            y=y,
            scoring=self._configured_scoring,
            passive_scorings=self._configured_passive_scorings,
            categorical_attributes=categorical_attributes,
            task_type=self.task_type,
            timeout_overall=self.timeout_overall,
            timeout_candidate=self.timeout_candidate,
            max_hpo_iterations=self.max_hpo_iterations,
            max_hpo_iterations_without_imp=self.max_hpo_iterations_without_imp,
            max_hpo_time_without_imp=self.max_hpo_time_without_imp
        )

    def reset(self, task: SupervisedTask):

        # initialize
        self._chosen_model = None
        self.task = task
        self.best_score_overall = -np.inf
        self._history = []

        # evaluator
        self.evaluator = self.get_evaluation_pool(task)

        # reset algorithm selector
        self.algorithm_selector.reset(task=task, evaluator=self.evaluator)

        # show start message
        self.logger.info(
            f"""Optimizing pipeline under the following conditions.
                {task.description}{self.algorithm_selector.description}{self.hp_optimizer.description}"""
        )

    def fit(self, X, y, categorical_features=None):

        # determine deadline
        self.start_time = time.time()
        deadline = self.start_time + self.timeout_overall if self.timeout_overall is not None else None

        # reset the task
        self.reset(self.get_task_from_data(X, y, categorical_features))

        # choose algorithms
        self._history = self.algorithm_selector.run(
            deadline=deadline
        )
        relevant_history = self._history[self._history[self.task.scoring["name"]].notna()]\
            if self._history is not None \
            else None
        if (
                isinstance(relevant_history, pd.DataFrame) and
                len(relevant_history) > 0 and
                relevant_history.iloc[0]["pipeline"] is not None
        ):
            self.steps_after_which_algorithm_selection_was_completed = len(self._history)

            # get candidate descriptor
            as_result_for_best_candidate = relevant_history.sort_values(self.task.scoring["name"]).iloc[-1]

            if (
                    deadline is None or
                    deadline is not None and deadline - time.time() >= as_result_for_best_candidate["runtime"] + 5
            ):

                # tune hyperparameters
                self.hp_optimizer.reset(
                    task=self.task,
                    runtime_of_default_config=as_result_for_best_candidate["runtime"],
                    config_space=self.algorithm_selector.get_config_space(as_result_for_best_candidate),
                    history_descriptor_creation_fun=lambda hp_config: self.algorithm_selector.create_history_descriptor(
                        as_result_for_best_candidate,
                        hp_config
                    ),
                    evaluator=self.evaluator,
                    is_pipeline_forbidden=self.algorithm_selector.is_pipeline_forbidden,
                    is_timeout_required=self.algorithm_selector.is_timeout_required
                )
                df_results_hpo = self.hp_optimizer.optimize(deadline=deadline)
                self._history = pd.concat([self._history, df_results_hpo], ignore_index=True)
            else:
                self.logger.info("No time remaining for hyperparameter optimization.")

            # get final pipeline and train it on full data
            self.logger.info("--------------------------------------------------")
            self.logger.info("Search Completed. Building final pipeline.")
            self.logger.info("--------------------------------------------------")
            offset = 1
            successful_training = False
            while not successful_training:
                try:
                    best_configuration = relevant_history.sort_values(self.task.scoring["name"]).iloc[-offset]
                    self._chosen_model = best_configuration["pipeline"]
                    self.logger.info(self._chosen_model)
                    self.logger.info("Now fitting the pipeline with all given data.")

                    # fit the best model
                    self._chosen_model.fit(X, y)
                    successful_training = True
                except KeyboardInterrupt:
                    raise
                except Exception:
                    exception = traceback.format_exc()
                    self.logger.warning(
                        f"There was an issue with training the best final pipeline, training the next best. "
                        f"The exception was {exception}"

                    )
                offset += 1

        else:
            self.logger.info("No model was chosen in first phase, so there is nothing to return for me ...")
        self.end_time = time.time()
        self.logger.info(f"Runtime was {self.end_time - self.start_time} seconds")

    def recover_model(self, pl=None, history_index=None):
        if pl is None and history_index is None:
            raise ValueError(
                "Provide a pipeline object or a history index to recover a model!"
            )
        if pl is None:
            pl = self._history.iloc[history_index]["pipeline"]
        pl.fit(self.task.X, self.task.y)
        return pl

    def predict(self, X):
        return self._chosen_model.predict(X)

    def predict_proba(self, X):
        return self._chosen_model.predict_proba(X)
