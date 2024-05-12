import json
from ConfigSpace.read_and_write import json as config_json
import numpy as np
from ..commons import get_hyperparameter_space_size
import traceback
from ConfigSpace import ConfigurationSpace


class HPOHelper:

    def __init__(self, search_space):

        self.config_spaces = {}

        self.step_names = []
        for step in search_space:
            step_name = step["name"]
            self.step_names.append(step_name)
            self.config_spaces[step_name] = {}

            for comp in step["components"]:
                config_space_as_string = json.dumps(comp["params"])
                self.config_spaces[step_name][comp["class"]] = config_json.read(config_space_as_string)

    def get_config_space_for_selected_algorithms(self, selected_algorithms):
        """

        :param selected_algorithms: dictionary where keys are slot names and vals are algorithm names
        :return:
        """
        cs = ConfigurationSpace()
        for step, selection in selected_algorithms.items():
            cs.add_configuration_space(
                prefix=step,
                configuration_space=self.config_spaces[step][selection]
            )
        return cs

    def get_hps_by_step(self, hp_config):

        entry = {}
        for step_name in self.step_names:
            relevant_keys = [k for k in hp_config.keys() if k.startswith(f"{step_name}:")]
            if relevant_keys:
                entry[step_name] = {k[len(step_name) + 1:]: hp_config[k] for k in relevant_keys}
            else:
                entry[step_name] = None
        return entry


class HPOProcess:

    def __init__(
            self,
            task_type,
            step_names,
            comps_by_steps,
            X,
            y,
            scoring,
            side_scores,
            evaluation_fun,
            execution_timeout,
            mandatory_pre_processing,
            max_time_without_imp,
            max_its_without_imp,
            min_its=10,
            logger_name=None,
            allow_exhaustive_search=True
    ):
        self.task_type = task_type
        self.step_names = step_names
        self.comps_by_steps = comps_by_steps  # list of components, in order of appearance in pipeline
        self.X = X
        self.y = y
        self.mandatory_pre_processing = mandatory_pre_processing
        self.execution_timeout = execution_timeout

        self.config_spaces = {}
        self.best_configs = {}
        self.space_size = 1
        for step_name, comp in self.comps_by_steps:
            config_space_as_string = json.dumps(comp["params"])
            self.config_spaces[step_name] = config_json.read(config_space_as_string)
            self.best_configs[step_name] = self.config_spaces[step_name].get_default_configuration()
            space_size_for_component = get_hyperparameter_space_size(self.config_spaces[step_name])
            if space_size_for_component > 0:
                if self.space_size < np.inf:
                    if space_size_for_component < np.inf:
                        self.space_size *= space_size_for_component
                    else:
                        self.space_size = np.inf

        self.eval_runtimes = []
        self.configs_since_last_imp = 0
        self.time_since_last_imp = 0
        self.evaled_configs = set([])
        self.active = self.space_size >= 1
        self.best_score = -np.inf
        self.max_time_without_imp = max_time_without_imp
        self.max_its_without_imp = max_its_without_imp
        self.min_its = min_its
        self.scoring = scoring
        self.pool = EvaluationPool(
            task_type,
            X,
            y,
            scoring=scoring,
            side_scores=side_scores,
            evaluation_fun=evaluation_fun
        )
        self.its = 0
        self.allow_exhaustive_search = allow_exhaustive_search

        if side_scores is None:
            self.side_scores = []
        elif isinstance(side_scores, list):
            self.side_scores = side_scores
        else:
            self.logger.warning("side scores was not given as list, casting it to a list of size 1 implicitly.")
            self.side_scores = [side_scores]

        # init logger
        self.logger = logging.getLogger('naiveautoml.hpo' if logger_name is None else logger_name)
        self.logger.info(f"Search space size is {self.space_size}. Active? {self.active}")

    def get_parametrized_pipeline(self, configs_by_comps):
        steps = []
        for step_name, comp in self.comps_by_steps:
            params_of_comp = configs_by_comps[step_name]
            this_step = (step_name, build_estimator(comp, params_of_comp, self.X, self.y))
            steps.append(this_step)
        return Pipeline(steps=self.mandatory_pre_processing + steps)

    def evalComp(self, configs_by_comps):
        try:
            status, scores, evaluation_report = self.pool.evaluate(
                pl=self.get_parametrized_pipeline(configs_by_comps),
                timeout=self.execution_timeout
            )
            return (
                status,
                scores,
                evaluation_report,
                None
            )
        except pynisher.WallTimeoutException:
            self.logger.info("TIMEOUT")
            return ("timeout",
                    {scoring: np.nan for scoring in [self.scoring] + self.side_scores},
                    {scoring: {} for scoring in [self.scoring] + self.side_scores},
                    None)
        except KeyboardInterrupt:
            raise
        except Exception:
            return (
                "exception",
                {scoring: np.nan for scoring in [self.scoring] + self.side_scores},
                {scoring: {} for scoring in [self.scoring] + self.side_scores},
                traceback.format_exc()
            )


