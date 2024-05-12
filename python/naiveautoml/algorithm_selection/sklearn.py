from .._interfaces import AlgorithmSelector, SupervisedTask

# core stuff
import json
import numpy as np
import pandas as pd
import importlib.resources as pkg_resources
from collections.abc import Iterable
import scipy as sp

# HPO and process control
from ConfigSpace.read_and_write import json as config_json
import time
import pynisher
import traceback

# progress bar
from tqdm import tqdm

# sklearn stuff
from sklearn.base import clone
from ._pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from ._sklearn_factory import build_estimator, is_pipeline_forbidden
from ._sklearn_hpo import HPOHelper


def is_component_defined_in_steps(steps, name):
    candidates = [s[1] for s in steps if s[0] == name]
    return len(candidates) > 0


def get_step_with_name(steps, name):
    candidates = [s for s in steps if s[0] == name]
    return candidates[0]


class SKLearnAlgorithmSelector(AlgorithmSelector):

    def __init__(self,
                 search_space=None,
                 standard_classifier=KNeighborsClassifier,
                 standard_regressor=LinearRegression,
                 show_progress=False,
                 opt_ordering=None,
                 logger=None,
                 strictly_naive=False,
                 sparse=True,
                 raise_errors=False
                 ):
        """

        :param standard_classifier:
        :param standard_regressor:
        :param show_progress:
        :param opt_ordering:
        :param sparse: whether data is treated sparsely in pre-processing.
        Default is `None`, and in that case, the sparsity is inferred automatically.
        It is tested whether the mandatory pipeline can execute `fit_transform` without a numpy memory exception.
        If so, no sparsity is applied.
        Otherwise, it is assumed that not enough memory is available, and sparsity is enabled.
        """
        super().__init__()
        self.logger = logger

        self._configured_search_space = search_space

        self.show_progress = show_progress
        self.opt_ordering = opt_ordering
        self.standard_classifier = standard_classifier
        self.standard_regressor = standard_regressor
        self.strictly_naive = strictly_naive
        self.sparse = sparse
        self.raise_errors = raise_errors

        # state variables
        self.start_time = None
        self.task = None
        self.evaluator = None
        self.mandatory_pre_processing = None
        self.hpo_helper = None
        self.y_encoded = None
        self._history = None
        self.best_score_overall = None

    def reset(self, task: SupervisedTask, evaluator):

        # memorize task and evaluator
        self.task = task
        self.evaluator = evaluator

        # encode targets if necessary
        self.y_encoded = self.substitute_targets(task.y.copy())

        # register search space
        task_type = task.inferred_task_type
        self.logger.info(f"Automatically inferred task type: {task_type}")

        ''' search_space is a string or a list of dictionaries
            -   if it is a dict, the last one for the learner and all the others for pre-processing.
                Each dictionary has an entry "name" and an entry "components",
                which is a list of components with their parameters.
            - if it is a string, a json file with the name of search_space is read in with the same semantics
         '''
        if self._configured_search_space is None or self._configured_search_space == "auto-sklearn":
            json_str = pkg_resources.read_text('naiveautoml', 'searchspace-' + task_type + '.json')
            self.search_space = json.loads(json_str)
        else:
            if isinstance(self._configured_search_space, str):
                with open(self._configured_search_space) as f:
                    self.search_space = json.load(f)

        # register the HPO helper
        self.hpo_helper = HPOHelper(self.search_space)

        # determine mandatory pre-processing
        self.mandatory_pre_processing = self.get_mandatory_preprocessing(task.X, task.y, task.categorical_attributes)

        # reset history
        self._history = []
        self.best_score_overall = -np.inf

    def run(self, deadline=None):

        self.start_time = time.time()

        # run over all the elements of the pipeline
        self.logger.info("--------------------------------------------------")
        self.logger.info("Choosing Algorithm for each slot")
        self.logger.info("--------------------------------------------------")
        decisions = []
        components_with_score = {}

        if self.show_progress:
            print("Progress for algorithm selection:")
            pbar = tqdm(total=sum([len(step["components"]) for step in self.search_space]))

        # retrieve ordering of slots for optimization
        if self.opt_ordering is None:
            opt_ordering = ["learner"]
            for step in self.search_space:
                if step["name"] != "learner":
                    opt_ordering.append(step["name"])
            self.opt_ordering = opt_ordering

        best_score = -np.inf
        for step_index, step_name in enumerate(self.opt_ordering):

            # create list of components to try for this slot
            step = [step for step in self.search_space if step["name"] == step_name][0]
            self.logger.info("--------------------------------------------------")
            self.logger.info(f"Selecting component for step with name: {step_name}")
            self.logger.info("--------------------------------------------------")

            # find best default parametrization for this slot (depending on choice of previously configured slots)
            decision = None
            for comp in step["components"]:
                if deadline is not None:
                    remaining_time = deadline - 10 - time.time()
                    if remaining_time is not None and remaining_time < 0:
                        self.logger.info("Timeout approaching. Not evaluating anymore for this stage.")
                        break
                    else:
                        self.logger.info(
                            f"Evaluating {comp['class'] if comp is not None else None}."
                            f"Timeout: {self.task.timeout_candidate}. Remaining time: {remaining_time}"
                        )

                # get and evaluate pipeline for this step
                pl = self.get_pipeline_for_decision_in_step(step_name, comp, self.task.X, self.task.y, decisions)

                eval_start_time = time.time()
                if is_pipeline_forbidden(self.task, pl):
                    self.logger.info(f"Preventing evaluation of forbidden pipeline {pl}")
                    scores = {scoring["name"]: np.nan for scoring in [self.task.scoring] + self.task.passive_scorings}
                    self.evaluator.tellEvaluation(pl, [scores[self.task.scoring["name"]]], None, time.time())
                    status, scores, evaluation_report, exception = "avoided", scores, None, None
                else:
                    if self.task.timeout_candidate is None:
                        timeout = None
                    else:
                        timeout = min(self.task.timeout_candidate, remaining_time if deadline is not None else 10 ** 10)
                    status, scores, evaluation_report, exception = self.evaluator.evaluate(pl, timeout)

                runtime = time.time() - eval_start_time
                if scores is None:
                    scores = {
                        scoring["name"]: np.nan for scoring in
                        [self.task.scoring] +
                        (self.task.passive_scorings if self.task.passive_scorings is not None else [])
                    }
                    evaluation_report = {
                        scoring["name"]: {} for scoring in
                        [self.task.scoring] +
                        (self.task.passive_scorings if self.task.passive_scorings is not None else [])
                    }
                score = scores[self.task.scoring["name"]]
                self.logger.info(
                    f"Observed score of {score} for default configuration of {None if comp is None else comp['class']}"
                )

                # update history
                summary = {
                    "time": time.time(),
                    "runtime": runtime,
                    "pipeline": clone(pl),
                    "default_hp": True
                }
                summary.update(scores)
                summary.update({
                    "new_best": score > self.best_score_overall,
                    "evaluation_report": evaluation_report,
                    "status": status,
                    "exception": exception
                })
                for step in self.search_space:
                    step_name_tmp = step["name"]
                    if step_name_tmp == step_name:
                        d = comp["class"]
                    else:
                        d = [_[1] for _ in decisions if _[0] == step_name_tmp]
                        if d:
                            d = d[0]["class"]
                        else:
                            d = None
                    summary[f"{step_name_tmp}_class"] = d
                    summary[f"{step_name_tmp}_hps"] = None
                self._history.append(summary)

                # update best score
                if not np.isnan(score) and score > best_score:
                    self.logger.debug("This is a NEW BEST SCORE!")
                    best_score = score
                    components_with_score[step_name] = score
                    decision = comp
                    if score > self.best_score_overall:
                        self.best_score_overall = score
                        self.logger.debug(f"Updating new best internal pipeline to {pl}")
                        self.pl = pl

                # update progress bar
                if self.show_progress:
                    pbar.update(1)

            if decision is None:
                if step_name == "learner":
                    self.logger.error(
                        "No learner was chosen in the initial phase."
                        "This is typically caused by too low timeouts or bugs in a custom scoring function."
                    )
                    break
                self.logger.debug("No component chosen for this slot. Leaving it blank")
            else:
                self.logger.debug(f"Added {decision['class']} as the decision for step {step_name}")
                decisions.append((step_name, decision))

        # ordering decisions by their order in the pipeline
        decisions_tmp = [d for d in decisions]
        decisions = []
        for step in self.search_space:
            if is_component_defined_in_steps(decisions_tmp, step["name"]):
                decisions.append(get_step_with_name(decisions_tmp, step["name"]))

        self.decisions = decisions
        self.components_with_score = components_with_score
        self.logger.info(
            "Algorithm Selection ready."
            "Decisions: " +
            "".join([
                "\n\t" + str((d[0], d[1]["class"])) + " with performance " + str(components_with_score[d[0]])
                for d in decisions])
        )

        # close progress bar
        if self.show_progress:
            pbar.close()

        # compile history
        keys = list(self._history[0].keys())
        return pd.DataFrame({key: [e[key] for e in self._history] for key in keys}, columns=keys)

    def get_config_space(self, as_report):
        space = self.hpo_helper.get_config_space_for_selected_algorithms({
            step['name']: as_report[f"{step['name']}_class"]
            for step in self.search_space
            if as_report[f"{step['name']}_class"] is not None
        })
        return space

    def create_history_descriptor(self, base_pl_descriptor, hp_config):
        descriptor = base_pl_descriptor.copy()
        hpo_entries = self.hpo_helper.get_hps_by_step(hp_config)
        for k, v in hpo_entries.items():
            descriptor[f"{k}_hps"] = v

        # now build pipeline object
        steps = []
        for step in self.search_space:
            step_name = step["name"]
            if base_pl_descriptor[f"{step_name}_class"] is not None:
                comp = [c for c in step["components"] if c["class"] == base_pl_descriptor[f"{step_name}_class"]][0]
                steps.append(
                    (
                        step_name,
                        build_estimator(
                            comp,
                            hpo_entries[step_name],
                            X=self.task.X,
                            y=self.task.y
                        )
                    )
                )
        pl = Pipeline(steps=self.mandatory_pre_processing + steps)
        descriptor["pipeline"] = pl
        return descriptor

    @staticmethod
    def get_preprocessing_steps(
            categorical_features,
            missing_values_per_feature,
            numeric_transformer,
            numeric_features,
            sparse
    ):
        if not isinstance(categorical_features, Iterable):
            raise ValueError(f"categorical_features must be iterable but is {type(categorical_features)}")
        if not isinstance(missing_values_per_feature, Iterable):
            raise ValueError(f"missing_values_per_feature must be iterable but is {type(missing_values_per_feature)}")
        if not isinstance(sparse, bool):
            raise ValueError(f"`sparse` must be a bool but is {type(sparse)}")
        if len(categorical_features) > 0 or sum(missing_values_per_feature) > 0:
            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("binarizer", OneHotEncoder(handle_unknown='ignore', sparse_output=sparse)),

            ])
            return [("impute_and_binarize", ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            ))]
        else:
            return []

    def get_mandatory_preprocessing(self, X=None, y=None, categorical_features=None):

        if (X is None or y is None) and (self.X is None or self.y is None):
            raise ValueError(
                "Cannot infer pre-processing without data having been set. "
                "Use the reset method to do so or pass arguments X or y."
            )

        if X is None:
            X = self.X
        if y is None:
            y = self.y

        # determine categorical attributes and necessity of binarization
        sparse_training_data = sp.sparse.issparse(X)
        if isinstance(X, pd.DataFrame):
            if categorical_features is None:
                categorical_features = list(X.select_dtypes(exclude=np.number).columns)
            else:
                categorical_features = [c if not isinstance(c, int) else X.columns[c] for c in categorical_features]
            numeric_features = [c for c in X.columns if c not in categorical_features]

        elif isinstance(X, np.ndarray) or sparse_training_data:
            if categorical_features is None:
                types = [set([type(v) for v in r]) for r in X.T]
                categorical_features = [c for c, t in enumerate(types) if len(t) != 1 or list(t)[0] == str]
            numeric_features = [c for c in range(X.shape[1]) if c not in categorical_features]
        else:
            raise TypeError(
                f"Given data X is of type {type(X)} but must be pandas dataframe, numpy array or sparse scipy matrix.")

        # check necessity of imputation
        if isinstance(X, sp.sparse.spmatrix):
            missing_values_per_feature = X.shape[0] - X.getnnz(axis=0)
        else:
            missing_values_per_feature = np.sum(pd.isnull(X), axis=0)
        self.logger.info(f"There are {len(categorical_features)} categorical features, which will be binarized.")
        self.logger.info(f"Missing values for the different attributes are {missing_values_per_feature}.")
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

        # get preprocessing steps
        try_dense = self.sparse in [False, None]
        steps = self.get_preprocessing_steps(
            categorical_features,
            missing_values_per_feature,
            numeric_transformer,
            numeric_features,
            not try_dense
        )
        if not steps or self.sparse is not None:
            return steps
        try:
            Pipeline(steps).fit_transform(X, y)
            return steps
        except np.core._exceptions._ArrayMemoryError:
            return self.get_preprocessing_steps(
                categorical_features,
                missing_values_per_feature,
                numeric_transformer,
                numeric_features,
                sparse=True
            )
        except Exception:
            raise

    def get_standard_learner_instance(self, X, y):
        return self.standard_classifier() if self.inferred_task_type == "classification" else self.standard_regressor()




    def get_shuffled_version_of_search_space(self):
        search_space = []
        for step in self.search_space:
            comps = step["components"].copy()
            random.shuffle(comps)
            search_space.append(step)
        return search_space

    def check_combinations(self, X, y):

        pool = self.get_evaluation_pool(X, y)
        algorithms_per_stage = []
        names = []
        for step in self.search_space:
            names.append(step["name"])
            cands = []
            if step["name"] != "learner":
                cands.append(None)
            cands.extend([get_class(comp["class"]) for comp in step["components"]])
            algorithms_per_stage.append(cands)

        for combo in it.product(*algorithms_per_stage):
            pl = Pipeline(steps=[(names[i], clazz()) for i, clazz in enumerate(combo) if clazz is not None])
            if pool.is_pipeline_forbidden(pl):
                self.logger.debug("SKIP FORBIDDEN")
            else:
                pool.evaluate(pl, timeout=self.execution_timeout)

    def get_instances_of_currently_selected_components_per_step(self, hpo_process, X, y):
        steps = []
        for step_name, comp in hpo_process.comps_by_steps:
            params = hpo_process.get_best_config(step_name)
            steps.append((step_name, build_estimator(comp, params, X, y)))
        return steps

    def get_pipeline_for_decision_in_step(self, step_name, comp, X, y, decisions):

        config = config_json.read(json.dumps(comp["params"])).get_default_configuration() if comp is not None else None

        if self.strictly_naive:  # strictly naive case

            # build pipeline to be evaluated here
            if step_name == "learner":
                steps = [("learner", build_estimator(comp, config, X, y))]
            elif comp is None:
                steps = [("learner", self.get_standard_learner_instance(X, y))]
            else:
                steps = [
                    (step_name, build_estimator(comp, config, X, y)),
                    ("learner", self.get_standard_learner_instance(X, y))
                ]
            return Pipeline(steps=self.mandatory_pre_processing + steps)

        else:  # semi-naive case (consider previous decisions)
            steps_tmp = [(s[0], build_estimator(s[1], None, X, y)) for s in decisions]
            if comp is not None:
                steps_tmp.append((step_name, build_estimator(comp, config, X, y)))
            steps_ordered = []
            for step_inner in self.search_space:
                if is_component_defined_in_steps(steps_tmp, step_inner["name"]):
                    steps_ordered.append(get_step_with_name(steps_tmp, step_inner["name"]))
            return Pipeline(steps=self.mandatory_pre_processing + steps_ordered)

    def build_pipeline(self, hpo_process, X, y):
        steps = self.get_instances_of_currently_selected_components_per_step(hpo_process, X, y)
        pl = Pipeline(self.mandatory_pre_processing + steps)
        self.logger.debug(f"Original final pipeline is: {pl}")
        i = 0
        while hpo_process.pool.is_pipeline_forbidden(pl):
            i += 1
            self.logger.debug("Invalid pipeline, removing first element!")
            pl = Pipeline(steps=self.mandatory_pre_processing + steps[i:])
        return pl

    def get_pipeline_descriptor(self, pl):
        descriptor = []
        pl_step_names = [s[0] for s in pl.steps]
        for step in self.opt_ordering:
            if step in pl_step_names:
                component = pl[pl_step_names.index(step)]

                comp_name = component.__class__.__name__
                comp_params = component.get_params()

                has_default_hps = True
                #if self.hpo_process is None:
                #else:
#
#                    has_default_hps = all(
#                        (hp_name not in comp_params) or (comp_params[hp_name] == hp_desc.default_value)
#                        for hp_name, hp_desc in self.hpo_process.config_spaces[step].items()
#                    )  # actually limited because it does not work for mapped hyperparameters!
#
                descriptor.extend([comp_name, comp_params, has_default_hps])

            else:
                descriptor.extend([None, None, None])
        return descriptor

    def substitute_targets(self, y):
        if self.task.inferred_task_type is None:
            raise Exception("Task has not been inferred yet. Run reset to do so.")

        if self.task.inferred_task_type == "regression":
            if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number):
                if isinstance(y, sp.sparse.spmatrix):
                    y = y.toarray().astype(float).reshape(-1)
                else:
                    try:
                        y = np.array([float(v) for v in y])
                    except ValueError:
                        raise Exception(
                            "Identified a regression task, but the target object y cannot be cast to a numpy array."
                        )
                self.logger.info(
                    "Detected a regression problem, and converted nun-numeric vector to numpy array. "
                    f"It has now shape {y.shape}."
                )

        # for classification tasks, internally create a numerical version of the labels to avoid problems
        else:
            y = OrdinalEncoder().fit_transform(np.array([y]).T).T[0]
        return y

    @property
    def description(self):

        # print overview
        summary = ""
        for step in self.search_space:
            summary += "\n" + step["name"]
            for comp in step["components"]:
                summary += "\n\t" + comp['class']
        return summary

