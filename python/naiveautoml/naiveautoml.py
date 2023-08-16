import random
import scipy.sparse
from tqdm import tqdm
import importlib.resources as pkg_resources

# sklearn
from sklearn.compose import ColumnTransformer

# naiveautoml commons
from naiveautoml.commons import *
import numpy as np
import traceback


class NaiveAutoML:

    def __init__(self,
                 search_space=None,
                 scoring=None,
                 side_scores=None,
                 evaluation_fun=None,
                 num_cpus=8,
                 execution_timeout=300,
                 max_hpo_iterations=100,
                 max_hpo_iterations_without_imp=100,
                 max_hpo_time_without_imp=1800,
                 timeout=None,
                 standard_classifier=sklearn.neighbors.KNeighborsClassifier,
                 standard_regressor=sklearn.linear_model.LinearRegression,
                 logger_name=None,
                 show_progress=False,
                 opt_ordering=None,
                 strictly_naive: bool = False,
                 sparse: bool = None,
                 task_type: str = "auto",
                 raise_errors: bool = False):
        """

        :param search_space:
        :param scoring:
        :param side_scores:
        :param evaluation_fun:
        :param num_cpus:
        :param execution_timeout:
        :param max_hpo_iterations:
        :param max_hpo_iterations_without_imp:
        :param max_hpo_time_without_imp:
        :param timeout:
        :param standard_classifier:
        :param standard_regressor:
        :param logger_name:
        :param show_progress:
        :param opt_ordering:
        :param strictly_naive:
        :param sparse: whether data is treated sparsely in pre-processing.
                Default is `None`, and in that case, the sparsity is inferred automatically.
                It is tested whether the mandatory pipeline can execute `fit_transform` without a numpy memory exception.
                If so, no sparsity is applied.
                Otherwise, it is assumed that not enough memory is available, and sparsity is enabled.

        :param task_type:
        :param raise_errors:
        """

        self.search_space = search_space
        if isinstance(search_space, str):
            with open(search_space) as f:
                self.search_space = json.load(f)

        # check validity of scorings
        self.scoring = scoring
        self.side_scores = side_scores
        for s in [scoring] + (side_scores if side_scores is not None else []):
            if s is not None:
                build_scorer(s) # check whether this scorer can be built
        
        self.evaluation_fun = evaluation_fun
        self.num_cpus = num_cpus
        self.execution_timeout = execution_timeout
        self.max_hpo_iterations = max_hpo_iterations
        self.max_hpo_iterations_without_imp = max_hpo_iterations_without_imp
        self.max_hpo_time_without_imp = max_hpo_time_without_imp
        self.strictly_naive = strictly_naive
        self.timeout = timeout
        self.show_progress = show_progress
        self.stage_entrypoints = {}
        self.standard_classifier = standard_classifier
        self.standard_regressor = standard_regressor
        self.raise_errors = raise_errors

        # state variables
        self.start_time = None
        self.deadline = None
        self.best_score_overall = None
        self.chosen_model = None
        self.chosen_attributes = None
        
        # mandatory pre-processing steps
        self.sparse = sparse
        self.mandatory_pre_processing = None
        
        self.task_type = task_type
        self.opt_ordering = opt_ordering

        # init logger
        self.logger_name = logger_name
        self.logger = logging.getLogger('naiveautoml' if logger_name is None else logger_name)

    def get_task_type(self, X, y):
        """
        :param X: the descriptions of the instances
        :param y: the labels of the instances
        :return:
        """
        # infer task type
        if self.task_type == "auto":
            if type(self.scoring) == str:
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
            else:
                return "regression" if len(np.unique(y)) > 100 else "classification"
        else:
            return self.task_type

    def get_evaluation_pool(self, X, y):
        task_type = self.get_task_type(X, y)
        return EvaluationPool(
            task_type,
            X,
            y,
            scoring=self.scoring,
            side_scores=self.side_scores,
            evaluation_fun=self.evaluation_fun,
            logger_name=None if self.logger_name is None else self.logger_name + ".pool"
        )

    def get_standard_learner_instance(self, X, y):
        return self.standard_classifier() if self.get_task_type(X, y) == "classification" else self.standard_regressor()
        
    def register_search_space(self, X, y):
        
        task_type = self.get_task_type(X, y)
        self.logger.info(f"Automatically inferred task type: {task_type}")
        
        ''' search_space is a string or a list of dictionaries
            -   if it is a dict, the last one for the learner and all the others for pre-processing.
                Each dictionary has an entry "name" and an entry "components",
                which is a list of components with their parameters.
            - if it is a string, a json file with the name of search_space is read in with the same semantics
         '''
        json_str = pkg_resources.read_text('naiveautoml', 'searchspace-' + task_type + '.json')
        self.search_space = json.loads(json_str)
        
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
            pl = sklearn.pipeline.Pipeline(steps=[(names[i], clazz()) for i, clazz in enumerate(combo) if clazz is not None])
            if is_pipeline_forbidden(pl):
                self.logger.debug("SKIP FORBIDDEN")
            else:
                pool.evaluate(pl, timeout=self.execution_timeout)
                
    def get_instances_of_currently_selected_components_per_step(self, hpo_processes, X, y):
        steps = []
        for step in self.search_space:
            step_name = step["name"]
            if step_name in hpo_processes:
                hpo = hpo_processes[step_name]
                comp = hpo.comp
                params = hpo.get_best_config()
                steps.append((step_name, build_estimator(comp, params, X, y)))
        return steps
    
    def get_pipeline_for_decision_in_step(self, step_name, comp, X, y, decisions):
        
        if self.strictly_naive: # strictly naive case
            
            # build pipeline to be evaluated here
            if step_name == "learner":
                steps = [("learner", build_estimator(comp, None, X, y))]
            elif comp is None:
                steps = [("learner", self.get_standard_learner_instance(X, y))]
            else:
                steps = [(step_name, build_estimator(comp, None, X, y)), ("learner", self.get_standard_learner_instance(X, y))]
            return Pipeline(steps=self.mandatory_pre_processing + steps)
        
        else: # semi-naive case (consider previous decisions)
            steps_tmp = [(s[0], build_estimator(s[1], None, X, y)) for s in decisions]
            if comp is not None:
                steps_tmp.append((step_name, build_estimator(comp, None, X, y)))
            steps_ordered = []
            for step_inner in self.search_space:
                if is_component_defined_in_steps(steps_tmp, step_inner["name"]):
                    steps_ordered.append(get_step_with_name(steps_tmp, step_inner["name"]))
            return Pipeline(steps=self.mandatory_pre_processing + steps_ordered)

    def build_pipeline(self, hpo_processes, X, y):
        steps = self.get_instances_of_currently_selected_components_per_step(hpo_processes, X, y)
        pl = Pipeline(self.mandatory_pre_processing + steps)
        self.logger.debug(f"Original final pipeline is: {pl}")
        i = 0
        while is_pipeline_forbidden(pl):
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
                descriptor.append(component.__class__.__name__)
                descriptor.append(component.get_params())
            else:
                descriptor.extend([None, None])
        return descriptor
    
    def choose_algorithms(self, X, y):
        
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

        for step_index, step_name in enumerate(self.opt_ordering):

            # create list of components to try for this slot
            step = [step for step in self.search_space if step["name"] == step_name][0]
            self.logger.info("--------------------------------------------------")
            self.logger.info(f"Selecting component for step with name: {step_name}")
            self.logger.info("--------------------------------------------------")
            if step_name not in ["learner"]:
                components = [None] + step["components"]
            else:
                components = step["components"]
            
            # find best default parametrization for this slot (depending on choice of previously configured slots)
            pool = self.get_evaluation_pool(X, y)
            best_score = -np.inf
            decision = None
            for comp in components:
                if self.deadline is not None:
                    remaining_time = self.deadline - 10 - time.time()
                    if remaining_time is not None and remaining_time < 0:
                        self.logger.info("Timeout approaching. Not evaluating anymore for this stage.")
                        break
                    else:
                        self.logger.info(f"Evaluating {comp['class'] if comp is not None else None}. Timeout: {self.execution_timeout}. Remaining time: {remaining_time}")
                
                # get and evaluate pipeline for this step
                pl = self.get_pipeline_for_decision_in_step(step_name, comp, X, y, decisions)
                exception = None
                timeout = False
                status = "ok"
                try:
                    if self.execution_timeout is None:
                        timeout = None
                    else:
                        timeout = min(self.execution_timeout, remaining_time if self.deadline is not None else 10**10)
                    scores = pool.evaluate(pl, timeout)
                except KeyboardInterrupt:
                    raise
                except pynisher.WallTimeoutException:
                    self.logger.info("TIMEOUT! No result observed for candidate.")
                    timeout = True
                    status = "timeout"
                except:
                    exception = traceback.format_exc()
                    status = "exception"
                    if self.raise_errors:
                        raise
                    else:
                        self.logger.warning(f"Observed exception during the evaluation. The trace is as follows:\n{exception}")
                if status != "ok":
                    scores = {scoring: np.nan for scoring in [self.scoring] + (self.side_scores if self.side_scores is not None else [])}
                score = scores[get_scoring_name(self.scoring)]
                self.logger.info(f"Observed score of {score} for default configuration of {None if comp is None else comp['class']}")
                
                # update history
                self.history.append({
                    "time": time.time() - self.start_time,
                    "pl": self.get_pipeline_descriptor(pl),
                    "score_internal": score,
                    "scores": scores,
                    "new_best": score > self.best_score_overall,
                    "status": status,
                    "exception": exception
                })
                
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
                    self.logger.error("No learner was chosen in the initial phase. This is typically caused by too low timeouts or bugs in a custom scoring function (if applicable).")
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
        self.logger.info("Algorithm Selection ready. Decisions: " + "".join(["\n\t" + str((d[0], d[1]["class"])) + " with performance " + str(components_with_score[d[0]]) for d in decisions]))
        
        # close progress bar
        if self.show_progress:
            pbar.close()
    
    def tune_parameters(self, X, y):
        
        # now conduct HPO until there is no local improvement or the deadline is hit
        self.logger.info("--------------------------------------------------")
        self.logger.info("Entering HPO phase")
        self.logger.info("--------------------------------------------------")
        task_type = self.get_task_type(X, y)
        
        # read variables from state
        decisions = self.decisions
        components_with_score = self.components_with_score
        
        # create HPO processes for each slot, taking into account the default parametrized component of each other slot
        self.hpo_processes = hpo_processes = {}
        step_names = [d[0] for d in decisions]
        for step_name, comp in decisions:
            if step_name == "learner":
                other_instances = [(step_name, None)]
            else:
                other_instances = [(step_name, None), ("learner", self.get_standard_learner_instance(X, y))]
            index = 0 # it is (rather by coincidence) the first step we want to optimize
            hpo = HPOProcess(
                task_type,
                step_name,
                comp,
                X,
                y,
                scoring=self.scoring,
                side_scores=self.side_scores,
                evaluation_fun=self.evaluation_fun,
                execution_timeout=self.execution_timeout,
                mandatory_pre_processing=self.mandatory_pre_processing,
                other_step_component_instances=other_instances,
                index_in_steps=index,
                max_time_without_imp=self.max_hpo_time_without_imp,
                max_its_without_imp=self.max_hpo_iterations_without_imp,
                allow_exhaustive_search=(self.max_hpo_iterations is None),
                logger_name=None if self.logger_name is None else self.logger_name + ".hpo"
            )
            hpo.best_score = components_with_score[step_name] # performance of default config
            hpo_processes[step_name] = hpo
        
        # starting HPO process
        opt_round = 1
        rs = np.random.RandomState()
        active_for_optimization = [name for name, hpo in hpo_processes.items() if hpo.active]
        round_runtimes = []
        if self.show_progress:
            print("Progress for parameter tuning:")
            pbar = tqdm(total = self.max_hpo_iterations)
            
        while active_for_optimization and (self.max_hpo_iterations is None or opt_round <= self.max_hpo_iterations):
            self.logger.info(f"Entering optimization round {opt_round}")
            if self.deadline is not None:
                remaining_time = self.deadline - (np.mean(round_runtimes) if round_runtimes else 0) - 10 - time.time()
                if remaining_time < 0:
                    self.logger.info("Timeout almost exhausted, stopping HPO phase")
                    break
                self.logger.debug("Remaining time is: " + str(remaining_time) + "s.")
            else:
                remaining_time = None
            
            round_start = time.time()
            inactive = []
            for name in active_for_optimization:
                hpo = hpo_processes[name]
                self.logger.debug(f"Stepping HPO for {name}")
                try:
                    res = hpo.step(remaining_time)
                    if res is not None:
                        pl, status, scores, runtime, exception = res
                        score = scores[get_scoring_name(self.scoring)]
                        if score > self.best_score_overall:
                            self.best_score_overall = score
                        self.history.append({
                            "time": time.time() - self.start_time,
                            "pl": self.get_pipeline_descriptor(pl),
                            "score_internal": score,
                            "scores": scores,
                            "new_best": score > self.best_score_overall,
                            "status": status,
                            "exception": exception}
                        )
                    if not hpo.active:
                        self.logger.info(f"Deactivating {name}")
                        inactive.append(name)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.logger.error(f"An error occurred in the HPO step: {e}")
                    raise
                    
            round_runtimes.append(time.time() - round_start)
            for name in inactive:
                active_for_optimization.remove(name)
            opt_round += 1
            newPl = self.build_pipeline(hpo_processes, X, y)
            if str(newPl) != str(self.pl):
                self.logger.info(f"Updating new best internal pipeline to {newPl}")
                self.pl = newPl
            
            # update progress bar
            if self.show_progress:
                pbar.update(1)
         
        # close progress bar for HPO
        if self.show_progress:
            pbar.close()

    def get_mandatory_preprocessing(self, X, y, categorical_features):

        # determine categorical attributes and necessity of binarization
        sparse_training_data = isinstance(X, (scipy.sparse.csr.csr_matrix, scipy.sparse.lil.lil_matrix))
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
        missing_values_per_feature = np.sum(pd.isnull(X), axis=0)
        self.logger.info(f"There are {len(categorical_features)} categorical features, which will be binarized.")
        self.logger.info(f"Missing values for the different attributes are {missing_values_per_feature}.")
        numeric_transformer = Pipeline([("imputer", sklearn.impute.SimpleImputer(strategy="median"))])

        # get preprocessing steps
        try_dense = self.sparse in [False, None]
        steps = self.get_preprocessing_steps(categorical_features, missing_values_per_feature, numeric_transformer, numeric_features, not try_dense)
        if not steps or self.sparse is not None:
            return steps
        try:
            Pipeline(steps).fit_transform(X, y)
            return steps
        except np.core._exceptions._ArrayMemoryError:
            return self.get_preprocessing_steps(categorical_features, missing_values_per_feature, numeric_transformer,
                                                 numeric_features, sparse=True)
        except:
            raise

    @staticmethod
    def get_preprocessing_steps(categorical_features, missing_values_per_feature, numeric_transformer, numeric_features, sparse):
        if type(sparse) != bool:
            raise ValueError(f"`sparse` must be a bool but is {type(sparse)}")
        if len(categorical_features) > 0 or sum(missing_values_per_feature) > 0:
            categorical_transformer = Pipeline([
                ("imputer", sklearn.impute.SimpleImputer(strategy="most_frequent")),
                ("binarizer", sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=sparse)),

            ])
            return [("impute_and_binarize", ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            ))]
        else:
            return []

    def reset(self, X, y, categorical_features=None):
        # register search space
        self.register_search_space(X, y)

        # initialize
        self.pl = None
        self.best_score_overall = -np.inf
        self.history = []
        self.start_time = time.time()
        self.deadline = self.start_time + self.timeout if self.timeout is not None else None
        if self.scoring is None:
            task_type = self.get_task_type(X, y)
            if task_type == "classification":
                self.scoring = "roc_auc" if len(np.unique(y)) == 2 else "neg_log_loss"
            else:
                self.scoring = "neg_mean_squared_error"

        # show start message
        self.logger.info(f"""Optimizing pipeline for data with shape {X.shape}.
                Timeout: {self.timeout}
                Timeout per execution: {self.execution_timeout}
                Scoring: {self.scoring}""")

        self.mandatory_pre_processing = self.get_mandatory_preprocessing(X, y, categorical_features)

        # print overview
        summary = ""
        for step in self.search_space:
            summary += "\n" + step["name"]
            for comp in step["components"]:
                summary += "\n\t" + comp['class']
        self.logger.info(f"These are the components used by NaiveAutoML in the upcoming process (by steps):{summary}")

    def fit(self, X, y, categorical_features=None):

        self.reset(X, y, categorical_features)
        
        # choose algorithms
        self.choose_algorithms(X, y)
        if self.decisions:
            self.steps_after_which_algorithm_selection_was_completed = len(self.history)

            # tune parameters
            self.tune_parameters(X, y)

            # train final pipeline
            self.logger.info("--------------------------------------------------")
            self.logger.info("Search Completed. Building final pipeline.")
            self.logger.info("--------------------------------------------------")
            self.pl = self.build_pipeline(self.hpo_processes, X, y)
            self.logger.info(self.pl)
            self.logger.info("Now fitting the pipeline with all given data.")
            while len(self.pl.steps) > 0:
                try:
                    self.pl.fit(X, y)
                    break
                except:
                    self.logger.warning("There was a problem in building the pipeline, cutting it one down!")
                    self.pl = Pipeline(steps=self.pl.steps[1:])
                    self.logger.warning("new pipeline is:", self.pl)

            if len(self.pl.steps) > 0:
                self.chosen_model = self.pl
            else:
                self.logger.warning("For some reason (should happen), the final pipeline was reduced to empty.")
        else:
            self.logger.info("No model was chosen in first phase, so there is nothing to return for me ...")
        self.end_time = time.time()

        # compile history
        history_keys = ["time"]
        for step in self.opt_ordering:
            history_keys.append(step + "_class")
            history_keys.append(step + "_hps")
        history_keys.extend([self.scoring] + (self.side_scores if self.side_scores is not None else []) + ["new_best", "status", "exception"])

        history_rows = []
        for row in self.history:
            row_formatted = [row["time"]] + row["pl"] + [row["score_internal"]]
            if self.side_scores is not None:
                for s in self.side_scores:
                    row_formatted.append(row["scores"][s])
            row_formatted.extend([row["new_best"], row["status"], row["exception"]])
            history_rows.append(row_formatted)
        self.history = pd.DataFrame(history_rows, columns=history_keys)
        self.logger.info(f"Runtime was {self.end_time - self.start_time} seconds")
        
    def eval_history(self, X, y):
        pool = self.get_evaluation_pool(X, y)
        scores = []
        for entry in self.history:
            scores.append(pool.evaluate(entry["pl"]))
        return scores

    def predict(self, X):
        return self.pl.predict(X)
    
    def predict_proba(self, X):
        return self.pl.predict_proba(X)