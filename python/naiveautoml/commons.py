import numpy as np
import pandas as pd
import logging
import warnings
import itertools as it
import os
import psutil
import scipy.sparse
import time

# sklearn
import sklearn
import sklearn.model_selection
import sklearn.base
import sklearn.feature_selection
import sklearn.kernel_approximation
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.naive_bayes
import sklearn.tree
import sklearn.svm
import sklearn.neural_network
import sklearn.discriminant_analysis
import sklearn.linear_model
import sklearn.ensemble
import sklearn.cluster
import sklearn.gaussian_process
from sklearn.pipeline import Pipeline
from sklearn.metrics import get_scorer
from sklearn.metrics import make_scorer

# configspace
import ConfigSpace.hyperparameters
from ConfigSpace.read_and_write import json as config_json
import json
import traceback

import pynisher


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    try:
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)
    except Exception:
        parts = kls.split('.')
        parts[2] = "_data"
        module = ".".join(parts[:-1])
        m = __import__(module)
        for comp in parts[1:]:
            m = getattr(m, comp)

    return m


def is_component_defined_in_steps(steps, name):
    candidates = [s[1] for s in steps if s[0] == name]
    return len(candidates) > 0


def get_step_with_name(steps, name):
    candidates = [s for s in steps if s[0] == name]
    return candidates[0]


def get_scoring_name(scoring):
    return scoring if isinstance(scoring, str) else scoring["name"]


def build_scorer(scoring):
    return get_scorer(scoring) if isinstance(scoring, str) else make_scorer(
        **{key: val for key, val in scoring.items() if key != "name"})


def get_evaluation_fun(instance, evaluation_fun):

    from .evaluators import\
        LccvValidator, KFold, Mccv

    is_small_dataset = instance.X.shape[0] < 2000

    if evaluation_fun is None:
        if is_small_dataset:
            instance.logger.info("This is a small dataset, choosing mccv-5 for evaluation")
            return Mccv(instance, n_splits=5)
        else:
            instance.logger.info("Dataset is not small. Using LCCV-80 for evaluation")
            return LccvValidator(instance, 0.8)

    elif evaluation_fun == "lccv-80":
        return LccvValidator(instance, 0.8)
    elif evaluation_fun == "lccv-90":
        return LccvValidator(instance, 0.9)
    elif evaluation_fun == "kfold_5":
        return KFold(instance, n_splits=5)
    elif evaluation_fun == "kfold_3":
        return KFold(instance, n_splits=3)
    elif evaluation_fun == "mccv_1":
        return Mccv(instance, n_splits=1)
    elif evaluation_fun == "mccv_3":
        return Mccv(instance, n_splits=3)
    elif evaluation_fun == "mccv_5":
        return Mccv(instance, n_splits=5)
    else:
        return evaluation_fun


class EvaluationPool:

    def __init__(self,
                 task_type,
                 X,
                 y,
                 scoring,
                 evaluation_fun=None,
                 side_scores=None,
                 tolerance_tuning=0.05,
                 tolerance_estimation_error=0.01,
                 logger_name=None,
                 use_caching=True
                 ):
        domains_task_type = ["classification", "regression"]
        if task_type not in domains_task_type:
            raise ValueError(f"task_type must be in {domains_task_type}")
        self.task_type = task_type
        self.logger = logging.getLogger('naiveautoml.evalpool' if logger_name is None else logger_name)

        # disable warnings by default
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')

        if not isinstance(X, (
                pd.DataFrame,
                np.ndarray,
                scipy.sparse.csr_matrix,
                scipy.sparse.csc_matrix,
                scipy.sparse.lil_matrix
        )):
            raise TypeError(f"X must be a numpy array but is {type(X)}")
        if y is None:
            raise TypeError("Parameter y must not be None")

        self.X = X
        self.y = y
        self.scoring = scoring
        if side_scores is None:
            self.side_scores = []
        elif isinstance(side_scores, list):
            self.side_scores = side_scores
        else:
            self.logger.warning("side scores was not given as list, casting it to a list of size 1 implicitly.")
            self.side_scores = [side_scores]
        self.evaluation_fun = get_evaluation_fun(self, evaluation_fun)
        self.bestScore = -np.inf
        self.tolerance_tuning = tolerance_tuning
        self.tolerance_estimation_error = tolerance_estimation_error
        self.cache = {}
        self.use_caching = use_caching

    def tellEvaluation(self, pl, scores, evaluation_report, timestamp):
        spl = str(pl)
        self.cache[spl] = (spl, scores, evaluation_report, timestamp)
        score = np.mean(scores)
        if score > self.bestScore:
            self.bestScore = score
            self.best_spl = spl

    def cross_validate(self, pl, X, y, scorings, errors="raise"):  # just a wrapper to ease parallelism
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            if self.task_type == "classification":
                splitter = sklearn.model_selection.StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
            elif self.task_type:
                splitter = sklearn.model_selection.KFold(n_splits=5, random_state=None, shuffle=True)
            scores = {get_scoring_name(scoring): [] for scoring in scorings}
            for train_index, test_index in splitter.split(X, y):

                X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
                y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
                y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

                pl_copy = sklearn.base.clone(pl)
                pl_copy.fit(X_train, y_train)

                # compute values for each metric
                for scoring in scorings:
                    scorer = build_scorer(scoring)
                    try:
                        score = scorer(pl_copy, X_test, y_test)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        score = np.nan
                        if errors == "message":
                            self.logger.info(f"Observed exception in validation of pipeline {pl_copy}. Placing nan.")
                        else:
                            raise

                    scores[get_scoring_name(scoring)].append(score)
            return scores
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.logger.error(f"Observed an error: {e}")
                return None
            else:
                raise

    def evaluate(self, pl, timeout=None):

        # disable warnings by default
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')

        try:

            if self.is_pipeline_forbidden(pl):
                self.logger.info(f"Preventing evaluation of forbidden pipeline {pl}")
                scores = {get_scoring_name(scoring): np.nan for scoring in [self.scoring] + self.side_scores}
                evaluation_report = None
                if hasattr(self.evaluation_fun, "update"):
                    self.evaluation_fun.update(pl, scores)
                return "avoided", scores, evaluation_report

            process = psutil.Process(os.getpid())
            mem = int(process.memory_info().rss / 1024 / 1024)
            self.logger.info(
                f"Initializing evaluation of {pl}. Current memory consumption {mem}MB. Now awaiting results."
            )

            start_outer = time.time()
            spl = str(pl)
            if self.use_caching and spl in self.cache:
                out = {get_scoring_name(scoring): np.nan for scoring in [self.scoring] + self.side_scores}
                out[get_scoring_name(self.scoring)] = np.round(np.mean(self.cache[spl][1]), 4)
                return "cache", out, self.cache[spl][2]

            timestamp = time.time()
            if timeout is not None:
                if timeout > 1:
                    with pynisher.limit(self.evaluation_fun, wall_time=timeout) as limited_evaluation:
                        if hasattr(self.evaluation_fun, "errors"):
                            scores, evaluation_report = limited_evaluation(
                                pl,
                                self.X,
                                self.y,
                                [self.scoring] + self.side_scores,
                                errors="ignore"
                            )
                        else:
                            scores, evaluation_report = limited_evaluation(
                                pl,
                                self.X,
                                self.y,
                                [self.scoring] + self.side_scores
                            )
                else:  # no time left
                    raise pynisher.WallTimeoutException()
            else:
                scores, evaluation_report = self.evaluation_fun(pl, self.X, self.y, [self.scoring] + self.side_scores)

            # here we give the evaluator the chance to update itself
            # this looks funny, but it is done because the evaluation could have been done with a copy of the evaluator
            if hasattr(self.evaluation_fun, "update"):
                self.evaluation_fun.update(pl, scores)

            # if no score was observed, return results here
            if scores is None:
                return ("failed",
                        {get_scoring_name(scoring): np.nan for scoring in [self.scoring] + self.side_scores},
                        {get_scoring_name(scoring): {} for scoring in [self.scoring] + self.side_scores})
            runtime = time.time() - start_outer

            if not isinstance(scores, dict):
                raise TypeError(f"""
                scores is of type {type(scores)} but must be a dictionary
                with entries for {get_scoring_name(self.scoring)}. Probably you inserted an
                evaluation_fun argument that does not return a proper dictionary."""
                                )

            self.logger.info(f"Completed evaluation of {spl} after {runtime}s. Scores are {scores}")
            self.tellEvaluation(pl, scores[get_scoring_name(self.scoring)], evaluation_report, timestamp)
            return "ok", scores, evaluation_report

        # if there was an exception, then tell the evaluator function about a nan
        except Exception:
            if hasattr(self.evaluation_fun, "update"):
                self.evaluation_fun.update(pl, {
                    get_scoring_name(scoring): np.nan for scoring in [self.scoring] + self.side_scores
                })
            raise

    def is_pipeline_forbidden(self, pl):

        # forbid pipelines with SVMs if the main scoring function requires probabilities
        if pl["learner"].__class__ in [sklearn.svm.SVC, sklearn.svm.LinearSVC]:
            if build_scorer(self.scoring)._response_method == "predict_proba":
                return True

        # forbid pipelines with scalers and trees
        if "data-pre-processor" in [e[0] for e in pl.steps]:
            if pl["learner"].__class__ in [
                sklearn.tree.DecisionTreeClassifier,
                sklearn.tree.DecisionTreeRegressor,
                sklearn.ensemble.ExtraTreesRegressor,
                sklearn.ensemble.ExtraTreesClassifier,
                sklearn.ensemble.HistGradientBoostingClassifier,
                sklearn.ensemble.HistGradientBoostingRegressor,
                sklearn.ensemble.GradientBoostingClassifier,
                sklearn.ensemble.RandomForestClassifier,
                sklearn.ensemble.RandomForestRegressor
            ] and pl["data-pre-processor"].__class__ in [
                sklearn.preprocessing.RobustScaler,
                sklearn.preprocessing.StandardScaler,
                sklearn.preprocessing.MinMaxScaler,
                sklearn.preprocessing.QuantileTransformer
            ]:
                return True  # scaling has no effect onf tree-based classifiers

        # certain pipeline combos are generally forbidden
        forbidden_combos = [
            {
                "data-pre-processor": sklearn.preprocessing.PowerTransformer,
                "feature-pre-processor": sklearn.feature_selection.SelectPercentile
            },
            {
                "data-pre-processor": sklearn.preprocessing.PowerTransformer,
                "feature-pre-processor": sklearn.feature_selection.GenericUnivariateSelect
            },
            {
                "feature-pre-processor": sklearn.decomposition.FastICA,
                "classifier": sklearn.naive_bayes.MultinomialNB
            },
            {
                "feature-pre-processor": sklearn.decomposition.PCA,
                "classifier": sklearn.naive_bayes.MultinomialNB
            },
            {
                "feature-pre-processor": sklearn.decomposition.KernelPCA,
                "classifier": sklearn.naive_bayes.MultinomialNB
            },
            {
                "feature-pre-processor": sklearn.kernel_approximation.RBFSampler,
                "classifier": sklearn.naive_bayes.MultinomialNB
            },
            {
                "feature-pre-processor": sklearn.kernel_approximation.Nystroem,
                "classifier": sklearn.naive_bayes.MultinomialNB
            },
            {
                "data-pre-processor":  sklearn.preprocessing.PowerTransformer,
                "classifier": sklearn.naive_bayes.MultinomialNB
            },
            {
                "data-pre-processor": sklearn.preprocessing.StandardScaler,
                "classifier": sklearn.naive_bayes.MultinomialNB
            },
            {
                "data-pre-processor": sklearn.preprocessing.RobustScaler,
                "classifier": sklearn.naive_bayes.MultinomialNB
            }
        ]

        representation = {}
        for step_name, obj in pl.steps:
            representation[step_name] = obj.__class__

        for combo in forbidden_combos:
            matches = True
            for key, val in combo.items():
                if key not in representation or representation[key] != val:
                    matches = False
                    break
            if matches:
                return True
        return False


def fullname(o):
    module = o.__module__
    if module is None or module == str.__class__.__module__:
        return o.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__name__


def check_true(p: str) -> bool:
    if p in ("True", "true", 1, True):
        return True
    return False


def check_false(p: str) -> bool:
    if p in ("False", "false", 0, False):
        return True
    return False


def check_none(p: str) -> bool:
    if p in ("None", "none", None):
        return True
    return False


def check_for_bool(p: str, allow_non_bool=False) -> bool:
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
        if allow_non_bool:
            return p
        raise ValueError("%s is not a bool" % str(p))


def build_estimator(comp, params, X, y):
    if params is None:
        if get_class(comp["class"]) == sklearn.svm.SVC:
            params = {"kernel": config_json.read(json.dumps(comp["params"])).get_hyperparameter("kernel").value}
        else:
            return get_class(comp["class"])()
    return compile_pipeline_by_class_and_params(get_class(comp["class"]), params, X, y)


def compile_pipeline_by_class_and_params(clazz, params, X, y):
    if clazz == sklearn.cluster.FeatureAgglomeration:
        pooling_func_mapping = dict(mean=np.mean, median=np.median, max=np.max)
        n_clusters = int(params["n_clusters"])
        n_clusters = min(n_clusters, X.shape[1])
        pooling_func = pooling_func_mapping[params["pooling_func"]]
        metric = params["metric"]
        linkage = params["linkage"]
        return sklearn.cluster.FeatureAgglomeration(
            n_clusters=n_clusters,
            metric=metric,
            linkage=linkage,
            pooling_func=pooling_func
        )

    if clazz == sklearn.feature_selection.SelectPercentile:
        percentile = int(float(params["percentile"]))
        score_func = params["score_func"]
        if score_func == "chi2":
            score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            score_func = sklearn.feature_selection.f_classif
        elif score_func == "mutual_info":
            score_func = sklearn.feature_selection.mutual_info_classif
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info'), ""but is: %s" % score_func)
        return sklearn.feature_selection.SelectPercentile(score_func=score_func, percentile=percentile)

    if clazz == sklearn.preprocessing.MinMaxScaler:
        return sklearn.preprocessing.MinMaxScaler()

    if clazz == sklearn.preprocessing.StandardScaler:
        return sklearn.preprocessing.StandardScaler()

    if clazz == sklearn.preprocessing.RobustScaler:
        return sklearn.preprocessing.RobustScaler(quantile_range=(params["q_min"], params["q_max"]), copy=False,)

    if clazz == sklearn.decomposition.PCA:
        n_components = float(params["keep_variance"])
        whiten = check_for_bool(params["whiten"])
        return sklearn.decomposition.PCA(n_components=n_components, whiten=whiten, copy=True)

    if clazz == sklearn.decomposition.KernelPCA:
        return sklearn.decomposition.KernelPCA(**params, copy_X=True)

    if clazz == sklearn.decomposition.FastICA:
        algorithm = params["algorithm"]
        fun = params["fun"]
        whiten = check_for_bool(params["whiten"], allow_non_bool=True)
        n_components = int(params["n_components"]) if whiten else None
        return sklearn.decomposition.FastICA(
            n_components=n_components,
            algorithm=algorithm,
            fun=fun,
            whiten=whiten)

    if clazz == sklearn.preprocessing.PolynomialFeatures:
        include_bias = check_for_bool(params["include_bias"])
        interaction_only = check_for_bool(params["interaction_only"])
        degree = int(params["degree"])
        return sklearn.preprocessing.PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only
        )

    if clazz == sklearn.preprocessing.QuantileTransformer:
        return sklearn.preprocessing.QuantileTransformer(**params)

    if clazz == sklearn.preprocessing.PowerTransformer:
        return sklearn.preprocessing.PowerTransformer()

    if clazz == sklearn.preprocessing.Normalizer:
        return sklearn.preprocessing.Normalizer()

    if clazz == sklearn.kernel_approximation.RBFSampler:
        return sklearn.kernel_approximation.RBFSampler(**params)

    if clazz == sklearn.kernel_approximation.Nystroem:
        return sklearn.kernel_approximation.Nystroem(**params)

    if clazz == sklearn.feature_selection.VarianceThreshold:
        return sklearn.feature_selection.VarianceThreshold()

    if clazz == sklearn.feature_selection.GenericUnivariateSelect:
        alpha = params["alpha"]
        mode = params["mode"] if "mode" in params else None
        score_func = params["score_func"]
        if score_func == "chi2":
            score_func = sklearn.feature_selection.chi2
        elif score_func == "f_classif":
            score_func = sklearn.feature_selection.f_classif
        elif score_func == "mutual_info_classif":
            score_func = sklearn.feature_selection.mutual_info_classif
            # mutual info classif constantly crashes without mode percentile
            mode = 'percentile'
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info_classif') "
                             "for classification "
                             "but is: %s " % (score_func))
        return sklearn.feature_selection.GenericUnivariateSelect(score_func=score_func, param=alpha, mode=mode)

    if clazz == sklearn.neighbors.KNeighborsClassifier:
        return sklearn.neighbors.KNeighborsClassifier(**params)

    if clazz == sklearn.naive_bayes.BernoulliNB:
        alpha = float(params["alpha"])
        fit_prior = check_for_bool(params["fit_prior"])
        return sklearn.naive_bayes.BernoulliNB(alpha=alpha, fit_prior=fit_prior)

    if clazz == sklearn.naive_bayes.GaussianNB:
        return sklearn.naive_bayes.GaussianNB(**params)

    if clazz == sklearn.naive_bayes.MultinomialNB:
        alpha = float(params["alpha"])
        fit_prior = check_for_bool(params["fit_prior"])
        return sklearn.naive_bayes.MultinomialNB(alpha=alpha, fit_prior=fit_prior)

    if clazz == sklearn.tree.DecisionTreeClassifier:
        criterion = params["criterion"]
        max_features = float(params["max_features"])
        # Heuristic to set the tree depth
        if check_none(params["max_depth_factor"]):
            max_depth_factor = None
        else:
            num_features = X.shape[1]
            max_depth_factor = int(params["max_depth_factor"])
            max_depth_factor = max(
                1,
                int(np.round(max_depth_factor * num_features, 0)))
        min_samples_split = int(params["min_samples_split"])
        min_samples_leaf = int(params["min_samples_leaf"])
        if check_none(params["max_leaf_nodes"]):
            max_leaf_nodes = None
        else:
            max_leaf_nodes = int(params["max_leaf_nodes"])
        min_weight_fraction_leaf = float(params["min_weight_fraction_leaf"])
        min_impurity_decrease = float(params["min_impurity_decrease"])

        return sklearn.tree.DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth_factor,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=None)

    if clazz == sklearn.linear_model.ARDRegression:
        params = dict(params).copy()
        params["fit_intercept"] = check_for_bool(params["fit_intercept"])
        return sklearn.linear_model.ARDRegression(**params)

    if clazz == sklearn.neighbors.KNeighborsRegressor:
        return sklearn.neighbors.KNeighborsRegressor(**params)

    if clazz == sklearn.tree.DecisionTreeRegressor:
        criterion = params["criterion"]
        max_features = float(params["max_features"])
        if check_none(params["max_depth_factor"]):
            max_depth_factor = params["max_depth_factor"] = None
        else:
            num_features = X.shape[1]
            max_depth_factor = int(params["max_depth_factor"])
            max_depth_factor = max(
                1, int(np.round(params["max_depth_factor"] * num_features, 0))
            )
        min_samples_split = int(params["min_samples_split"])
        min_samples_leaf = int(params["min_samples_leaf"])
        if check_none(params["max_leaf_nodes"]):
            max_leaf_nodes = None
        else:
            max_leaf_nodes = int(params["max_leaf_nodes"])
        min_weight_fraction_leaf = float(params["min_weight_fraction_leaf"])
        min_impurity_decrease = float(params["min_impurity_decrease"])

        return sklearn.tree.DecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth_factor,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_impurity_decrease=min_impurity_decrease
        )

    if clazz == sklearn.ensemble.AdaBoostRegressor:
        n_estimators = int(params["n_estimators"])
        learning_rate = float(params["learning_rate"])
        max_depth = int(params["max_depth"])
        base_estimator = sklearn.tree.DecisionTreeRegressor(max_depth=max_depth)

        return sklearn.ensemble.AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )

    if clazz == sklearn.ensemble.ExtraTreesRegressor:
        n_estimators = 10**3
        if params["criterion"] not in ("squared_error", "friedman_mse", "absolute_error", "poisson"):
            raise ValueError(
                "'criterion' is not in ('squared_error', 'friedman_mse', 'absolute_error', 'poisson'): "
                f"{params['criterion']}"
            )

        if check_none(params["max_depth"]):
            max_depth = None
        else:
            max_depth = int(params["max_depth"])

        if check_none(params["max_leaf_nodes"]):
            max_leaf_nodes = None
        else:
            max_leaf_nodes = int(params["max_leaf_nodes"])

        min_samples_leaf = int(params["min_samples_leaf"])
        min_samples_split = int(params["min_samples_split"])
        max_features = float(params["max_features"])
        min_impurity_decrease = float(params["min_impurity_decrease"])
        min_weight_fraction_leaf = float(params["min_weight_fraction_leaf"])
        bootstrap = check_for_bool(params["bootstrap"])

        return sklearn.ensemble.ExtraTreesRegressor(
            n_estimators=n_estimators,
            criterion=params["criterion"],
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_impurity_decrease=min_impurity_decrease,
            oob_score=False,
            n_jobs=1
        )

    if clazz == sklearn.gaussian_process.GaussianProcessRegressor:
        alpha = float(params["alpha"])

        kernel = sklearn.gaussian_process.kernels.RBF()

        # Instantiate a Gaussian Process model
        return sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            optimizer="fmin_l_bfgs_b",
            alpha=alpha,
            copy_X_train=True,
            normalize_y=True,
        )

    if clazz == sklearn.ensemble.HistGradientBoostingRegressor:
        learning_rate = float(params["learning_rate"])
        quantile = float(params["quantile"]) if "quantile" in params else None
        max_iter = 1000
        min_samples_leaf = int(params["min_samples_leaf"])

        if check_none(params["max_depth"]):
            max_depth = None
        else:
            max_depth = int(params["max_depth"])
        if check_none(params["max_leaf_nodes"]):
            max_leaf_nodes = None
        else:
            max_leaf_nodes = int(params["max_leaf_nodes"])
        max_bins = int(params["max_bins"])
        l2_regularization = float(params["l2_regularization"])
        tol = float(params["tol"])
        if check_none(params["scoring"]):
            scoring = None
        else:
            scoring = params["scoring"]

        if params["early_stop"] == "off":
            n_iter_no_change = 1
            validation_fraction_ = None
        elif params["early_stop"] == "train":
            n_iter_no_change = int(params["n_iter_no_change"])
            validation_fraction_ = None
        elif params["early_stop"] == "valid":
            n_iter_no_change = int(params["n_iter_no_change"])
            validation_fraction_ = float(params["validation_fraction"])
        else:
            raise ValueError("early_stop should be either off, train or valid")

        return sklearn.ensemble.HistGradientBoostingRegressor(
            loss=params["loss"],
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            max_bins=max_bins,
            l2_regularization=l2_regularization,
            tol=tol,
            scoring=scoring,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction_,
            verbose=False,
            warm_start=False,
            quantile=quantile
        )

    if clazz == sklearn.svm.LinearSVC:
        penalty = params["penalty"]
        loss = params["loss"]
        multi_class = params["multi_class"]
        C = float(params["C"])
        tol = float(params["tol"])
        dual = check_for_bool(params["dual"])
        fit_intercept = check_for_bool(params["fit_intercept"])
        intercept_scaling = float(params["intercept_scaling"])

        return sklearn.svm.LinearSVC(
            penalty=penalty,
            loss=loss,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            multi_class=multi_class
        )

    if clazz == sklearn.svm.SVC:
        kernel = params["kernel"]
        if len(params) == 1:
            return sklearn.svm.SVC(kernel=kernel, probability=False)

        C = float(params["C"])
        if "degree" not in params or params["degree"] is None:
            degree = 3
        else:
            degree = int(params["degree"])
        if params["gamma"] is None:
            gamma = 0.0
        else:
            gamma = float(params["gamma"])
        if "coef0" not in params or params["coef0"] is None:
            coef0 = 0.0
        else:
            coef0 = float(params["coef0"])
        tol = float(params["tol"])
        max_iter = int(params["max_iter"])
        shrinking = check_for_bool(params["shrinking"])

        return sklearn.svm.SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            decision_function_shape='ovr',
            probability=False
        )

    if clazz == sklearn.svm.LinearSVR:
        C = float(params["C"])
        tol = float(params["tol"])
        epsilon = float(params["epsilon"])

        dual = check_for_bool(params["dual"])
        fit_intercept = check_for_bool(params["fit_intercept"])
        intercept_scaling = float(params["intercept_scaling"])

        return sklearn.svm.LinearSVR(
            epsilon=epsilon,
            loss=params["loss"],
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
        )

    if clazz == sklearn.svm.SVR:
        C = float(params["C"])
        epsilon = float(params["epsilon"])
        tol = float(params["tol"])
        shrinking = check_for_bool(params["shrinking"])
        degree = int(params["degree"]) if "degree" in params else 3
        gamma = float(params["gamma"]) if "gamma" in params else 0.1
        if "coef0" not in params or check_none(params["coef0"]):
            coef0 = 0.0
        else:
            coef0 = float(params["coef0"])
        max_iter = int(params["max_iter"])

        # Calculate the size of the kernel cache (in MB) for sklearn's LibSVM.
        # The cache size is calculated as 2/3 of the available memory
        # (which is calculated as the memory limit minus the used memory)
        cache_size = 200

        return sklearn.svm.SVR(
            kernel=params["kernel"],
            C=C,
            epsilon=epsilon,
            tol=tol,
            shrinking=shrinking,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            cache_size=cache_size
        )

    if clazz == sklearn.neural_network.MLPRegressor:
        max_iter = 10**4
        hidden_layer_depth = int(params["hidden_layer_depth"])
        num_nodes_per_layer = int(params["num_nodes_per_layer"])
        hidden_layer_sizes = tuple(
            num_nodes_per_layer for i in range(hidden_layer_depth)
        )
        activation = str(params["activation"])
        alpha = float(params["alpha"])
        learning_rate_init = float(params["learning_rate_init"])
        early_stopping = str(params["early_stopping"])
        if params["early_stopping"] == "train":
            validation_fraction = 0.0
            tol = float(params["tol"])
            n_iter_no_change = int(params["n_iter_no_change"])
            early_stopping_val = False
        elif params["early_stopping"] == "valid":
            validation_fraction = float(params["validation_fraction"])
            tol = float(params["tol"])
            n_iter_no_change = int(params["n_iter_no_change"])
            early_stopping_val = True
        else:
            raise ValueError(
                "Set early stopping to unknown value %s" % params["early_stopping"]
            )
        # elif self.early_stopping == "off":
        #     self.validation_fraction = 0
        #     self.tol = 10000
        #     self.n_iter_no_change = self.max_iter
        #     self.early_stopping_val = False

        solver = params["solver"]

        try:
            batch_size = int(params["batch_size"])
        except ValueError:
            batch_size = str(params["batch_size"])

        shuffle = check_for_bool(params["shuffle"])
        beta_1 = float(params["beta_1"])
        beta_2 = float(params["beta_2"])
        epsilon = float(params["epsilon"])
        beta_1 = float(params["beta_1"])

        # initial fit of only increment trees
        return sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            shuffle=shuffle,
            warm_start=False,
            early_stopping=early_stopping_val,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            beta_1=beta_2,
            beta_2=beta_1,
            epsilon=epsilon,
            # We do not use these, see comments below in search space
            # momentum=self.momentum,
            # nesterovs_momentum=self.nesterovs_momentum,
            # power_t=self.power_t,
            # learning_rate=self.learning_rate,
            # max_fun=self.max_fun
        )

    if clazz == sklearn.ensemble.RandomForestRegressor:
        n_estimators = 10**3
        if check_none(params["max_depth"]):
            max_depth = None
        else:
            max_depth = int(params["max_depth"])

        min_samples_split = int(params["min_samples_split"])
        min_samples_leaf = int(params["min_samples_leaf"])

        max_features = float(params["max_features"])

        bootstrap = check_for_bool(params["bootstrap"])

        if check_none(params["max_leaf_nodes"]):
            max_leaf_nodes = None
        else:
            max_leaf_nodes = int(params["max_leaf_nodes"])

        min_impurity_decrease = float(params["min_impurity_decrease"])

        return sklearn.ensemble.RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=params["criterion"],
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
            bootstrap=bootstrap,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=False,
        )

    if clazz == sklearn.linear_model.SGDRegressor:
        alpha = float(params["alpha"])
        l1_ratio = float(params["l1_ratio"]) if "l1_ratio" in params else 0.15
        epsilon = float(params["epsilon"]) if "epsilon" in params else 0.1
        eta0 = float(params["eta0"]) if "eta0" in params else 0.1
        power_t = float(params["power_t"]) if "power_t" in params else 0.25
        average = check_for_bool(params["average"])
        fit_intercept = check_for_bool(params["fit_intercept"])
        tol = float(params["tol"])

        return sklearn.linear_model.SGDRegressor(
            loss=params["loss"],
            penalty=params["penalty"],
            alpha=alpha,
            fit_intercept=fit_intercept,
            tol=tol,
            learning_rate=params["learning_rate"],
            l1_ratio=l1_ratio,
            epsilon=epsilon,
            eta0=eta0,
            power_t=power_t,
            shuffle=True,
            average=average,
            warm_start=False,
        )

    if clazz == sklearn.discriminant_analysis.LinearDiscriminantAnalysis:
        if params["shrinkage"] in (None, "none", "None"):
            shrinkage_ = None
            solver = 'lsqr'
        elif params["shrinkage"] == "auto":
            shrinkage_ = 'auto'
            solver = 'lsqr'
        elif params["shrinkage"] == "manual":
            shrinkage_ = float(params["shrinkage_factor"])
            solver = 'lsqr'
        else:
            raise ValueError()

        tol = float(params["tol"])
        return sklearn.discriminant_analysis.LinearDiscriminantAnalysis(shrinkage=shrinkage_, tol=tol, solver=solver)

    if clazz == sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis:
        return sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(**params)

    if clazz == sklearn.neural_network.MLPClassifier:
        max_iter = 512  # hard coded in auto-sklearn
        hidden_layer_depth = int(params["hidden_layer_depth"])
        num_nodes_per_layer = int(params["num_nodes_per_layer"])
        hidden_layer_sizes = tuple(params["num_nodes_per_layer"] for i in range(params["hidden_layer_depth"]))
        activation = str(params["activation"])
        alpha = float(params["alpha"])
        learning_rate_init = float(params["learning_rate_init"])
        early_stopping = str(params["early_stopping"])
        tol = float(params["tol"])
        if early_stopping == "train":
            validation_fraction = 0.0
            n_iter_no_change = int(params["n_iter_no_change"])
            early_stopping_val = False
        elif early_stopping == "valid":
            validation_fraction = float(params["validation_fraction"])
            n_iter_no_change = int(params["n_iter_no_change"])
            early_stopping_val = True
        else:
            raise ValueError("Set early stopping to unknown value %s" % early_stopping)
        try:
            batch_size = int(params["batch_size"])
        except ValueError:
            batch_size = str(params["batch_size"])

        solver = params["solver"]
        shuffle = check_for_bool(params["shuffle"])
        beta_1 = float(params["beta_1"])
        beta_2 = float(params["beta_2"])
        epsilon = float(params["epsilon"])
        verbose = False

        # initial fit of only increment trees
        return sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            shuffle=shuffle,
            verbose=verbose,
            warm_start=True,
            early_stopping=early_stopping_val,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            beta_1=beta_2,
            beta_2=beta_1,
            epsilon=epsilon
        )

    if clazz == sklearn.linear_model.SGDClassifier:
        max_iter = 1024
        loss = params["loss"]
        penalty = params["penalty"]
        alpha = float(params["alpha"])
        l1_ratio = float(params["l1_ratio"]) if "l1_ratio" in params and params["l1_ratio"] is not None else 0.15
        epsilon = float(params["epsilon"]) if "epsilon" in params and params["epsilon"] is not None else 0.1
        eta0 = float(params["eta0"]) if "eta0" in params and params["eta0"] is not None else 0.01
        power_t = float(params["power_t"]) if "power_t" in params and params["power_t"] is not None else 0.5
        average = check_for_bool(params["average"])
        fit_intercept = check_for_bool(params["fit_intercept"])
        tol = float(params["tol"])
        learning_rate = params["learning_rate"]

        return sklearn.linear_model.SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            learning_rate=learning_rate,
            l1_ratio=l1_ratio,
            epsilon=epsilon,
            eta0=eta0,
            power_t=power_t,
            shuffle=True,
            average=average,
            warm_start=True
        )

    if clazz == sklearn.linear_model.PassiveAggressiveClassifier:
        max_iter = 1024  # fixed in auto-sklearn
        average = check_for_bool(params["average"])
        fit_intercept = check_for_bool(params["fit_intercept"])
        tol = float(params["tol"])
        C = float(params["C"])
        loss = params["loss"]

        return sklearn.linear_model.PassiveAggressiveClassifier(
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            loss=loss,
            shuffle=True,
            warm_start=True,
            average=average,
        )

    if clazz == sklearn.ensemble.RandomForestClassifier:
        criterion = params["criterion"]
        if "n_estimators" in params and params["n_estimators"] is not None:
            n_estimators = int(params["n_estimators"])
        else:
            n_estimators = 512
        if check_none(params["max_depth"]):
            max_depth = None
        else:
            max_depth = int(params["max_depth"])

        min_samples_split = int(params["min_samples_split"])
        min_samples_leaf = int(params["min_samples_leaf"])
        min_weight_fraction_leaf = float(params["min_weight_fraction_leaf"])

        if params["max_features"] not in ("sqrt", "log2", "auto"):
            max_features = int(X.shape[1] ** float(params["max_features"]))
        else:
            max_features = params["max_features"]

        bootstrap = check_for_bool(params["bootstrap"])

        if check_none(params["max_leaf_nodes"]):
            max_leaf_nodes = None
        else:
            max_leaf_nodes = int(params["max_leaf_nodes"])

        min_impurity_decrease = float(params["min_impurity_decrease"])

        # initial fit of only increment trees
        return sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            bootstrap=bootstrap,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=True)

    if clazz == sklearn.ensemble.HistGradientBoostingClassifier:
        learning_rate = float(params["learning_rate"])
        max_iter = int(params["max_iter"]) if "max_iter" in params else 512
        min_samples_leaf = int(params["min_samples_leaf"])
        loss = params["loss"]
        scoring = params["scoring"]
        if check_none(params["max_depth"]):
            max_depth = None
        else:
            max_depth = int(params["max_depth"])
        if check_none(params["max_leaf_nodes"]):
            max_leaf_nodes = None
        else:
            max_leaf_nodes = int(params["max_leaf_nodes"])
        max_bins = int(params["max_bins"])
        l2_regularization = float(params["l2_regularization"])
        tol = float(params["tol"])
        if check_none(params["scoring"]):
            scoring = None
        if params["early_stop"] == "off":
            n_iter_no_change = 1
            validation_fraction_ = None
            early_stopping_ = False
        elif params["early_stop"] == "train":
            n_iter_no_change = int(params["n_iter_no_change"])
            validation_fraction_ = None
            early_stopping_ = True
        elif params["early_stop"] == "valid":
            n_iter_no_change = int(params["n_iter_no_change"])
            validation_fraction = float(params["validation_fraction"])
            early_stopping_ = True
            n_classes = len(np.unique(y))
            if validation_fraction * X.shape[0] < n_classes:
                validation_fraction_ = n_classes
            else:
                validation_fraction_ = params["validation_fraction"]
        else:
            raise ValueError("early_stop should be either off, train or valid")

        # initial fit of only increment trees
        return sklearn.ensemble.HistGradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            max_bins=max_bins,
            l2_regularization=l2_regularization,
            tol=tol,
            scoring=scoring,
            early_stopping=early_stopping_,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction_,
            warm_start=True
        )

    if clazz == sklearn.ensemble.ExtraTreesClassifier:
        n_estimators = 512
        max_features = int(X.shape[1] ** float(params["max_features"]))
        if params["criterion"] not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): ""%s" % params["criterion"])

        if check_none(params["max_depth"]):
            max_depth = None
        else:
            max_depth = int(params["max_depth"])
        if check_none(params["max_leaf_nodes"]):
            max_leaf_nodes = None
        else:
            max_leaf_nodes = int(params["max_leaf_nodes"])

        criterion = params["criterion"]
        min_samples_leaf = int(params["min_samples_leaf"])
        min_samples_split = int(params["min_samples_split"])
        max_features = float(params["max_features"])
        min_impurity_decrease = float(params["min_impurity_decrease"])
        min_weight_fraction_leaf = float(params["min_weight_fraction_leaf"])
        oob_score = check_for_bool(params["oob_score"]) if "oob_score" in params else False
        bootstrap = check_for_bool(params["bootstrap"])

        return sklearn.ensemble.ExtraTreesClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_impurity_decrease=min_impurity_decrease,
            oob_score=oob_score,
            warm_start=True
        )

    # allow instantiation without specific rule only for components without config space
    raise ValueError(f"No rule defined for component {clazz}. But received params {params}")


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


def get_all_configurations(config_spaces):
    configs_by_comps = {}
    for step_name, config_space in config_spaces.items():
        names = []
        domains = []
        for hp in config_space.get_hyperparameters():
            names.append(hp.name)
            if isinstance(hp, (
                    ConfigSpace.hyperparameters.UnParametrizedHyperparameter,
                    ConfigSpace.hyperparameters.Constant
            )):
                domains.append([hp.value])

            if isinstance(hp, ConfigSpace.hyperparameters.CategoricalHyperparameter):
                domains.append(list(hp.choices))
            elif isinstance(hp, ConfigSpace.hyperparameters.IntegerHyperparameter):
                domains.append(list(range(hp.lower, hp.upper + 1)))
            else:
                raise TypeError(f"Unsupported hyperparameter type {type(hp)}")

        # compute product
        configs = []
        for combo in it.product(*domains):
            configs.append({name: combo[i] for i, name in enumerate(names)})
        configs_by_comps[step_name] = configs
    return configs_by_comps


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

    def step(self, remaining_time=None):
        self.its += 1

        if not self.active:
            raise Exception("Cannot step inactive HPO Process")

        self.logger.info(f"Starting {self.its}-th HPO step. Currently best known score is {self.best_score}")

        # draw random parameters
        configs_by_comps = {}
        for step_name, comp in self.comps_by_steps:
            config_space = self.config_spaces[step_name]
            sampled_config = config_space.sample_configuration(1)
            params = {}
            for hp in config_space.get_hyperparameters():
                if hp.name in sampled_config:
                    params[hp.name] = sampled_config[hp.name]
            configs_by_comps[step_name] = params
            del params

        # evaluate configured pipeline
        time_start_eval = time.time()
        status, scores, evaluation_report, exception = self.evalComp(configs_by_comps)
        if not isinstance(scores, dict):
            raise TypeError(f"""
The scores must be a dictionary as a function of the scoring functions. Observed type is {type(scores)}: {scores}
""")
        score = scores[get_scoring_name(self.scoring)]
        runtime = time.time() - time_start_eval
        self.eval_runtimes.append(runtime)
        self.logger.info(f"Observed score of {score} for params {configs_by_comps}")
        if score > self.best_score:
            self.logger.info("This is a NEW BEST SCORE!")
            self.best_score = score
            self.time_since_last_imp = 0
            self.configs_since_last_imp = 0
            self.best_configs = configs_by_comps.copy()
        else:
            self.configs_since_last_imp += 1
            self.time_since_last_imp += runtime
            if self.its >= self.min_its and (
                    self.time_since_last_imp > self.max_time_without_imp or
                    self.configs_since_last_imp > self.max_its_without_imp
            ):
                self.logger.info(
                    f"No improvement within {self.time_since_last_imp}s"
                    f" or within {self.max_its_without_imp} steps."
                    "Stopping HPO here."
                )
                self.active = False
                return (
                    self.get_parametrized_pipeline(configs_by_comps),
                    "no_imp",
                    {s: np.nan for s in [self.scoring] + self.side_scores},
                    None,
                    None
                )

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
                configs = get_all_configurations(self.config_spaces)
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
        return self.get_parametrized_pipeline(configs_by_comps), status, scores, evaluation_report, runtime, exception

    def get_best_config(self, step_name):
        return self.best_configs[step_name]
