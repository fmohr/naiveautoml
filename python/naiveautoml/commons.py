# core
import pandas as pd
import logging, warnings
import itertools as it
import os, psutil
import scipy.sparse
import time

# sklearn
import sklearn
from sklearn import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import get_scorer
from sklearn.metrics import make_scorer


# timeout functions
from func_timeout import func_timeout, FunctionTimedOut

# configspace
import ConfigSpace
from ConfigSpace.util import *
from ConfigSpace.read_and_write import json as config_json
import json
import traceback

def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
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
    return scoring if type(scoring) == str else scoring["name"]


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
                 logger_name=None
                 ):
        domains_task_type = ["classification", "regression"]
        if task_type not in domains_task_type:
            raise ValueError(f"task_type must be in {domains_task_type}")
        self.task_type = task_type
        self.logger = logging.getLogger('naiveautoml.evalpool' if logger_name is None else logger_name)
        if X is None:
            raise Exception("Parameter X must not be None")
        if y is None:
            raise Exception("Parameter y must not be None")
        if type(X) != pd.DataFrame and type(X) != np.ndarray and type(X) != scipy.sparse.csr.csr_matrix and type(X) != scipy.sparse.csc.csc_matrix and type(X) != scipy.sparse.lil.lil_matrix:
            raise Exception("X must be a numpy array but is " + str(type(X)))
        self.X = X
        self.y = y
        self.scoring = scoring
        if side_scores is None:
            self.side_scores = []
        elif type(side_scores) == list:
            self.side_scores = side_scores
        else:
            self.logger.warning("side scores was not given as list, casting it to a list of size 1 implicitly.")
            self.side_scores = [side_scores]
        self.bestScore = -np.inf
        self.tolerance_tuning = tolerance_tuning
        self.tolerance_estimation_error = tolerance_estimation_error
        self.cache = {}
        self.evaluation_fun = self.cross_validate if evaluation_fun is None else evaluation_fun

    def tellEvaluation(self, pl, scores, timestamp):
        spl = str(pl)
        self.cache[spl] = (pl, scores, timestamp)
        score = np.mean(scores)
        if score > self.bestScore:
            self.bestScore = score        
            self.best_spl = spl
                        
    def cross_validate(self, pl, X, y, scorings, errors="raise"): # just a wrapper to ease parallelism
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if type(scorings) != list:
                scorings = [scorings]

            if self.task_type == "classification":
                splitter = sklearn.model_selection.StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
            elif self.task_type:
                splitter = sklearn.model_selection.KFold(n_splits=5, random_state=None, shuffle=True)
            scores = {get_scoring_name(scoring): [] for scoring in scorings}
            for train_index, test_index in splitter.split(X, y):
                
                X_train = X.iloc[train_index] if type(X) == pd.DataFrame else X[train_index]
                y_train = y.iloc[train_index] if type(y) == pd.Series else y[train_index]
                X_test = X.iloc[test_index] if type(X) == pd.DataFrame else X[test_index]
                y_test = y.iloc[test_index] if type(y) == pd.Series else y[test_index]

                pl_copy = sklearn.base.clone(pl)
                pl_copy.fit(X_train, y_train)
                
                # compute values for each metric
                for scoring in scorings:
                    scorer = get_scorer(scoring) if type(scoring) == str else make_scorer(**{key: val for key, val in scoring.items() if key != "name"})
                    try:
                        score = scorer(pl_copy, X_test, y_test)
                    except KeyboardInterrupt:
                        raise
                    except:
                        if errors in ["message", "ignore"]:
                            if errors == "message":
                                self.logger.info(f"Observed exception in validation of pipeline {pl_copy}. Placing nan.")
                            score = np.nan
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
        if is_pipeline_forbidden(pl):
            self.logger.info(f"Preventing evaluation of forbidden pipeline {pl}")
            return {get_scoring_name(scoring): np.nan for scoring in [self.scoring] + self.side_scores}

        process = psutil.Process(os.getpid())
        self.logger.info(f"Initializing evaluation of {pl} with current memory consumption {int(process.memory_info().rss / 1024 / 1024)} MB. Now awaiting results.")
        
        start_outer = time.time()
        spl = str(pl)
        if spl in self.cache:
            out = {get_scoring_name(scoring): np.nan for scoring in [self.scoring] + self.side_scores}
            out[get_scoring_name(self.scoring)] = np.round(np.mean(self.cache[spl][1]), 4)
            return out
        timestamp = time.time()
        if timeout is not None:
            scores = func_timeout(timeout, self.evaluation_fun, (pl, self.X, self.y, [self.scoring] + self.side_scores))
        else:
            scores = self.evaluation_fun(pl, self.X, self.y, [self.scoring] + self.side_scores)
        if scores is None:
            return {get_scoring_name(scoring): np.nan for scoring in [self.scoring] + self.side_scores}
        runtime = time.time() - start_outer
        if type(scores) != dict:
            raise ValueError(f"scores is of type {type(scores)} but must be a dictionary with entries for {get_scoring_name(self.scoring)}.\nProbably you inserted an evaluation_fun argument that does not return a proper dictionary.")
        self.logger.info(f"Completed evaluation of {spl} after {runtime}s. Scores are {scores}")
        self.tellEvaluation(pl, scores[get_scoring_name(self.scoring)], timestamp)
        return {scoring: np.round(np.mean(scores[scoring]), 4) for scoring in scores}

    def getBestCandidate(self):
        return self.getBestCandidates(1)[0]
        
    def getBestCandidates(self, n):
        candidates = sorted([key for key in self.cache], key=lambda k: np.mean(self.cache[k][1]), reverse=True)
        return [self.cache[c] for c in candidates[:n]]
    
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


def check_for_bool(p: str) -> bool:
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
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
        affinity = params["affinity"]
        linkage = params["linkage"]
        return sklearn.cluster.FeatureAgglomeration(n_clusters=n_clusters, affinity=affinity, linkage=linkage, pooling_func=pooling_func)
    
    
    
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
    
    if clazz == sklearn.preprocessing.RobustScaler:
        return sklearn.preprocessing.RobustScaler(quantile_range=(params["q_min"], params["q_max"]), copy=False,)
    
    if clazz == sklearn.decomposition.PCA:
        n_components = float(params["keep_variance"])
        whiten = check_for_bool(params["whiten"])
        return sklearn.decomposition.PCA(n_components=n_components, whiten=whiten, copy=True)
    
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
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
    
    if clazz == sklearn.ensemble.ExtraTreesRegressor:
        n_estimators = 10**3
        if params["criterion"] not in ("mse", "friedman_mse", "mae"):
            raise ValueError(
                "'criterion' is not in ('mse', 'friedman_mse', "
                "'mae): %s" % self.criterion
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
        thetaL = float(params["thetaL"])
        thetaU = float(params["thetaU"])

        #n_features = X.shape[1]
        kernel = sklearn.gaussian_process.kernels.RBF(
            #length_scale=[1.0] * n_features,
            #length_scale_bounds=[(thetaL, thetaU)] * n_features,
        )

        # Instanciate a Gaussian Process model
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
            n_iter_no_change = 0
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
            
        return sklearn.svm.LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, multi_class=multi_class)
    
    if clazz == sklearn.svm.SVC:
        kernel = params["kernel"]
        if len(params) == 1:
            return sklearn.svm.SVC(kernel=kernel, probability=True)
        
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
        max_iter = float(params["max_iter"])
        shrinking = check_for_bool(params["shrinking"])
        
        return sklearn.svm.SVC(C=C, kernel=kernel,degree=degree,gamma=gamma,coef0=coef0,shrinking=shrinking,tol=tol, max_iter=max_iter, decision_function_shape='ovr', probability=True)

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
        if not "coef0" in params or check_none(params["coef0"]):
            coef0 = 0.0
        else:
            coef0 = float(params["coef0"])
        max_iter = int(params["max_iter"])
        
        # Calculate the size of the kernel cache (in MB) for sklearn's LibSVM.
        # The cache size is calculated as 2/3 of the available memory
        # (which is calculated as the memory limit minus the used memory)
        try:
            # Retrieve memory limits imposed on the process
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)

            if soft > 0:
                # Convert limit to units of megabytes
                soft /= 1024 * 1024

                # Retrieve memory used by this process
                maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024

                # In MacOS, the MaxRSS output of resource.getrusage in bytes;
                # on other platforms, it's in kilobytes
                if sys.platform == "darwin":
                    maxrss = maxrss / 1024

                cache_size = (soft - maxrss) / 1.5

                if cache_size < 0:
                    cache_size = 200
            else:
                cache_size = 200
        except Exception:
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
                "Set early stopping to unknown value %s" % self.early_stopping
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
            solver = 'svd'
        elif params["shrinkage"] == "auto":
            shrinkage_ = 'auto'
            solver = 'lsqr'
        elif params["shrinkage"] == "manual":
            shrinkage_ = float(params["shrinkage_factor"])
            solver = 'lsqr'
        else:
            raise ValueError(self.shrinkage)

        tol = float(params["tol"])
        return sklearn.discriminant_analysis.LinearDiscriminantAnalysis(shrinkage=shrinkage_, tol=tol, solver=solver)
        
    if clazz == sklearn.neural_network.MLPClassifier:
        _fully_fit = False
        max_iter = 512 # hard coded in auto-sklearn
        hidden_layer_depth = int(params["hidden_layer_depth"])
        num_nodes_per_layer = int(params["num_nodes_per_layer"])
        hidden_layer_sizes = tuple(params["num_nodes_per_layer"]
                                        for i in range(params["hidden_layer_depth"]))
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

        return sklearn.linear_model.SGDClassifier(loss=loss,
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
                                           warm_start=True)
    
    
    if clazz == sklearn.linear_model.PassiveAggressiveClassifier:
        max_iter = 1024 # fixed in auto-sklearn
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
        n_estimators = int(params["n_estimators"]) if "n_estimators" in params and params["n_estimators"] is not None else 512
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
    
    if clazz == sklearn.ensemble.GradientBoostingClassifier:
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
            n_iter_no_change = 0
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
            raise ValueError("'criterion' is not in ('gini', 'entropy'): ""%s" % self.criterion)

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

        return sklearn.ensemble.ExtraTreesClassifier(n_estimators=n_estimators,
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
             warm_start=True)
        
    else:
        return clazz(**params)

    
    
    
    
    
def get_hyperparameter_space_size(config_space):
    hps = config_space.get_hyperparameters();
    if not hps:
        return 0
    size = 1
    for hp in hps:
        if type(hp) in [ConfigSpace.hyperparameters.UnParametrizedHyperparameter, ConfigSpace.hyperparameters.Constant]:
            continue
            
        if type(hp) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
            size *= len(list(hp.choices))
        elif issubclass(hp.__class__, ConfigSpace.hyperparameters.IntegerHyperparameter):
            size *= (hp.upper - hp.lower + 1)
        else:
            return np.inf
    return size


def get_all_configurations(config_space):
    names = []
    domains = []
    for hp in config_space.get_hyperparameters():
        names.append(hp.name)
        if type(hp) in [ConfigSpace.hyperparameters.UnParametrizedHyperparameter, ConfigSpace.hyperparameters.Constant]:
            domains.append([hp.value])
            
        if type(hp) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
            domains.append(list(hp.choices))
        elif issubclass(hp.__class__, ConfigSpace.hyperparameters.IntegerHyperparameter):
            domains.append(list(range(hp.lower, hp.upper + 1)))
        else:
            raise Exception("Unsupported type " + str(type(hp)))
    
    # compute product
    configs = []
    for combo in it.product(*domains):
        configs.append({name: combo[i] for i, name in enumerate(names)})
    return configs

def is_pipeline_forbidden(pl):
    forbidden_combos = [
        {"feature-pre-processor": sklearn.decomposition.FastICA, "classifier": sklearn.naive_bayes.MultinomialNB},
        {"feature-pre-processor": sklearn.decomposition.PCA, "classifier": sklearn.naive_bayes.MultinomialNB},
        {"feature-pre-processor": sklearn.decomposition.KernelPCA, "classifier": sklearn.naive_bayes.MultinomialNB},
        {"feature-pre-processor": sklearn.kernel_approximation.RBFSampler, "classifier": sklearn.naive_bayes.MultinomialNB},
        {"feature-pre-processor": sklearn.kernel_approximation.Nystroem, "classifier": sklearn.naive_bayes.MultinomialNB},
        {"data-pre-processor": 
sklearn.preprocessing.PowerTransformer, "classifier": sklearn.naive_bayes.MultinomialNB},
        {"data-pre-processor": sklearn.preprocessing.StandardScaler, "classifier": sklearn.naive_bayes.MultinomialNB},
        {"data-pre-processor": sklearn.preprocessing.RobustScaler, "classifier": sklearn.naive_bayes.MultinomialNB}
    ]
    
    representation = {}
    for step_name, obj in pl.steps:
        representation[step_name] = obj.__class__
    
    for combo in forbidden_combos:
        matches = True
        for key, val in combo.items():
            if not key in representation or representation[key] != val:
                matches = False
                break
        if matches:
            return True
    return False

class HPOProcess:
    
    def __init__(self, task_type, step_name, comp, X, y, scoring, side_scores, evaluation_fun, execution_timeout, mandatory_pre_processing, other_step_component_instances, index_in_steps, max_time_without_imp, max_its_without_imp, min_its = 10, logger_name = None, allow_exhaustive_search = True):
        self.task_type = task_type
        self.step_name = step_name
        self.index_in_steps = index_in_steps
        self.comp = comp
        self.X = X
        self.y = y
        self.mandatory_pre_processing = mandatory_pre_processing
        self.other_step_component_instances = other_step_component_instances
        self.execution_timeout = execution_timeout
        config_space_as_string = json.dumps(comp["params"])
        self.config_space = config_json.read(config_space_as_string)
        self.space_size = get_hyperparameter_space_size(self.config_space)
        self.eval_runtimes = []
        self.configs_since_last_imp = 0
        self.time_since_last_imp = 0
        self.evaled_configs = set([])
        self.active = self.space_size >= 1
        self.best_score = -np.inf
        self.best_params = None
        self.max_time_without_imp = max_time_without_imp
        self.max_its_without_imp = max_its_without_imp
        self.min_its = min_its
        self.scoring = scoring
        self.pool = EvaluationPool(task_type, X, y, scoring = scoring, side_scores = side_scores, evaluation_fun = evaluation_fun)
        self.its = 0
        self.allow_exhaustive_search = allow_exhaustive_search
        
        if side_scores is None:
            self.side_scores = []
        elif type(side_scores) == list:
            self.side_scores = side_scores
        else:
            self.logger.warning("side scores was not given as list, casting it to a list of size 1 implicitly.")
            self.side_scores = [side_scores]
        
        # init logger
        self.logger = logging.getLogger('naiveautoml.hpo' if logger_name is None else logger_name)
        self.logger.info(f"Search space size for {comp['class']} is {self.space_size}")
        
    def get_parametrized_pipeline(self, params):
        this_step = (self.step_name, build_estimator(self.comp, params, self.X, self.y))
        steps = [s for s in self.other_step_component_instances]
        steps[self.index_in_steps] = this_step
        return Pipeline(steps=self.mandatory_pre_processing + steps)
        
    def evalComp(self, params):
        try:
            return "ok", self.pool.evaluate(self.get_parametrized_pipeline(params), timeout=self.execution_timeout), None
        except FunctionTimedOut:
            self.logger.info("TIMEOUT")
            return "timeout", {scoring: np.nan for scoring in [self.scoring] + self.side_scores}, None
        except KeyboardInterrupt:
            raise
        except:
            return "exception", {scoring: np.nan for scoring in [self.scoring] + self.side_scores}, traceback.format_exc()

        
    def step(self, remaining_time = None):
        self.its += 1
        
        if not self.active:
            raise Exception("Cannot step inactive HPO Process")
        
        # draw random parameters
        params = {}
        sampled_config = self.config_space.sample_configuration(1)
        for hp in self.config_space.get_hyperparameters():
            if hp.name in sampled_config:
                params[hp.name] = sampled_config[hp.name]

        # evaluate configured pipeline
        time_start_eval = time.time()
        status, scores, exception = self.evalComp(params)
        if type(scores) != dict:
            raise ValueError(f"The scores must be a dictionary as a function of the scoring functions. Observed type is {type(scores)}: {scores}")
        score = scores[get_scoring_name(self.scoring)]
        runtime = time.time() - time_start_eval
        self.eval_runtimes.append(runtime)
        self.logger.info(f"Observed score of {score} for {self.comp['class']} with params {params}")
        if score > self.best_score:
            self.logger.info("This is a NEW BEST SCORE!")
            self.best_score = score
            self.time_since_last_imp = 0
            self.configs_since_last_imp = 0
            self.best_params = params
        else:
            self.configs_since_last_imp += 1
            self.time_since_last_imp += runtime
            if self.its >= self.min_its and (self.time_since_last_imp > self.max_time_without_imp or self.configs_since_last_imp > self.max_its_without_imp):
                self.logger.info("No improvement within " + str(self.time_since_last_imp) + "s or within " + str(self.max_its_without_imp) + " steps. Stopping HPO here.")
                self.active = False
                return
            
        # check whether we do a quick exhaustive search and then disable this module
        if len(self.eval_runtimes) >= 10:
            total_expected_runtime = self.space_size * np.mean(self.eval_runtimes)
            if self.allow_exhaustive_search and total_expected_runtime < np.inf and (remaining_time is None or total_expected_runtime < remaining_time):
                self.active = False
                self.logger.info(f"Expected time to evaluate all configurations is only {total_expected_runtime}. Doing exhaustive search.")
                configs = get_all_configurations(self.config_space)
                self.logger.info(f"Now evaluation all {len(configs)} possible configurations.")
                for params in configs:
                    status, scores, exception = self.evalComp(params)
                    score = scores[self.scoring]
                    self.logger.info(f"Observed score of {score} for {self.comp['class']} with params {params}")
                    if score > self.best_score:
                        self.logger.info("This is a NEW BEST SCORE!")
                        self.best_score = score
                        self.best_params = params
                self.logger.info("Configuration space completely exhausted.")
        return self.get_parametrized_pipeline(params), status, scores, runtime, exception
    
    def get_best_config(self):
        return self.best_params