import openml
import pandas as pd

import sklearn as sk
import sklearn.ensemble
import sklearn.decomposition
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn import *

import pebble
from func_timeout import func_timeout, FunctionTimedOut
import time
from datetime import datetime

import ConfigSpace
from ConfigSpace.util import *
from ConfigSpace.read_and_write import json as config_json
import json

import itertools as it

import os, psutil
import multiprocessing
from concurrent.futures import TimeoutError

import scipy.sparse

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
	def __init__(self, *args, **kwargs):
	    kwargs['context'] = NoDaemonContext()
	    super(NestablePool, self).__init__(*args, **kwargs)
	    print("Established a nestable pool with params: " + str(*args))

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass
        
class NoDaemonContext(type(multiprocessing.get_context("spawn"))):
	Process = NoDaemonProcess
        
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

def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0].dropna()
    y = df[ds.default_target_attribute].values
    
    categorical_attributes = df.select_dtypes(exclude=['number']).columns
    expansion_size = 1
    for att in categorical_attributes:
        expansion_size *= len(pd.unique(df[att]))
        if expansion_size > 10**5:
            break
    
    if expansion_size < 10**5:
        X = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]]).values.astype(float)
    else:
        print("creating SPARSE data")
        dfSparse = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]], sparse=True)
        
        print("dummies created, now creating sparse matrix")
        X = lil_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        print("Done. shape is" + str(X.shape))
    return X, y

class EvaluationPool:

    def __init__(self, X, y, scoring, tolerance_tuning = 0.05, tolerance_estimation_error = 0.01):
        if X is None:
            raise Exception("Parameter X must not be None")
        if y is None:
            raise Exception("Parameter y must not be None")
        if type(X) != np.ndarray and type(X) != scipy.sparse.csr.csr_matrix and type(X) != scipy.sparse.lil.lil_matrix:
            raise Exception("X must be a numpy array but is " + str(type(X)))
        self.X = X
        self.y = y
        self.scoring = scoring
        self.bestScore = -np.inf
        self.tolerance_tuning = tolerance_tuning
        self.tolerance_estimation_error = tolerance_estimation_error
        self.cache = {}
        
    def merge(self, pool):
        num_entries_before = len(self.cache)
        for spl in pool.cache:
            pl, learning_curve, timestamp = pool.cache[spl]
            self.tellEvaluation(pl, learning_curve, timestamp)
        num_entries_after = len(self.cache)
        print("Adopted " + str(num_entries_after - num_entries_before) + " entries from external pool.")

    def tellEvaluation(self, pl, scores, timestamp):
        spl = str(pl)
        self.cache[spl] = (pl, scores, timestamp)
        score = np.mean(scores)
        if score > self.bestScore:
            self.bestScore = score        
            self.best_spl = spl
            
    def cross_validate(self, pl, X, y, scoring): # just a wrapper to ease parallelism
        try:
            scorer = sklearn.metrics.make_scorer(sklearn.metrics.log_loss if scoring == "neg_log_loss" else sklearn.metrics.roc_auc_score, greater_is_better = scoring != "neg_log_loss", needs_proba=True, labels=list(np.unique(y)))
            return sklearn.model_selection.cross_validate(pl, X, y, scoring=scorer, error_score="raise")
        except:
            raise
            #print("OBSERVED ERROR, EXECUTION ABORTED!")
            #return None

    def evaluate(self, pl, timeout=None, deadline=None, verbose=False):
        if is_pipeline_forbidden(pl):
            if verbose:
                print("Preventing evaluation of forbidden pipeline " + str(pl))
            return np.nan
        
        process = psutil.Process(os.getpid())
        if verbose:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Initializing evaluation of " + str(pl) + " with current memory consumption " + str(int(process.memory_info().rss / 1024 / 1024))  + "MB. Now awaiting results.")
        
        start_outer = time.time()
        spl = str(pl)
        if spl in self.cache:
            return np.round(np.mean(self.cache[spl][1]), 4)
        timestamp = time.time()
        if timeout is not None:
            result = func_timeout(timeout, self.cross_validate, (pl, self.X, self.y, self.scoring))
        else:
            result = self.cross_validate(pl, self.X, self.y, self.scoring)
        if result is None:
            return np.nan
        scores = result["test_score"]
        runtime = time.time() - start_outer
        if verbose:
            print("Completed evaluation of " + spl + " after " + str(runtime) + "s. Scores are", scores)
        self.tellEvaluation(pl, scores, timestamp)
        return np.round(np.mean(scores), 4)

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
            params = {"kernel": config_json.read(comp["params"]).get_hyperparameter("kernel").value}
            print("SVC invoked without params. Setting kernel explicitly to " + params["kernel"])
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
        
        print(kernel, "PROBA")

        return sklearn.svm.SVC(C=C, kernel=kernel,degree=degree,gamma=gamma,coef0=coef0,shrinking=shrinking,tol=tol, max_iter=max_iter, decision_function_shape='ovr', probability=True)
    
    
    
    
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
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
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
        #print(hp.name, type(hp))
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
    
    def __init__(self, step_name, comp, X, y, scoring, execution_timeout, other_step_component_instances, index_in_steps, max_time_without_imp, max_its_without_imp, min_its = 10):
        self.step_name = step_name
        self.index_in_steps = index_in_steps
        self.comp = comp
        self.X = X
        self.y = y
        self.other_step_component_instances = other_step_component_instances
        self.execution_timeout = execution_timeout
        config_space_as_string = comp["params"]
        self.config_space = config_json.read(config_space_as_string)
        self.space_size = get_hyperparameter_space_size(self.config_space)
        self.eval_runtimes = []
        self.configs_since_last_imp = 0
        self.time_since_last_imp = 0
        self.evaled_configs = set([])
        self.active = self.space_size >= 1
        print("Search space size for", comp["class"], self.space_size)
        self.best_score = -np.inf
        self.best_params = None
        self.max_time_without_imp = max_time_without_imp
        self.max_its_without_imp = max_its_without_imp
        self.min_its = min_its
        self.pool = EvaluationPool(X, y, scoring)
        self.its = 0
        
    def evalComp(self, params):
        this_step = (self.step_name, build_estimator(self.comp, params, self.X, self.y))
        steps = [s for s in self.other_step_component_instances]
        steps[self.index_in_steps] = this_step
        print(steps)
        try:
            return np.mean(self.pool.evaluate(Pipeline(steps=steps), timeout=self.execution_timeout))
        except FunctionTimedOut:
            print("TIMEOUT")
            return np.nan
        
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
        score = self.evalComp(params)
        runtime = time.time() - time_start_eval
        self.eval_runtimes.append(runtime)
        print("Observed score of", score, "for", self.comp["class"], "with params", params)
        if score > self.best_score:
            print("This is a NEW BEST SCORE!")
            self.best_score = score
            self.time_since_last_imp = 0
            self.configs_since_last_imp = 0
            self.best_params = params
        else:
            self.configs_since_last_imp += 1
            self.time_since_last_imp += runtime
            if self.its >= self.min_its and (self.time_since_last_imp > self.max_time_without_imp or self.configs_since_last_imp > self.max_its_without_imp):
                print("No improvement within " + str(self.time_since_last_imp) + "s or within " + str(self.max_its_without_imp) + " steps. Stopping HPO here.")
                self.active = False
                return
            
        # check whether we do a quick exhaustive search and then disable this module
        if len(self.eval_runtimes) >= 10:
            total_expected_runtime = self.space_size * np.mean(self.eval_runtimes)
            if total_expected_runtime < np.inf and (remaining_time is None or total_expected_runtime < remaining_time):
                self.active = False
                print("Expected time to evaluate all configurations is only", total_expected_runtime, "Doing exhaustive search.")
                configs = get_all_configurations(self.config_space)
                print("Now evaluation all " + str(len(configs)) + " possible configurations.")
                for params in configs:
                    score = self.evalComp(params)
                    print("Observed score of", score, "for", self.comp["class"], "with params", params)
                    if score > self.best_score:
                        print("This is a NEW BEST SCORE!")
                        self.best_score = score
                        self.best_params = params
                print("Configuration space completely exhausted.")
    
    def get_best_config(self):
        return self.best_params