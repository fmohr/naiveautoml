# core stuff
import json
import numpy as np

# HPO and process control
from ConfigSpace.read_and_write import json as config_json

# sklearn
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
