from ConfigSpace.read_and_write import json as config_json
from datetime import datetime
from sklearn.datasets import load_iris
from gama.genetic_programming.components import Individual, PrimitiveNode, Fitness
import gama.genetic_programming.compilers.scikitlearn
from typing import Callable, Tuple, Optional, Sequence
from sklearn.base import TransformerMixin, is_classifier
from sklearn.model_selection import ShuffleSplit, cross_validate, check_cv
from gama import GamaClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.naive_bayes import BernoulliNB

from gama.utilities.generic.timekeeper import TimeKeeper

from experimentutils import *

def get_mandatory_preprocessing(X, y):
    
    # determine fixed pre-processing steps for imputation and binarization (naml does this automatically)
    types = [set([type(v) for v in r]) for r in X.T]
    numeric_features = [c for c, t in enumerate(types) if len(t) == 1 and list(t)[0] != str]
    numeric_transformer = Pipeline([("imputer", sklearn.impute.SimpleImputer(strategy="median"))])
    categorical_features = [i for i in range(X.shape[1]) if i not in numeric_features]
    missing_values_per_feature = np.sum(pd.isnull(X), axis=0)
    if len(categorical_features) > 0 or sum(missing_values_per_feature) > 0:
        categorical_transformer = Pipeline([
            ("imputer", sklearn.impute.SimpleImputer(strategy="most_frequent")),
            ("binarizer", sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse = True)),

        ])
        mandatory_pre_processing = [("impute_and_binarize", ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        ))]
    else:
        mandatory_pre_processing = []
    return mandatory_pre_processing

def get_gama_search_space(file):
    f = open(file)
    search_space = json.load(f)
    
    config = {}
    for step in search_space:
        for c in step["components"]:
            config_space = config_json.read(c["params"])
            
            config_of_component = {}
            for hp in config_space.get_hyperparameters():
                if type(hp) in [ConfigSpace.hyperparameters.UnParametrizedHyperparameter, ConfigSpace.hyperparameters.Constant]:
                    domain = [hp.value]

                elif type(hp) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
                    domain = list(hp.choices)
                
                else:
                    rs = np.random.RandomState()
                    domain = sorted(list(set([hp.sample(rs) for i in range(10**4)])))
                
                config_of_component[hp.name] = domain
            
            config[get_class(c["class"])] = config_of_component
    return config


def fit_patched(self, x, y, warm_start = None):
    """ Find and fit a model to predict target y from X.
    Various possible machine learning pipelines will be fit to the (X,y) data.
    Using Genetic Programming, the pipelines chosen should lead to gradually
    better models. Pipelines will internally be validated using cross validation.
    After the search termination condition is met, the best found pipeline
    configuration is then used to train a final model on all provided data.
    Parameters
    ----------
    x: pandas.DataFrame or numpy.ndarray, shape = [n_samples, n_features]
        Training data. All elements must be able to be converted to float.
    y: pandas.DataFrame, pandas.Series or numpy.ndarray, shape = [n_samples,]
        Target values.
        If a DataFrame is provided, assumes the first column contains target values.
    warm_start: List[Individual], optional (default=None)
        A list of individual to start the search  procedure with.
        If None is given, random start candidates are generated.
    """
    self._time_manager = TimeKeeper(self._time_manager.total_time)
    self._x, self._y = x, y
    # with self._time_manager.start_activity(
    #         "preprocessing", activity_meta=["default"]
    # ):
    #     x, self._y = format_x_y(x, y)
    #     self._inferred_dtypes = x.dtypes
    #     is_classification = hasattr(self, "_label_encoder")
    #     self._x, self._basic_encoding_pipeline = basic_encoding(
    #         x, is_classification
    #     )
    #     self._fixed_pipeline_extension = basic_pipeline_extension(
    #         self._x, is_classification
    #     )
    #     self._operator_set._safe_compile = partial(
    #         self._operator_set._compile,
    #         preprocessing_steps=self._fixed_pipeline_extension,
    #     )
    #
    #     store_pipelines = (
    #             self._evaluation_library._m is None or self._evaluation_library._m > 0
    #     )
    #     if store_pipelines and self._x.shape[0] * self._x.shape[1] > 6_000_000:
    #         # if m > 0, we are storing models for each evaluation. For this size
    #         # KNN will create models of about 76Mb in size, which is too big, so
    #         # we exclude it from search:
    #         log.info("Excluding KNN from search because the dataset is too big.")
    #         from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    #
    #         self._pset["prediction"] = [
    #             p
    #             for p in self._pset["prediction"]
    #             if p.identifier not in [KNeighborsClassifier, KNeighborsRegressor]
    #         ]
        # if store_pipelines and self._x.shape[1] > 50:
        #     log.info("Data has too many features to include PolynomialFeatures")
        #     from sklearn.preprocessing import PolynomialFeatures
        #
        #     self._pset["data"] = [
        #         p
        #         for p in self._pset["data"]
        #         if p.identifier not in [PolynomialFeatures]
        #     ]
    fit_time = int(
        (1 - self._post_processing.time_fraction)
        * self._time_manager.total_time_remaining
    )
    with self._time_manager.start_activity(
            "search",
            time_limit=fit_time,
            activity_meta=[self._search_method.__class__.__name__],
    ):
        self._search_phase(warm_start, timeout=fit_time)
    with self._time_manager.start_activity(
            "postprocess",
            time_limit=int(self._time_manager.total_time_remaining),
            activity_meta=[self._post_processing.__class__.__name__],
    ):
        best_individuals = list(
            reversed(
                sorted(
                    self._final_pop,
                    key=lambda ind: ind.fitness.values,
                )
            )
        )
        
        mandatory_pre_processing = get_mandatory_preprocessing(x, y)
        
        for ind in best_individuals:
            ind.pipeline = Pipeline(mandatory_pre_processing + ind.pipeline.steps)
        self._post_processing.dynamic_defaults(self)
        self.model = self._post_processing.post_process(
            self._x,
            self._y,
            self._time_manager.total_time_remaining,
                best_individuals,
        )
    if not self._store == "all":
        to_clean = dict(nothing="all", logs="evaluations", models="logs")
        self.cleanup(to_clean[self._store])
    return self

def prepare_for_prediction_patched(self, x):
    if isinstance(x, np.ndarray):
        x = self._np_to_matching_dataframe(x)
    return x


def predict_patched(self, x: pd.DataFrame):
    """ Predict the target for input X.
    Parameters
    ----------
    x: pandas.DataFrame
        A dataframe with the same number of columns as the input to `fit`.
    Returns
    -------
    numpy.ndarray
        Array with predictions of shape (N,) where N is len(X).
    """
    y = self.model.predict(x)  # type: ignore
    return y