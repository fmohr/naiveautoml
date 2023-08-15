import logging
import naiveautoml
import numpy as np
import sklearn.datasets
from sklearn import *

import scipy.sparse

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

import unittest
from parameterized import parameterized
import itertools as it
import time
import openml
import pandas as pd


def get_dataset(openmlid, as_numpy = True):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]
    num_rows = len(df)
        
    # prepare label column as numpy array
    print(f"Read in data frame. Size is {len(df)} x {len(df.columns)}.")
    X = df.drop(columns=[ds.default_target_attribute])
    y = df[ds.default_target_attribute]
    
    if as_numpy:
        X = X.values
        y = y.values
    print(f"Data is of shape {X.shape}.")
    return X, y
    
    
class TestNaiveAutoML(unittest.TestCase):
    
    def setUpClass():
        # setup logger for this test suite
        logger = logging.getLogger('naml_test')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        log_level = logging.WARN
        
        # configure naml logger (by default set to WARN, change it to DEBUG if tests fail)
        naml_logger = logging.getLogger("naml")
        naml_logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        naml_logger.addHandler(ch)
        
        # configure naml logger (by default set to WARN, change it to DEBUG if tests fail)
        naml_eval_logger = logging.getLogger("naiveautoml.evalpool")
        naml_eval_logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        naml_eval_logger.addHandler(ch)
        
        # configure naml logger (by default set to WARN, change it to DEBUG if tests fail)
        naml_eval_logger = logging.getLogger("naiveautoml.hpo")
        naml_eval_logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        naml_eval_logger.addHandler(ch)
        
    def setUp(self):
        self.logger = logging.getLogger("naml_test")
        self.naml_logger = logging.getLogger("naml")
        self.num_seeds = 5

        
        
        
        
        
    @parameterized.expand([
            (61,),
            (188,), # eucalyptus. Very important because has both missing values and categorical attributes
            
        ])
    def test_acceptance_of_dataframe(self, openmlid):
        X, y = get_dataset(openmlid, as_numpy = False)
        naml = naiveautoml.NaiveAutoML(logger_name="naml", timeout=15, max_hpo_iterations=1, show_progress=True)
        naml.fit(X, y)
        
    @parameterized.expand([
            (188,), # eucalyptus. Very important because has both missing values and categorical attributes
        ])
    def test_definition_of_own_categorical_attributes_in_dataframe(self, openmlid):
        X, y = get_dataset(openmlid, as_numpy = False)
        naml = naiveautoml.NaiveAutoML(logger_name="naml", timeout=15, max_hpo_iterations=1, show_progress=True)
        
        if openmlid == 188:
            categorical_features_by_string = ["Abbrev", "Locality", "Map_Ref", "Latitude", "Altitude", "Sp"] # Altitude is normally not categorical
            categorical_features_by_index = [list(X.columns).index(c) for c in categorical_features_by_string]
        naml.fit(X, y, categorical_features=categorical_features_by_index)
        naml.fit(X, y, categorical_features=categorical_features_by_string)
        
    @parameterized.expand([
            (188,), # eucalyptus. Very important because has both missing values and categorical attributes
        ])
    def test_definition_of_own_categorical_attributes_in_numpy(self, openmlid):
        X, y = get_dataset(openmlid, as_numpy = True)
        naml = naiveautoml.NaiveAutoML(logger_name="naml", timeout=15, max_hpo_iterations=1, show_progress=True)
        
        if openmlid == 188:
            categorical_features = [0, 2, 3, 4, 5, 9] # Altitude (5) is normally not categorical
            
        naml.fit(X, y, categorical_features=categorical_features)
        
        

    def test_constant_algorithms_in_hpo_phase(self):
        """
        This function checks that the algorithms are not changed during the HPO phase.

        :return:
        """

        X, y = get_dataset(61)

        # run naml
        naml = naiveautoml.NaiveAutoML(logger_name="naml", timeout=60, max_hpo_iterations=10, show_progress=True)
        naml.fit(X, y)
        history = naml.history.iloc[naml.steps_after_which_algorithm_selection_was_completed:]
        self.assertTrue(len(pd.unique(history["learner_class"])) <= 2)
        self.assertTrue(len(pd.unique(history["data-pre-processor_class"])) <= 2)
        self.assertTrue(len(pd.unique(history["feature-pre-processor_class"])) <= 2)
        
        
    """
        This checks naiveautoml convergence time and performance
    """
    #, , 1475, , , , , 4541, , , , 4135, 40978, 40996, 41027, 40981, 40982, 40983, 40984, 40701, 40670, 40685, 40900,  1111, 42732, 42733, 42734, 40498, 41161, 41162, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41142, 41143, 41144, 41145, 41146, 41147, 41150, 41156, 41157, 41158,  41159, 41138, 54, 181, 188, 1461, 1494, 1464, 12, 23, 3, 1487, 40668, 1067, 1049, 40975, 31
    
    ''' [
            (openmlid, expected runtime, expected performance),
            (61, 5, 0.95),
            (188, 60, 0.5) # eucalyptus. Very important because has both missing values and categorical attributes
            (1485, 240, 0.82),
            (1515, 240, 0.85),
            (1468, 120, 0.94),
            (1489, 180, 0.89),
            (23512, 600, 0.65),
            (23517, 600, 0.5),
            (4534, 180, 0.92),
            (4538, 400, 0.66),
            (4134, 400, 0.79),
            
        ]
    '''

    @parameterized.expand([
            (61, 7, 0.9),
            (188, 260, 0.5), # eucalyptus. Very important because has both missing values and categorical attributes
            #(1485, 240, 0.82),
            #(1515, 240, 0.85),
            #(1468, 120, 0.94),
            #(1489, 180, 0.89),
            #(23512, 600, 0.65),
            #(23517, 600, 0.5),
            #(4534, 180, 0.92),
            #(4538, 400, 0.66),
            #(4134, 400, 0.79),
            
        ])
    def test_naml_results_classification(self, openmlid, exp_runtime, exp_result):
        X, y = get_dataset(openmlid)
        self.logger.info(f"Start result test for NaiveAutoML on classification dataset {openmlid}")
        
            
        # run naml
        scores = []
        runtimes = []
        for seed in range(1, self.num_seeds + 1):
            
            # create split
            self.logger.debug(f"Running test on seed {seed}/{self.num_seeds}")
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
            
            # run naml
            start = time.time()
            naml = naiveautoml.NaiveAutoML(logger_name="naml", execution_timeout=10, max_hpo_iterations=10, show_progress=True)
            naml.fit(X_train, y_train)
            end = time.time()
            runtime = end - start
            runtimes.append(runtime)
            
            # compute test performance
            self.logger.debug(f"finished training on seed {seed} after {int(np.round(runtime))}s. Now computing performance of solution.")
            y_hat = naml.predict(X_test)
            score = sklearn.metrics.accuracy_score(y_test, y_hat)
            scores.append(score)
            self.logger.debug(f"finished test on seed {seed}. Test score for this run is {score}")
            
        # check conditions
        runtime_mean = int(np.round(np.mean(runtimes)))
        score_mean = np.round(np.mean(scores), 2)
        self.assertTrue(runtime_mean <= exp_runtime, msg=f"Permitted runtime exceeded. Expected was {exp_runtime}s but true runtime was {runtime_mean}")
        self.assertTrue(score_mean >= exp_result, msg=f"Returned solution was bad. Expected was at least {exp_result}s but true avg score was {score_mean}")
        self.logger.info(f"Test on dataset {openmlid} finished. Mean runtimes was {runtime_mean}s and avg accuracy was {score_mean}")
        
    @parameterized.expand([
            (41021, 90, 550), # moneyball
            #(183, 260, 15), # abalone
            (212, 260, 15) # diabetes, has decimal targets
            
        ])
    def test_naml_results_regression(self, openmlid, exp_runtime, exp_result):
        X, y = get_dataset(openmlid)
        self.logger.info(f"Start result test for NaiveAutoML on regression dataset {openmlid}")
        
            
        # run naml
        scores = []
        runtimes = []
        for seed in range(1, self.num_seeds + 1):
            
            # create split
            self.logger.debug(f"Running test on seed {seed}/{self.num_seeds}")
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
            
            # run naml
            start = time.time()
            naml = naiveautoml.NaiveAutoML(logger_name="naml", timeout=120, max_hpo_iterations=10, show_progress=True, task_type="regression")
            naml.fit(X_train, y_train)
            end = time.time()
            runtime = end - start
            runtimes.append(runtime)
            
            # compute test performance
            self.logger.debug(f"finished training on seed {seed} after {int(np.round(runtime))}s. Now computing performance of solution.")
            y_hat = naml.predict(X_test)
            score = sklearn.metrics.mean_squared_error(y_test, y_hat)
            scores.append(score)
            self.logger.debug(f"finished test on seed {seed}. Test score for this run is {score}")
            
        # check conditions
        runtime_mean = int(np.round(np.mean(runtimes)))
        score_mean = np.round(np.mean(scores), 2)
        self.assertTrue(runtime_mean <= exp_runtime, msg=f"Permitted runtime exceeded. Expected was {exp_runtime}s but true runtime was {runtime_mean}")
        self.assertTrue(score_mean <= exp_result, msg=f"Returned solution was bad. Expected was at most {exp_result} but true avg score was {score_mean}")
        self.logger.info(f"Test on dataset {openmlid} finished. Mean runtimes was {runtime_mean}s and avg accuracy was {score_mean}")

        
        
        
        
    @parameterized.expand([
            (61, 10, 0.9),
            #(188, 60, 0.5), # eucalyptus. Very important because has both missing values and categorical attributes
            #(1485, 240, 0.82),
            #(1515, 240, 0.85),
            #(1468, 120, 0.94),
            #(1489, 180, 0.89),
            #(23512, 600, 0.65),
            #(23517, 600, 0.5),
            #(4534, 180, 0.92),
            #(4538, 400, 0.66),
            #(4134, 400, 0.79),
            
        ])
    def test_individual_scoring(self, openmlid, exp_runtime, exp_result):
        return
        X, y = get_dataset(openmlid)
        self.logger.info(f"Start result test for NaiveAutoML on classification dataset {openmlid}")
        
        scoring1 = {
                "name": "accuracy",
                "score_func": lambda y, y_pred: np.count_nonzero(y == y_pred) / len(y),
                "greater_is_better": True,
                "needs_proba": False,
                "needs_threshold": False
            }
        scoring2 = {
            "name": "errorrate",
            "score_func": lambda y, y_pred: np.count_nonzero(y != y_pred) / len(y),
            "greater_is_better": False,
            "needs_proba": False,
            "needs_threshold": False
        }
        scorer = sklearn.metrics.make_scorer(**{k:v for k, v in scoring1.items() if k != "name"})
        
            
        # run naml
        scores = []
        runtimes = []
        for seed in range(1, self.num_seeds + 1):
            
            # create split
            self.logger.debug(f"Running test on seed {seed}/{self.num_seeds}")
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
            
            # run naml
            start = time.time()
            naml = naiveautoml.NaiveAutoML(logger_name="naml", max_hpo_iterations=10, show_progress=True, scoring = scoring1, side_scores=[scoring2])
            naml.fit(X_train, y_train)
            end = time.time()
            runtime = end - start
            runtimes.append(runtime)
            
            # compute test performance
            self.logger.debug(f"finished training on seed {seed} after {int(np.round(runtime))}s. Now computing performance of solution.")
            score = scorer(naml, X_test, y_test)
            scores.append(score)
            self.logger.debug(f"finished test on seed {seed}. Test score for this run is {score}")
            
        # check conditions
        runtime_mean = int(np.round(np.mean(runtimes)))
        score_mean = np.round(np.mean(scores), 2)
        self.assertTrue(runtime_mean <= exp_runtime, msg=f"Permitted runtime exceeded. Expected was {exp_runtime}s but true runtime was {runtime_mean}")
        self.assertTrue(score_mean >= exp_result, msg=f"Returned solution was bad. Expected was at least {exp_result} but true avg score was {score_mean}")
        self.logger.info(f"Test on dataset {openmlid} finished. Mean runtimes was {runtime_mean}s and avg accuracy was {score_mean}")
        
        
        
    @parameterized.expand([
            (61, 10, 0.9),
            #(188, 60, 0.5), # eucalyptus. Very important because has both missing values and categorical attributes
            #(1485, 240, 0.82),
            #(1515, 240, 0.85),
            #(1468, 120, 0.94),
            #(1489, 180, 0.89),
            #(23512, 600, 0.65),
            #(23517, 600, 0.5),
            #(4534, 180, 0.92),
            #(4538, 400, 0.66),
            #(4134, 400, 0.79),
            
        ])
    def test_individual_evaluation(self, openmlid, exp_runtime, exp_result):
        
        X, y = get_dataset(openmlid)
        self.logger.info(f"Start result test for NaiveAutoML on classification dataset {openmlid}")
        
        def evaluation(pl, X, y, scoring_functions):
            return {s: np.mean(sklearn.model_selection.cross_validate(pl, X, y, scoring=s)["test_score"]) for s in scoring_functions}
        
        scorer = sklearn.metrics.get_scorer("accuracy")
        
        # run naml
        scores = []
        runtimes = []
        for seed in range(1, self.num_seeds + 1):
            
            # create split
            self.logger.debug(f"Running test on seed {seed}/{self.num_seeds}")
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
            
            # run naml
            start = time.time()
            naml = naiveautoml.NaiveAutoML(logger_name="naml", max_hpo_iterations=10, show_progress=True, evaluation_fun = evaluation)
            naml.fit(X_train, y_train)
            end = time.time()
            runtime = end - start
            runtimes.append(runtime)
            
            # compute test performance
            self.logger.debug(f"finished training on seed {seed} after {int(np.round(runtime))}s. Now computing performance of solution.")
            score = scorer(naml, X_test, y_test)
            scores.append(score)
            self.logger.debug(f"finished test on seed {seed}. Test score for this run is {score}")
            
        # check conditions
        runtime_mean = int(np.round(np.mean(runtimes)))
        score_mean = np.round(np.mean(scores), 2)
        self.assertTrue(runtime_mean <= exp_runtime, msg=f"Permitted runtime exceeded. Expected was {exp_runtime}s but true runtime was {runtime_mean}")
        self.assertTrue(score_mean >= exp_result, msg=f"Returned solution was bad. Expected was at least {exp_result} but true avg score was {score_mean}")
        self.logger.info(f"Test on dataset {openmlid} finished. Mean runtimes was {runtime_mean}s and avg accuracy was {score_mean}")