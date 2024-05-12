import logging
import naiveautoml
import numpy as np
import sklearn.datasets

import unittest
from parameterized import parameterized
import time
import openml
import pandas as pd

from typing import Callable


def get_dataset(openmlid, as_numpy = True):
    ds = openml.datasets.get_dataset(
        openmlid,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False
    )
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


def evaluate_randomly(pl, X, y, scoring_functions):
    rs = np.random.RandomState()
    return (
        {s["name"]: rs.random() for s in scoring_functions},  # scores
        {s["name"]: {} for s in scoring_functions}  # evaluation reports
    )

class TestNaiveAutoML(unittest.TestCase):
    
    @staticmethod
    def setUpClass():

        log_level_tester = logging.INFO

        # setup logger for this test suite
        logger = logging.getLogger('naml_test')
        logger.setLevel(log_level_tester)
        ch = logging.StreamHandler()
        ch.setLevel(log_level_tester)
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
        self.num_seeds = 3
        
    @parameterized.expand([
            (61,),
            (188,),  # eucalyptus. Very important because has both missing values and categorical attributes
            
        ])
    def test_acceptance_of_dataframe(self, openmlid):
        self.logger.info(f"Testing acceptance of dataframes")
        X, y = get_dataset(openmlid, as_numpy=False)
        naml = naiveautoml.NaiveAutoML(
            logger_name="naml",
            timeout_overall=15,
            max_hpo_iterations=1,
            show_progress=True,
            raise_errors=True
        )
        naml.fit(X, y)
        
    @parameterized.expand([
            (188,),  # eucalyptus. Very important because has both missing values and categorical attributes
        ])
    def test_definition_of_own_categorical_attributes_in_dataframe(self, openmlid):
        self.logger.info(f"Testing acceptance of definition of own categorical attributes in pandas")
        X, y = get_dataset(openmlid, as_numpy=False)
        naml = naiveautoml.NaiveAutoML(logger_name="naml", timeout_overall=15, max_hpo_iterations=1, show_progress=True)
        
        if openmlid == 188:
            categorical_features_by_string = ["Abbrev", "Locality", "Map_Ref", "Latitude", "Altitude", "Sp"]  # Altitude is normally not categorical
            categorical_features_by_index = [list(X.columns).index(c) for c in categorical_features_by_string]
        naml.fit(X, y, categorical_features=categorical_features_by_index)
        naml.fit(X, y, categorical_features=categorical_features_by_string)
        
    @parameterized.expand([
            (188,),  # eucalyptus. Very important because has both missing values and categorical attributes
        ])
    def test_definition_of_own_categorical_attributes_in_numpy(self, openmlid):
        self.logger.info(f"Testing acceptance of definition of own categorical attributes in numpy")
        X, y = get_dataset(openmlid, as_numpy=True)
        naml = naiveautoml.NaiveAutoML(logger_name="naml", timeout_overall=15, max_hpo_iterations=1, show_progress=True)
        
        if openmlid == 188:
            categorical_features = [0, 2, 3, 4, 5, 9] # Altitude (5) is normally not categorical
            
        naml.fit(X, y, categorical_features=categorical_features)

    @parameterized.expand([
        (61,),   # iris. Quick check for classification
        (531,),  # boston housing. Quick check for regression
        (6,),    # letter. Very important because it has many classes
        (188,),  # eucalyptus. Very important because has both missing values and categorical attributes
    ])
    def test_core_functionality(self, openmlid):
        self.logger.info(f"Start test for core functionality on {openmlid}")
        X, y = get_dataset(openmlid)
        naml = naiveautoml.NaiveAutoML(
            logger_name="naml",
            timeout_overall=60,
            timeout_candidate=5,
            max_hpo_iterations=2,
            show_progress=True,
            raise_errors=False
        )
        naml.fit(X, y)
        self.assertTrue(len(naml.leaderboard) > 0)
        self.logger.info(f"Finished test for core functionality on {openmlid}")

    def test_recoverability_of_pipelines(self):
        openmlid = 61
        self.logger.info(f"Testing recoverability of pipelines from history. On dataset {openmlid}")
        X, y = get_dataset(openmlid)
        naml = naiveautoml.NaiveAutoML(
            logger_name="naml",
            timeout_overall=60,
            max_hpo_iterations=2,
            show_progress=True,
            evaluation_fun=evaluate_randomly
        )
        naml.fit(X, y)
        for i, row in naml.leaderboard.iterrows():
            pl = naml.recover_model(history_index=i)
            pl.predict(X)
        pl = naml.recover_model(pl=sklearn.tree.DecisionTreeClassifier())
        pl.predict(X)
        print(naml.leaderboard)
        self.logger.info(f"Finished test for recoverability of pipelines from history on dataset {openmlid}")

    @parameterized.expand([
        (61,),
    ])
    def test_number_of_evaluations(self, openmlid):
        self.logger.info(f"Test that history will contain entries for all evaluations.")
        max_hpo_iterations = 5
        X, y = get_dataset(openmlid, as_numpy=True)
        naml = naiveautoml.NaiveAutoML(
            logger_name="naml",
            max_hpo_iterations=max_hpo_iterations,
            show_progress=True,
            evaluation_fun=evaluate_randomly
        )
        naml.fit(X, y)

        self.assertEqual(
            sum([len(s["components"]) for s in naml.algorithm_selector.search_space]) + max_hpo_iterations,
            len(naml.history)
        )

    def test_constant_algorithms_in_hpo_phase(self):
        """
        This function checks two things:
        1. that the algorithms are not changed during the HPO phase.
        2. that the algorithm pairing is the one that was best in phase 1.

        :return:
        """

        self.logger.info(f"Testing that all evaluations in HPO phase are for the same algorithms.")

        X, y = get_dataset(61)

        # run naml
        np.random.seed(round(time.time()))
        naml = naiveautoml.NaiveAutoML(
            logger_name="naml",
            timeout_overall=60,
            max_hpo_iterations=10,
            show_progress=True,
            evaluation_fun=evaluate_randomly
        )
        naml.fit(X, y)
        print(naml.history[["learner_class", "neg_log_loss"]])

        # check that there is only one combination of algorithms in the HPO phase
        history = naml.history.iloc[naml.steps_after_which_algorithm_selection_was_completed:]
        self.assertTrue(len(pd.unique(history["learner_class"])) == 1)
        self.assertTrue(len(pd.unique(history["data-pre-processor_class"])) == 1)
        self.assertTrue(len(pd.unique(history["feature-pre-processor_class"])) == 1)

        # get best solution from phase 1
        phase_1_solutions = naml.history.iloc[:naml.steps_after_which_algorithm_selection_was_completed]
        phase_1_solutions = phase_1_solutions[phase_1_solutions[naml.task.scoring["name"]].notna()]
        best_solution_in_phase_1 = phase_1_solutions.sort_values(naml.task.scoring["name"]).iloc[-1]

        for step in ["data-pre-processor", "feature-pre-processor", "learner"]:
            field = f"{step}_class"
            class_in_phase1 = best_solution_in_phase_1[field]
            class_in_phase2 = pd.unique(history[field])[0]
            self.assertEquals(
                class_in_phase1,
                class_in_phase2,
                f"Choice for {step} should conicide but is {class_in_phase1} in AS phase and {class_in_phase2} in HPO."
            )
        
        
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
            (61, 5, 0.9),
            (188, 50, 0.5),  # eucalyptus. Very important because has both missing values and categorical attributes
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
    def test_naml_results_classification(self, openmlid, exp_runtime_per_seed, exp_result):
        X, y = get_dataset(openmlid)
        self.logger.info(f"Start result test for NaiveAutoML on classification dataset {openmlid}")

        exp_runtime = self.num_seeds * exp_runtime_per_seed

        # run naml
        scores = []
        runtimes = []
        for seed in range(1, self.num_seeds + 1):
            
            # create split
            self.logger.debug(f"Running test on seed {seed}/{self.num_seeds}")
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
            
            # run naml
            start = time.time()
            naml = naiveautoml.NaiveAutoML(
                logger_name="naml",
                timeout_candidate=10,
                max_hpo_iterations=5,
                show_progress=True
            )
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
        self.assertTrue(runtime_mean <= exp_runtime, msg=f"Permitted runtime exceeded on dataset {openmlid}. Expected was {exp_runtime}s but true runtime was {runtime_mean}")
        self.assertTrue(score_mean >= exp_result, msg=f"Returned solution was bad on dataset {openmlid}. Expected was at least {exp_result}s but true avg score was {score_mean}")
        self.logger.info(f"Test on dataset {openmlid} finished. Mean runtimes was {runtime_mean}s and avg accuracy was {score_mean}")
        
    @parameterized.expand([
            (41021, 120, 650), # moneyball
            #(183, 260, 15), # abalone
            (212, 120, 15)  # diabetes, has decimal targets
            
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
            naml = naiveautoml.NaiveAutoML(
                logger_name="naml",
                timeout_overall=75,
                max_hpo_iterations=5,
                show_progress=True,
                task_type="regression",
                evaluation_fun="mccv_1"
            )
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
            (61, 30, 0.9),
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
        X, y = get_dataset(openmlid)
        self.logger.info(f"Testing individual scoring function on dataset {openml}")
        
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
        scorer = sklearn.metrics.make_scorer(**{k: v for k, v in scoring1.items() if k != "name"})

        # run naml
        scores = []
        runtimes = []
        for seed in range(1, self.num_seeds + 1):
            
            # create split
            self.logger.debug(f"Running test on seed {seed}/{self.num_seeds}")
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
            
            # run naml
            start = time.time()
            naml = naiveautoml.NaiveAutoML(
                logger_name="naml",
                max_hpo_iterations=10,
                show_progress=True,
                scoring=scoring1,
                timeout_candidate=2,
                timeout_overall=20,
                passive_scorings=[scoring2]
            )
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
        self.assertTrue(runtime_mean <= exp_runtime, msg=f"Permitted runtime exceeded on {openmlid}. Expected was {exp_runtime}s but true runtime was {runtime_mean}")
        self.assertTrue(score_mean >= exp_result, msg=f"Returned solution was bad on {openmlid}. Expected was at least {exp_result} but true avg score was {score_mean}")
        self.logger.info(f"Test on dataset {openmlid} finished. Mean runtimes was {runtime_mean}s and avg accuracy was {score_mean}")

    @parameterized.expand([
            (61, 30, 0.9),
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
        self.logger.info(
            f"Testing individual evaluation function on dataset {openml}. "
            f"Expecting {exp_result} after {exp_runtime}s"
        )
        
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
            naml = naiveautoml.NaiveAutoML(
                logger_name="naml",
                max_hpo_iterations=10,
                show_progress=True,
                evaluation_fun=evaluate_randomly,
                timeout_candidate=2
            )
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
        (61, 30, 0.9),
        # (188, 60, 0.5), # eucalyptus. Very important because has both missing values and categorical attributes
        # (1485, 240, 0.82),
        # (1515, 240, 0.85),
        # (1468, 120, 0.94),
        # (1489, 180, 0.89),
        # (23512, 600, 0.65),
        # (23517, 600, 0.5),
        # (4534, 180, 0.92),
        # (4538, 400, 0.66),
        # (4134, 400, 0.79),

    ])
    def test_individual_stateful_evaluation(self, openmlid, exp_runtime, exp_result):
        X, y = get_dataset(openmlid)
        self.logger.info(f"Start test of individual stateful evaluation function on dataset {openmlid}.")

        class Evaluator(Callable):

            def __init__(self):
                self.history = []

            def reset(self):
                self.history = []

            def __call__(self, pl, X, y, scoring_functions):
                results = {
                    s["name"]: np.random.rand()
                    for s in scoring_functions
                }
                evaluation_report = {
                    s["name"]: {} for s in scoring_functions
                }
                return results, evaluation_report

            def update(self, pl, results):
                self.history.append([pl, results])

        scorer = sklearn.metrics.get_scorer("accuracy")
        evaluation = Evaluator()

        # run naml
        scores = []
        runtimes = []
        for seed in range(1, self.num_seeds + 1):

            evaluation.reset()

            # create split
            self.logger.debug(f"Running test on seed {seed}/{self.num_seeds}")
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

            # run naml
            start = time.time()
            naml = naiveautoml.NaiveAutoML(logger_name="naml", max_hpo_iterations=10, show_progress=True,
                                           evaluation_fun=evaluation)
            naml.fit(X_train, y_train)
            end = time.time()
            runtime = end - start
            runtimes.append(runtime)

            # compute test performance
            self.logger.debug(
                f"finished training on seed {seed} after {int(np.round(runtime))}s. Now computing performance of solution.")
            score = scorer(naml, X_test, y_test)
            scores.append(score)
            self.logger.debug(f"finished test on seed {seed}. Test score for this run is {score}")

            self.assertEqual(len(naml.history[naml.history["status"] != "avoided"]), len(evaluation.history), "History lengths don't match!")

        # check conditions
        runtime_mean = int(np.round(np.mean(runtimes)))
        score_mean = np.round(np.mean(scores), 2)
        self.assertTrue(runtime_mean <= exp_runtime,
                        msg=f"Permitted runtime exceeded. Expected was {exp_runtime}s but true runtime was {runtime_mean}")

        # we also check the score, because the result here *should* be good. if not, the values might not be used
        self.assertTrue(score_mean >= exp_result,
                        msg=f"Returned solution was bad. Expected was at least {exp_result} but true avg score was {score_mean}")
        self.logger.info(
            f"Test on dataset {openmlid} finished. Mean runtimes was {runtime_mean}s and avg accuracy was {score_mean}")

    def test_searchspaces(self):

        for openmlid, task_type in {
            61: "classification",  # iris
            531: "regression"  # boston housing
        }.items():

            self.logger.info(f"Testing search space on {task_type} task on dataset {openmlid}")
            X, y = get_dataset(openmlid)

            scoring = "accuracy" if task_type == "classification" else "r2"

            naml = naiveautoml.NaiveAutoML(
                task_type=task_type,
                scoring=scoring,
                timeout_candidate=2,
                evaluation_fun="mccv_1"
            )
            task = naml.get_task_from_data(X, y, None)
            naml.reset(task)
            hp_optimizer = naml.hp_optimizer

            from naiveautoml.algorithm_selection._sklearn_hpo import HPOHelper
            helper = HPOHelper(search_space=naml.algorithm_selector.search_space)

            faked_as_info_raw = {
                f"{k['name']}_class": None for k in naml.algorithm_selector.search_space
            }

            for step in naml.algorithm_selector.search_space:
                step_name = step["name"]
                algo_set = step["components"]

                for algo in algo_set:

                    if "sklearn.neighbors._classification.KNeighborsClassifier" in algo["class"]:
                        continue

                    self.logger.info(f"Next algorithm: {algo['class']}")
                    selection = {step_name: algo["class"]}
                    if step_name != "learner":
                        selection.update({"learner": naml.algorithm_selector.search_space[2]["components"][3]["class"]})

                    faked_as_info = faked_as_info_raw.copy()
                    faked_as_info.update({
                        f"{k}_class": v for k, v in selection.items()
                    })

                    # get HPO process for supposed selection
                    hp_optimizer.reset(
                        task=task,
                        runtime_of_default_config=0,
                        config_space=helper.get_config_space_for_selected_algorithms(selection),
                        history_descriptor_creation_fun=lambda hp_config: naml.algorithm_selector.create_history_descriptor(faked_as_info, hp_config),
                        evaluator=naml.evaluator
                    )

                    # create and evaluate random configurations
                    for seed in range(10):
                        self.logger.debug(f"Evaluating seed {seed}")
                        s_result = hp_optimizer.step().iloc[0]
                        status = s_result["status"]
                        score = s_result[scoring]
                        exception = s_result["exception"]
                        self.assertFalse(
                            np.isnan(score) and status == "ok",
                            "Observed nan score even though status is ok"
                        )
                        self.logger.debug(f"{seed}: {status} {score}")
                        if status == "exception":
                            allowed_exception_texts = [
                                "There are significant negative eigenvalues",
                                "ValueError: array must not contain infs or NaNs",
                                "ValueError: illegal value in 4th argument of internal gesdd"
                            ]
                            if not any([t in exception for t in allowed_exception_texts]):
                                self.logger.exception(exception)
                                self.fail("Status must not be an (not white-listed) exception!")

                        if status not in ["ok", "timeout"]:
                            self.logger.warning(f"Observed uncommon status \"{status}\".")

    @parameterized.expand([
        (61, )
    ])
    def test_process_leak(self, openmlid):
        X, y = get_dataset(openmlid)
        self.logger.info(f"Start test of individual stateful evaluation function on dataset {openmlid}.")

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.8)
        for i in range(1, 21):
            self.logger.info(f"Run {i}-th instance")
            automl = naiveautoml.NaiveAutoML(
                evaluation_fun="mccv_1",
                show_progress=True,
                timeout_overall=30
            )
            automl.fit(X_train, y_train)
