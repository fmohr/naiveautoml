import openml
import pandas as pd
import scipy.sparse
import numpy as np
from naiveautoml.commons import *

import json
import sklearn
from sklearn import *
from scipy.sparse import lil_matrix
import itertools as it

from tqdm import tqdm

import random
import logging

from multiprocessing import Pool
import multiprocessing.context
from multiprocessing import get_context

def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]
    num_rows = len(df)
    
    # impute missing values
    cateogry_columns=df.select_dtypes(include=['object', 'category']).columns.tolist()
    obsolete_cols = []
    for column in df:
        if df[column].isnull().any():
            if(column in cateogry_columns):
                if (all(df[column].values == None)):
                    obsolete_cols.append(column)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].median(), inplace=True)
    df = df.drop(columns=obsolete_cols)
    
    # prepare label column as numpy array
    y = np.array(list(df[ds.default_target_attribute].values))
    
    print(f"Read in data frame. Size is {len(df)} x {len(df.columns)}.")
    
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
        
    if X.shape[0] != num_rows:
        raise Exception("Number of rows not ok!")
    return X, y
    
class PipelineSampler:
    
    def __init__(self, search_space_file, X, y, seed):
        self.X = X
        self.y = y
        self.search_space = []
        self.search_space_description = json.load(open(search_space_file))
        self.seed = np.random.randint(10**9) if seed is None else seed
        self.rs = np.random.RandomState(self.seed)
        
        # build all possible structural pipelines
        choices = []
        self.step_names = []
        for step in self.search_space_description:
            self.step_names.append(step["name"])
            choice_in_step = []
            if step["name"] != "classifier":
                choice_in_step.append(None)
            choice_in_step.extend(step["components"])
            choices.append(choice_in_step)
        self.pl_choices = list(it.product(*choices))
    
    ''' Samples a pipeline according to the weights
    '''
    def sample(self, do_build=True):
        
        pl = random.sample(self.pl_choices, 1)[0]
        steps = []
        for name, comp in zip(self.step_names, pl):
            if comp is not None:
                config_space = config_json.read(comp["params"])
                config_space.random = self.rs
                sampled_config = config_space.sample_configuration(1)
                params = {}
                for hp in config_space.get_hyperparameters():
                    if hp.name in sampled_config:
                        params[hp.name] = sampled_config[hp.name]
                steps.append((name, build_estimator(comp, params, self.X, self.y)))
        return sklearn.pipeline.Pipeline(steps)
    

    
    
class RandomSearch():
    
    def __init__(self, searchspace, seed, timeout_total, timeout_per_eval, scoring, side_scorings):
        self.searchspace = searchspace
        self.seed = seed
        self.timeout_total = timeout_total
        self.timeout_per_eval = timeout_per_eval
        self.scoring = scoring
        self.side_scorings = side_scorings
    
    def fit(self, X, y):
        start_time = time.time()
        sampler = PipelineSampler(self.searchspace, X, y, self.seed)
        deadline = start_time + self.timeout_total
        pool = EvaluationPool(X, y, self.scoring, self.side_scorings, tolerance_tuning = 0.05, tolerance_estimation_error = 0.01)
        
        self.history = []
        self.best_score = -np.inf
        self.best_solution = None
        while time.time() < deadline - 10:
            pl = sampler.sample()
            print("Now evaluating " + str(pl).replace("\n", ""))
            print(f"Remaining time: {deadline - time.time()}s")
            try:
                scores = pool.evaluate(pl, deadline=deadline, timeout=self.timeout_per_eval)
            except KeyboardInterrupt:
                raise
            except:
                print("Observed error in execution of pipeline.")
                scores = {scoring: np.nan for scoring in [self.scoring] + self.side_scorings}
            score = scores[self.scoring]
            now = time.time()
            self.history.append([now - start_time, str(pl), scores])
            print(f"Observed scores {scores} for pipeline " + str(pl).replace("\n", "") + f" after {now - start_time}s")
            if score > self.best_score:
                self.best_score = score
                self.best_solution = pl
                print("This is a NEW BEST score.")
        
        # search finished, train final model on all data
        self.best_solution.fit(X, y)
    
    def predict(self, X):
        return self.best_solution.predict(X)
    
    def predict_proba(self, X):
        return self.best_solution.predict_proba(X)
    
    
class GridSearch():
    
    def __init__(self, search_space_file, timeout_total, timeout_per_eval, scoring, side_scorings, logger_name="grid", num_cpus=1):
        self.timeout_total = timeout_total
        self.timeout_per_eval = timeout_per_eval
        self.scoring = scoring
        self.side_scorings = side_scorings
        self.logger = logging.getLogger(logger_name)
        self.num_cpus = num_cpus
        
        # build all possible structural pipelines
        self.search_space_description = json.load(open(search_space_file))
        choices_pp = []
        choices_cl = None
        self.step_names = []
        for step in self.search_space_description:
            self.step_names.append(step["name"])
            choice_in_step = []
            if step["name"] != "classifier":
                choice_in_step.append(None)
            choice_in_step.extend(step["components"])
            
            if step["name"] != "classifier":
                choices_pp.append(choice_in_step)
            else:
                choices_cl = choice_in_step
        
        self.pp_combos = list(it.product(*choices_pp))
        self.classifiers = choices_cl
        
    def get_classifier(self, classifier_descriptor):
        config_space = config_json.read(classifier_descriptor["params"])
        sampled_config = config_space.sample_configuration(1)
        params = {}
        for hp in config_space.get_hyperparameters():
            params[hp.name] = hp.default_value
        return build_estimator(classifier_descriptor, params, self.X_trans, self.y_trans)
        
        
    def evaluate_candidate(self, classifier_descriptor):
        
        try:
            # check timeout
            if time.time() > self.deadline - 10:
                self.logger.info("Approaching timeout, stopping.")
                return

            # build classifier
            classifier = self.get_classifier(classifier_descriptor)

            # build pipeline
            steps_here = self.steps + [("classifier", classifier)]
            pl = Pipeline(steps_here)
            self.logger.info("Now evaluating " + str(pl).replace("\n", "").replace("\t", " "))
            self.logger.info(f"Remaining time: {self.deadline - time.time()}s")
            scores = self.pool.evaluate(Pipeline([("classifier", classifier)]), deadline=self.deadline, timeout=self.timeout_per_eval - self.runtime_transform)

            timestamp = time.time() - self.start_time
            return timestamp, scores
    
        except KeyboardInterrupt:
            raise
        except:
            self.logger.info("Observed error in execution of pipeline.")
            return
        
    def exec_pp_pipeline(self, timeout, pl_pp, X, y):
        return func_timeout(timeout, pl_pp.fit_transform, (X, y))
        
    def fit(self, X, y):
        start_time = time.time()
        self.start_time = start_time
        
        deadline = start_time + self.timeout_total
        self.deadline = deadline
        
        self.history = []
        self.best_score = -np.inf
        self.best_solution = None
        
        # create pool
        p = Pool(self.num_cpus)
        blacklists = {}
        
        # iterate over all possible pipelines
        for step_descriptors in tqdm(self.pp_combos):
            
            
            # check timeout
            if time.time() > deadline - 10:
                self.logger.info("Approaching timeout, stopping.")
                break
            self.logger.info(f"Remaining time: {deadline - time.time()}s")
            
            # build pipeline to transform data
            steps = []
            for pp_name, pp_descriptor in zip(self.step_names[:-1], step_descriptors):
                if pp_descriptor is not None:
                    steps.append((pp_name, get_class(pp_descriptor["class"])()))
            
            # transform data
            starttime_transform = time.time()
            if steps:
                pl_pp = Pipeline(steps)
                timeout = int(min(self.deadline - time.time(), self.timeout_per_eval))
                self.logger.info(f"Transforming data for {steps} with timeout {timeout}s")
                try:
                    handle = p.apply_async(self.exec_pp_pipeline, (timeout, pl_pp, X, y,))
                    X_trans = handle.get(timeout=timeout)
                except KeyboardInterrupt:
                    raise
                except (FunctionTimedOut, multiprocessing.context.TimeoutError):
                    self.logger.info("Timeout observed during transformation. No pipeline with this pre-processing can be executed in the allowed time frame. Canceling.")
                    continue
                except:
                    self.logger.warning(f"Error with feature transformation {steps}. Skipping.")
                    #raise
                    continue
                self.logger.info("Done, now training models.")
            else:
                X_trans = X
            runtime_transform = time.time() - starttime_transform
            self.runtime_transform = runtime_transform
            self.logger.info(f"Feature transformation took {runtime_transform}s. Discounting this from the timeout on the classifier evaluations.")
            
            # efficient pool that relies on previous computations
            self.X_trans = X_trans
            self.y_trans = y
            self.pool = EvaluationPool(X_trans, y, self.scoring, self.side_scorings, tolerance_tuning = 0.05, tolerance_estimation_error = 0.01)
            self.steps = steps

            
            # now run pool
            if self.num_cpus <= 1:
                for classifier_descriptor in self.classifiers:
                    timestamp, scores = self.evaluate_candidate(classifier_descriptor)

                    steps_here = self.steps + [("classifier", self.get_classifier(classifier_descriptor))]
                    pl = Pipeline(steps_here)
                    self.history.append([timestamp, str(pl), scores])
                    self.logger.info(f"Observed scores {scores} for pipeline " + str(pl).replace("\n", "") + f" after {timestamp}s")
                    score = scores[self.scoring]
                    if score > self.best_score:
                        self.best_score = score
                        self.best_solution = pl
                        self.logger.info("This is a NEW BEST score.")
                self.logger.info(f"History has now length {len(self.history)}")
            else:
                self.logger.info(f"Launching pool with {self.num_cpus} threads.")
                

                
                # enqueue jobs
                enqueue_time = time.time()
                multiple_results = [p.apply_async(self.evaluate_candidate, (c,)) for c in self.classifiers]
                
                # collect results
                for classifier_descriptor, handle in zip(self.classifiers, multiple_results):
                    self.logger.debug("Entering loop.")

                    # wait for result to arrive
                    try:
                        local_timeout = self.timeout_per_eval - runtime_transform#max(.1, (enqueue_time + (self.timeout_per_eval - runtime_transform)) - time.time())
                        self.logger.info(f"Waiting at most {local_timeout}s for result of {classifier_descriptor['class']}")
                        result = handle.get(timeout=local_timeout)
                        if result is None:
                            continue
                        timestamp, scores = result
                    except multiprocessing.context.TimeoutError:
                        self.logger.info("Timeout observed while waiting for result")
                        continue
                    except KeyboardInterrupt:
                        raise
                    except:
                        raise

                    self.logger.debug("Result received.")

                    steps_here = self.steps + [("classifier", self.get_classifier(classifier_descriptor))]
                    pl = Pipeline(steps_here)
                    self.history.append([timestamp, str(pl), scores])
                    self.logger.info(f"Observed scores {scores} for pipeline " + str(pl).replace("\n", "") + f" after {timestamp}s")
                    score = scores[self.scoring]
                    if score > self.best_score:
                        self.best_score = score
                        self.best_solution = pl
                        self.logger.info("This is a NEW BEST score.")
                
                self.logger.info(f"History has now length {len(self.history)}")
        
        # enforce that this thing is closed
        #p.close()
        #p.terminate()
        #p.join()
        del p # I know that this is not well done but join sometimes hangs here. This is the work-around
        
        
        # search finished, train final model on all data
        self.logger.info(f"Optimization completed. Chosen pipeline is {self.best_solution} with score {self.best_score}. Now training this on the full data.")
        self.best_solution.fit(X, y)
        self.logger.info("Done. Pipeline trained on full data.")
    
    def predict(self, X):
        return self.best_solution.predict(X)
    
    def predict_proba(self, X):
        return self.best_solution.predict_proba(X)