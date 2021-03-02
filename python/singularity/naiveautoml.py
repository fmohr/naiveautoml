import numpy as np
import pandas as pd
import random

import sklearn as sk
import sklearn.ensemble
import sklearn.decomposition
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn import *

import pebble
from func_timeout import func_timeout, FunctionTimedOut
from galocal import geneticalgorithm as ga
import time

import json

import itertools as it

import os, psutil
import multiprocessing
from concurrent.futures import TimeoutError

import sys

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


def getLearner(pl):

    #pl should be a 5-tuple: (attributes, baseLearner, baseLearnerParams, metaLearner, metaLearnerParams)
    #clazz = get_class(pl[1])
    params = {} if pl[3] is None else pl[3]
    base_learner = pl[2](**params)
    if len(pl) > 4:
        learner = pl[4](base_learner) # configure the meta-learner, if defined
    else:
        learner = base_learner
    return learner

def approximateBestPossiblePerformance(learning_curve_dict, target_size, max_possible_slope=-0.01 / 1000, estimator=lambda x: np.mean(x)):
    anchor_points = sorted([p for p in learning_curve_dict if len(learning_curve_dict[p]) > 0])

    mean_dict = {}
    lowest_score = 1
    lowest_score_anchor = None
    for p in anchor_points:
        s = estimator(learning_curve_dict[p])
        mean_dict[p] = s
        if s <= lowest_score: # use less-equals here, because we want to use the biggest possible one
            lowest_score = s
            lowest_score_anchor = p
    if len(mean_dict) < 2:
        return 0

    # compute maximum slope pairwise
    max_slope_pairwise = -np.inf
    for i, s1 in enumerate(anchor_points):
        y1 = mean_dict[s1]
        for s2 in anchor_points[i+1:]:
            y2 = mean_dict[s2]
            slope = min(max_possible_slope, (y2-y1) / (s2-s1))
            max_slope_pairwise = max(slope, max_slope_pairwise)

    # compute slope of last leg
    p1 = anchor_points[-2]
    p2 = anchor_points[-1]
    slope_of_last_leg = (mean_dict[p1] - mean_dict[p2]) / (p1 -p2)

    # take the better value
    max_slope = min(max_slope_pairwise, slope_of_last_leg)

    # estimate bound
    last_anchor = lowest_score_anchor
    best_possible_improvement = (target_size - last_anchor) * max_slope
    return mean_dict[last_anchor] + best_possible_improvement

def getBestAverageScore(learning_curve):
    best = 1
    for size in learning_curve:
        best = min(best, np.mean(learning_curve[size]))
    return best

class EvaluationPool:

    def __init__(self, X, y, anchor_points=[(100,5), (200,5), (500,4), (1000,3), (10000,2), (np.inf,2)], tolerance_tuning = 0.05, tolerance_estimation_error = 0.01, iterative = True):
        if X is None:
            raise Exception("Parameter X must not be None")
        if y is None:
            raise Exception("Parameter y must not be None")
        if type(X) != np.ndarray and type(X) != scipy.sparse.csr.csr_matrix and type(X) != scipy.sparse.lil.lil_matrix:
            raise Exception("X must be a numpy array but is " + str(type(X)))
        self.X = X
        self.y = y
        self.anchor_points = [p for p in anchor_points if p[0] <= X.shape[0] * 0.7]
        if len(self.anchor_points) < len(anchor_points):
            self.anchor_points.append((int(X.shape[0] * 0.7), 2))
        self.bestScore = 1.0
        self.tolerance_tuning = tolerance_tuning
        self.tolerance_estimation_error = tolerance_estimation_error
        self.cache = {}
        self.iterative = iterative
        print("Created a pool with training data of shape " + str(X.shape) + " and anchor point configuration " + str(self.anchor_points))
        
    def merge(self, pool):
        num_entries_before = len(self.cache)
        for spl in pool.cache:
            pl, learning_curve, timestamp = pool.cache[spl]
            self.tellEvaluation(pl, learning_curve, timestamp)
        num_entries_after = len(self.cache)
        print("Adopted " + str(num_entries_after - num_entries_before) + " entries from external pool.")

    def tellEvaluation(self, pl, learning_curve, timestamp):
        spl = str(pl)
        self.cache[spl] = (pl, learning_curve, timestamp)
        score = getBestAverageScore(learning_curve)
        if score < self.bestScore:
            self.bestScore = score

            
    def trainAndPredict(self, learner, X, y, train_size):
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, train_size=train_size / X.shape[0])
        try:
            learner.fit(X_train, y_train)
            y_hat = learner.predict(X_valid)
            cv_result = 1 - sklearn.metrics.accuracy_score(y_valid, y_hat)
        except:
            print("EXCEPTION, MAYBE A TIMEOUT", str(sys.exc_info()[0]), "\n",  str(sys.exc_info()))
            cv_result = 1 # punish failed exceptions with 1
        return cv_result
        

    def evaluate(self, pl, bestScoreValue, X = None, timeout=60, deadline=None):
        process = psutil.Process(os.getpid())
        if not X is None and type(X) != np.ndarray and type(X)!= scipy.sparse.csr.csr_matrix and type(X) != scipy.sparse.lil.lil_matrix:
            raise Exception("X must be an array, but is " + str(type(X)) + ": " + str(X))
        
        print("Initializing evaluationg of " + str(pl) + " on " + str(X) +". " + str(process.memory_info().rss / 1024 / 1024)  + "MB. Now awaiting results.")
        start_outer = time.time()
        
        if X is None:
            X = self.X

        spl = str(pl)
        if spl in self.cache:
            return self.cache[spl][1]
        
        # scale the data if defined
        if not pl[0] is None:
            print("Scaling attributes of size " + str(X.shape))
            X_scaled = pl[0].transform(X)
        else:
            X_scaled = self.X
            
        # reduce the data if define
        if not pl[1] is None:
            X_reduced = X_scaled[:,pl[1]]
        else:
            X_reduced = X_scaled

        # collect learning curve results
        learning_curve_dict = {}
        timeout_detected = False
        max_train_size = self.anchor_points[-1][0]
        
        anchor_points = self.anchor_points if self.iterative else [(int(X.shape[0] * 0.7), 5)]
        for size, num_repetitions in anchor_points:
            if timeout_detected:
                break
            
            scores_at_size = []
            learning_curve_dict[size] = scores_at_size
            for seed in range(num_repetitions):
                inst = getLearner(pl)
                
                
                start_inner = time.time()
                timeout_here = timeout if deadline is None else min(timeout, deadline - start_inner)
                print("Evaluating " + str(pl) + " for " + str(timeout_here) + "s on size " + str(size) + "/" + str(X_reduced.shape[0]) + " with seed " + str(seed))
                if timeout_here > 0:
                    try:
                        score = func_timeout(timeout, self.trainAndPredict, (inst, X_reduced, self.y, size))
                        runtime = time.time() - start_inner
    #                    print("Evaluation result ready after " + str(runtime) + "s")
                        if runtime > timeout + 5:
                            print("TIMEOUT VIOLATED! Runtime was " + str(runtime))
                    except FunctionTimedOut as error:
                        print("Timeout detected. Stopping evaluation of " + spl + ". Learning curve is: " + str(learning_curve_dict))
                        timeout_detected = True
                        break
                else:
                    timeout_detected = True
                    break
                scores_at_size.append(score)
                
            estimatedBestPerformance = approximateBestPossiblePerformance(learning_curve_dict, max_train_size)
            if estimatedBestPerformance - (self.tolerance_tuning + self.tolerance_estimation_error) > bestScoreValue.value:
                print("PRUNING " + spl + " since current performance is " + str(learning_curve_dict) + " and best performance can be " + str(estimatedBestPerformance) + " for training on " + str(self.X.shape[0]) + " datapoints, which is worse than " + str(bestScoreValue.value))
                self.cache[spl] = (pl, learning_curve_dict, time.time())
                print("Added to cache. Cache size now " + str(len(self.cache)))
                return learning_curve_dict
            else:
                print("Not enough evidence to prune candidated based on best possible performance " + str(np.round(estimatedBestPerformance, 2)) + " and best known performance " + str(bestScoreValue.value))
            
            if len(scores_at_size) > 0:
                score = np.mean(scores_at_size)
                cbs = bestScoreValue.value # we do not lock here. Maybe we should, but we then need the Manager object
                print(cbs, "vs", score)
                if score < cbs:
                    bestScoreValue.value = score
                    self.bestScore = score
            else:
                del learning_curve_dict[size]
        timestamp = time.time()
        
        runtime = time.time() - start_outer
        print("Completed evaluation of " + spl + " after " + str(runtime) + "s. Learning curve is " + str(learning_curve_dict))
        self.tellEvaluation(pl, learning_curve_dict, timestamp)
        return learning_curve_dict

    def getBestCandidates(self, n):
        candidates = sorted([key for key in self.cache], key=lambda k: getBestAverageScore(self.cache[k][1]))
        return [self.cache[c] for c in candidates[:n]]

def fullname(o):
  # o.__module__ + "." + o.__class__.__qualname__ is an example in
  # this context of H.L. Mencken's "neat, plausible, and wrong."
  # Python makes no guarantees as to whether the __module__ special
  # attribute is defined, so we take a more circumspect approach.
  # Alas, the module name is explicitly excluded from __qualname__
  # in Python 3.

    module = o.__module__
    if module is None or module == str.__class__.__module__:
        return o.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__name__

class NaiveAutoML:

    def __init__(self, scaling = False, filtering=False, wrapping=False, metalearning=False, tuning=False, validation=None, num_cpus = 8, execution_timeout = 10, timeout = None, iterative_evaluations = True):
        self.scaling = scaling
        self.filtering = filtering
        self.wrapping = wrapping
        self.metalearning = metalearning
        self.tuning = tuning
        self.validation = validation
        self.iterative_evaluations = iterative_evaluations

        self.chosen_model = None
        self.chosen_attributes = None
        self.num_cpus = num_cpus
        self.execution_timeouts = execution_timeout
        self.timeout = timeout
        self.stage_entrypoints = {}

    def fit(self, X, y):
        
        self.start_time = time.time()
        self.deadline = self.start_time + self.timeout

        self.baseLearners = [
            (sklearn.svm.LinearSVC, {}),
            (sklearn.tree.DecisionTreeClassifier, {}),
            (sklearn.tree.ExtraTreeClassifier, {}),
            (sklearn.linear_model.LogisticRegression, {}),
            (sklearn.linear_model.PassiveAggressiveClassifier, {}),
            (sklearn.linear_model.Perceptron, {}),
            (sklearn.linear_model.RidgeClassifier, {}),
            (sklearn.linear_model.SGDClassifier, {}),
            (sklearn.neural_network.MLPClassifier, {}),
            (sklearn.discriminant_analysis.LinearDiscriminantAnalysis, {}),
            (sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis, {}),
            (sklearn.naive_bayes.BernoulliNB, {}),
            (sklearn.naive_bayes.MultinomialNB, {}),
            (sklearn.neighbors.KNeighborsClassifier, {})
        ]
        self.finalizedEnsembles = [
            (sklearn.ensemble.ExtraTreesClassifier, {}),
            (sklearn.ensemble.RandomForestClassifier, {}),
            (sklearn.ensemble.GradientBoostingClassifier, {})
        ]
        
        self.metaLearners = [
            sklearn.ensemble.BaggingClassifier,
            sklearn.ensemble.AdaBoostClassifier
        ]
        
        self.sparse_training_data = type(X) == scipy.sparse.csr.csr_matrix or type(X) == scipy.sparse.lil.lil_matrix

        
        ## INITIALIZATION
        if self.validation is None:
            X_train, X_valid, y_train, y_valid = (X, None, y, None)
        else:
            X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, train_size=1 - self.validation)
        print("Fitting NaiveAutoml model. Configs:")
        print("\tTimeout: " + str(self.timeout) + " (Deadline " + str(self.deadline) + ")")
        print("\tScaling: " + str(self.scaling))
        print("\tFiltering: " + str(self.filtering))
        print("\tWrapping: " + str(self.wrapping))
        print("\tMeta-Learners: " + str(self.metalearning))
        print("\tTuning: " + str(self.tuning))
        print("Validation configuration: " + str(self.validation))
        print("X_train: " + str(X_train.shape))
        print("X_valid: " + ("-" if X_valid is None else str(X_valid.shape)))
        self.pools = []

        
        #  PROBING STAGE
        self.stage_entrypoints["probing"] = time.time()
        self.pools.append(self.runProbingStage(X_train, y_train))
        print("Best 10 candidates after probing stage.")
        for c in self.pools[-1].getBestCandidates(10):
            print("\t" + str(c[0]) + ": " + str(getBestAverageScore(c[1])) + " (" + str(c[1]) + ")")
        
        
        ### SECTION FOR OPTIONAL STAGES ###
        
        # 0. Feature Scaling
        self.stage_entrypoints["scaling"] = time.time()
        if self.scaling:
            print("Scaling Stage initialized.")
            self.pools.append(self.runScalingStage(X_train, y_train))
            print("Scaling Stage completed. Best 10 candidates after stage:")
            for c in self.pools[-1].getBestCandidates(10):
                print("\t" + str(c[0]) + ": " + str(getBestAverageScore(c[1])) + " (" + str(c[1]) + ")")
            
        else:
            print("Scaling Stage skipped.")
        
        # 1. Filtering
        self.stage_entrypoints["filtering"] = time.time()
        if self.filtering:
            print("Do Filtering")
            self.pools.append(self.runFilteringStage(X_train, y_train))
            print("Filtering Completed. Best 10 candidates after this stage:")
            for c in self.pools[-1].getBestCandidates(10):
                print("\t" + str(c[0]) + ": " + str(getBestAverageScore(c[1])) + " (" + str(c[1]) + ")")
        else:
            print("Skip Filtering")
            self.filter_results = None
            
        
        # 2. Consolidation
        
        # 3. Meta-Learner
        self.stage_entrypoints["meta"] = time.time()
        if self.metalearning:
            print("Building Meta-Learners (homogeneous ensembles)")
            self.pools.append(self.runMetaLearnerStage(X_train, y_train))
            print("Meta-Learner Stage Completed. Best 10 candidates in the new pool:")
            for c in self.pools[-1].getBestCandidates(10):
                    print("\t" + str(c[0]) + ": " + str(getBestAverageScore(c[1])) + " (" + str(c[1]) + ")")
        else:
            print("Skipping Meta-Learner Stage")
                 
        
        # 4. Wrapping
        self.stage_entrypoints["wrapping"] = time.time()
        if self.wrapping:
            print("Do Wrapping")
            self.pools.append(self.runWrappingStage(X_train, y_train))
            print("Wrapping Completed")
        else:
            print("Skip Wrapping")
        
        # 5. Parameter Tuning
        self.stage_entrypoints["tuning"] = time.time()
        if self.tuning:
            print("Do Parameter Tuning")
            self.pools.append(self.runTuningStage(X_train, y_train))
            print("Parameter Tuning Completed")
        else:
            print("Skip Parameter Tuning")

        # 6. Model Selection
        self.stage_entrypoints["validation"] = time.time()
        can_do_validation = time.time() < self.deadline
        if self.validation is None or not can_do_validation:
            if not can_do_validation:
                print("Cannot do real validation phase due to timeout. Just selecting the best internal model.")
            print("Selecting model. No validation fold available. 10 best available candidates with in-sample errors:")
            best_score = np.inf
            for pool in self.pools:
                for c in pool.getBestCandidates(10):
                    print("\t" + str(c[0]) + ": " + str(getBestAverageScore(c[1])) + " (" + str(c[1]) + ")")
                model = pool.getBestCandidates(1)[0]
                print("Best candidate in this pool:", model)
                score = getBestAverageScore(model[1])
                if score < best_score:
                    print("Better than currently best. Updating best solution.")
                    self.chosen_model = model[0]
                    best_score = score
        else:
            self.chosen_model = self.runModelSelectionStage(X_train, y_train, X_valid, y_valid)
        print("Naive AutoML has finished its search process. Building chosen model " + str(self.chosen_model) + " on full dataset.")
        
        self.finish_time = time.time()
        self.chosen_attributes = self.chosen_model[1]
        X_scaled = X if self.chosen_model[0] is None else self.chosen_model[0].transform(X)
        self.built_model = getLearner(self.chosen_model)
        X_projected = X_scaled if self.chosen_attributes is None else X_scaled[:,self.chosen_attributes]
        self.built_model.fit(X_projected, y)

    def predict(self, X):
        X_scaled = X if self.chosen_model[0] is None else self.chosen_model[0].transform(X)
        X_projected = X_scaled if self.chosen_attributes is None else X_scaled[:,self.chosen_attributes]
        return self.built_model.predict(X_projected)

    def evaluateCandidateSet(self, candidates, epool):
        process = psutil.Process(os.getpid())
        remaining_time = self.deadline - time.time()
        
        print("Evaluating candidate set of " + str(len(candidates)) + " entries. Current memory usage: " + str(process.memory_info().rss / 1024 / 1024)  + "MB. Time to deadline is " + str(remaining_time) + "s. Now awaiting results.")
        
        if self.num_cpus > 1:
            
            if remaining_time > 0:
        
                tpool = NestablePool(self.num_cpus)
                learning_curves_handles = []
                manager = multiprocessing.Manager()
                bestScoreValue = manager.Value("f", epool.bestScore)
                print("initialize best score with " + str(bestScoreValue.value))
                for i, pl in enumerate(candidates):
                    handle = tpool.apply_async(epool.evaluate, (pl,bestScoreValue,None,self.execution_timeouts, self.deadline))
                    print("Enqueued candidate " + str(i + 1) + "/" + str(len(candidates)) + ": " + str(pl))
                    learning_curves_handles.append((pl, handle))



                print("Scheduled " + str(len(learning_curves_handles)) + " jobs and have as many handles. Current memory consumption: " + str(process.memory_info().rss / 1024 / 1024)  + "MB. Now awaiting results.")

                # collect learning curves
                for pl, lc in learning_curves_handles:
                    print("Waiting for result of " + str(pl))
                    start_wait = time.time()
                    try:
                        score = lc.get()
                        end_wait = time.time()
                        epool.tellEvaluation(pl, score, time.time())
                        print("Registered result of " + str(pl) + " after " + str(end_wait - start_wait) + "s: " + str(score))
                    except:
                        print("Observed an error! Adding empty learning curve. Here is the error info: " + str(sys.exc_info()[0]), "\n",  str(sys.exc_info()))
                        epool.tellEvaluation(pl, {}, time.time())
                print("Closing pool.")
                tpool.close()
                print("Joining pool")
                tpool.join()
            else:
                print("No time left, adding empty learning curves for candidates")
                for i, pl in enumerate(candidates):
                    epool.tellEvaluation(pl, {}, time.time())
            
        else:
            print("No parallelism, executing the task in this very thread.")
            manager = multiprocessing.Manager()
            bestScoreValue = manager.Value("f", epool.bestScore)
            for i, pl in enumerate(candidates):
                if remaining_time > 0:
                    lc = epool.evaluate(pl,bestScoreValue,None,self.execution_timeouts,self.deadline)
                    print(lc)
                    epool.tellEvaluation(pl, lc, time.time())
                else:
                    print("No time left, adding empty learning curves for candidate")
                    epool.tellEvaluation(pl, {}, time.time())
                    
        return epool

    def runProbingStage(self, X, y):
        epool = EvaluationPool(X, y, iterative=self.iterative_evaluations)
        print("Starting probing stage")
        self.evaluateCandidateSet([(None, None, bl[0], bl[1]) for bl in self.baseLearners + self.finalizedEnsembles], epool)
        print("Probing completed.")
        return epool
    
    def runScalingStage(self, X, y):
        scalers = [sk.preprocessing.StandardScaler, sk.preprocessing.Normalizer, sk.preprocessing.PowerTransformer, sk.preprocessing.QuantileTransformer, sk.preprocessing.MinMaxScaler]
        pilots = [sklearn.svm.LinearSVC, sklearn.neighbors.KNeighborsClassifier]
        best_previous_pipelines = sorted(self.pools[-1].cache.items(), key=lambda item: getBestAverageScore(item[1][1]))
        
        additional_pilot_index = 0
        while (len(pilots) < 5) & (not best_previous_pipelines[additional_pilot_index][1][0][2] in pilots):
            additional_learner = best_previous_pipelines[additional_pilot_index][1][0][2]
            print("Adding learner from previously best pipeline to pilots:", additional_learner, "with learner")
            pilots.append(additional_learner)
            additional_pilot_index += 1
        
        epool = EvaluationPool(X, y, iterative=self.iterative_evaluations)
        epool.merge(self.pools[-1])
        
        # evaluate all pilots for all scaling techniques
        candidates = []
        scalermap = {}
        for s in scalers:
            if self.sparse_training_data:
                print("Preparing scaler " + str(s))
                if s == sk.preprocessing.StandardScaler:
                    print("Setting with_mean to " + str(~self.sparse_training_data))
                    scaler = s(with_mean=False)
                if s == sk.preprocessing.PowerTransformer:
                    print("Power transformer not supported for sparse data!")
                    continue
            else:
                print("Creating plain object")
                scaler = s()
             
            print("Fitting scaler " + str(s))
            scaler.fit(X)
            scalermap[s] = scaler
            for pilot in pilots:
                candidates.append((scaler, None, pilot, {}))
        print("Evaluating pilots")
        self.evaluateCandidateSet(candidates, epool)
        print("All pilots evaluated. Now checking for which scalers we also check other learners than the pilots.")
        
        # create second pool with all pipelines for the "chosen" scalers
        candidates = []
        for s in scalermap:
            scaler = scalermap[s]
            considered = False
            for pilot in pilots:
                key = str((None, None, pilot, {}))
                score_bare = getBestAverageScore(self.pools[0].cache[key][1])
                #print("Bare score of pilot " + str(pilot) + " is " + str(score_bare))
                pl = (scaler, None, pilot, {})
                lc = epool.cache[str(pl)][1]
                score = getBestAverageScore(lc)
                improvement = score_bare - score
                print("Score and improvement of " + str(pl) + " are " + str(score) + " and " + str(improvement) + " respectively")
                if improvement >= 0.01:
                    candidates.extend([(scaler, None, l[0], l[1]) for l in self.baseLearners + self.finalizedEnsembles if not l in pilots])
                    print("Positive improvement for at least one pilot on scaler " + str(s) + ", considering this scaling for other candidates")
                    break
        print("Evaluating " + str(len(candidates)) + " further candidates on scaled data with hope for improvement")
        self.evaluateCandidateSet(candidates, epool)
        print("Finished evaluation of candidates.")
        return epool
            

    def runFilteringStage(self, X, y):
        tpool = NestablePool(self.num_cpus)
        X_red = X[:1000]
        y_red = y[:1000]
        results = []
        self.filter_results = {}
        result_f_map = {}
        for f in [
                sklearn.feature_selection.f_classif,
                sklearn.feature_selection.mutual_info_classif,
                sklearn.feature_selection.chi2
            ]:
            r = tpool.apply_async(self.runFilter, (X_red, y_red, f))
            result_f_map[r] = f
            results.append(r)
        tpool.close()
        tpool.join()
        print("Filterings completed. Identifying best feature pilot set.")
        best_score = 1
        best_att_selection = None
        for r in results:
            attributes, n, score, args = r.get(timeout = self.execution_timeouts)
            f = result_f_map[r]
            self.filter_results[str(f)] = {"ranking": args, "bestn": n, "score": score}
            if score < best_score:
                best_score = score
                best_att_selection = attributes

        print("Now evaluating all candidates on reduced set: " + str(best_att_selection))
        epool = EvaluationPool(X, y, iterative=self.iterative_evaluations)
        epool.merge(self.pools[-1])
        candidateSet = []
        for spl in epool.cache:
            pl_template = epool.cache[spl][0]
            print(pl_template)
            pl = (pl_template[0], best_att_selection, pl_template[2], pl_template[3])
            print("Enqueuing pipeline " + str(pl) + " for evaluation.")
            candidateSet.append(pl)
        self.evaluateCandidateSet(candidateSet, epool)
        return epool

    def runFilter(self, X, y, f, max_iterations_without_improvement = 20):
        try:
            gus = sklearn.feature_selection.GenericUnivariateSelect(score_func=f)
            gus.fit(X, y)
        except:
            print("Could not fit the selector!")
            gus = None
        
        if gus is None:
            return (None, np.inf, 1.0, [])
        
        pilot = (sklearn.neighbors.KNeighborsClassifier, {"n_neighbors": 1})
        #pilot = sklearn.svm.LinearSVC
        
        try:
            args = list(np.argsort(gus.scores_))
            args.reverse()
            epool = EvaluationPool(X, y, iterative=self.iterative_evaluations)

            best_combo = None
            best_n = 0
            best_score = 1
            iterations_without_improvement = 0
            manager = multiprocessing.Manager()
            bestScoreValue = manager.Value("f", epool.bestScore)

            for n in range(1, len(args) + 1):
                lc = epool.evaluate((None, args[:n], pilot[0], pilot[1]), bestScoreValue)
                if len(lc) > 0:
                    score = getBestAverageScore(lc)
                    print(score)
                    if score < best_score:
                        best_score = score
                        best_n = n
                        best_combo = args[:n]
                        iterations_without_improvement = 0
                    else:
                        iterations_without_improvement += 1
                        if iterations_without_improvement > max_iterations_without_improvement:
                            print("no improvement in " + str(max_iterations_without_improvement) + " iterations. Stopping")
                            break
                else:
                    print("no learning curve obtained; probably due to timeouts. Stopping filtering here.")
                    break
                    
            return (best_combo, best_n, best_score, args)
                
        except:
            print("ERRRROROORRR")
            raise Exception("An exception occurred with input matrix X:\n" + str(X))


            #self.evaluateCandidateSet(self.baseLearners + self.finalizedEnsembles, epool)
    
    def runWrappingStage(self, X, y):
        epool = EvaluationPool(X, y, iterative=self.iterative_evaluations)
        epool.merge(self.pools[-1])
        
        manager = multiprocessing.Manager()
        bestScoreValue = manager.Value("f", epool.bestScore)
        
        timeout = 300
        deadline = time.time() + timeout
        
        for c, entry in sorted(self.pools[-1].cache.items(), key=lambda item: getBestAverageScore(item[1][1])):
            if time.time() > deadline - 5:
                print("Stopping due to timeout for wrapping stage")
                break
                
            pl = entry[0]
            lc = entry[1]
            print("Apply Wrapping for " + str(pl) + " with scores " + str(lc))
            
            # start from filtering results
            model=ga(function=lambda x: getBestAverageScore(epool.evaluate((pl[0], list(np.where(x)[0]), pl[2], pl[3]), bestScoreValue)),dimension=X.shape[1],variable_type='bool',progress_bar=False, convergence_curve = False, algorithm_parameters={"population_size": 10, "timeout": min(60, int(deadline - time.time())), "mutation_probability": 1 / X.shape[1], "max_iteration_without_improv": 20})
            
            # adding init solutions from filtering if available
            if self.filter_results is None:
                print("No filtering conducted, starting feature selection from scratch.")
            else:
                print("Rankings obtained from filtering: " + str(self.filter_results))    
                init_solutions = []
                for fil in self.filter_results:
                    chosen_attributes = self.filter_results[fil]["ranking"][:self.filter_results[fil]["bestn"]]
                    encoding = [x in chosen_attributes for x in range(X.shape[1])]
                    if not encoding in init_solutions:
                        print("Adding initial solution with " + str(sum(encoding)) + " flags: " + str(encoding))
                        init_solutions.append(encoding)
                for encoding in init_solutions:
                    model.add_init_solution(encoding)
            
            # run the actual optimization
            print("Running GA to optimize features for " + str(c))
            try:
                model.run()
                print("GA ready.")
            except:
                print("Observed error during GA execution, maybe just a timeout")
                
        return epool
    
    def runMetaLearnerStage(self, X, y):
        epool = EvaluationPool(X, y, iterative=self.iterative_evaluations)
        epool.merge(self.pools[-1])
        
        manager = multiprocessing.Manager()
        bestScoreValue = manager.Value("f", epool.bestScore)
        
        #timeout = 300
        
        #deadline = time.time() + timeout
        
        pipeline_templates = sorted(self.pools[-1].cache.items(), key=lambda item: getBestAverageScore(item[1][1]))
        
        candidates = []
        for pl_encoding, pl_info in pipeline_templates:
            pl = pl_info[0]
            print("Starting off",pl)
            for meta_learner in self.metaLearners:
                candidates.append((pl[0], pl[1], pl[2], pl[3], meta_learner))
        
        self.evaluateCandidateSet(candidates, epool)
        
        return epool
    
    def runTuningStage(self, X, y):
        epool = EvaluationPool(X, y, iterative=self.iterative_evaluations)
        epool.merge(self.pools[-1])
        
        manager = multiprocessing.Manager()
        bestScoreValue = manager.Value("f", epool.bestScore)
        
        parammap = {
            sklearn.svm.LinearSVC: [
                {
                    "name": "penalty",
                    "type": "cat",
                    "values": ["l1", "l2"]
                },
                {
                    "name": "loss",
                    "type": "cat",
                    "values": ["hinge", "squared_hinge"]
                },
                {
                    "name": "C",
                    "type": "double-exp",
                    "min": -20,
                    "max": 20
                },
                {
                    "name": "dual",
                    "type": "cat",
                    "values": [True, False]
                }
            ],
            sklearn.tree.DecisionTreeClassifier: [
                {
                    "name": "criterion",
                    "type": "cat",
                    "values": ["gini", "entropy"]
                },
                {
                    "name": "splitter",
                    "type": "cat",
                    "values": ["best", "random"]
                },
                {
                    "name": "max_depth",
                    "type": "cat",
                    "values": [None, 1, 10, 100]
                },
                {
                    "name": "max_features",
                    "type": "cat",
                    "values": [None, 1, 2, 5, 10, "auto", "sqrt", "log2"]
                }
            ],
            sklearn.tree.ExtraTreeClassifier: [
                {
                    "name": "criterion",
                    "type": "cat",
                    "values": ["gini", "entropy"]
                },
                {
                    "name": "splitter",
                    "type": "cat",
                    "values": ["best", "random"]
                },
                {
                    "name": "max_depth",
                    "type": "cat",
                    "values": [None, 1, 10, 100]
                }
            ],
            sklearn.linear_model.LogisticRegression: [
                {
                    "name": "penalty",
                    "type": "cat",
                    "values": ["l1", "l2", "elasticnet", "none"]
                },
                {
                    "name": "C",
                    "type": "double-exp",
                    "min": -20,
                    "max": 20
                },
                {
                    "name": "dual",
                    "type": "cat",
                    "values": [True, False]
                }
            ],
            sklearn.linear_model.PassiveAggressiveClassifier: [
                {
                    "name": "C",
                    "type": "double-exp",
                    "min": -20,
                    "max": 20
                }
            ],
            sklearn.linear_model.Perceptron: [
                {
                    "name": "penalty",
                    "type": "cat",
                    "values": ["l1", "l2", "elasticnet", "none"]
                },
                {
                    "name": "alpha",
                    "type": "double-exp",
                    "min": -10,
                    "max": 10
                },
                {
                    "name": "eta0",
                    "type": "double-exp",
                    "min": -10,
                    "max": 10
                }
            ],
            sklearn.linear_model.RidgeClassifier: [
                {
                    "name": "normalize",
                    "type": "cat",
                    "values": [True, False]
                },
                {
                    "name": "alpha",
                    "type": "double-exp",
                    "min": -10,
                    "max": 10
                }
            ],
            sklearn.linear_model.SGDClassifier: [
                {
                    "name": "penalty",
                    "type": "cat",
                    "values": ["l1", "l2", "elasticnet", "none"]
                },
                {
                    "name": "loss",
                    "type": "cat",
                    "values": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
                },
                {
                    "name": "alpha",
                    "type": "double-exp",
                    "min": -10,
                    "max": 10
                }
            ],
            sklearn.neural_network.MLPClassifier: [
                {
                    "name": "hidden_layer_sizes",
                    "type": "cat",
                    "values": [(100,),(100,100,), (200,), (200,200,), (100,100,100,), (200,200,200,)]
                },
                {
                    "name": "activation",
                    "type": "cat",
                    "values": ["identity", "logistic", "tanh", "relu"]
                }
            ],
            sklearn.discriminant_analysis.LinearDiscriminantAnalysis: [
            ],
            sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis: [
                {
                    "name": "reg_param",
                    "type": "double",
                    "min": 0,
                    "max": 1
                }
            ],
            sklearn.naive_bayes.BernoulliNB: [
                {
                    "name": "alpha",
                    "type": "double-exp",
                    "min": -10,
                    "max": 10
                }
            ],
            sklearn.naive_bayes.MultinomialNB: [
                {
                    "name": "alpha",
                    "type": "double-exp",
                    "min": -10,
                    "max": 10
                }
            ],
            sklearn.neighbors.KNeighborsClassifier: [
                {
                    "name": "n_neighbors",
                    "type": "cat",
                    "values": [1, 2, 3, 4, 5, 6, 7, 8, 10,16, 32]
                }
            ],
            sklearn.ensemble.ExtraTreesClassifier: [
                {
                    "name": "n_estimators",
                    "type": "cat",
                    "values": [10, 100, 500, 1000]
                },
                {
                    "name": "criterion",
                    "type": "cat",
                    "values": ["gini", "entropy"]
                },
                {
                    "name": "bootstrap",
                    "type": "cat",
                    "values": [True, False]
                },
                {
                    "name": "max_depth",
                    "type": "cat",
                    "values": [None, 1, 10, 100]
                }
                
            ],
            sklearn.ensemble.RandomForestClassifier: [
                {
                    "name": "n_estimators",
                    "type": "cat",
                    "values": [10, 100, 500, 1000]
                },
                {
                    "name": "criterion",
                    "type": "cat",
                    "values": ["gini", "entropy"]
                },
                {
                    "name": "bootstrap",
                    "type": "cat",
                    "values": [True, False]
                },
                {
                    "name": "oob_score",
                    "type": "cat",
                    "values": [True, False]
                },
                {
                    "name": "max_depth",
                    "type": "cat",
                    "values": [None, 1, 10, 100]
                },
                {
                    "name": "max_features",
                    "type": "cat",
                    "values": [None, 1, 2, 5, 10, "auto", "sqrt", "log2"]
                }
                
            ],
            sklearn.ensemble.GradientBoostingClassifier: [
                {
                    "name": "n_estimators",
                    "type": "cat",
                    "values": [10, 100, 500, 1000]
                },
                {
                    "name": "loss",
                    "type": "cat",
                    "values": ["deviance", "exponential"]
                },
                {
                    "name": "learning_rate",
                    "type": "double-exp",
                    "min": -10,
                    "max": 10
                },
                {
                    "name": "subsample",
                    "type": "double",
                    "min": 0.1,
                    "max": 1
                },
                {
                    "name": "criterion",
                    "type": "cat",
                    "values": ["friedman_mse", "mse", "mae"]
                }
            ]
        }
        
        timeout = 300
        deadline = time.time() + timeout
        
        pipeline_templates = sorted(self.pools[-1].cache.items(), key=lambda item: getBestAverageScore(item[1][1]))
        
        print("Tuning templates in the following order for at most " + str(timeout) + "s")
        for c, entry in pipeline_templates:
            print(entry[0])
        
        for c, entry in pipeline_templates:
            remaining_time = deadline - time.time()
            if remaining_time <= 5:
                print("Stopping due to timeout for tuning stage!")
                break
            pl = entry[0]
            lc = entry[1]
            print("Apply Tuning to " + str(pl) + " with scores " + str(lc))
            print("Remaining time: " + str(remaining_time) + "s")
            
            learner = pl[2]
            
            params_for_learner = parammap[learner] if learner in parammap else None
            if params_for_learner is None:
                print("No parameters defined for " + str(learner) + ", skipping tunint phase!")
                continue
            print("Tuning parameters of " + str(learner))
            
            # draw random configurations
            candidates = []
            
            # check whether we can enumerate
            enumerable = False
            if all([t["type"] == "cat" for t in params_for_learner]):
                domains = [t["values"] for t in params_for_learner]
                num_vals = 1
                for domain in domains:
                    num_vals *= len(domain)
                enumerable = num_vals < 1000
            
            if enumerable:
                names = [t["name"] for t in params_for_learner]
                for combo in it.product(*domains):
                    pmap = {}
                    for i, n in enumerate(names):
                        pmap[n] = combo[i]
                    candidates.append((pl[0], pl[1], pl[2], pmap))
                
            else:
                for i in range(10):
                    pmap = {}
                    for p in params_for_learner:
                        if p["type"] == "cat":
                            pmap[p["name"]] = random.sample(p["values"], 1)[0]
                        elif p["type"] == "double":
                            pmap[p["name"]] = (p["max"] - p["min"]) * np.random.rand() + p["min"]
                        elif p["type"] == "double-exp":
                            pmap[p["name"]] = np.exp(np.random.uniform(p["min"], p["max"]))
                        elif p["type"] == "int":
                            pmap[p["name"]] = np.randint(p["min"], p["max"])

                    candidates.append((pl[0], pl[1], pl[2], pmap))
            
            self.evaluateCandidateSet(candidates, epool)
                
        return epool
        

    def runModelSelectionStage(self, X_train, y_train, X_valid, y_valid):
        candidates = self.pools[-1].getBestCandidates(10)
        best_score = 1
        output = None
        
        satisfaction_validation = min(1, X_valid.shape[0] / 10000)
        influence_validation = satisfaction_validation + X_valid.shape[0] / (X_train.shape[0] + X_valid.shape[0]) * (1 - satisfaction_validation)
        influence_training = 1 - influence_validation
        print("Running validation stage. Validation data has shape " + str(X_valid.shape) + ". This yields a satisfaction of " + str(satisfaction_validation) + " and then, given the split sizes of " + str(X_train.shape) + " and " + str(X_valid.shape) + " , an influence of " + str(influence_validation) + " for the validation set and " + str(influence_training) + " for the training set performance.")
        for pl, lc, timestamp in candidates:
            if not pl[0] is None:
                X_train_scaled = pl[0].transform(X_train)
                X_valid_scaled = pl[0].transform(X_valid)
            else:
                X_train_scaled = X_train
                X_valid_scaled = X_valid
                
            attribute_set = range(X_train_scaled.shape[1]) if pl[1] is None else pl[1]
            learner = getLearner(pl)
            learner.fit(X_train_scaled[:,attribute_set], y_train)
            y_hat = learner.predict(X_valid_scaled[:,attribute_set])
            score_validation = 1 - sklearn.metrics.accuracy_score(y_valid, y_hat)
            score_training = getBestAverageScore(lc)
            
            score = influence_validation * score_validation + influence_training * score_training
            print("Validation score of " + str(pl) + " is " + str(score_validation) + " (influence " + str(influence_validation) + "). Internal score was " + str(score_training) + " (influence " + str(influence_training) + "). Effective score is " + str(score))
            if score < best_score:
                best_score = score
                output = pl
        return output

    def getPools(self):
        return self.pools
    
    def getHistory(self):
        entries = []
        for spl in self.pools[-1].cache:
            pl, lc, timestamp = self.pools[-1].cache[spl]
            score = getBestAverageScore(lc)
            entries.append((int(np.round(1000 * (timestamp - self.start_time))), score))
        return entries
    
    def getStageRuntimeInfo(self):
        entries = {}
        for stage in self.stage_entrypoints:
            entries[stage] = {
                "entrytime": int(np.round(1000 * (self.stage_entrypoints[stage] - self.start_time)))
            }
        stages = ["probing", "scaling", "filtering", "meta", "wrapping", "tuning", "validation"]
        for i, stage in enumerate(stages):
            if i < len(stages) - 1:
                entries[stage]["runtime"] = entries[stages[i+1]]["entrytime"] - entries[stage]["entrytime"]
            else:
                entries[stage]["runtime"] = np.round((self.finish_time - self.start_time) * 1000) - entries[stage]["entrytime"]
        return entries