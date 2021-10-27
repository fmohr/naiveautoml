import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
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
import time


import scipy as sp
import numpy as np
import pandas as pd
import itertools as it
import sys

from scipy.sparse import lil_matrix

import itertools as it

import multiprocessing
from concurrent.futures import TimeoutError
import ast

import resource


from commons import *

class SimpleOptimizer:
    
    def __init__(self, search_space, scoring, datapreprocessor_class, featurepreprocessor_class, predictor_class, hpo):
        self.datapreprocessor_class = datapreprocessor_class
        self.featurepreprocessor_class = featurepreprocessor_class
        self.predictor_class= predictor_class
        self.comps = []
        print(datapreprocessor_class, featurepreprocessor_class, predictor_class)
        if datapreprocessor_class is not None:
            self.comps.append([comp for comp in search_space[0]["components"] if comp["class"][comp["class"].rindex("."):] == datapreprocessor_class[datapreprocessor_class.rindex("."):]][0])
        if featurepreprocessor_class is not None:
            self.comps.append([comp for comp in search_space[1]["components"] if comp["class"][comp["class"].rindex("."):] == featurepreprocessor_class[featurepreprocessor_class.rindex("."):]][0])
        self.comps.append([comp for comp in search_space[2]["components"] if comp["class"][comp["class"].rindex("."):] == predictor_class[predictor_class.rindex("."):]][0])
        
        # decode hpo space and compute size
        self.config_spaces = {}
        self.space_size = 1
        for comp in self.comps:
            config_space_as_string = comp["params"]
            self.config_spaces[comp["class"]] = config_json.read(config_space_as_string)
            self.space_size *= get_hyperparameter_space_size(self.config_spaces[comp["class"]])
        print("Space size is", self.space_size)
        self.best_params = {comp["class"]: None for comp in self.comps}
        self.best_score = -np.inf
        self.hpo = hpo
        
        self.max_time_without_imp = 3600 * 12
        self.max_its_without_imp = 10**4
        self.min_its = 10
        self.its = 0
        
        self.active = True
        self.eval_runtimes = []
        self.history = []
        
    def getPipeline(self, params, X, y):
        steps = [(str(i), build_estimator(comp, params[comp["class"]], X, y)) for i, comp in enumerate(self.comps)]
        return Pipeline(steps=steps)
    
    def evalComp(self, params, X, y):
        
        pl = self.getPipeline(params, X, y)
        try:
            return np.mean(self.pool.evaluate(pl, timeout=86400))
        except FunctionTimedOut:
            print("TIMEOUT")
            return np.nan
        
    def step(self, X, y, remaining_time = None):
        self.its += 1
        
        if not self.active:
            raise Exception("Cannot step inactive HPO Process")
        
        # draw random parameters
        params = {}
        for comp in self.comps:
            sampled_config = self.config_spaces[comp["class"]].sample_configuration(1)
            params[comp["class"]] = {}
            for hp in self.config_spaces[comp["class"]].get_hyperparameters():
                if hp.name in sampled_config:
                    params[comp["class"]][hp.name] = sampled_config[hp.name]

        # evaluate configured pipeline
        time_start_eval = time.time()
        score = self.evalComp(params, X, y)
        runtime = time.time() - time_start_eval
        self.history.append((params, score, runtime))
        self.eval_runtimes.append(runtime)
        print("Observed score of", score, "for params", params)
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
            if remaining_time is None or total_expected_runtime < remaining_time:
                self.active = False
                print("Expected time to evaluate all configurations is only", total_expected_runtime, "Doing exhaustive search.")
                configs = get_all_configurations(self.config_space)
                print("Now evaluation all " + str(len(configs)) + " possible configurations.")
                for params in configs:
                    score = self.evalComp(params, X, y)
                    print("Observed score of", score, "for", self.comp["class"], "with params", params)
                    if score > self.best_score:
                        print("This is a NEW BEST SCORE!")
                        self.best_score = score
                        self.best_params = params
                print("Configuration space completely exhausted.")
    
    def get_best_config(self):
        return self.best_params
    
    def fit(self, X, y):
        
        self.pool = EvaluationPool(X, y, scoring)

        # fit the default configuration
        time_start_eval = time.time()
        score = self.evalComp({comp["class"]: None for comp in self.comps}, X, y)
        runtime = time.time() - time_start_eval
        self.history.append(({}, score, runtime))
        print("Score for default pipeline was", score)
        
        # if hpo is active, tune the pipeline now
        if self.hpo:
            
            print("Now doing hyper-parameter optimization for the pipeline!")
            while self.active:
                self.step(X, y, 10000)
                print(len(self.history))
            
        else:
            print("No HPO activated, ready!")
        print(self.comps, score)
        print(self.best_params)
        self.pl = self.getPipeline(self.best_params, X, y)
        print("Training final pipeline", self.pl)
        self.pl.fit(X, y)
        
    
    def predict(self, X):
        return self.pl.predict(X)
    
    def predict_proba(self, X):
        return self.pl.predict_proba(X)
        

if __name__ == '__main__':
    
    # timeouts
    timeout = 86400    
    print("Timeout:", timeout)
    
    # memory limits
    memory_limit = 6 * 1024
    print("Setting memory limit to " + str(memory_limit) + "MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    
    # pipeline specification
    options = sys.argv[1].split(" ")
    print("Options:", options)
    datapreprocessor = None if options[0] == "None" else options[0]
    featurepreprocessor = None if options[1] == "None" else options[1]
    predictor = options[2]
    hpo = options[3] == "true"
    print(datapreprocessor, featurepreprocessor, predictor, hpo)
    
    # folder
    folder = sys.argv[-3][:sys.argv[-3].rindex("/")]
    print("Folder is:",folder)
    
    dfTrain = pd.read_csv(sys.argv[-3])
    dfTest = pd.read_csv(sys.argv[-2])
    labelColumn = sys.argv[-1]
    
    labels = pd.unique(dfTrain[labelColumn])
    non_target_cols = [c for c in dfTrain.columns if not c == labelColumn]
    
    dfUnion = pd.concat([dfTrain[non_target_cols], dfTest[non_target_cols]], ignore_index=True)
    
    # expand categorical attribute; check whether this needs to be done sparse!
    categorical_attributes = dfUnion.select_dtypes(exclude=['number']).columns
    expansion_size = 1
    for att in categorical_attributes:
        expansion_size *= len(pd.unique(dfUnion[att]))
        if expansion_size > 10**5:
            break
    if expansion_size < 10**5:
        X = pd.get_dummies(dfUnion[[c for c in dfUnion.columns if c != labelColumn]]).fillna(0).values.astype(float)
    else:
        print("creating SPARSE data")
        dfSparse = pd.get_dummies(dfUnion[[c for c in dfUnion.columns if c != labelColumn]], sparse=True).fillna(0)
        
        print("dummies created, now creating sparse matrix")
        X = lil_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        print("Done. shape is" + str(X.shape))
    
    print("Training on " + str(len(dfTrain)) + " instances and validating on " + str(len(dfTest)) + " instances.")
    print("Class label:",labelColumn)
    
    X_train = X[:len(dfTrain),:]
    X_test =  X[len(dfTrain):,:]
    y_train = dfTrain[labelColumn].values
    y_test = dfTest[labelColumn].values
    print(str(X_train.shape[0]) + " training instances and " + str(X_test.shape[0]) + " test instances.")
    
    print("Number of classes in train data:", len(pd.unique(y_train)))
    print("Number of classes in test data:", len(pd.unique(y_test)))
    
    metric_sk = sklearn.metrics.roc_auc_score if len(labels) == 2  else sklearn.metrics.log_loss
    scoring = "roc_auc" if len(labels) == 2 else "neg_log_loss"
    
    # read in search space
    search_space = json.load(open("searchspace.json"))
    simple_optimizer = SimpleOptimizer(search_space, scoring, datapreprocessor, featurepreprocessor, predictor, hpo)
    
    print("Optimizer created, starting training.")
    simple_optimizer.fit(X_train, y_train)
    y_hat = simple_optimizer.predict(X_test)
    y_hat_proba = simple_optimizer.predict_proba(X_test)
    
    if metric_sk == sklearn.metrics.roc_auc_score:
        y_hat_proba = y_hat_proba[:,1]
    error_rate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
    requested_metric = metric_sk(y_test, y_hat_proba, labels=labels)
    
    # write results to file
    result = {
        "errorrate": error_rate,
        "metric": requested_metric,
        "history": simple_optimizer.history
    }
    
    if datapreprocessor is None:
        result["chosenparameters_datapreprocessor"] = None
        result["chosenparameters_featurepreprocessor"] = None if featurepreprocessor is None else simple_optimizer.best_params[simple_optimizer.comps[0]["class"]]
        
    else:
        result["chosenparameters_datapreprocessor"] = simple_optimizer.best_params[simple_optimizer.comps[0]["class"]]
        result["chosenparameters_featurepreprocessor"] = None if featurepreprocessor is None else simple_optimizer.best_params[simple_optimizer.comps[1]["class"]]
    result["chosenparameters_predictor"] = simple_optimizer.best_params[simple_optimizer.comps[-1]["class"]]
    with open(folder + "/results.json", "w") as outfile: 
        json.dump(result, outfile)
    print("Results dumped into", outfile)