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


from seminaiveautoml import *
from commons import *


if __name__ == '__main__':
    
    # timeouts
    timeout = int(sys.argv[4])
    if timeout in [120, 3600]:
        execution_timeout = 300
    elif timeout == 86400:
        execution_timeout = 1200
    else:
        raise Exception("Invalid timeout: " + str(timeout))
    
    print("Timeout:", timeout)
    print("Timeout for evaluation:", execution_timeout)
    
    # memory limits
    memory_limit = 20 * 1024
    print("Setting memory limit to " + str(memory_limit) + "MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    
    # folder
    folder = sys.argv[1][:sys.argv[1].rindex("/")]
    print("Folder is:",folder)
    
    dfTrain = pd.read_csv(sys.argv[1])
    dfTest = pd.read_csv(sys.argv[2])
    labelColumn = sys.argv[3]
    
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
    print("Timeout per execution:", execution_timeout)
    
    X_train = X[:len(dfTrain),:]
    X_test =  X[len(dfTrain):,:]
    y_train = dfTrain[labelColumn].values
    y_test = dfTest[labelColumn].values
    print(str(X_train.shape[0]) + " training instances and " + str(X_test.shape[0]) + " test instances.")
    
    print("Number of classes in train data:", len(pd.unique(y_train)))
    print("Number of classes in test data:", len(pd.unique(y_test)))
    
    metric_sk = sklearn.metrics.roc_auc_score if len(labels) == 2  else sklearn.metrics.log_loss
    scoring = "roc_auc" if len(labels) == 2 else "neg_log_loss"
    
    automl = SemiNaiveAutoML("searchspace.json", ["classifier", "data-pre-processor", "feature-pre-processor"], scoring, execution_timeout=execution_timeout, timeout=timeout)
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    y_hat_proba = automl.predict_proba(X_test)
    
    if metric_sk == sklearn.metrics.roc_auc_score:
        y_hat_proba = y_hat_proba[:,1]
    error_rate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
    requested_metric = metric_sk(y_test, y_hat_proba, labels=labels)
    
    # serialize error rate into file
    print("Error Rate:", error_rate)
    print("Requested Metric:", requested_metric)
    f = open(folder + "/error_rate.txt", "w")
    f.write(str(error_rate))
    f.close()
    f = open(folder + "/score.txt", "w")
    f.write(str(requested_metric))
    f.close()
    
    # write chosen model into file
    model = automl.chosen_model
    print("Chosen Model:", str(model))
    f = open(folder + "/model.txt", "w")
    f.write(str(model))
    f.close()
    
    # write online data into file
    score_history = automl.eval_history(X_train, y_train)
    history = []
    for i, score in enumerate(score_history):
        history_entry = automl.history[i]
        if not np.isnan(score):
            history.append([history_entry["time"], str(history_entry["pl"]), score])
    print("History        : " + str(history))
                                 
    with open(folder + "/onlinedata.txt", "w") as outfile: 
        json.dump(history, outfile)