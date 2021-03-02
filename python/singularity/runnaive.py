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


from naiveautoml import *


if __name__ == '__main__':
    execution_timeout = 60
    timeout = 60 * 60
    
    
    memory_limit = 28 * 1024
    print("Setting memory limit to " + str(memory_limit) + "MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    
    print("Args: " + str(sys.argv))
    
    options = ast.literal_eval(sys.argv[1])
    print("Options: " + str(options))
    enable_scaling = options[0]
    enable_filtering = options[1]
    enable_meta = options[2]
    enable_wrapping = options[3]
    enable_tuning = options[4]
    enable_validation = options[5]
    iterative_evaluations = options[6]
    
    folder = sys.argv[2][:sys.argv[2].rindex("/")]
    print("Folder is:",folder)
    
    dfTrain = pd.read_csv(sys.argv[2])
    dfTest = pd.read_csv(sys.argv[3])
    labelColumn = sys.argv[4]
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
        X = pd.get_dummies(dfUnion[[c for c in dfUnion.columns if c != labelColumn]]).values.astype(float)
    else:
        print("creating SPARSE data")
        dfSparse = pd.get_dummies(dfUnion[[c for c in dfUnion.columns if c != labelColumn]], sparse=True)
        
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
    
    automl = NaiveAutoML(scaling = enable_scaling, filtering=enable_filtering, metalearning=enable_meta, wrapping=enable_wrapping,tuning=enable_tuning,validation=enable_validation, num_cpus = 1, execution_timeout = execution_timeout, iterative_evaluations = iterative_evaluations, timeout=timeout)
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    score = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
    
    # serialize error rate into file
    print("Error Rate:", score)
    f = open(folder + "/score.txt", "w")
    f.write(str(score))
    f.close()
    model = automl.chosen_model
    
    # write chosen model into file
    print("Chosen Model:", str(model))
    f = open(folder + "/model.txt", "w")
    f.write(str(model))
    f.close()
    
    # write online data into file
    runtime_info = automl.getStageRuntimeInfo()
    history = automl.getHistory()
    history = [[h[0],h[1]] for h in sorted(history,key=lambda t: t[0])]
    history_reduced = history.copy()
    i = 0
    while i < len(history_reduced):
        if i > 0 and history_reduced[i][1] >= history_reduced[i-1][1]:
            del history_reduced[i]
        else:
            i += 1
    print("History        : " + str(history))
    print("History Reduced: " + str(history_reduced))
                                 
    online_data = {
        "history": history_reduced,
        "stageruntimes": {s: runtime_info[s]["runtime"] for s in runtime_info}
    }
    f = open(folder + "/onlinedata.txt", "w")
    f.write(str(online_data).replace("'", "\""))
    f.close()