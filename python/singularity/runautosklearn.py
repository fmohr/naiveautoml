import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import openml
from autosklearn.metrics import accuracy, balanced_accuracy, precision, recall, f1, roc_auc, log_loss

import scipy as sp
import numpy as np
import pandas as pd
import itertools as it
import sys
import arff
import codecs

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

import os
import json
import re
import numpy as np
import datetime
import time
import copy

if __name__ == '__main__':
    
    folder = sys.argv[1][:sys.argv[1].rindex("/")]
    print("Folder is:",folder)
    
    dfTrain = pd.read_csv(sys.argv[1])
    dfTest = pd.read_csv(sys.argv[2])
    labelColumn = sys.argv[3]
    timeout = int(sys.argv[4])
    if timeout in [120, 3600]:
        timeout_per_eval = 300
    elif timeout == 86400:
        timeout_per_eval = 1200
    else:
        raise Exception("Unsupported timeout " + str(timeout))
    
    
    labels = pd.unique(dfTrain[labelColumn])
    non_target_cols = [c for c in dfTrain.columns if not c == labelColumn]
    
    dfUnion = pd.concat([dfTrain[non_target_cols], dfTest[non_target_cols]], ignore_index=True)
    
    # expand categorical attribute; check whether this needs to be done sparse!
    categorical_attributes = dfUnion.select_dtypes(exclude=['number']).columns
    print("Categorical attributes:", categorical_attributes)
    expansion_size = 1
    for att in categorical_attributes:
        expansion_size *= len(pd.unique(dfUnion[att]))
        print(pd.unique(dfUnion[att]))
        if expansion_size > 10**5:
            break
    if expansion_size < 10**5:
        X = pd.get_dummies(dfUnion[[c for c in dfUnion.columns if c != labelColumn]]).fillna(0).values.astype(float)
    else:
        print("creating SPARSE data")
        dfSparse = pd.get_dummies(dfUnion[[c for c in dfUnion.columns if c != labelColumn]], sparse=True).fillna(0)
        
        print("dummies created, now creating sparse matrix")
        X = csr_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        print("Done. shape is" + str(X.shape))
    
    print("Training auto-sklearn on " + str(len(dfTrain)) + " instances and validating on " + str(len(dfTest)) + " instances.")
    print("Class label:",labelColumn, str(len(labels)) + " Labels:", labels)
    metric_ask = autosklearn.metrics.roc_auc if len(labels) == 2  else autosklearn.metrics.log_loss
    metric_sk = sklearn.metrics.roc_auc_score if len(labels) == 2  else sklearn.metrics.log_loss
    print("Metric:", metric_sk)
    print("Overall Timeout:", timeout)
    print("Timeout per eval:",timeout_per_eval)
    
    # reading logging config
    with open('logging.json') as f:
        logging_conf = json.load(f)
    logging_conf["version"] = int(logging_conf["version"])
    logging_conf["disable_existing_loggers"] = logging_conf["disable_existing_loggers"].lower() == "true"
    
    # prepare data
    X_train = X[:len(dfTrain),:]
    X_test =  X[len(dfTrain):,:]
    y_train = dfTrain[labelColumn].values
    y_test = dfTest[labelColumn].values
    print(str(X_train.shape[0]) + " training instances and " + str(X_test.shape[0]) + " test instances.")
    
    print("Number of classes in train data:", len(pd.unique(y_train)))
    print("Number of classes in test data:", len(pd.unique(y_test)))
    
    # prepare and run auto-sklearn as an optimizer (vanilla version) with
    #  - 5-fold CV
    #  - timeout per evaluation: 5 minutes for timeouts with 1h and 20 minutes for a runtime of 24h
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeout,per_run_time_limit=timeout_per_eval,ensemble_size=1,ensemble_nbest=1,metric=metric_ask,logging_config=logging_conf,tmp_folder=folder+"/asklearn_tmp",output_folder=folder+"/asklearn_out",delete_tmp_folder_after_terminate=False, delete_output_folder_after_terminate=False,scoring_functions=[accuracy, balanced_accuracy, precision, recall, f1, roc_auc, log_loss],memory_limit=20 * 1024,resampling_strategy='cv',resampling_strategy_arguments={'folds': 5},
    initial_configurations_via_metalearning=0, exclude_estimators=["passive_aggressive", "sgd"])
    print("Launching auto-sklearn")
    print(automl)
    starttime = time.time()
    automl.fit(X_train, y_train, dataset_name="naiveautoml")
    
    # compute test error and chosen model
    print("Ready, now computing its loss")
    y_hat = automl.predict(X_test)
    y_hat_proba = automl.predict_proba(X_test)
    print("Test Vector has size", y_test.shape)
    print("Pred Vector has size", y_hat.shape)
    print("Prob Vector has size", y_hat_proba.shape)
    print("Prob Vector is:", y_hat_proba)
    if metric_ask == autosklearn.metrics.roc_auc:
        y_hat_proba = y_hat_proba[:,1]
    error_rate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
    requested_metric = metric_sk(y_test, y_hat_proba, labels=labels)
    
    print("Error Rate:", error_rate)
    print("Requested Metric:", requested_metric)
    f = open(folder + "/error_rate.txt", "w")
    f.write(str(error_rate))
    f.close()
    f = open(folder + "/score.txt", "w")
    f.write(str(requested_metric))
    f.close()
    model = automl.show_models()
    print("Chosen Model:", str(model))
    f = open(folder + "/model.txt", "w")
    f.write(model)
    f.close()
    
    # write away cv performance history
    stats = {}
    for key in automl.cv_results_:
        stats[key] = [str(v) for v in automl.cv_results_[key]]
    times = []
    for run_key, run_value in automl.automl_.runhistory_.data.items():
        times.append(run_value.endtime - starttime)
    stats["timestamps"] = times
    with open(folder + "/onlinedata.txt", "w") as outfile: 
        json.dump(stats, outfile)