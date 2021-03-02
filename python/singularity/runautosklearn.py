import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import openml

import scipy as sp
import numpy as np
import pandas as pd
import itertools as it
import sys
import arff
import codecs

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

if __name__ == '__main__':

    
    timeout_per_eval = 60
    timeout = 2 * 60

    
    folder = sys.argv[1][:sys.argv[1].rindex("/")]
    print("Folder is:",folder)
    
    dfTrain = pd.read_csv(sys.argv[1])
    dfTest = pd.read_csv(sys.argv[2])
    labelColumn = sys.argv[3]

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
        X = pd.get_dummies(dfUnion[[c for c in dfUnion.columns if c != labelColumn]]).values.astype(float)
    else:
        print("creating SPARSE data")
        dfSparse = pd.get_dummies(dfUnion[[c for c in dfUnion.columns if c != labelColumn]], sparse=True)
        
        print("dummies created, now creating sparse matrix")
        X = csr_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        print("Done. shape is" + str(X.shape))
    
    print("Training auto-sklearn on " + str(len(dfTrain)) + " instances and validating on " + str(len(dfTest)) + " instances.")
    print("Class label:",labelColumn)
    print("Overall Timeout:", timeout)
    print("Timeout per eval:",timeout_per_eval)
    
    
    X_train = X[:len(dfTrain),:]
    X_test =  X[len(dfTrain):,:]
    y_train = dfTrain[labelColumn].values
    y_test = dfTest[labelColumn].values
    print(str(X_train.shape[0]) + " training instances and " + str(X_test.shape[0]) + " test instances.")
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeout,per_run_time_limit=timeout_per_eval,ensemble_size=1,ensemble_nbest=1)
    print("Launching auto-sklearn")
    print(automl)
    automl.fit(X_train, y_train)
    print("Ready, now computing its loss")
    y_hat = automl.predict(X_test)
    score = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
    print("Error Rate:", score)
    f = open(folder + "/score.txt", "w")
    f.write(str(score))
    f.close()
    model = automl.show_models()
    print("Chosen Model:", str(model))
    f = open(folder + "/model.txt", "w")
    f.write(model)
    f.close()
    
