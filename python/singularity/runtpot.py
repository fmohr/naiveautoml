import tpot
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import scipy as sp
import numpy as np
import pandas as pd
import itertools as it
import sys

if __name__ == '__main__':

    

    timeout = 30
    
    folder = sys.argv[1][:sys.argv[1].rindex("/")]
    print("Folder is:",folder)
    
    dfTrain = pd.read_csv(sys.argv[1])
    dfTest = pd.read_csv(sys.argv[2])
    labelColumn = sys.argv[3]
    print("Training on " + str(len(dfTrain)) + " instances and validating on " + str(len(dfTest)) + " instances.")
    print("Class label:",labelColumn)
    print("Timeout:", timeout)
    
    X_train = dfTrain[[c for c in dfTrain.columns if not c == labelColumn]].values
    X_test = dfTest[[c for c in dfTrain.columns if not c == labelColumn]].values
    y_train = dfTrain[labelColumn].values
    y_test = dfTest[labelColumn].values
    print(str(len(X_train)) + " training instances and " + str(len(X_test)) + " test instances.")
    automl = tpot.TPOTClassifier(max_time_mins=int(np.ceil(timeout / 60)))
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    score = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
    print("Error Rate:", score)
    f = open(folder + "/score.txt", "w")
    f.write(str(score))
    f.close()
    model = automl.fitted_pipeline_
    print("Chosen Model:", str(model))
    f = open(folder + "/model.txt", "w")
    f.write(model)
    f.close()
    
