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


from gamawrapper import *

from sklearn.experimental import enable_hist_gradient_boosting  # noqa


if __name__ == '__main__':
    # timeouts
    timeout = int(sys.argv[4])
    if timeout in [20, 120, 3600]:
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
    
    search_space_file = "searchspace.json"
    search_space_original = json.load(open(search_space_file))
    search_space_gama = get_gama_search_space(search_space_file)
    allowed_data_preprocessors = list(set([get_class(c["class"]) for c in search_space_original[0]["components"]]))
    allowed_feature_preprocessors = list(set([get_class(c["class"]) for c in search_space_original[1]["components"]]))
    allowed_classifiers = list(set([get_class(c["class"]) for c in search_space_original[2]["components"]]))
    allowed_classifiers.append(sklearn.ensemble.HistGradientBoostingClassifier)
    
    # varialbe to track best internal score (being this a list is just a trick to make it available in the monkey patched function)
    best_internal_score = [-np.inf]
    
    # create GAMA-specific functions
    # these must be defined here, cannot be in a module, because they rely on X and y and must be top level to be pickled
    # setup new compiler function

    # work-around for original evaluation:
    original_evaluation = gama.genetic_programming.compilers.scikitlearn.evaluate_pipeline

    def monkey_patch_evaluate(pipeline, *args, **kwargs):
        
        # check that pipeline steps are ok with search space definition
        classes = [s[1].__class__ for s in pipeline.steps]
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Checking validity of pipeline with components", [c.__name__ for c in classes])
        estimator_class = classes[-1]
        if estimator_class not in allowed_classifiers:
            raise TypeError(f"Classifier must not be " + str(estimator_class.__name__))
        elif len(pipeline.steps) == 2:
            pre_processor = classes[-2]
            if pre_processor not in allowed_data_preprocessors and pre_processor not in allowed_feature_preprocessors:
                raise TypeError(f"Pre-Processor must not be " + str(pre_processor.__name__))
        elif len(pipeline.steps) > 2:
            feature_pre_processor = classes[-2]
            data_pre_processor = classes[-3]
            if data_pre_processor not in allowed_data_preprocessors:
                raise TypeError(f"Data-Pre-Processor must not be " + str(data_pre_processor.__name__))
            if feature_pre_processor not in allowed_feature_preprocessors:
                raise TypeError(f"Feature-Pre-Processor must not be " + str(feature_pre_processor.__name__))

            
        # If all is good, use the original evaluation function
        score = original_evaluation(pipeline, *args, **kwargs)
        if score[1][0] > best_internal_score[0]:
            best_internal_score[0] = score[1][0]
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Finished evaluation. Score is", score[1][0],"Best seen score is", best_internal_score[0])
        return score

    gama.genetic_programming.compilers.scikitlearn.evaluate_pipeline = monkey_patch_evaluate
    
    def primitive_node_to_sklearn(primitive_node: PrimitiveNode) -> object:
        hyperparameters = {
            terminal.output: terminal.value for terminal in primitive_node._terminals
        }
        return compile_pipeline_by_class_and_params(primitive_node._primitive.identifier, hyperparameters, X_train, y_train)

    def new_compiler(
        individual: Individual,
        parameter_checks=None,
        preprocessing_steps: Sequence[Tuple[str, TransformerMixin]] = None,
    ) -> Pipeline:
        steps = [
            (str(i), primitive_node_to_sklearn(primitive))
            for i, primitive in enumerate(individual.primitives)
        ]    
        if preprocessing_steps:
            steps = steps + list(reversed(preprocessing_steps))
        steps = list(reversed(steps))
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Evaluating", len(steps),"step pipeline")
        return Pipeline(steps)

    # create a patched GAMA object
    tmp_folder = folder + "/gama"
    automl = GamaClassifier(max_total_time=timeout, max_eval_time=execution_timeout, config=search_space_gama, n_jobs=1, scoring=scoring, output_directory=tmp_folder,max_memory_mb=memory_limit)
    automl._operator_set._compile = new_compiler
    GamaClassifier.fit = fit_patched
    GamaClassifier._prepare_for_prediction = prepare_for_prediction_patched
    GamaClassifier._predict = predict_patched
    
    
    # run GAMA
    start_time = time.time()
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
    model = automl.model
    print("Chosen Model:", str(model))
    f = open(folder + "/model.txt", "w")
    f.write(str(model))
    f.close()
    
    # decode and write online data
    dfResults = pd.read_csv(tmp_folder + "/evaluations.log", delimiter=";")
    history = []
    for i, row in dfResults.iterrows():
        timestamp_eval_finish = datetime.timestamp(datetime.strptime(row["t_start"], "%Y-%m-%d %H:%M:%S,%f")) + row["t_wallclock"]
        relative_time_finish = timestamp_eval_finish - start_time
        score = float(row["score"][1:row["score"].index(",")])
        if score == -np.inf:
            score = -10**6
        elif score == np.inf:
            score = 10**6
        history.append([relative_time_finish, row["pipeline"], score])
    with open(folder + "/onlinedata.txt", "w") as outfile: 
        json.dump(history, outfile)