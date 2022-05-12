from multiprocessing import set_start_method
if __name__ == '__main__':
    set_start_method("spawn")

# core stuff
import argparse
import os
import resource
import logging

# gama
from gamawrapper import *
from sklearn.experimental import enable_hist_gradient_boosting  # noqa

# auto-sklearn
import autosklearn.classification
from autosklearn.metrics import accuracy, balanced_accuracy, precision, recall, f1, roc_auc, log_loss

# naiveautoml
import naiveautoml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=int, required=True)
    parser.add_argument('--algorithm', type=str, choices=['rf', 'random', 'auto-sklearn', 'gama', 'grid', 'naive', 'semi-naive'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout_per_eval', type=int, default=300)
    parser.add_argument('--timeout_total', type=int, default=300)
    parser.add_argument('--folder', type=str, default='./tmp/')
    parser.add_argument('--prob_dp', type=float, default=0.25)
    parser.add_argument('--prob_fp', type=float, default=0.25)
    parser.add_argument('--metric', type=str, choices=["neg_log_loss", "roc_auc"], default="neg_log_loss")
    return parser.parse_args()


def get_learner(args,  X_train, y_train, scoring):
    
    name = args.algorithm
    
    
    if name == "rf":
        return sklearn.ensemble.RandomForestClassifier()
    
    elif name == "random":
        return RandomSearch("searchspace.json", args.seed, args.timeout_total, args.timeout_per_eval, scoring, side_scorings = ["accuracy", "roc_auc"])
    
    elif name == "grid":
        return GridSearch("searchspace.json", args.timeout_total * 2, args.timeout_per_eval, scoring, side_scorings = ["accuracy", "roc_auc"], logger_name = "experimenter.grid", num_cpus = 4)
    
    elif name == "auto-sklearn":
        
        # reading logging config
        with open('logging-autosklearn.json') as f:
            logging_conf = json.load(f)
        logging_conf["version"] = int(logging_conf["version"])
        logging_conf["disable_existing_loggers"] = logging_conf["disable_existing_loggers"].lower() == "true"
        
        # configure metric
        metric_ask = autosklearn.metrics.roc_auc if scoring == "roc_auc" else autosklearn.metrics.log_loss
        
        return autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=args.timeout_total,
            per_run_time_limit=args.timeout_per_eval,
            ensemble_size=1, # no ensembling
            ensemble_nbest=1,
            metric=metric_ask,
            logging_config=logging_conf,
            tmp_folder=args.folder+"/asklearn_tmp",
            delete_tmp_folder_after_terminate=False,
            scoring_functions=[accuracy, balanced_accuracy, precision, recall, f1, roc_auc, log_loss],
            memory_limit=60 * 1024,
            resampling_strategy='cv',
            resampling_strategy_arguments={'folds': 5},
            initial_configurations_via_metalearning=0, # disable warm-starting
            exclude={"classifier": ["passive_aggressive", "sgd"]}
        )
    
    elif name == "gama":
        
        # create a patched GAMA object
        tmp_folder = args.folder + "/gama"
        automl = GamaClassifier(max_total_time=timeout, max_eval_time=execution_timeout, config=search_space_gama, n_jobs=1, scoring=scoring, output_directory=tmp_folder,max_memory_mb=memory_limit)
        automl._operator_set._compile = new_compiler
        GamaClassifier.fit = fit_patched
        GamaClassifier._prepare_for_prediction = prepare_for_prediction_patched
        GamaClassifier._predict = predict_patched
        return automl
    
    elif name == "mosaic":
        return mosaic_ml.automl.AutoML(
            time_budget=args.timeout_total,
            time_limit_for_evaluation=args.timeout_per_eval,
            memory_limit=memory_limit,
            seed=args.seed,
            ensemble_size=1,
            verbose=True
        )
    
    elif name in ["naive", "semi-naive"]:
        strictly_naive = name == "naive"
        return naiveautoml.NaiveAutoML(
            search_space = "searchspace.json",
            scoring = scoring,
            side_scores = ["accuracy", "roc_auc"],
            num_cpus = 1,
            execution_timeout = args.timeout_per_eval,
            logger_name = "experimenter.naml",
            max_hpo_iterations = np.inf,
            timeout = args.timeout_total,
            standard_classifier=sklearn.neighbors.KNeighborsClassifier,
            show_progress = False,
            opt_ordering = None,
            strictly_naive=strictly_naive,
            sparse = True)
    
    raise ValueError(f"No factory for learner {name}")

def get_model(args, automl):
    
    name = args.algorithm
    
    if name == "rf":
        return automl
    if name in ["random", "grid"]:
        return automl.best_solution
    if name == "gama":
        return automl.model
    if name == "auto-sklearn":
        return automl.show_models()
    if name in ["naive", "semi-naive"]:
        return automl.chosen_model
    
    raise ValueError(f"No method to compute model for {name}")

def get_history(args, learner, starttime):
    
    name = args.algorithm
    
    if name == "rf":
        return []
    
    if name in ["random", "grid"]:
        return [[e[0], str(e[1]), e[2]] for e in learner.history]
    
    elif name == "auto-sklearn":
        stats = {}
        for key in learner.cv_results_:
            stats[key] = [str(v) for v in learner.cv_results_[key]]
        times = []
        for run_key, run_value in automl.automl_.runhistory_.data.items():
            times.append(run_value.endtime - starttime)
        stats["timestamps"] = times
        return stats
        
    elif name == "gama":
        
        tmp_folder = args.folder + "/gama"
        
        # decode and write online data
        history = []
        dfResults = pd.read_csv(tmp_folder + "/evaluations.log", delimiter=";")
        for i, row in dfResults.iterrows():
            timestamp_eval_finish = datetime.timestamp(datetime.strptime(row["t_start"], "%Y-%m-%d %H:%M:%S,%f")) + row["t_wallclock"]
            relative_time_finish = timestamp_eval_finish - start_time
            score = float(row["score"][1:row["score"].index(",")])
            if score == -np.inf:
                score = -10**6
            elif score == np.inf:
                score = 10**6
            history.append([relative_time_finish, row["pipeline"], score])
        return history
    
    elif name in ["naive", "semi-naive"]:
        return [[e["time"], str(e["pl"]), e["scores"]] for e in learner.history]
        
    else:
        raise ValueError(f"No history logic for {name}")


if __name__ == '__main__':
    
    # avoid deadlocks in parallelization
    #set_start_method("spawn")
    
    # get params
    args = parse_args()
    
    # ger logger
    logger = logging.getLogger('experimenter')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # read timeouts
    timeout = args.timeout_total
    execution_timeout = args.timeout_per_eval
    logger.info(f"Dataset: {args.dataset_id}")
    logger.info(f"Metric: {args.metric}")
    logger.info(f"Timeout: {timeout}")
    logger.info(f"Timeout for evaluation: {execution_timeout}")
    
    # memory limits
    memory_limit = 20 * 1024
    logger.info(f"Setting memory limit to {memory_limit}MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    
    # show CPU settings
    for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"]:
        logger.info(f"\t{v}: {os.environ[v] if v in os.environ else 'n/a'}")
    
    # folder
    folder = args.folder
    logger.info(f"Folder is: {folder}")
    
    # get dataset
    X, y = get_dataset(args.dataset_id)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.9, random_state=args.seed)
    logger.info(f"{X_train.shape[0]} training instances and {X_test.shape[0]} test instances.")
    logger.info(f"Number of classes in train data: {len(pd.unique(y_train))}")
    logger.info(f"Number of classes in test data: {len(pd.unique(y_test))}")
    labels = list(pd.unique(y))
    
    # get metric
    scoring = args.metric
    metric_sk = sklearn.metrics.roc_auc_score if scoring == "roc_auc" else sklearn.metrics.log_loss
    
    
    '''
        START GAMA PATCH LOGIC
        this logic is necessary here to make gama work properly
        looks crazy but it's the way it is. The problem is that
            * the function requires access to the training data AND
            * the function must be defined on the root level
    '''
    search_space_file = "searchspace.json"
    search_space_original = json.load(open(search_space_file))
    search_space_gama = get_gama_search_space(search_space_file)
    allowed_data_preprocessors = list(set([get_class(c["class"]) for c in search_space_original[0]["components"]]))
    allowed_feature_preprocessors = list(set([get_class(c["class"]) for c in search_space_original[1]["components"]]))
    allowed_classifiers = list(set([get_class(c["class"]) for c in search_space_original[2]["components"]]))
    allowed_classifiers.append(sklearn.ensemble.HistGradientBoostingClassifier)

    original_evaluation = gama.genetic_programming.compilers.scikitlearn.evaluate_pipeline
    
    
    # varialbe to track best internal score (being this a list is just a trick to make it available in the monkey patched function)
    best_internal_score = [-np.inf]

    def monkey_patch_evaluate(pipeline, *args, **kwargs):

        # check that pipeline steps are ok with search space definition
        classes = [s[1].__class__ for s in pipeline.steps]
        logger.info(f"Checking validity of pipeline with components {[c.__name__ for c in classes]}")
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
        logger.info(f"Finished evaluation. Score is {score[1][0]}. Best seen score is {best_internal_score[0]}")
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
        logger.info(f"Evaluating {len(steps)} step pipeline")
        return Pipeline(steps)
    
    
    '''
        END GAMA PATCH LOGIC
    '''
    
    
    
    # get tool
    automl = get_learner(args, X_train, y_train, scoring)
    
    # run tool
    start_time = time.time()
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    y_hat_proba = automl.predict_proba(X_test)
    
    if metric_sk == sklearn.metrics.roc_auc_score:
        y_hat_proba = y_hat_proba[:,1]
    error_rate = 1 - sklearn.metrics.accuracy_score(y_test, y_hat)
    requested_metric = metric_sk(y_test, y_hat_proba, labels=labels)
    
    # serialize error rate into file
    logger.info(f"Error Rate: {error_rate}")
    logger.info(f"Requested Metric: {requested_metric}")
    f = open(folder + "/error_rate.txt", "w")
    f.write(str(error_rate))
    f.close()
    f = open(folder + "/score.txt", "w")
    f.write(str(requested_metric))
    f.close()
    
    # write chosen model into file
    model = get_model(args, automl)
    logger.info(f"Chosen Model: {model}")
    f = open(folder + "/model.txt", "w")
    f.write(str(model))
    f.close()
    
    # get history
    history = get_history(args, automl, start_time)
    logger.info(f"History: {history}")
    with open(folder + "/onlinedata.txt", "w") as outfile: 
        json.dump(history, outfile)