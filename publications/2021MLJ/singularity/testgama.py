from commons import *
from gamawrapper import *
import resource

if __name__ == '__main__':
    
    openmlid = 61
    #openmlid = 4534
    #openmlid = 41147
    #openmlid = 4541
    #openmlid = 1457
    #openmlid =1489
    #openmlid = 1515
    #openmlid = 1485
    #openmlid = 1590
    #openmlid = 41027
    #openmlid = 23512
    #X, y = getDataset(61)
    #X, y = getDataset(41145)
    
    memory_limit = 10 * 1024
    print("Setting memory limit to " + str(memory_limit) + "MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    
    print("Reading dataset " + str(openmlid))
    X, y = get_dataset(openmlid)
    print("Done, starting.")
    scoring = "neg_log_loss"
    metric = sklearn.metrics.log_loss
    scores = []
    for seed in range(0, 1):
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, train_size=0.9, random_state = seed)
        
        
        # create GAMA-specific functions
        # these must be defined here, cannot be in a module, because they rely on X and y and must be top level to be pickled
        # setup new compiler function
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
            print("Evaluating", len(steps),"step pipeline")
            return Pipeline(steps)

        def get_gama(X, y, gama_params):

            # setup search space
            search_space = get_gama_search_space("searchspace.json")
            gama_params["config"] = search_space
        
        # create GAMA object
        tmp_folder = "myownout"
        automl = GamaClassifier(max_total_time=15, max_eval_time=5, config=get_gama_search_space("searchspace.json"), n_jobs=1, scoring="neg_log_loss", output_directory=tmp_folder)
        automl._operator_set._compile = new_compiler
        
        # now run GAMA
        print("Running GAMA")
        start_time = time.time()
        automl.fit(X_train, y_train)
        print("GAMA finished")
        
        # decode online data
        dfResults = pd.read_csv(tmp_folder + "/evaluations.log", delimiter=";")
        online_data = []
        for i, row in dfResults.iterrows():
            timestamp_eval_finish = datetime.timestamp(datetime.strptime(row["t_start"], "%Y-%m-%d %H:%M:%S,%f")) + row["t_wallclock"]
            relative_time_finish = timestamp_eval_finish - start_time
            online_data.append([relative_time_finish, row["pipeline"], row["score"]])
        #print(dfResults[['t_start', 't_wallclock', 't_process', 'score']])
        #print(dfResults.columns)
        print(online_data)
        
        
        y_hat = automl.predict_proba(X_valid)
        #for i, pred in enumerate(y_hat):
         #   print(pred, "(ground truth is " + str(y_valid[i]))
        score = metric(y_valid, y_hat, labels=np.unique(y))
        print("Test score:", score)
        scores.append(score)

    print(scores)
    print(np.mean(scores))