from naiveautoml import NaiveAutoML
from experiment_utils import get_dataset
from py_experimenter.result_processor import ResultProcessor
from sklearn.metrics import precision_score
import time
import sklearn

def run_ex(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    start_time = time.time()
    seed = parameters['seed']
    X, y = get_dataset(parameters['dataset_id'])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.8, random_state=seed)
    automl = NaiveAutoML(evaluation_fun=parameters['eval_func'])
    automl.fit(X_train, y_train)
    y_predict = automl.predict(X_test)
    score = precision_score(y_test, y_predict, average='micro')
    end_time = time.time()
    run_time = end_time - start_time
    result_processor.process_results({'start_time': start_time})
    result_processor.process_results({'end_time': end_time})
    result_processor.process_results({'run_time': run_time})
    result_processor.process_results({'pipeline': automl.chosen_model})
    result_processor.process_results({'score': score})
    result_processor.process_results({'history': automl.history})
 