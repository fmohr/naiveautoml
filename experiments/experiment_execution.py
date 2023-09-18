from naiveautoml import NaiveAutoML
from experiment_utils import get_dataset
from py_experimenter.result_processor import ResultProcessor
import sklearn

def run_ex(parameters: dict, result_processor: ResultProcessor, custon_config: dict):
    seed = parameters['seed']
    X, y = get_dataset(parameters['dataset_id'])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.8, random_state=seed)
    automl = NaiveAutoML()
    automl.fit(X_train, y_train)
    history = automl.history.iloc[-1]
    result = {}
    for column, value in history.items():
        if column == 'status':
            column = 'statusml'
        result[column.replace('-', '_')] = value
    if 'roc_auc' not in result:
        result['roc_au c'] = None
    result_processor.process_results({'learning_df': history})
 