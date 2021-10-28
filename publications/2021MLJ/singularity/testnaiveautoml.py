from naiveautoml import * 
from scipy.sparse import lil_matrix
import openml

def getDataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0].dropna()
    y = df[ds.default_target_attribute].values
    
    categorical_attributes = df.select_dtypes(exclude=['number']).columns
    expansion_size = 1
    for att in categorical_attributes:
        expansion_size *= len(pd.unique(df[att]))
        if expansion_size > 10**5:
            break
    
    if expansion_size < 10**5:
        X = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]]).values.astype(float)
    else:
        print("creating SPARSE data")
        dfSparse = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]], sparse=True)
        
        print("dummies created, now creating sparse matrix")
        X = lil_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        print("Done. shape is" + str(X.shape))
    return X, y
    
#timeout = 60 * 5
import resource
         
if __name__ == '__main__':
    
    openmlid = 61
    #openmlid = 4534
    #openmlid = 41147
    #openmlid = 4541
    #openmlid = 1457
    #openmlid = 1515
    #openmlid = 1485
    #openmlid = 41027
    #openmlid = 23512
    #X, y = getDataset(61)
    #X, y = getDataset(41145)
    
    memory_limit = 10 * 1024
    print("Setting memory limit to " + str(memory_limit) + "MB")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024)) 
    
    print("Reading dataset " + str(openmlid))
    X, y = getDataset(openmlid)
    print("Done, starting.")
    scoring = "neg_log_loss"
    metric = sklearn.metrics.log_loss
    scores = []
    for seed in range(0, 1):
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, train_size=0.9, random_state = seed)
        naml = NaiveAutoML("searchspace.json", scoring, num_cpus = 8, execution_timeout = 300, timeout=3600, standard_classifier=sklearn.tree.DecisionTreeClassifier)
        naml.fit(X_train, y_train)
        print("Chosen model: " + str(naml.pl))
        print("Now creating predictions for " + str() + " instances.")
        print("History length:", len(naml.history))
        print("Eval history")
        try:
            score_history = naml.eval_history(X_train, y_train)
            online_data = []
            for i, score in enumerate(score_history):
                pl_json = naml.history[i].copy()
                pl_json["score"] = score
                online_data.append(pl_json)
            print(online_data)
        except:
            print("An error occured in the evaluation of the history.")
        
        y_hat = naml.predict_proba(X_valid)
        #for i, pred in enumerate(y_hat):
         #   print(pred, "(ground truth is " + str(y_valid[i]))
        score = metric(y_valid, y_hat, labels=np.unique(y))
        print("Test score:", score)
        scores.append(score)

    print(scores)
    print(np.mean(scores))
