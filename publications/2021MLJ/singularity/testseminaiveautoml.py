from seminaiveautoml import * 
from scipy.sparse import lil_matrix
from commons import *
    
#timeout = 60 * 5
import resource
         
if __name__ == '__main__':
    
    #openmlid = 61
    #openmlid = 4534
    #openmlid = 41147
    #openmlid = 4541
    #openmlid = 1457
    openmlid =1489
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
    X, y = getDataset(openmlid)
    print("Done, starting.")
    scoring = "neg_log_loss"
    metric = sklearn.metrics.log_loss
    scores = []
    for seed in range(0, 1):
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, train_size=0.9, random_state = seed)
        naml = SemiNaiveAutoML("searchspace.json", ["classifier", "data-pre-processor", "feature-pre-processor"], scoring, num_cpus = 8, execution_timeout = 300, timeout=3600)
        naml.fit(X_train, y_train)
        print("Chosen model: " + str(naml.pl))
        print("Now creating predictions for " + str() + " instances.")
        print("History length:", len(naml.history))
        print("Eval history:", naml.history)
        score_history = naml.eval_history(X_train, y_train)
        online_data = []
        for i, score in enumerate(score_history):
            pl_json = naml.history[i].copy()
            pl_json["score"] = score
            online_data.append(pl_json)
        print(online_data)
        
        y_hat = naml.predict_proba(X_valid)
        #for i, pred in enumerate(y_hat):
         #   print(pred, "(ground truth is " + str(y_valid[i]))
        score = metric(y_valid, y_hat, labels=np.unique(y))
        print("Test score:", score)
        scores.append(score)

    print(scores)
    print(np.mean(scores))
