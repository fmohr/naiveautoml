if __name__ == '__main__':
    import naiveautoml
    import sklearn.datasets
    naml = naiveautoml.NaiveAutoML(show_progress=True)
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    naml.fit(X, y)
    for x in naml.history.iloc[-1].items():
        print(x)