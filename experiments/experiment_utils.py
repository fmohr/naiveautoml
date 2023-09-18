import openml

def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]

    X = df.drop(columns=[ds.default_target_attribute]).values
    y = df[ds.default_target_attribute].values

    return X, y