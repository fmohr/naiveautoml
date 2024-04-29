# Naive AutoML

## Why to use?
`naiveautoml` has significant performance advantages over other tools like auto-sklearn in the short run and is hardly outperformed in the long run.
The following figures show average advantages of `naiveautoml` with vanilla auto-sklearn on the AutoML benchmark datasets over 24 hours (here for mean advantage in cross-entropy over 62 datasets); positive values indicate that `naiveautoml` is better than another approach.
Vertical lines are visual aids for 10 minutes and 1h respectively.
As can be seen, `naiveautoml` is substantially better in the short run (first hour) than both `autosklearn` and a `random search` and is not outperformed in the long run.
It is slightly outperformed by random forests in the (very) short run but outperforms random forests in the long run.

![Results for the comparison with auto-sklearn on the AutoML benchmark datasets](https://github.com/fmohr/naiveautoml/blob/master/publications/2022MLJ/plots/advantages.jpg)

Note that `naiveautoml` does not ask you for *any* parametrization (not even a timeout); it still provides you with the possibility to customize a lot of its behavior. Given our exhaustive empirical results, you can be confident that you will get at most of the times results that are comparable to what you would get with other tools.

Another great feature of `naiveautoml` is that you can use it directly on data with missing values or categorical attributes. `naiveautoml` will try to automatically detect columns with values others than numbers and treat them as categorical.
Missing values will be imputed per default with the median value of a column (for numerical attributes) or the mode (for categorical attributes).

## Python
Install via `pip install naiveautoml.`
The current version is 0.0.29.

We highly recommend to check out the [usage example python notebook](https://github.com/fmohr/naiveautoml/blob/master/python/usage-example.ipynb).

Finding an optimal model for your data is then as easy as running:

```python
import naiveautoml
import sklearn.datasets
naml = naiveautoml.NaiveAutoML()
X, y = sklearn.datasets.load_iris(return_X_y=True)
naml.fit(X, y)
print(naml.chosen_model)
```

To get the **history** of considered pipelines, together with a (relative) timestamp and internal validation scores, you can access the history:

```python
print(naml.history)
```

Want to limit the **number of candidates considered during hyper-parameter tuning**?

```python
naml = naiveautoml.NaiveAutoML(max_hpo_iterations=20)
```
Want to put a **timeout**? Specify it *in seconds* (should be always bigger than 10s to avoid strange side effects).

```python
naml = naiveautoml.NaiveAutoML(timeout=20)
```
The timeout for pipeline evaluations is 10 seconds per default. You can modify this timeout on single pipeline evaluations with

```python
naml = naiveautoml.NaiveAutoML(execution_timeout=20)
```

This can also be **combined** with `max_hpo_iterations`.

Want to see the **progress bar** for the optimization process?

```python
naml = naiveautoml.NaiveAutoML(show_progress=True)
```

Want logging?

```python
# configure logger
import logging
logger = logging.getLogger('naiveautoml')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
```

### Scoring functions
By default, log-loss is used for classification (AUROC in the case of binary classification). To use a custom scoring function, pass it in the constructor:

```python
naml = naiveautoml.NaiveAutoML(scoring="accuracy")
```

To additionally evaluate other scoring functions (not used to rank candidates), you can use a list of `side_scores`:
```python
naml = naiveautoml.NaiveAutoML(scoring="accuracy", side_scores=["neg_log_loss", "f1_score"])
```

You can also pass a custom scoring function through a dictionary:

```python
scorer = make_scorer(**{
            "name": "accuracy",
            "score_func": lambda y, y_pred: np.count_nonzero(y == y_pred).mean(),
            "greater_is_better": True,
            "needs_proba": False,
            "needs_threshold": False
        })
naml = naiveautoml.NaiveAutoML(scoring=scorer)
```

### Custom Categorical Features
Naive AutoML determines the categorical attributes automatically as far as possible.
However, sometimes even columns consisting only of numbers should be treated as categorical attributes.
To pass an explicit list of attributes that should be treated as categoricals, use the `categorical_features` parameter in the `fit` function:

```python
naml.fit(X_df, y, categorical_features=["name_of_first_categorical_attribute", "name_of_second_categorical_attribute"])
```
alternatively (or if your data is a numpy array), you can use the index of the column:
```python
naml.fit(X_df, y, categorical_features=[4, 9])
```

## Citing naive automl
Please use the reference from the Machine Learning Journal to cite Naive AutoML:

https://link.springer.com/article/10.1007/s10994-022-06200-0#article-info

```
@article{mohr2022naive,
  title={{Naive Automated Machine Learning}},
  author={Mohr, Felix and Wever, Marcel},
  journal={Machine Learning},
  pages={1131--1170},
  year={2022},
  publisher={Springer},
  volume={112},
  issue={4}
}

