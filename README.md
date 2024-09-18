# Naive AutoML
![https://github.com/github/docs/actions/workflows/python-publish.yml/badge.svg](https://github.com/fmohr/naiveautoml/actions/workflows/python-publish.yml/badge.svg)

`naiveautoml` is a tool to find optimal machine learning pipelines for
- classification tasks (binary, multi-class, or multi-label) and
- regression tasks.

Other than most AutoML tools, `naiveautoml` has no (also no implicit) definitions of timeouts. While timeouts can optionally provided, `naiveautoml` will simply stop as soon as it believes that no better pipeline can be found; this can be surprisingly quick.

## Python
Install via `pip install naiveautoml.`
The current version is 0.1.3.

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

The task type (here classification) is derived automatically, but it can also be specified via `task_type` with values in `classification`, `regression` or `multilabel-indicator` to be sure.

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
naml = naiveautoml.NaiveAutoML(timeout_overall=20)
```
You can modify the pipeline timeout on single pipeline evaluations with

```python
naml = naiveautoml.NaiveAutoML(timeout_candidate=20)
```
*However*, be aware that on many pipelines this time out is *not enforced* since this not safely possible without memory leakage or malfunction.

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

To additionally evaluate other scoring functions (not used to rank candidates), you can use a list of `passive_scorings`:
```python
naml = naiveautoml.NaiveAutoML(scoring="accuracy", passive_scorings=["neg_log_loss", "f1_score"])
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

