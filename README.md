# naiveautoml

## Why to use?
`naiveautoml` has significant performance advantages over other tools like auto-sklearn in the short run and is hardly outperformed in the long run.
The following figures show average advantages of `naiveautoml` with vanilla auto-sklearn on the AutoML benchmark datasets over 24 hours (here for mean advantage in cross-entropy over 62 datasets); positive values indicate that `naiveautoml` is better than another approach.
Vertical lines are visual aids for 10 minutes and 1h respectively.
As can be seen, `naiveautoml` is substantially better in the short run (first hour) than both `autosklearn` and a `random search` and is not substantially outperformed in the long run.
It is slightly outperformed by random forests in the (very) short run but outperforms random forests in the long run.

![Results for the comparison with auto-sklearn on the AutoML benchmark datasets](https://github.com/fmohr/naiveautoml/blob/master/publications/2022MLJ/plots/advantages.jpg)

Note that `naiveautoml` does not ask you for *any* parametrization (not even a timeout); it still provides you with the possibility to customize a lot of its behavior. Given our exhaustive empirical results, you can be confident that you will get at most of the times results that are comparable to what you would get with other tools.

## Python
Install via `pip install naiveautoml.`
The current version is 0.0.8.

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

## Citing naive automl
The current option to cite naiveautoml is

```
@inproceedings{mohr2021replacing,
  title={Replacing the Ex-Def Baseline in AutoML by Naive AutoML},
  author={Mohr, Felix and Wever, Marcel},
  booktitle={8th ICML Workshop on Automated Machine Learning (AutoML)},
  year={2021}
}
```
Note that the implementation deviates in some aspects from the workshop paper.
A follow-up paper with the exact implementation used here is currently under review.
