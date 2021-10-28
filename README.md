# naiveautoml

## Why to use?
`naiveautoml` has significant performance advantages over other tools like auto-sklearn in the short run and is hardly outperformed in the long run.
The following figures show average advantages of `naiveautoml` with vanilla auto-sklearn on the AutoML benchmark datasets over 24 hours (left with AUROC for binary datasets and right with cross-entropy for multi-class datasets).
Vertical lines are visual aids for 10 minutes and 1h respectively.
As can be seen, `naiveautoml` is substantially better in the short run (first hour) and is not substantially outperformed in the long run.
![Results for the comparison with auto-sklearn on the AutoML benchmark datasets](https://github.com/fmohr/naiveautoml/blob/master/publications/2021MLJ/plots/advantage-sklearn.jpg)

## Python
Install via `pip install naiveautoml.`
The current version is 0.0.4.

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
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
