import warnings

from lccv import lccv
import numpy as np
import pandas as pd
import sklearn

from .commons import\
    get_scoring_name, build_scorer


class LccvValidator:

    def __init__(self, instance, train_size=0.8):
        self.r = -np.inf
        self.instance = instance
        self.train_size = train_size

    def __call__(self, pl, X, y, scorings, errors="message"):
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            try:
                score, score_est, elc, elcm = lccv(
                    pl,
                    X,
                    y,
                    r=self.r,
                    base_scoring=scorings[0],
                    additional_scorings=scorings[1:],
                    target_anchor=self.train_size
                )
                if not np.isnan(score) and score > self.r:
                    self.r = score

                results_at_highest_anchor = elcm.df[elcm.df["anchor"] == np.max(elcm.df["anchor"])].mean(
                    numeric_only=True)
                results = {
                    s: np.round(np.mean(results_at_highest_anchor[f"score_test_{s}"]), 4)
                    if not np.isnan(score) else np.nan
                    for s in scorings
                }
                evaluation_history = {
                    s: elc if not np.isnan(score) else np.nan for s in scorings
                }

                # return the object itself, so that it can be overwritten in the pool (necessary because of pynisher)
                return results, evaluation_history
            except KeyboardInterrupt:
                raise
            except Exception:
                if errors == "message":
                    self.instance.logger.info(f"Observed exception in validation of pipeline {pl}.")
                else:
                    raise
            return {s: np.nan for s in scorings}, {s: {} for s in scorings}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None, None
            else:
                raise


class KFold:
    def __init__(self, instance, n_splits):
        self.instance = instance
        self.n_splits = n_splits

    def __call__(self, pl, X, y, scorings, errors="raise"):
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            if self.instance.task_type == "classification":
                splitter = sklearn.model_selection.StratifiedKFold(
                    n_splits=self.n_splits,
                    random_state=None,
                    shuffle=True
                )
            elif self.instance.task_type:
                splitter = sklearn.model_selection.KFold(n_splits=self.n_splits, random_state=None, shuffle=True)
            scores = {get_scoring_name(scoring): [] for scoring in scorings}
            for train_index, test_index in splitter.split(X, y):

                X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
                y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
                y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

                pl_copy = sklearn.base.clone(pl)
                pl_copy.fit(X_train, y_train)

                for scoring in scorings:
                    scorer = build_scorer(scoring)
                    try:
                        score = scorer(pl_copy, X_test, y_test)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        score = np.nan
                        if errors == "message":
                            self.instance.logger.info(
                                f"Observed exception in validation of pipeline {pl_copy}. Placing nan."
                            )
                        else:
                            raise

                    scores[get_scoring_name(scoring)].append(score)
            return {k: np.round(np.mean(v), 4) for k, v in scores.items()}, scores
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None, None
            else:
                raise


class Mccv:
    def __init__(self, instance, n_splits):
        self.instance = instance
        self.n_splits = n_splits

    def __call__(self, pl, X, y, scorings, errors="raise"):
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            if self.instance.task_type == "classification":
                splitter = sklearn.model_selection.StratifiedShuffleSplit(
                    n_splits=self.n_splits,
                    train_size=0.8,
                    random_state=None
                )
            elif self.instance.task_type:
                splitter = sklearn.model_selection.ShuffleSplit(
                    n_splits=self.n_splits,
                    train_size=0.8,
                    random_state=None
                )
            scores = {get_scoring_name(scoring): [] for scoring in scorings}
            for train_index, test_index in splitter.split(X, y):

                X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
                y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
                y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

                pl_copy = sklearn.base.clone(pl)
                pl_copy.fit(X_train, y_train)

                for scoring in scorings:
                    scorer = build_scorer(scoring)
                    try:
                        score = scorer(pl_copy, X_test, y_test)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        score = np.nan
                        if errors == "message":
                            self.instance.logger.info(
                                f"Observed exception in validation of pipeline {pl_copy}. Placing nan."
                            )
                        else:
                            raise

                    scores[get_scoring_name(scoring)].append(score)
            return {k: np.round(np.mean(v), 4) for k, v in scores.items()}, scores
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None, None
            else:
                raise
