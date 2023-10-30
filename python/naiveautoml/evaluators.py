import warnings

from lccv import lccv
import numpy as np
import pandas as pd
import sklearn

from .commons import\
    get_scoring_name, build_scorer

class Lccv_validator:

    def __init__(self, instance):
        self.r = -np.inf
        self.instance = instance

    def lccv_validate(self, pl, X, y, scorings, errors="raise"):  # just a wrapper to ease parallelism
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            try:
                score, _, _, elcm = lccv(
                    pl,
                    X,
                    y,
                    r=self.r,
                    base_scoring=scorings[0],
                    additional_scorings=scorings[1:]
                )
                if not np.isnan(score) and score > self.r:
                    self.r = score

                results_at_highest_anchor = elcm.df[elcm.df["anchor"] == np.max(elcm.df["anchor"])].mean(
                    numeric_only=True)
                results = {
                    s: results_at_highest_anchor[f"score_test_{s}"] if not np.isnan(score) else np.nan for s in scorings
                }
                return results
            except KeyboardInterrupt:
                raise
            except Exception:
                if errors == "message":
                    self.instance.logger.info(f"Observed exception in validation of pipeline {pl}.")
                else:
                    raise
            return results
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None
            else:
                raise

class Kfold_5:
    def __init__(self, instance):
        self.instance = instance

    def kfold_5_validate(self, pl, X, y, scorings, errors="raise"):  # just a wrapper to ease parallelism
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            if self.instance.task_type == "classification":
                splitter = sklearn.model_selection.StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
            elif self.instance.task_type:
                splitter = sklearn.model_selection.KFold(n_splits=5, random_state=None, shuffle=True)
            scores = {get_scoring_name(scoring): [] for scoring in scorings}
            for train_index, test_index in splitter.split(X, y):

                X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
                y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
                y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

                pl_copy = sklearn.base.clone(pl)
                pl_copy.fit(X_train, y_train)

                # compute values for each metric
                for scoring in scorings:
                    scorer = build_scorer(scoring)
                    try:
                        score = scorer(pl_copy, X_test, y_test)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        score = np.nan
                        if errors == "message":
                            self.instance.logger.info(f"Observed exception in validation of pipeline {pl_copy}. Placing nan.")
                        else:
                            raise

                    scores[get_scoring_name(scoring)].append(score)
            return scores
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None
            else:
                raise

class Kfold_3:
    def __init__(self, instance):
        self.instance = instance

    def kfold_3_validate(self, pl, X, y, scorings, errors="raise"):  # just a wrapper to ease parallelism
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            if self.instance.task_type == "classification":
                splitter = sklearn.model_selection.StratifiedKFold(n_splits=3, random_state=None, shuffle=True)
            elif self.instance.task_type:
                splitter = sklearn.model_selection.KFold(n_splits=3, random_state=None, shuffle=True)
            scores = {get_scoring_name(scoring): [] for scoring in scorings}
            for train_index, test_index in splitter.split(X, y):

                X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
                y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
                y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

                pl_copy = sklearn.base.clone(pl)
                pl_copy.fit(X_train, y_train)

                # compute values for each metric
                for scoring in scorings:
                    scorer = build_scorer(scoring)
                    try:
                        score = scorer(pl_copy, X_test, y_test)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        score = np.nan
                        if errors == "message":
                            self.instance.logger.info(f"Observed exception in validation of pipeline {pl_copy}. Placing nan.")
                        else:
                            raise

                    scores[get_scoring_name(scoring)].append(score)
            return scores
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None
            else:
                raise

class Mccv_1:
    def __init__(self, instance):
        self.instance = instance

    def mccv_1_validate(self, pl, X, y, scorings, errors="raise"):  # just a wrapper to ease parallelism
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            if self.instance.task_type == "classification":
                splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, random_state=None)
            elif self.instance.task_type:
                splitter = sklearn.model_selection.ShuffleSplit(n_splits=1, random_state=None)
            scores = {get_scoring_name(scoring): [] for scoring in scorings}
            for train_index, test_index in splitter.split(X, y):

                X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
                y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
                y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

                pl_copy = sklearn.base.clone(pl)
                pl_copy.fit(X_train, y_train)

                # compute values for each metric
                for scoring in scorings:
                    scorer = build_scorer(scoring)
                    try:
                        score = scorer(pl_copy, X_test, y_test)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        score = np.nan
                        if errors == "message":
                            self.instance.logger.info(f"Observed exception in validation of pipeline {pl_copy}. Placing nan.")
                        else:
                            raise

                    scores[get_scoring_name(scoring)].append(score)
            return scores
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None
            else:
                raise

class Mccv_3:
    def __init__(self, instance):
        self.instance = instance

    def mccv_3_validate(self, pl, X, y, scorings, errors="raise"):  # just a wrapper to ease parallelism
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            if self.instance.task_type == "classification":
                splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=3, random_state=None)
            elif self.instance.task_type:
                splitter = sklearn.model_selection.ShuffleSplit(n_splits=3, random_state=None)
            scores = {get_scoring_name(scoring): [] for scoring in scorings}
            for train_index, test_index in splitter.split(X, y):

                X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
                y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
                y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

                pl_copy = sklearn.base.clone(pl)
                pl_copy.fit(X_train, y_train)

                # compute values for each metric
                for scoring in scorings:
                    scorer = build_scorer(scoring)
                    try:
                        score = scorer(pl_copy, X_test, y_test)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        score = np.nan
                        if errors == "message":
                            self.instance.logger.info(f"Observed exception in validation of pipeline {pl_copy}. Placing nan.")
                        else:
                            raise

                    scores[get_scoring_name(scoring)].append(score)
            return scores
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None
            else:
                raise

class Mccv_5:
    def __init__(self, instance):
        self.instance = instance

    def mccv_5_validate(self, pl, X, y, scorings, errors="raise"):  # just a wrapper to ease parallelism
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            if self.instance.task_type == "classification":
                splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=5, random_state=None)
            elif self.instance.task_type:
                splitter = sklearn.model_selection.ShuffleSplit(n_splits=5, random_state=None)
            scores = {get_scoring_name(scoring): [] for scoring in scorings}
            for train_index, test_index in splitter.split(X, y):

                X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
                y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
                y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

                pl_copy = sklearn.base.clone(pl)
                pl_copy.fit(X_train, y_train)

                # compute values for each metric
                for scoring in scorings:
                    scorer = build_scorer(scoring)
                    try:
                        score = scorer(pl_copy, X_test, y_test)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        score = np.nan
                        if errors == "message":
                            self.instance.logger.info(f"Observed exception in validation of pipeline {pl_copy}. Placing nan.")
                        else:
                            raise

                    scores[get_scoring_name(scoring)].append(score)
            return scores
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None
            else:
                raise