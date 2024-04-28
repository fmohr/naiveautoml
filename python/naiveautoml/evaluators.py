import warnings

from lccv import lccv
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from .stoppers._lcmodel_stopper import LCModelStopper

from .commons import\
    get_scoring_name, build_scorer


class EarlyDiscardingValidator:

    def __init__(
            self,
            instance,
            stopper="lce",
            train_size=0.8,
            repetitions_per_anchor=5,
    ):
        self.instance = instance

        if isinstance(stopper, str):
            accepted_stoppers = ["lce", "pfn"]
            if stopper not in accepted_stoppers:
                raise ValueError(f"'stopper' must be in {accepted_stoppers} but is {stopper}")
            self.stopper = LCModelStopper(
                max_steps=10**3  # not sure why this is being used
            )
        else:
            self.stopper = stopper
        self.repetitions_per_anchor = repetitions_per_anchor

        self.max_anchor = int(train_size * instance.X.shape[0])
        self.scorings = [instance.scoring] + instance.side_scores

    def get_subject(self, pl):
        return str(pl)

    def __call__(self, pl, X, y, scorings, errors="message"):
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')

        # configure schedule
        if self.stopper.best_objective == -np.inf:
            schedule = [self.max_anchor]
        else:
            base = np.sqrt(2)
            min_exp = 8
            max_exp = int(np.log(self.max_anchor) / np.log(base))
            schedule = [int(base**i) for i in range(min_exp, max_exp + 1)]
            if self.max_anchor not in schedule:
                schedule.append(self.max_anchor)

        # prepare scoring functions
        subject = self.get_subject(pl)
        scores = None
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]
            base_scoring = scorings[0]
            scorers = [build_scorer(scoring) for scoring in scorings]

            for budget in schedule:

                if self.stopper.stop(subject=subject):
                    print("Stopper says we should stop!")
                    break

                scores = {
                    s_name: [] for s_name in scorings
                }

                sss = StratifiedShuffleSplit(n_splits=self.repetitions_per_anchor, train_size=budget)
                for train_index, val_index in sss.split(X, y):
                    pl_copy = sklearn.base.clone(pl)
                    pl_copy.fit(X[train_index], y[train_index])

                    # compute scores
                    for s_name, s_fun in zip(scorings, scorers):
                        scores[s_name].append(
                            s_fun(pl_copy, X[val_index], y[val_index])
                        )

                self.stopper.observe(
                    subject=subject,
                    budget=budget,
                    objective=np.mean(scores[base_scoring])
                )

        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                    self.instance.logger.exception(e)
                return ({s: np.nan for s in scorings}, {s: {} for s in scorings})
            else:
                raise

        if scores is None or budget != self.max_anchor:
            print(f"Returning nan at budget {budget}")
            return {s: np.nan for s in self.scorings}, {s: {} for s in scorings}
        print(str(pl))
        return {s: e[-1] for s, e in scores.items()}, {s: {} for s in scorings}

    def update(self, pl, score):
        self.stopper.observe(
            subject=self.get_subject(pl),
            budget=self.max_anchor,
            objective=score[self.instance.scoring]
        )


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
    
    def update(self, pl, scores):
        self.r = max([scores[s] for s in scores])


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
