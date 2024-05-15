import logging
import warnings

from lccv import lccv
import numpy as np
import pandas as pd
import sklearn


class LccvValidator:

    def __init__(self, instance, train_size=0.8):
        self.r = -np.inf
        self.instance = instance
        self.train_size = train_size

    def __call__(self, pl, X, y, scorings, error_treatment="raise"):
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
            except Exception as e:
                if error_treatment != "raise":
                    msg = f"Observed an error: {e}"
                    if error_treatment == "info":
                        self.instance.logger.info(msg)
                    elif error_treatment == "warning":
                        self.instance.logger.warn(msg)
                    elif error_treatment == "error":
                        self.instance.logger.warn(msg)
                else:
                    raise
            return {s: np.nan for s in scorings}, {s: {} for s in scorings}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if error_treatment != "raise":
                msg = f"Observed an error: {e}"
                if error_treatment == "info":
                    self.instance.logger.info(msg)
                elif error_treatment == "warning":
                    self.instance.logger.warn(msg)
                elif error_treatment == "error":
                    self.instance.logger.warn(msg)
                return None, None
            else:
                raise


class SplitBasedEvaluator:

    def __init__(self, task_type, splitter, logger_name):
        self.task_type = task_type
        self.splitter = splitter
        self.logger = logging.getLogger(logger_name)

    def __call__(self, pl, X, y, scorings, error_treatment="raise"):
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            # compute scores
            scores = self.evaluate_splits(
                pl=pl,
                X=X,
                y=y,
                scorings=scorings,
                error_treatment=error_treatment
            )
            return {k: np.round(np.mean(v), 4) for k, v in scores.items()}, scores
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if error_treatment != "raise":
                msg = f"Observed an error: {e}"
                if error_treatment == "info":
                    self.instance.logger.info(msg)
                elif error_treatment == "warning":
                    self.instance.logger.warn(msg)
                elif error_treatment == "error":
                    self.instance.logger.warn(msg)
                return None, None
            else:
                raise

    def evaluate_splits(self, pl, X, y, scorings, error_treatment):
        scores = {scoring["name"]: [] for scoring in scorings}
        for train_index, test_index in self.splitter.split(X, y):
            split_results = self.evaluate_split(
                pl=pl,
                X=X,
                y=y,
                train_index=train_index,
                test_index=test_index,
                scorings=scorings,
                error_treatment=error_treatment
            )
            for key, val in split_results.items():
                scores[key].append(val)
        return scores

    def evaluate_split(self, pl, X, y, train_index, test_index, scorings, error_treatment):

        X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
        y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
        X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
        y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

        pl_copy = sklearn.base.clone(pl)
        pl_copy.fit(X_train, y_train)

        if self.task_type == "multilabel-indicator":
            y_hat = pl_copy.predict(X_test)

        out = {}
        for scoring in scorings:
            scorer = scoring["fun"]
            try:

                if self.task_type == "multilabel-indicator":
                    s_name = scoring["name"]
                    if scoring["fun"] is not None:
                        self.logger.warning(
                            f"There is an explicitly specified scoring function for {s_name}. "
                            f"In multi-label classification, scoring functions are ignored."
                        )
                    if s_name == "f1_macro":
                        out[s_name] = sklearn.metrics.f1_score(y_test, y_hat, average="macro")
                    elif s_name == "accuracy":
                        out[s_name] = sklearn.metrics.accuracy_score(y_test, y_hat)
                    elif s_name == "neg_hamming_loss":
                        out[s_name] = sklearn.metrics.hamming_loss(y_test, y_hat) * -1
                    else:
                        raise ValueError(f"Unsupported multi-label metric {scoring['name']}.")

                else:
                    out[scoring["name"]] = scorer(pl_copy, X_test, y_test)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if error_treatment != "raise":
                    msg = f"Observed an error: {e}"
                    if error_treatment == "info":
                        self.instance.logger.info(msg)
                    elif error_treatment == "warning":
                        self.instance.logger.warn(msg)
                    elif error_treatment == "error":
                        self.instance.logger.warn(msg)
                    out[scoring["name"]] = np.nan
                else:
                    raise
        return out


class KFold(SplitBasedEvaluator):

    def __init__(self, task_type, n_splits, logger_name="naml.evaluator"):

        # define splitter
        if task_type in ["classification"]:
            splitter = sklearn.model_selection.StratifiedKFold(
                n_splits=n_splits,
                random_state=None,
                shuffle=True
            )
        elif task_type in ["regression", "multilabel-indicator"]:
            splitter = sklearn.model_selection.KFold(n_splits=n_splits, random_state=None, shuffle=True)
        else:
            raise ValueError(f"Unsupported task type {task_type}")

        super().__init__(task_type=task_type, splitter=splitter, logger_name=logger_name)


class Mccv(SplitBasedEvaluator):

    def __init__(self, task_type, n_splits, logger_name="naml.evaluator"):

        if task_type in ["classification"]:
            splitter = sklearn.model_selection.StratifiedShuffleSplit(
                n_splits=n_splits,
                train_size=0.8,
                random_state=None
            )
        elif task_type in ["regression", "multilabel-indicator"]:
            splitter = sklearn.model_selection.ShuffleSplit(
                n_splits=n_splits,
                train_size=0.8,
                random_state=None
            )
        else:
            raise ValueError(f"Unsupported task type {task_type}")
        super().__init__(task_type=task_type, splitter=splitter, logger_name=logger_name)
