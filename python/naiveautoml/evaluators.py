import logging
import warnings

from lccv import lccv
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from .stoppers._lcmodel_stopper import LCModelStopper
from .stoppers._lcpfn_stopper import LCPFNStopper


class LccvValidator:

    def __init__(self, task_type, logger_name="naml.evaluator", train_size=0.8, repetitions_per_anchor=5, random_state=None):
        self.r = -np.inf
        self.task_type = task_type
        self.logger = logging.getLogger(logger_name)
        self.train_size = train_size
        self.repetitions_per_anchor = repetitions_per_anchor
        self.random_state = random_state
        print(random_state)

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
                    base_scoring=scorings[0]["name"],
                    additional_scorings=[s["name"] for s in scorings[1:]],
                    target_anchor=self.train_size,
                    max_evaluations=self.repetitions_per_anchor,
                    seed=self.random_state
                )
                if not np.isnan(score) and score > self.r:
                    self.r = score

                results_at_highest_anchor = elcm.df[elcm.df["anchor"] == np.max(elcm.df["anchor"])].mean(
                    numeric_only=True)
                results = {
                    s["name"]: np.round(np.mean(results_at_highest_anchor[f"score_test_{s['name']}"]), 4)
                    if not np.isnan(score) else np.nan
                    for s in scorings
                }
                evaluation_report = {
                    s["name"]: elc if not np.isnan(score) else np.nan for s in scorings
                }

                # return the object itself, so that it can be overwritten in the pool (necessary because of pynisher)
                return results, evaluation_report
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if error_treatment != "raise":
                    msg = f"Observed an error: {e}"
                    if error_treatment == "info":
                        self.logger.info(msg)
                    elif error_treatment == "warning":
                        self.logger.warn(msg)
                    elif error_treatment == "error":
                        self.logger.warn(msg)
                else:
                    raise
            return {s["name"]: np.nan for s in scorings}, {s["name"]: {} for s in scorings}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if error_treatment != "raise":
                msg = f"Observed an error: {e}"
                if error_treatment == "info":
                    self.logger.info(msg)
                elif error_treatment == "warning":
                    self.logger.warn(msg)
                elif error_treatment == "error":
                    self.logger.warn(msg)
                return None, None
            else:
                raise

    def update(self, pl, scores):
        self.r = max([scores[s] for s in scores])


class SplitBasedEvaluator:

    def __init__(self, task_type, splitter, logger_name, random_state=None):
        self.task_type = task_type
        self.splitter = splitter
        self.logger = logging.getLogger(logger_name)
        self.random_state = random_state

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
                    self.logger.info(msg)
                elif error_treatment == "warning":
                    self.logger.warn(msg)
                elif error_treatment == "error":
                    self.logger.warn(msg)
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
                        self.logger.info(msg)
                    elif error_treatment == "warning":
                        self.logger.warn(msg)
                    elif error_treatment == "error":
                        self.logger.warn(msg)
                    out[scoring["name"]] = np.nan
                else:
                    raise
        return out


class KFold(SplitBasedEvaluator):

    def __init__(self, task_type, n_splits, logger_name="naml.evaluator", random_state=None):

        # define splitter
        if task_type in ["classification"]:
            splitter = sklearn.model_selection.StratifiedKFold(
                n_splits=n_splits,
                random_state=self.random_state,
                shuffle=True
            )
        elif task_type in ["regression", "multilabel-indicator"]:
            splitter = sklearn.model_selection.KFold(n_splits=n_splits, random_state=self.random_state, shuffle=True)
        else:
            raise ValueError(f"Unsupported task type {task_type}")

        super().__init__(task_type=task_type, splitter=splitter, logger_name=logger_name, random_state=random_state)


class Mccv(SplitBasedEvaluator):

    def __init__(self, task_type, n_splits, logger_name="naml.evaluator", random_state=None):

        if task_type in ["classification"]:
            splitter = sklearn.model_selection.StratifiedShuffleSplit(
                n_splits=n_splits,
                train_size=0.8,
                random_state=random_state
            )
        elif task_type in ["regression", "multilabel-indicator"]:
            splitter = sklearn.model_selection.ShuffleSplit(
                n_splits=n_splits,
                train_size=0.8,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported task type {task_type}")
        super().__init__(task_type=task_type, splitter=splitter, logger_name=logger_name, random_state=random_state)


class EarlyDiscardingValidator:

    def __init__(
            self,
            task_type,
            logger_name="naml.evaluator",
            stopper="lce",
            train_size=0.8,
            repetitions_per_anchor=5,
            random_state=None
    ):
        self.task_type = task_type
        self.logger = logging.getLogger(logger_name)
        self.train_size = train_size
        self.random_state = random_state

        if isinstance(stopper, str):
            accepted_stoppers = ["lce", "pfn"]
            if stopper not in accepted_stoppers:
                raise ValueError(f"'stopper' must be in {accepted_stoppers} but is {stopper}")
            elif stopper == "lce":
                self.stopper = LCModelStopper(
                    max_steps=10**3,
                    random_state=self.random_state
                )
            elif stopper == "pfn":
                self.stopper = LCPFNStopper(
                    max_steps=10**3,
                    random_state=self.random_state
                )
        else:
            self.stopper = stopper
        self.repetitions_per_anchor = repetitions_per_anchor
        self.max_anchor = None

    def get_subject(self, pl):
        return str(pl)

    def __call__(self, pl, X, y, scorings, error_treatment="raise"):
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')

        max_anchor = int(self.train_size * X.shape[0])

        # configure schedule
        if self.stopper.best_objective == -np.inf:
            schedule = [max_anchor]
        else:
            base = np.sqrt(2)
            min_exp = 8
            max_exp = int(np.log(max_anchor) / np.log(base))
            schedule = [int(base**i) for i in range(min_exp, max_exp + 1) if int(base**i) >= np.unique(y).shape[0]]
            if max_anchor not in schedule:
                schedule.append(max_anchor)

        # prepare scoring functions
        subject = self.get_subject(pl)
        scores = None
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]
            base_scoring = scorings[0]
            
            evaluation_report = {
                s_name["name"]: [] for s_name in scorings
            }

            for budget in schedule:

                if self.stopper.stop(subject=subject):
                    print("Stopper says we should stop!")
                    self.max_anchor = budget
                    break

                scores = {
                    s_name["name"]: [] for s_name in scorings
                }

                sss = StratifiedShuffleSplit(
                    n_splits=self.repetitions_per_anchor,
                    train_size=budget,
                    random_state=self.random_state
                    )

                for train_index, val_index in sss.split(X, y):
                    pl_copy = sklearn.base.clone(pl)
                    pl_copy.fit(X[train_index], y[train_index])

                    # compute scores
                    for scoring in scorings:
                        score = scoring["fun"](pl_copy, X[val_index], y[val_index])
                        scores[scoring["name"]].append(score)
                
                evaluation_report[scoring["name"]].append({budget: np.mean(scores[base_scoring["name"]])})

                self.stopper.observe(
                    subject=subject,
                    budget=budget,
                    objective=np.mean(scores[base_scoring["name"]])
                )

        except KeyboardInterrupt:
            raise
        except Exception as e:
            if error_treatment != "raise":
                msg = f"Observed an error: {e}"
                if error_treatment == "info":
                    self.logger.info(msg)
                elif error_treatment == "warning":
                    self.logger.warn(msg)
                elif error_treatment == "error":
                    self.logger.warn(msg)
            else:
                raise
            return {s["name"]: np.nan for s in scorings}, {s["name"]: {} for s in scorings}

        if scores is None or budget != max_anchor:
            print(f"Returning nan at budget {budget}")
            return {s["name"]: np.nan for s in scorings}, {s["name"]: {} for s in scorings}
        print(str(pl))
        return {s: e[-1] for s, e in scores.items()}, evaluation_report

    def update(self, pl, score):
        self.stopper.observe(
            subject=self.get_subject(pl),
            budget=self.max_anchor,
            objective=list(score.values())[0]
        )