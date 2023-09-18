import numpy as np
import warnings
from lccv import lccv


class Wrapper:

    def __init__(self):
        self.r = -np.inf

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
                    self.logger.info(f"Observed exception in validation of pipeline {pl}.")
                else:
                    raise
            return results
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.logger.error(f"Observed an error: {e}")
                return None
            else:
                raise
