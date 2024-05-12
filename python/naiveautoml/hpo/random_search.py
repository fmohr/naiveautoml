from .._interfaces import HPOptimizer
import time
from tqdm import tqdm
import numpy as np
import pandas as pd


class RandomHPO(HPOptimizer):

    def __init__(self, logger):
        super().__init__(logger)
        self.its = 0

    def step(self, remaining_time=None):
        self.its += 1
        if not self.active:
            raise Exception("HPO not active anymore.")

        self.logger.info(f"Starting {self.its}-th HPO step. Currently best known score is {self.best_score}")

        # draw random parameters
        candidate_config = self.config_space.sample_configuration(1)
        candidate_history_entry = self.create_history_descriptor(candidate_config)
        candidate_history_entry["time"] = time.time()
        candidate_pipeline = candidate_history_entry["pipeline"]

        # evaluate configured pipeline
        time_start_eval = time.time()
        status, scores, evaluation_report, exception = self.evaluator.evaluate(candidate_pipeline)
        if not isinstance(scores, dict):
            raise TypeError(f"""
The scores must be a dictionary as a function of the scoring functions. Observed type is {type(scores)}: {scores}
""")
        runtime = time.time() - time_start_eval

        score = scores[self.task.scoring["name"]]
        self.logger.info(f"Observed score of {score} for params {candidate_config}")

        candidate_history_entry.update({
            "runtime": runtime,
            "status": status,
            "evaluation_report": evaluation_report,
            "exception": exception
        })
        candidate_history_entry.update(scores)

        # check whether this is a new best solution
        if score > self.best_score:
            self.logger.info("This is a NEW BEST SCORE!")
            self.best_score = score
            self.time_since_last_imp = 0
            self.configs_since_last_imp = 0
            self.best_config = candidate_history_entry
        else:
            self.configs_since_last_imp += 1
            self.time_since_last_imp += runtime

            if (
                    self.time_since_last_imp > self.task.max_hpo_time_without_imp or
                    self.configs_since_last_imp > self.task.max_hpo_iterations_without_imp
            ):
                self.logger.info(
                    f"No improvement within {self.task.max_hpo_time_without_imp}s"
                    f" or within {self.task.max_hpo_iterations_without_imp} steps."
                    "Stopping HPO here."
                )
                self.active = False

        return pd.DataFrame({k: [v] for k, v in candidate_history_entry.items()})

    @property
    def description(self):
        return "Random HPO"
