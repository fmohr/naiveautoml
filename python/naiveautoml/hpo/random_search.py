from .._interfaces import HPOptimizer
import time
import pandas as pd
import numpy as np


class RandomHPO(HPOptimizer):

    def __init__(self, show_progress=False, logger=None):
        super().__init__(show_progress=show_progress, logger=logger)
        self.its = None
        self.configs_since_last_imp = None
        self.time_since_last_imp = None

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.its = 0
        self.configs_since_last_imp = 0
        self.time_since_last_imp = 0

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

        time_start_eval = time.time()
        if self.is_pipeline_forbidden(candidate_pipeline):
            status = "avoided"
            scores = {s["name"]: np.nan for s in [self.task.scoring] + self.task.passive_scorings}
            evaluation_report = None
            exception = None
        else:

            # determine whether a timeout is to be applied
            timeout = self.task.timeout_candidate if self.is_timeout_required(candidate_pipeline) else None

            # evaluate configured pipeline
            status, scores, evaluation_report, exception = self.evaluator.evaluate(
                pl=candidate_pipeline,
                timeout=timeout
            )
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

        if self.show_progress:
            self.pbar.update(1)
            if self.its == self.task.max_hpo_iterations:
                self.pbar.close()

        return pd.DataFrame({k: [v] for k, v in candidate_history_entry.items()})

    @property
    def description(self):
        return "Random HPO"
