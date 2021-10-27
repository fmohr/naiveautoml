import numpy as np
import pandas as pd
import random

import sklearn as sk
import sklearn.ensemble
import sklearn.decomposition
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn import *

from tqdm import tqdm

import time
from datetime import datetime

import ConfigSpace
from ConfigSpace.util import *
from ConfigSpace.read_and_write import json as config_json
import json

import itertools as it

import os, psutil
from func_timeout import func_timeout, FunctionTimedOut

import scipy.sparse


from naiveautoml.commons import *
import importlib.resources as pkg_resources

import logging

class NaiveAutoML:

    def __init__(self, search_space = None, scoring = None, num_cpus = 8, execution_timeout = 10, max_hpo_iterations = 100, timeout = None, standard_classifier=sklearn.neighbors.KNeighborsClassifier, logger_name = None, show_progress = False, opt_ordering = None, strictly_naive=False):
        
        ''' search_space is a string or a list of dictionaries
            - if it is a dict, the last one for the learner and all the others for pre-processing. Each dictionary has an entry "name" and an entry "components", which is a list of components with their parameters.
            - if it is a string, a json file with the name of search_space is read in with the same semantics
         '''
        if search_space is None or type(search_space) == str:
            if search_space is None:
                json_str = pkg_resources.read_text('naiveautoml', 'searchspace.json')
                search_space = json.loads(json_str)
            elif type(search_space) == str:
                f = open(search_space)
                search_space = json.load(f)
            
            # randomly shuffle elements in the search space
            self.search_space = []
            for step in search_space:
                comps = step["components"]
                random.shuffle(comps)
                self.search_space.append(step)
            
        else:
            self.search_space = search_space
        self.scoring = scoring
        self.num_cpus = num_cpus
        self.execution_timeout = execution_timeout
        self.max_hpo_iterations = max_hpo_iterations
        self.strictly_naive = strictly_naive
        self.timeout = timeout
        self.show_progress = show_progress
        
        self.chosen_model = None
        self.chosen_attributes = None
        self.stage_entrypoints = {}
        self.standard_classifier = standard_classifier
        
        # define ordering of slots for optimization
        if opt_ordering is None:
            opt_ordering = ["classifier"]
            for step in self.search_space:
                if step["name"] != "classifier":
                    opt_ordering.append(step["name"])
        self.opt_ordering = opt_ordering
        
        ## init logger
        self.logger = logging.getLogger('naiveautoml' if logger_name is None else logger_name)
        
    def check_combinations(self, X, y):
        
        pool = EvaluationPool(X, y, self.scoring)
        algorithms_per_stage = []
        names = []
        for step in self.search_space:
            names.append(step["name"])
            cands = []
            if step["name"] != "classifier":
                cands.append(None)
            cands.extend([get_class(comp["class"]) for comp in step["components"]])
            algorithms_per_stage.append(cands)
            
        for combo in it.product(*algorithms_per_stage):
            pl = sklearn.pipeline.Pipeline(steps=[(names[i], clazz()) for i, clazz in enumerate(combo) if clazz is not None])
            if is_pipeline_forbidden(pl):
                self.logger.debug("SKIP FORBIDDEN")
            else:
                pool.evaluate(pl, timeout=self.execution_timeout)
                
    def get_instances_of_currently_selected_components_per_step(self, hpo_processes, X, y):
        steps = []
        for step in self.search_space:
            step_name = step["name"]
            if step_name in hpo_processes:
                hpo = hpo_processes[step_name]
                comp = hpo.comp
                params = hpo.get_best_config()
                steps.append((step_name, build_estimator(comp, params, X, y)))
        return steps
    
    def get_pipeline_for_decision_in_step(self, step_name, comp, X, y, decisions):
        
        if self.strictly_naive: ## strictly naive case
            
            # build pipeline to be evaluated here
            if step_name == "classifier":
                steps = [("classifier", build_estimator(comp, None, X, y))]
            elif comp is None:
                steps = [("classifier", self.standard_classifier())]
            else:
                steps = [(step_name, build_estimator(comp, None, X, y)), ("classifier", self.standard_classifier())]
            return Pipeline(steps = steps)
        
        else: ## semi-naive case (consider previous decisions)
            steps_tmp = [(s[0], build_estimator(s[1], None, X, y)) for s in decisions]
            if comp is not None:
                steps_tmp.append((step_name, build_estimator(comp, None, X, y)))
            steps_ordered = []
            for step_inner in self.search_space:
                if is_component_defined_in_steps(steps_tmp, step_inner["name"]):
                    steps_ordered.append(get_step_with_name(steps_tmp, step_inner["name"]))
            return Pipeline(steps = steps_ordered)
    
    
    def build_pipeline(self, hpo_processes, X, y):
        steps = self.get_instances_of_currently_selected_components_per_step(hpo_processes, X, y)
        pl = Pipeline(steps)
        self.logger.debug(f"Original final pipeline is: {pl}")
        while is_pipeline_forbidden(pl):
            self.logger.debug("Invalid pipeline, removing first element!")
            pl = Pipeline(steps=pl.steps[1:])
        return pl
    
    def choose_algorithms(self, X, y):
        
        # run over all the elements of the pipeline
        self.logger.info("--------------------------------------------------")
        self.logger.info("Choosing Algorithm for each slot")
        self.logger.info("--------------------------------------------------")
        decisions = []
        components_with_score = {}
        
                
        if self.show_progress:
            print("Progress for algorithm selection:")
            pbar = tqdm(total=sum([len(step["components"]) for step in self.search_space]))
                
        for step_index, step_name in enumerate(self.opt_ordering):
            
            # create list of components to try for this slot
            step = [step for step in self.search_space if step["name"] == step_name][0]
            self.logger.debug(f"Selecting component for step with name: {step_name}")
            if not step_name in ["classifier"]:
                components = [None] + step["components"]
            else:
                components = step["components"]
            
            # find best default parametrization for this slot (depending on choice of previously configured slots)
            pool = EvaluationPool(X, y, self.scoring)
            best_score = -np.inf
            decision = None
            for comp in components:
                if self.deadline is not None:
                    remaining_time = self.deadline - 10 - time.time()
                    if remaining_time is not None and remaining_time < 0:
                        self.logger.info("Timeout approaching. Not evaluating anymore for this stage.")
                        break
                    else:
                        self.logger.info(f"Evaluating {comp['class'] if comp is not None else None}. Timeout: {self.execution_timeout}. Remaining time: {remaining_time}")
                
                # get and evaluate pipeline for this step
                pl = self.get_pipeline_for_decision_in_step(step_name, comp, X, y, decisions)
                try:
                    score = pool.evaluate(pl, self.execution_timeout)
                except FunctionTimedOut:
                    self.logger.debug("TIMEOUT!")
                    score = np.nan
                self.logger.debug(f"Observed score of {score} for default configuration of {None if comp is None else comp['class']}")
                
                # update history
                self.history.append({"time": time.time() - self.start_time, "pl": pl, "score_internal": score, "new_best": score > self.best_score_overall})
                
                # update best score
                if not np.isnan(score) and score > best_score:
                    self.logger.debug("This is a NEW BEST SCORE!")
                    best_score = score
                    components_with_score[step_name] = score
                    decision = comp
                    if score > self.best_score_overall:
                        self.best_score_overall = score
                        self.logger.debug(f"Updating new best internal pipeline to {pl}")
                        self.pl = pl    
                    
                # update progress bar
                if self.show_progress:
                    pbar.update(1)
                    
            if decision is None:
                self.logger.debug("No component chosen for this slot. Leaving it blank")
            else:
                self.logger.debug(f"Added {decision['class']} as the decision for step {step_name}")
                decisions.append((step_name, decision))
        
        # ordering decisions by their order in the pipeline
        decisions_tmp = [d for d in decisions]
        decisions = []
        for step in self.search_space:
            if is_component_defined_in_steps(decisions_tmp, step["name"]):
                decisions.append(get_step_with_name(decisions_tmp, step["name"]))
        
        self.decisions = decisions
        self.components_with_score = components_with_score
        self.logger.info("Algorithm Selection ready. Decisions: " + "".join(["\n\t" + str((d[0], d[1]["class"])) + " with performance " + str(components_with_score[d[0]]) for d in decisions]))
        
        # close progress bar
        if self.show_progress:
            pbar.close()
    
    def tune_parameters(self, X, y):
        
        # now conduct HPO until there is no local improvement or the deadline is hit
        self.logger.info("--------------------------------------------------")
        self.logger.info("Entering HPO phase")
        self.logger.info("--------------------------------------------------")
        
        # read variables from state
        decisions = self.decisions
        components_with_score = self.components_with_score
        
        # create HPO processes for each slot, taking into account the default parametrized component of each other slot
        self.hpo_processes = hpo_processes = {}
        step_names = [d[0] for d in decisions]
        for step_name, comp in decisions:
            if step_name == "classifier":
                other_instances = [(step_name, None)]
            else:
                other_instances = [(step_name, None), ("classifier", self.standard_classifier())]
            index = 0 # it is (rather by coincidence) the first step we want to optimize
            hpo = HPOProcess(step_name, comp, X, y, self.scoring, self.execution_timeout, other_instances, index, 1800, 1000)
            hpo.best_score = components_with_score[step_name] # performance of default config
            hpo_processes[step_name] = hpo
        
        # starting HPO process
        opt_round = 1
        rs = np.random.RandomState()
        active_for_optimization = [name for name, hpo in hpo_processes.items() if hpo.active]
        round_runtimes = []
        if self.show_progress:
            print("Progress for parameter turning:")
            pbar = tqdm(total = self.max_hpo_iterations)
            
        while active_for_optimization and (self.max_hpo_iterations is None or opt_round <= self.max_hpo_iterations):
            self.logger.info("Entering optimization round " + str(opt_round))
            if self.deadline is not None:
                remaining_time = self.deadline - (np.mean(round_runtimes) if round_runtimes else 0) - 10 - time.time()
                if remaining_time < 0:
                    self.logger.info("Timeout almost exhausted, stopping HPO phase")
                    break
                self.logger.debug("Remaining time is: " + str(remaining_time) + "s.")
            else:
                remaining_time = None
            
            round_start = time.time()
            inactive = []
            for name in active_for_optimization:
                hpo = hpo_processes[name]
                self.logger.debug(f"Stepping HPO for {name}")
                pl, score, runtime = hpo.step(remaining_time)
                if score > self.best_score_overall:
                    self.best_score_overall = score
                self.history.append({"time": time.time() - self.start_time, "pl": pl, "score_internal": score, "new_best": score > self.best_score_overall})
                if not hpo.active:
                    self.logger.info(f"Deactivating {name}")
                    inactive.append(name)
            round_runtimes.append(time.time() - round_start)
            for name in inactive:
                active_for_optimization.remove(name)
            opt_round += 1
            newPl = self.build_pipeline(hpo_processes, X, y)
            if str(newPl) != str(self.pl):
                self.logger.info(f"Updating new best internal pipeline to {newPl}")
                self.pl = newPl
            
            # update progress bar
            if self.show_progress:
                pbar.update(1)
         
        # close progress bar for HPO
        if self.show_progress:
            pbar.close()
        

    def fit(self, X, y):
        
        # initialize
        self.pl = None
        self.best_score_overall = -np.inf
        self.history = []
        self.start_time = time.time()
        self.deadline = self.start_time + self.timeout if self.timeout is not None else None
        self.sparse_training_data = type(X) == scipy.sparse.csr.csr_matrix or type(X) == scipy.sparse.lil.lil_matrix
        if self.scoring is None:
            self.scoring = "auc_roc" if len(np.unique(y)) == 2 else "neg_log_loss"
        
        # print overview
        summary = ""
        for step in self.search_space:
            summary += "\n" + step["name"]
            for comp in step["components"]:
                summary += "\n\t" + comp['class']
        self.logger.info(f"These are the components used by NaiveAutoML in the upcoming process (by steps):{summary}")
        
        # choose algorithms
        self.choose_algorithms(X, y)
        
        # tune parameters
        self.tune_parameters(X, y)
                
        # train final pipeline
        self.logger.info("--------------------------------------------------")
        self.logger.info("Search Completed. Building final pipeline.")
        self.logger.info("--------------------------------------------------")
        self.pl = self.build_pipeline(self.hpo_processes, X, y)
        self.logger.info(self.pl)
        self.logger.info("Now fitting the pipeline with all given data.")
        while True:
            try:
                self.pl.fit(X, y)
                break
            except:
                self.logger.warning("There was a problem in building the pipeline, cutting it one down!")
                self.pl = Pipeline(steps=self.pl.steps[1:])
                self.logger.warning("new pipeline is:", self.pl)
            
        self.end_time = time.time()
        self.chosen_model = self.pl
        self.logger.info(f"Runtime was {self.end_time - self.start_time} seconds")
        
    def eval_history(self, X, y):
        pool = EvaluationPool(X, y, self.scoring)
        scores = []
        for entry in self.history:
            scores.append(pool.evaluate(entry["pl"]))
        return scores

    def predict(self, X):
        return self.pl.predict(X)
    
    def predict_proba(self, X):
        return self.pl.predict_proba(X)