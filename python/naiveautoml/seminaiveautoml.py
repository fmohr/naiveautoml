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

import time

import ConfigSpace
from ConfigSpace.util import *
from ConfigSpace.read_and_write import json as config_json
import json

import itertools as it

import os, psutil
from func_timeout import func_timeout, FunctionTimedOut

import scipy.sparse


from commons import *

class SemiNaiveAutoML:

    def __init__(self, search_space, opt_ordering, scoring, num_cpus = 8, execution_timeout = 10, timeout = None):
        
        ''' search_space is a string or a list of dictionaries
            - if it is a dict, the last one for the learner and all the others for pre-processing. Each dictionary has an entry "name" and an entry "components", which is a list of components with their parameters.
            - if it is a string, a json file with the name of search_space is read in with the same semantics
         '''
        if type(search_space) == str:
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
        self.opt_ordering = opt_ordering
        self.scoring = scoring
        self.num_cpus = num_cpus
        self.execution_timeout = execution_timeout
        self.timeout = timeout
        
        self.chosen_model = None
        self.chosen_attributes = None
        self.stage_entrypoints = {}
        
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
            print(pl)
            if is_pipeline_forbidden(pl):
                print("SKIP FORBIDDEN")
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
                
    def build_pipeline(self, hpo_processes, X, y, verbose=False):
        steps = self.get_instances_of_currently_selected_components_per_step(hpo_processes, X, y)
        pl = Pipeline(steps)
        if verbose:
            print("Original final pipeline is:", pl)
        while is_pipeline_forbidden(pl):
            if verbose:
                print("Invalid pipeline, removing first element!")
            pl = Pipeline(steps=pl.steps[1:])
        return pl

    def fit(self, X, y):
        
        # initialize
        self.pl = None
        history = []
        self.start_time = time.time()
        self.deadline = self.start_time + self.timeout
        self.sparse_training_data = type(X) == scipy.sparse.csr.csr_matrix or type(X) == scipy.sparse.lil.lil_matrix
        
        # print overview
        for step in self.search_space:
            print(step["name"], "(with order in optimization:", self.opt_ordering.index(step["name"]), ")")
            for comp in step["components"]:
                print("\t", comp["class"])
        
        # run over all the elements of the pipeline
        print("--------------------------------------------------")
        print("Choosing Algorithm for each slot")
        print("--------------------------------------------------")
        decisions = []
        components_with_score = {}
        best_score_overall = -np.inf
        for step_index, step_name in enumerate(self.opt_ordering):
            
            # create list of components to try for this slot
            step = [step for step in self.search_space if step["name"] == step_name][0]
            print("Selecting component for step with name:", step_name)
            if not step_name in ["classifier"]:
                components = [None] + step["components"]
            else:
                components = step["components"]
            
            # find best default parametrization for this slot (depending on choice of previously configured slots)
            pool = EvaluationPool(X, y, self.scoring)
            best_score = -np.inf
            decision = None
            for comp in components:
                remaining_time = self.deadline - 10 - time.time()
                if remaining_time < 0:
                    print("\tTimeout approaching. Not evaluating anymore for this stage.")
                    break
                    
                # build pipeline to be evaluated here
                steps_tmp = [(s[0], build_estimator(s[1], None, X, y)) for s in decisions]
                if comp is not None:
                    steps_tmp.append((step_name, build_estimator(comp, None, X, y)))
                steps_ordered = []
                for step_inner in self.search_space:
                    if is_component_defined_in_steps(steps_tmp, step_inner["name"]):
                        steps_ordered.append(get_step_with_name(steps_tmp, step_inner["name"]))
                pl = Pipeline(steps = steps_ordered)
                try:
                    score = pool.evaluate(pl, self.execution_timeout)
                except FunctionTimedOut:
                    print("TIMEOUT!")
                    score = np.nan
                print("\tObserved score of", score, " for default configuration of",None if comp is None else comp["class"])
                if not np.isnan(score) and score > best_score:
                    print("\tThis is a NEW BEST SCORE!")
                    best_score = score
                    components_with_score[step_name] = score
                    decision = comp
                    if score > best_score_overall:
                        print("\tUpdating new best internal pipeline to", pl)
                        self.pl = pl
                        history.append({"time": time.time() - self.start_time, "pl": pl})
            if decision is None:
                print("No component chosen for this slot. Leaving it blank")
            else:
                print("Added", decision["class"], "as the decision for step", step_name)
                decisions.append((step_name, decision))
        
        # ordering decisions by their order in the pipeline
        decisions_tmp = [d for d in decisions]
        decisions = []
        print(components_with_score)
        for step in self.search_space:
            if is_component_defined_in_steps(decisions_tmp, step["name"]):
                decisions.append(get_step_with_name(decisions_tmp, step["name"]))
        print("Algorithm Selection ready. Decisions:", "".join(["\n\t" + str((d[0], d[1]["class"])) + " with performance " + str(components_with_score[d[0]]) for d in decisions]))
            
        # now conduct HPO until there is no local improvement or the deadline is hit
        print("--------------------------------------------------")
        print("Entering HPO phase")
        print("--------------------------------------------------")
        
        # create HPO processes for each slot, taking into account the default parametrized component of each other slot
        hpo_processes = {}
        step_names = [d[0] for d in decisions]
        for step_name, comp in decisions:
            index = step_names.index(step_name)
            other_instances = [(d[0], build_estimator(d[1], None, X, y)) if d[0] != step_name else None for d in decisions]
            hpo = HPOProcess(step_name, comp, X, y, self.scoring, self.execution_timeout, other_instances, index, 1800, 1000)
            hpo.best_score = components_with_score[step_name] # performance of default config
            hpo_processes[step_name] = hpo
        
        # starting HPO process
        opt_round = 1
        rs = np.random.RandomState()
        active_for_optimization = [name for name, hpo in hpo_processes.items() if hpo.active]
        round_runtimes = []
        while active_for_optimization:
            remaining_time = self.deadline - (np.mean(round_runtimes) if round_runtimes else 0) - 10 - time.time()
            if remaining_time < 0:
                print("Timeout almost exhausted, stopping HPO phase")
                break
                
            print("Entering optimization round " + str(opt_round) + ". Remaining time is: " + str(remaining_time) + "s.")
            round_start = time.time()
            inactive = []
            for name in active_for_optimization:
                hpo = hpo_processes[name]
                print("Stepping HPO for", name)
                hpo.step(remaining_time)
                if not hpo.active:
                    print("deactivating " + name)
                    inactive.append(name)
            round_runtimes.append(time.time() - round_start)
            for name in inactive:
                active_for_optimization.remove(name)
            opt_round += 1
            newPl = self.build_pipeline(hpo_processes, X, y)
            if str(newPl) != str(self.pl):
                print("Updating new best internal pipeline to", newPl)
                self.pl = newPl
                history.append({"time": time.time() - self.start_time, "pl": newPl})
                
        # train final pipeline
        print("--------------------------------------------------")
        print("Search Completed. Building final pipeline.")
        print("--------------------------------------------------")
        self.pl = self.build_pipeline(hpo_processes, X, y, verbose = True)
        print(self.pl)
        print("Now fitting the pipeline with all given data.")
        while True:
            try:
                self.pl.fit(X, y)
                break
            except:
                print("There was a problem in building the pipeline, cutting it one down!")
                self.pl = Pipeline(steps=self.pl.steps[1:])
                print("new pipeline is:", self.pl)
            
        self.end_time = time.time()
        self.chosen_model = self.pl
        print("Runtime was", self.end_time - self.start_time, "seconds")
        self.history = history
        
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