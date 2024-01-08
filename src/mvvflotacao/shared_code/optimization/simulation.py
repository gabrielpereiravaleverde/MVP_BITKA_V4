import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from .optimizer import *
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

class Scenarios:
    def __init__(self, num_scenarios):
        self.num_scenarios = num_scenarios

    def choose_scenarios(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.sample(self.num_scenarios)
    
class ClusterScenarios(Scenarios):
    def __init__(self, num_scenarios, features):
        super().__init__(num_scenarios)
        self.features = features

    def choose_scenarios(self, data: pd.DataFrame) -> pd.DataFrame:
        cluster_data = data[self.features]
        cluster_data = MinMaxScaler().fit_transform(cluster_data)
        kmeans = KMeans(n_clusters = self.num_scenarios).fit(cluster_data)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, cluster_data)
        closest_samples = data.iloc[closest]
        return closest_samples

class Simulation:
    def __init__(self, scenario: Scenarios, decision_variables, target, baseline_searcher_builder, searcher_builder, evaluator_builder):
        self.baseline_searcher_builder = baseline_searcher_builder
        self.searcher_builder = searcher_builder
        self.scenario = scenario
        self.evaluator_builder = evaluator_builder
        self.decision_variables =  decision_variables
        self.target = target

    def simulate(self, data: pd.DataFrame):
        results = {'scenario' : [], 'searcher_state' : [], 'baseline_state' : [], 'searcher_name' : [], 'baseline_name' : []}
        instances = self.scenario.choose_scenarios(data)

        for idx in tqdm(instances.index):
            fixed_data = instances[instances.index == idx]
            evaluator = self.evaluator_builder(fixed_data.drop(columns = self.decision_variables + [self.target]))

            baseline_searcher = self.baseline_searcher_builder(evaluator)
            s_baseline = baseline_searcher.optimize()

            searcher = self.searcher_builder(evaluator)
            s = searcher.optimize()

            results['scenario'].append(idx)

            for k in self.decision_variables + [self.target]:
                if k not in results.keys():
                    results[k] = []
                results[k].append(fixed_data[k].values[0])

            for k, v in dict(zip([f"{searcher.get_name()}_" + x for x in s.columns], s.values.ravel())).items():
                if k not in results.keys():
                    results[k] = []
                results[k].append(v)

            for k, v in dict(zip([f"{baseline_searcher.get_name()}_" + x for x in s_baseline.columns], s_baseline.values.ravel())).items():
                if k not in results.keys():
                    results[k] = []
                results[k].append(v)

            results['baseline_state'].append(baseline_searcher.get_history())
            results['searcher_state'].append(searcher.get_history())

            results['searcher_name'].append(searcher.get_name())
            results['baseline_name'].append(baseline_searcher.get_name())

        return results