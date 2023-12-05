import pandas as pd 
import numpy as np
from pathlib import Path
import os
from optimizer import *
from simulation import *
import yaml
from kedro.io import DataCatalog
import pickle

def testing_gridsearch(cols, data):
    fixed_data = data.drop(columns = cols + ['target']).sample(n = 1)
    evaluator = ModelEvaluate(model = model, fixed_data = fixed_data)

    decision_variables = {}
    for f in cols: 
        decision_variables[f] = {'method' : 'linear', 'min' : data[f].quantile(0.1), 'max' : data[f].quantile(0.9), 'steps' : 10}

    gridsearcher = GridSearchOptimizer(decision_variables = decision_variables, objective_function = evaluator)
    s = gridsearcher.optimize()

    return s

def testing_bayesian(cols, data):
    fixed_data = data.drop(columns = cols + ['target']).sample(n = 1)
    evaluator = ModelEvaluate(model = model, fixed_data = fixed_data)

    decision_variables = {}
    for f in cols: 
        decision_variables[f] = {'method' : 'linear', 'min' : data[f].quantile(0.1), 'max' : data[f].quantile(0.9)}

    bayes_searcher = BayesianOptimizer(decision_variables = decision_variables, objective_function = evaluator, max_time_min = 5, max_it = 1000)
    s = bayes_searcher.optimize()

    return s

if __name__ == "__main__":
    print("Running optimization")

    CATALOG_PATH = Path("./app/catalog.yml") #str(Path("../../catalog.yml"))
    with open(CATALOG_PATH, 'r') as file:
        catalog_conf = yaml.safe_load(file)
    catalog = DataCatalog.from_config(catalog_conf)

    data = catalog.load("conc_cd_full_data")
    model = catalog.load("ebm_conc_cd")

    cols = ['VAZAO_AR_ROUGHER_COND', "Espumante (g/t)_CD", "ESPUMA_ROUGHER_COND"]

    print("Gridsearch testing...")
    print(testing_gridsearch(cols, data))
    print()

    print("Bayesian Opt testing...")
    print(testing_bayesian(cols, data))
    print()