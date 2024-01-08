import pandas as pd 
import numpy as np
import yaml
from pathlib import Path
import os
from optimization.optimizer import *
from optimization.simulation import *
import yaml
from kedro.io import DataCatalog
import pickle
from libs.utils.utils import *

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
    with open('app/data/00_metadata/etapas.yaml', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
    model_variables = model.explain_global().columns

    result_dict = generate_variable_dict(model_variables, yaml_data, include_all=False)
    decision_variables = []
    for category in result_dict.values():
        for subcategory in category.values():
            for var_info in subcategory:
                decision_variables.append(var_info[0])  # Adiciona apenas o nome da variável

    print("Gridsearch testing...")
    print(testing_gridsearch(decision_variables, data))
    print()

    print("Bayesian Opt testing...")
    print(testing_bayesian(decision_variables, data))
    print()