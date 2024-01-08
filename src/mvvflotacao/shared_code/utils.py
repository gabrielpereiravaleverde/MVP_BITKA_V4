import pandas as pd
import logging
import json
import os
import yaml
import git

from kedro.io import DataCatalog
from pathlib import Path
from interpret.glassbox import ExplainableBoostingRegressor
from catboost import CatBoostRegressor
from ..shared_code.modeling import Ensemble, ConformalModel

log = logging.getLogger(__name__)


def get_metadata_df_from_dict(metadata: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(metadata, orient='index').stack().to_frame()
    df = df.explode(0).reset_index()
    df.columns = ['step', 'table', 'column']
    return df


def split_X_y(train_data: pd.DataFrame) -> pd.DataFrame:
    """
    Split data in X (features) and y (target).

    Parameters:
    - train_data (pd.DataFrame): The training data. Must include a 'target' column.

    Returns:
    - X: feature dataframe
    - y: target dataframe
    """

    X = train_data[[c for c in train_data.columns if c != 'target']].copy()

    y = train_data['target'].copy()

    return X, y


def generate_pickle_dictionary(model,
                               algorithm: str,
                               name: str = None):

    path_components = [os.getcwd(), 'conf', 'base',
                       'parameters_generate_models_inputs.yml']

    file_path = Path(*path_components)
    with open(file_path) as f:
        params = yaml.safe_load(f)

    try:
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha

        results = {
            'algorithm': algorithm,
            'model': model,
            'params': params[f'generate_dataset_for_{name}'] if name is not None else '',
            'commit_hash':  commit_hash
        }
    except:
        results = {
            'algorithm': algorithm,
            'model': model,
            'params': params[f'generate_dataset_for_{name}'] if name is not None else '',
        } 

    return results


def load_hyperparameters(model, params):

    if (params is not None and 'latest' in params.keys() and params['latest'] == True):
        path_components = [os.getcwd(), 'data', '05_model_input',
                           model, 'latest_hyperparameters', 'hyperparameter.json']

        file_path = Path(*path_components)
        try:
            with open(file_path, 'r') as file:
                hyperparameters = json.load(file)
            success_message = (f"The latest hyperparameters for the model {model}"
                               f" at the path {file_path} were loaded successfully.")
            log.info(success_message)
        except:
            log.warn(
                (f"You selected 'latest: True' in the 'parameters_data_science.yml' file"
                 f", but the file was not found at: {file_path}")
            )
            hyperparameters = None
    elif (params is not None and 'hyperparameters_path' in params.keys()):
        path_components = [os.getcwd(), 'data', '05_model_input', model,
                           *params['hyperparameters_path'].split('/')]
        file_path = Path(*path_components)
        try:
            with open(file_path, 'r') as file:
                hyperparameters = json.load(file)
            success_message = (f"The hyperparameters selected for the model {model}"
                               f" at the path {file_path} were loaded successfully.")
            log.info(success_message)
        except:
            log.warn(
                (f"You passed a path for the hyperparameters in the"
                 f" 'parameters_data_science.yml' file, but the file was not found"
                 f" at: {file_path}.")
            )
            hyperparameters = None
    else:
        hyperparameters = None
        log.warn(
            "In the 'train_ebm_model' section of the parameters_data_science.yml file:"
            "\n - To choose specific hyperparameters, add 'hyperparameters_path'"
            " and pass the relative path inside '05_model_input'. "
            "\n - To use the latest hyperparameters, set 'latest: True'.")

    return hyperparameters


def train_ebm_model(train_data: pd.DataFrame,
                    train_data_s0: pd.DataFrame,
                    train_mode: str = "randomly", # by_date
                    hyperparameters: dict = None,
                    params: str = None,
                    model_name: str = None) -> ExplainableBoostingRegressor:
    """
    Train an Explainable Boosting Regressor (EBM) model on the provided data.

    Parameters:
    - train_data (pd.DataFrame): The training data. Must include a 'target' column.

    Returns:
    - ExplainableBoostingRegressor: The trained EBM model.
    """

    X, y = split_X_y(train_data_s0)

    if hyperparameters is None:
        hyperparameters = load_hyperparameters(model_name, params)

    if hyperparameters is None:
        ebm = ExplainableBoostingRegressor()
        log.warn(f"Using default hyperparameters for model '{model_name}'.")
    else:
        ebm = ExplainableBoostingRegressor(inner_bags=hyperparameters['inner_bags'],
                                           outer_bags=hyperparameters['outer_bags'],
                                           max_bins=hyperparameters['max_bins'],
                                           max_leaves=hyperparameters['max_leaves'],
                                           smoothing_rounds=hyperparameters['smoothing_rounds'],
                                           n_jobs=4)
    ebm.fit(X, y)

    pickle = generate_pickle_dictionary(ebm, 'EBM', model_name)

    return pickle


def conformal_model(
        model: ExplainableBoostingRegressor,
        train_data: pd.DataFrame,
        train_data_s0: pd.DataFrame,
        params: dict, 
        run_conformal_nodes:bool,
        train_mode = "randomly", # by_date
		name=None) -> ConformalModel.BaseConformalModel:

    if params.get(f"{train_mode}_dataset_s0") == None or params.get(f"{train_mode}_dataset_s0"):
        X, y = split_X_y(train_data_s0)
    else:
        X, y = split_X_y(train_data)

    model = ConformalModel.MapieConformal(base_model=model['model'],
                                          significance=0.05,
                                          params=dict(method="plus", cv=10))  # cv_plus
    if run_conformal_nodes:
        model.fit(X, y)

    pickle = generate_pickle_dictionary(model, 'MapieConformal', name)

    return pickle


supported_models = {
    'ebm': lambda hyperparameters: ExplainableBoostingRegressor() if hyperparameters is None else ExplainableBoostingRegressor(**dict({'n_jobs': 4}, **hyperparameters)),
    'catboost': lambda hyperparameters: CatBoostRegressor() if hyperparameters is None else CatBoostRegressor(**dict({'n_jobs': 4}, **hyperparameters)),
}


def build_model(model_name, params, hyperparameters_params, hyperparameters=None):
    if hyperparameters is None:
        current_hyperparameters = None
    else:
        current_hyperparameters = hyperparameters.copy()

    if params['type'] == 'ensemble':
        models = []
        features_ensemble = []
        weights = []
        for model_params in params['models']:
            models.append(build_model(model_name, model_params,
                          hyperparameters_params, hyperparameters))
            features_ensemble.append(model_params['features'])
            if weights is not None and "weight" in model_params.keys():
                weights.append(model_params['weight'])
            else:
                weights = None

        return Ensemble.EnsembleEstimator(estimators=models, features=features_ensemble, weights=weights)
    else:
        if params['type'] not in supported_models.keys():
            raise Exception(
                f"Model {params['type']} not currently supported. The models currently supported are {', '.join(supported_models.keys())}")
        if current_hyperparameters is None:
            current_hyperparameters = load_hyperparameters(
                model_name, hyperparameters_params)
        return supported_models[params['type']](current_hyperparameters)


def train_model(train_data: pd.DataFrame,
                train_data_s0: pd.DataFrame,
                params: dict, 
                train_mode: str = "randomly", # by_date
                model_name: str = None, 
                hyperparameters_params: dict = None, 
                hyperparameters = None):
    
    model = build_model(model_name, params, hyperparameters_params, hyperparameters)
    if params.get(f"{train_mode}_dataset_s0") == None or params.get(f"{train_mode}_dataset_s0"):
        X, y = split_X_y(train_data_s0)
    else:
        X, y = split_X_y(train_data)
    
    if params.get('features') != None and len(params['features']) > 0:
        model.fit(X[params['features']], y)
    else:
        model.fit(X, y)
    pickle = generate_pickle_dictionary(model, params['type'], model_name)
    return pickle


def get_params(name, path_components=[os.getcwd(), 'conf']):

    with open(Path(*(path_components + ['base', 'catalog.yml']))) as f:
        conf_catalog = yaml.safe_load(f)

    with open(Path(*(path_components + ['local', 'credentials.yml']))) as f:
        credentials = yaml.safe_load(f)

    catalog = DataCatalog.from_config(conf_catalog, credentials)

    params = catalog.load(name)

    return params
