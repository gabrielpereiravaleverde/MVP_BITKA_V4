import pandas as pd
import numpy as np
import logging
import pickle
import gc
import os

from interpret.glassbox import ExplainableBoostingRegressor
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from sklearn.metrics import mean_squared_error
from pathlib import Path
from math import sqrt

# Import custom modules
from ...shared_code.utils import split_X_y, train_ebm_model, train_model

# Initialize the logger
logger = logging.getLogger(__name__)


def split_data(full_data: pd.DataFrame) -> pd.DataFrame:
    """
    Split the data into training and testing sets. The split is done by sorting by 'DATA' and 
    taking the first 95% for training and the rest for testing.

    Parameters:
    - full_data (pd.DataFrame): The full dataset.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: The training data and the testing data.
    """
    full_data = full_data.sort_index()

    train_lenght = int(len(full_data)*0.85)
    train_data = full_data.head(train_lenght)
    test_data = full_data.tail(len(full_data) - train_lenght)

    return train_data, test_data


def calc_rmse(test: pd.DataFrame, model: ExplainableBoostingRegressor) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) of a model's predictions on the test data.

    Parameters:
    - test (pd.DataFrame): The test data.
    - model: The trained model.

    Returns:
    - float: The RMSE value.
    """

    # Predict using the final model on the original feature set

    X_test, y_test = split_X_y(test)
    y_pred = model.predict(X_test)

    return np.round(sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test)), 4)


def result_model(train: pd.DataFrame, test: pd.DataFrame, hyperparameters: dict, params_build_model):
    """
    Train an EBM model and calculate RMSE on test data using the given hyperparameters.

    Parameters:
    - train (pd.DataFrame): The training data.
    - test (pd.DataFrame): The test data.
    - hyperparameters (dict): Model hyperparameters.

    Returns:
    - float: RMSE value.
    """

    model = train_model(train, train, params = params_build_model, model_name = None, hyperparameters=hyperparameters)['model']

    return calc_rmse(test, model)


def optimize_ebm_model(systematic_sample_1,
                       systematic_sample_2,
                       systematic_sample_3,
                       model_name, params_opt, params_build_model):

    # Define hyperparameter choices
    hyperparameter_choices = {
        'inner_bags': [10, 20, 30, 40, 50],
        'outer_bags': [10, 20, 30, 40, 50],
        'max_bins': [32, 64, 128, 256, 512, 1024],
        'max_leaves': [2, 3, 4, 5],
        'smoothing_rounds': list(range(100, 2001, 100))
    }
    
    # Split data for each systematic sample
    train_1, test_1 = split_data(systematic_sample_1)
    train_2, test_2 = split_data(systematic_sample_2)
    train_3, test_3 = split_data(systematic_sample_3)

    # Define an objective function for hyperparameter optimization
    def objective_function(space):

        logger.info(f'\n Errors using {space} \n')

        result1 = result_model(train_1, test_1, hyperparameters=space, params_build_model = params_build_model)
        result2 = result_model(train_2, test_2, hyperparameters=space, params_build_model = params_build_model)
        result3 = result_model(train_3, test_3, hyperparameters=space, params_build_model = params_build_model)

        result_list = [result1, result2, result3]

        mean_error = np.round(np.mean(result_list), 2)
        std_error = np.round(np.std(result_list), 2)

        logger.info(
            f'Errors {result_list}  -   Mean Error: {mean_error} Standard Deviation: {std_error}')

        return {'loss': mean_error, 'status': STATUS_OK}

    # Define the hyperparameter search space using choices
    space = {
        'inner_bags': hp.choice('inner_bags', hyperparameter_choices['inner_bags']),
        'outer_bags': hp.choice('outer_bags', hyperparameter_choices['outer_bags']),
        'max_bins': hp.choice('max_bins', hyperparameter_choices['max_bins']),
        'max_leaves': hp.choice('max_leaves', hyperparameter_choices['max_leaves']),
        'smoothing_rounds': hp.choice('smoothing_rounds', hyperparameter_choices['smoothing_rounds'])
    }

    i = 1
    max_i = round(params_opt['max_iter']/params_opt['save_iter'])

    path_components = [os.getcwd(), 'data', '05_model_input',
                       model_name, 'hyperparameters_trials']
    file_path = Path(*path_components)

    if params_opt['load_trials']:
        try:
            with open(Path(file_path, 'trials.pickle'), 'rb') as file:
                trials = pickle.load(file)
            logger.info(
                f'Trial successfully loaded from the following location: {file_path}')
        except:
            trials = Trials()
            logger.warning((f'Unable to load trial from the specified location: {file_path}'
                            f'. Starting optimization from the beginning.'))
    else:
        trials = Trials()

    while i <= max_i:
        best = fmin(fn=objective_function, space=space, algo=tpe.suggest,
                    trials=trials, max_evals=i * params_opt['save_iter'])

        os.makedirs(file_path, exist_ok=True)
        with open(Path(file_path, 'trials.pickle'), 'wb') as file:
            pickle.dump(trials, file)
            logger.info(f'Optimization partially saved at: {file_path}')
        i += 1
        gc.collect()

    best_space = {param: hyperparameter_choices[param][best[param]] for param in space}

    logger.info(f"Best hyperparameters: {best_space}")

    return best_space, best_space