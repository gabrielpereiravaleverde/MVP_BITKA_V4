from typing import Dict, Callable, Any
import logging
import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from catboost import CatBoostRegressor
from mapie.regression import MapieRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from math import sqrt
from typing import Tuple, Dict
from typing import Any, Callable, Dict, Optional
from hyperopt.mongoexp import MongoTrials  # Required for parallel optimization
from ...shared_code.utils import load_hyperparameters
from ...shared_code.modeling import ConformalModel

logger = logging.getLogger(__name__)


def data_test_predict(test_data: pd.DataFrame, trained_model: object) -> pd.DataFrame:
    """
    Predicts the target values on test_data using the provided trained_model.

    Parameters:
    - test_data: DataFrame containing the features for which predictions are required.
    - trained_model: The trained ExplainableBoostingRegressor model.

    Returns:
    - DataFrame with original test_data and an additional column 'Prediction' containing the predictions.
    """
    X_test = test_data.drop(columns='target', errors='ignore')
    predictions = trained_model['model'].predict(X_test)
    test_data['Prediction'] = predictions

    return test_data


def data_full_predict(full_data: pd.DataFrame, trained_model: object) -> pd.DataFrame:
    """
    Predicts the target values on full_data using the provided trained_model.

    Parameters:
    - full_data: DataFrame containing the features for which predictions are required.
    - trained_model: The trained ExplainableBoostingRegressor model.

    Returns:
    - DataFrame with original full_data and an additional column 'Prediction' containing the predictions.
    """
    # Correct type
    full_data = full_data.set_index('DATA')
    full_data['DATA'] = np.array(range(len(full_data)))

    # Drop datetime64[ns] columns
    datetime_cols = full_data.columns[full_data.dtypes == 'datetime64[ns]']
    full_data = full_data.drop(columns=datetime_cols)

    # Convert Float64 columns to float64
    float64_cols = full_data.columns[full_data.dtypes == "Float64"]
    for col in float64_cols:
        full_data[col] = full_data[col].astype("float64")

    X = full_data.drop(columns='target', errors='ignore')
    predictions = trained_model['model'].predict(X)
    full_data['Prediction'] = predictions

    return full_data


def calculate_metrics(test_data: pd.DataFrame) -> dict[str, float]:
    """
    Calculate evaluation metrics for a test dataset.

    Parameters:
    -----------
    test_data : pd.DataFrame
        A DataFrame containing the 'target' and 'Prediction' columns.

    Returns:
    --------
    metrics : dict of str to float
        A dictionary containing the calculated metrics.

        - `date`: Current date and time as a string.
        - `rmse`: Root Mean Squared Error (RMSE) rounded to 4 decimal places.
        - `mae`: Mean Absolute Error (MAE) rounded to 4 decimal places.
        - `r2`: R-squared (R^2) score rounded to 4 decimal places.
    """

    test_data = test_data.dropna(subset=['target', 'Prediction'])

    return {
        'date': str(datetime.now()),
        'rmse': np.round(sqrt(mean_squared_error(test_data['target'], test_data['Prediction'])), 4),
        'mae': np.round(mean_absolute_error(test_data['target'], test_data['Prediction']), 4),
        'r2': np.round(mean_absolute_error(test_data['target'], test_data['Prediction']), 4)
    }


def conformal_inference(test_data: pd.DataFrame, trained_model, run_conformal_nodes: bool) -> pd.DataFrame:
    """
    Perform conformal inference on test_data using the trained_model.

    Parameters:
    - test_data (pd.DataFrame): The input test data.
    - trained_model (MapieRegressor): The trained conformal prediction model.
    - run_conformal_nodes (bool): Flag to determine whether to run conformal inference.

    Returns:
    - pd.DataFrame: The updated test data with conformal inference results.
    """
    if run_conformal_nodes:
        X_test = test_data.drop(columns=['target', 'Prediction'], errors='ignore')
        result = pd.DataFrame(trained_model['model'].predict(X_test), columns=[
                              'Prediction_conformal_inference', 'Min', 'Max'], index=X_test.index)
        test_data = pd.merge(test_data.reset_index(), result.reset_index(), on='DATA')

    return test_data


def create_valid_frame(train_data: pd.DataFrame, predicted_test_data: pd.DataFrame, run_conformal_nodes: bool) -> pd.DataFrame:
    """
    Generates a dataframe with predictions using the provided data.

    Parameters:
    - train_data: Training dataframe with the target.
    - predicted_test_data: Test dataframe with target and the 'predictions' column.

    Returns:
    - valid_df_merged: Dataframe containing both training and test data along with test predictions.
    """

    # Reset the index for the training data
    valid_df = train_data[['target']]

    # Subset of the dataframe
    if run_conformal_nodes:
        valid_df_test_period = predicted_test_data[[
            'DATA', 'target', 'Prediction', 'Prediction_conformal_inference', 'Min', 'Max']]
    else:
        valid_df_test_period = predicted_test_data[['target', 'Prediction']]

    return pd.merge(valid_df, valid_df_test_period, on=['DATA', 'target'], how='outer')
