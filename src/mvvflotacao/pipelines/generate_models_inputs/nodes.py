import pandas as pd
import numpy as np
import logging

import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings('ignore', category=PerformanceWarning)

from ...shared_code.utils import get_metadata_df_from_dict
from ..data_processing.nodes import remove_index_before_and_after


logger = logging.getLogger(__name__)


def select_features_from_step_columns(step_columns: list, model_name: str) -> list:
    """
    Select relevant feature columns from a given list based on exclusions and the target column for a specific model.

    Parameters:
    - step_columns (list): A list of columns names from which features will be selected.
    - model_name (str): The name of the model which determines the target column.

    Returns:
    - list: A list containing selected feature column names excluding specific patterns and including the target column.
    """
    target_column = get_target_name_for_model(model_name)

    return [c for c in step_columns
            if c == target_column or not any(ex in c for ex in [
                "CF", "REJ", "CONC_FLOT_", 'CONC_ROUG', "REC", 'CUT', 'CU_TOT'
            ])
            ]


def get_target_name_for_model(model_name: str) -> str:
    """
    Retrieves the target column name for a given model name.

    Parameters:
    - model_name (str): The name of the model for which the target column is desired.

    Returns:
    - str: The target column name corresponding to the model name.
    """
    targets = {
        'rej_rougher': 'REJ_ROUG_CU_TOT',
        'rec_global': 'REC_FLOT_CU',
        'rej_sc': 'REJ_SC_CL_CU_TOT',
        'conc_cl2': 'CONC_CLEA_II_CU_TOT',
        'rej_cl1': 'REJ_CLEA_I_CU_TOT',
        'conc_global': 'FLOT_CF_TEOR_CUT',
        'conc_cd': 'CONC_ROUG_FC01_CUT',
        'rec_metal': 'REC_FLOT_CU',
        'conc_cu': 'CONC_FLOT_CU_TOT'
    }

    return targets[model_name]


def correct_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the given DataFrame by performing the following actions:
    - Sets the 'DATA' column as the index.
    - Removes columns with dtype 'datetime64[ns]'.
    - Converts columns with dtype 'Float64' to 'float64'.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be processed.

    Returns:
    - pd.DataFrame: The processed DataFrame.
    """

    # Set 'DATA' column as the index
    try:
        df = df.set_index('DATA')
    except ValueError as e:
        print(f"Error setting 'DATA' column as index: {e}")

   # Drop columns with dtype 'datetime64[ns]'
    datetime_cols = df.columns[df.dtypes == 'datetime64[ns]']
    df = df.drop(columns=datetime_cols)

    # Convert 'Float64' columns to 'float64'
    float64_cols = df.columns[df.dtypes == "Float64"]
    for col in float64_cols:
        df[col] = df[col].astype("float64")

    return df


def remove_autocorrelation(df: pd.DataFrame, steps: int = 3, start: int = 0) -> pd.DataFrame:
    """
    Removes autocorrelation from a DataFrame using a systematic sample approach.

    Parameters:
    - df (pd.DataFrame): The input data.
    - steps (int): The step size used for removing autocorrelation. Default is 3.
    - start (int): The first observation of the systematic sample. Default is 0.

    Returns:
    - pd.DataFrame: The data with reduced autocorrelation (systematically sampled).
    """
    # Calculate the length of the DataFrame
    len_df = len(df)

    # Create a systematic sample filter based on steps and start
    filter_logic = np.unique([((x // steps) * steps + start) for x in range(len_df)])
    filter_logic = filter_logic[filter_logic < len_df]

    # Return filtered data with correct data types
    return df.iloc[filter_logic]


def create_lag_features(model_data: pd.DataFrame, num_lags: int = 2) -> pd.DataFrame:
    """
    Generates lag and squared features for the provided DataFrame.

    Parameters:
    - model_data (pd.DataFrame): The input data.
    - num_lags (int): The number of lagged features to create. Default is 2.

    Returns:
    - pd.DataFrame: The data with added lag and squared features.
    """
    # Create a copy of the input data to avoid modifying the original DataFrame
    new_model_data = model_data.copy()

    # Create lag features for each column (excluding 'target')
    for col in new_model_data.columns:
        if col != 'target':
            for lag in range(1, num_lags + 1):
                new_model_data[f'{col}_lag{lag}'] = new_model_data[col].shift(lag)

    # Remove rows with NaN values created by lag operations
    new_model_data = new_model_data.dropna()

    # Create squared features for each non-lagged and non-ignored column
    for col in new_model_data.columns:
        # Avoid creating squared features for the lagged columns and the 'DATA' column
        if '_lag' not in col and col not in ['DATA', 'target']:
            new_model_data[f'{col}_sqr2'] = np.power(new_model_data[col], num_lags)

    return new_model_data


def create_difflag_feature(model_data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new DataFrame with difference features based on lagged columns.

    Parameters:
    - model_data (pd.DataFrame): The input data with lagged features.

    Returns:
    - pd.DataFrame: The data with added difference features.
    """
    # Create a copy of the input data and reset the index to avoid issues during merging
    new_model_data = model_data.copy().reset_index(drop=True)

    # Identify lagged columns from the DataFrame
    columns = new_model_data.columns
    columns_lag1 = [col for col in columns if ('_lag1' in col) and ('DATA' not in col)]
    columns_lag2 = [col for col in columns if ('_lag2' in col) and ('DATA' not in col)]

    # Create new column names for the difference features
    columns_diff = [col.replace("_lag1", "_diff_lag1lag2") for col in columns_lag1]

    # Calculate the differences between lagged columns
    diff = np.array(new_model_data[columns_lag1]) - \
        np.array((new_model_data[columns_lag2]))
    df_diff = pd.DataFrame(data=diff, columns=columns_diff)

    # Merge the difference features back into the original DataFrame
    new_model_data = pd.merge(new_model_data, df_diff,
                              right_index=True, left_index=True)

    return new_model_data


def remove_outliers(data: pd.DataFrame):
    """
    Remove outliers based on the target column of the input dataset using the IQR method.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset with a 'target' column containing numerical values.

    Returns
    -------
    pd.DataFrame
        Dataset with outliers removed based on the 'target' column.
    """
    # Calculate bounds for outliers
    Q1 = data['target'].quantile(0.25)
    Q3 = data['target'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers from the train data
    data = data[(data['target'] >= lower_bound) & (
        data['target'] <= upper_bound)]

    return data


def filter_columns_with_lag(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Filter columns in a pandas DataFrame that contain 'lag' in their column names or it is named "target."

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame to filter.

    Returns:
    - pd.DataFrame: A new DataFrame containing only the columns with 'lag' in their names and the column target
    """

    lag_columns = [col for col in dataframe.columns if (
        ('lag' in col) or (col == "target"))]

    return dataframe[lag_columns]


def clean_dataset(data: pd.DataFrame, params):
    """
    Clean the input dataset based on specified criteria.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    features : list[str]
        List of feature columns.
    target : str
        Target column.
    lag : bool
        Whether to add lagged features.
    diff_lag : bool
        Whether to add lagged difference.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    # Select relevant columns for the model and rename the target
    new_data = data[[*params['features'], params['target']]].copy()
    new_data = new_data.rename({params['target']: 'target'}, axis=1)

    # Create new filters and features
    if params.get('filter_val_dardo_cd') == None or params['filter_val_dardo_cd']:
        if 'VAL_DARDO_CD' in params['features']:
            new_data = new_data[new_data['VAL_DARDO_CD'] < 50]

    if 'model_name' in params.keys() and params['model_name'] == "REC_FLOT_CU":
        new_data['RAZAO_CU_SOL_TOT'] = new_data['ALIM_FLOT_CU_SOL'] / \
            new_data['ALIM_FLOT_CU_TOT']

    # Remove columns with 40% or more null values
    new_data = new_data.dropna(axis=1, thresh=int(len(new_data)*0.60))
    # Log a warning if 'target' is not present in the columns
    if 'target' not in new_data.columns:
        logger.warning(
            'O target foi removido da base do conc_cd devido a quantidade de registros nulos!')

    # Create lag features if specified for the model
    if params['lag']:
        new_data = create_lag_features(new_data)

    # Create diffERENCE lag features if specified for the model
    if params['diff_lag']:
        new_data = create_difflag_feature(new_data)

    if params.get('remove_0_values') == None or params['remove_0_values']:
        indexes_to_remove = new_data[new_data['target'] == 0].index.to_list()
        new_data = remove_index_before_and_after(new_data, indexes_to_remove, n=3)

    if params.get('dropna') == None or params['dropna']:
        # Remove rows with null values after feature engineering
        new_data = new_data.dropna()

    return correct_type(new_data)


def generate_dataset_by_params(train_random: pd.DataFrame,
                               test_random: pd.DataFrame,
                               train_datetime: pd.DataFrame,
                               test_datetime: pd.DataFrame,
                               params: dict) -> pd.DataFrame:
    """
    Generates a dataset specific to the model.

    Parameters:
    - merged_data (pd.DataFrame): The merged data containing relevant columns.
    - metadata (pd.DataFrame): Metadata DataFrame containing step details.
    - create_lag_features_in (list): List of model names where lag features should be created.
    - create_difflag_features_in (list): List of model names where difference features should be created.

    Returns:
    - pd.DataFrame: The generated dataset with selected features and additional lag/difference features.
    """

    # Features
    features = [f"{feature}_{measure}" for feature in params['features']
                for measure in params['measures']]
    params['features'] = ['DATA', *params['features'], *features]

    train_random = clean_dataset(train_random, params)
    test_random = clean_dataset(test_random, params)

    train_datetime = clean_dataset(train_datetime, params)
    test_datetime = clean_dataset(test_datetime, params)

    full_data = pd.concat([train_datetime, test_datetime])

    if params['remove_outliers']['train']:
        train_random = remove_outliers(train_random)
        train_datetime = remove_outliers(train_datetime)
    if params['remove_outliers']['test']:
        test_random = remove_outliers(test_random)
        test_datetime = remove_outliers(test_datetime)

    if params.get("dynamic_feature") != None and len(params.get("dynamic_feature")) > 0:
        ddates = [pd.to_datetime(x) for x in params['dynamic_feature']]

        train_random = build_dynamic_feature(train_random, ddates)
        test_random = build_dynamic_feature(test_random, ddates)
        train_datetime = build_dynamic_feature(train_datetime, ddates)
        test_datetime = build_dynamic_feature(test_datetime, ddates)
        full_data = build_dynamic_feature(full_data, ddates)

    return train_random, test_random, train_datetime, test_datetime, full_data


def build_dynamic_feature(data: pd.DataFrame, ddates: list) -> pd.DataFrame:

    ddates = sorted(ddates, reverse=True)
    data['DYNAMIC'] = [0] * len(data)

    for i, ddate in enumerate(ddates):
        data['DYNAMIC'] = np.where(data.index <= ddate, i, data['DYNAMIC'])

    return data


def generate_dataset_from_steps(train_random_default: pd.DataFrame,
                                test_random_default: pd.DataFrame,
                                train_datetime_default: pd.DataFrame,
                                test_datetime_default: pd.DataFrame,
                                params: dict,
                                metadata: dict,
                                model_name: str) -> pd.DataFrame:
    """
    Generates a dataset for a specific model based on the provided steps and metadata.

    Parameters:
    - merged_data (pd.DataFrame): The merged data containing relevant columns.
    - metadata (dict): Metadata dictionary containing step details.
    - create_lag_features_in (list): List of model names where lag features should be created.
    - create_difflag_features_in (list): List of model names where difference features should be created.
    - steps (list): The list of steps for the model.
    - model_name (str): The name of the model.

    Returns:
    - pd.DataFrame: The generated dataset with selected features and additional lag/difference features.
    """
    # Extract relevant columns based on metadata and steps
    metadata_df = get_metadata_df_from_dict(metadata=metadata)
    step_columns = metadata_df[metadata_df.step.isin(
        params['steps'])]['column'].drop_duplicates().to_list()
    params['features'] = [col for col in select_features_from_step_columns(
        step_columns, model_name) if col != params['target']]

    train_random, test_random, train_datetime, test_datetime, full_data = generate_dataset_by_params(
        train_random_default, test_random_default, train_datetime_default, test_datetime_default, params)

    return train_random, test_random, train_datetime, test_datetime, full_data


def generate_dataset_for_conc_cd(train_random_default: pd.DataFrame,
                                 test_random_default: pd.DataFrame,
                                 train_datetime_default: pd.DataFrame,
                                 test_datetime_default: pd.DataFrame,
                                 params: dict,
                                 metadata: dict) -> pd.DataFrame:

    train_random, test_random, train_datetime, test_datetime, full_data = generate_dataset_by_params(
        train_random_default, test_random_default, train_datetime_default, test_datetime_default, params)

    return train_random, test_random, train_datetime, test_datetime, full_data


def generate_dataset_for_rec_global(train_random_default: pd.DataFrame,
                                    test_random_default: pd.DataFrame,
                                    train_datetime_default: pd.DataFrame,
                                    test_datetime_default: pd.DataFrame,
                                    params: dict,
                                    metadata: dict) -> pd.DataFrame:

    train_random, test_random, train_datetime, test_datetime, full_data = generate_dataset_by_params(
        train_random_default, test_random_default, train_datetime_default, test_datetime_default, params)

    return train_random, test_random, train_datetime, test_datetime, full_data
