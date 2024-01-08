
from typing import Dict, Tuple, List, Any, Union
import streamlit as st
import pandas as pd
import re
import time
from libs.otimizador.business_logic import *
import numpy as np

# Inicialização do estado da sessão
def initialize_state() -> None:
    """
    Initialize the session state variables for the Streamlit application.

    This function sets up the necessary variables in the Streamlit session state
    for tracking optimization results, user inputs, progress, and other relevant states.
    """
    if 'optimization_results' not in st.session_state:
        st.session_state['optimization_results'] = None
    if 'results_rendered' not in st.session_state:
        st.session_state['results_rendered'] = False
    if 'input_values_optimizer' not in st.session_state:
        st.session_state['input_values_optimizer'] = {}
    if 'progress' not in st.session_state:
        st.session_state['progress'] = 0
    if 'cu_teor' not in st.session_state:
        st.session_state['cu_teor'] = 0
    if 'best_values' not in st.session_state:
        st.session_state['best_values'] = {}
    if 'elapsed_time' not in st.session_state:
        st.session_state['elapsed_time'] = None
    if 'optimize_pressed' not in st.session_state:
        st.session_state['optimize_pressed'] = False
    if 'last_render' not in st.session_state:
        st.session_state.last_render = None
    if 'first_run' not in st.session_state:
        st.session_state.first_run = None
    if 'optimized' not in st.session_state:
        st.session_state.optimized = False

def display_name(variable_info: tuple) -> str:
    """
    Adjust the presentation of the variable name.

    Args:
        variable_info: Tuple containing variable name and unit.

    Returns:
        Adjusted variable name.
    """
    if not isinstance(variable_info, tuple) or len(variable_info) < 1:
        return ""

    variable_name = variable_info[0]  # Extrai o nome da variável da tupla

    for suffix in ["_FC-01", "_FC-02", "_FC-03", "_FC-05"]:
        variable_name = variable_name.replace(suffix, "")
    return variable_name

def should_add_variable(variable_info: tuple, decision_variables: list) -> bool:
    """
    Determine if a variable should be added based on its name.
    Expecting a tuple with (variable_name, unit).
    """
    if not isinstance(variable_info, tuple) or len(variable_info) < 1:
        return False

    variable_name = variable_info[0]  # Extract the variable name from the tuple

    # Condições de terminação
    ends_with_conditions = ("_lag1", "_lag2", "_sqr2")

    # Verificar se a variável está na lista de variáveis de decisão e não termina com os padrões específicos
    return bool(
        variable_name in decision_variables
        and not re.search(r'_lag\d+$', variable_name)
        and not variable_name.endswith(ends_with_conditions)
    )

def render_variable_input(variable: str, default_limits: Tuple[float, float],) -> Tuple[float, float]:
    """Render input fields for the variable and return the user's input."""
    format_string = "%.4f" if "DEPRESSOR" in variable else None
    min_value = st.number_input("Limite Inferior", key=f"min_{variable}", value=default_limits[0], format=format_string)
    max_value = st.number_input("Limite Superior", key=f"max_{variable}", value=default_limits[1], format=format_string)
    return min_value, max_value

def render_restriction_inputs(etapas: Dict[str, Dict[str, List[str]]], decision_variables:list, default_limits: Tuple[float, float], data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Render inputs for setting constraints on optimization variables and collect user-defined restrictions.
    
    The function identifies variables based on naming patterns (utilizing the `should_add_variable` helper function) and 
    then displays input fields to the user to set constraints. Variables ending in "_sqr2" have their constraints derived 
    from the square of the original variable's constraints.

    Args:
    - etapas (Dict[str, Dict[str, List[str]]]): A dictionary containing optimization stages and their corresponding variable groups.
    - default_limits (Tuple[float, float]): A tuple containing the default min and max constraints.
    - data (pd.DataFrame): DataFrame containing data for optimization (note: this argument is currently unused in the function).

    Returns:
    - Dict[str, Tuple[float, float]]: A dictionary where the keys are variable names and the values are tuples containing 
    user-defined min and max constraints for each variable.
    
    Note: 
    The function utilizes the helper functions `should_add_variable` and `render_variable_input` to streamline operations.
    """
    user_restrictions = {}
    unique_input_variables = set()
    for etapa, variable_groups in etapas.items():
        for variable_type, variable_tuples in variable_groups.items():
            for variable_tuple in variable_tuples:
                variable, _ = variable_tuple  # Extração do nome da variável da tupla

                if should_add_variable(variable_tuple, decision_variables):  # Passando a tupla para a função
                    unique_input_variables.add(variable)
                    with st.expander(f"Configurações para {display_name(variable_tuple)} - Etapa {etapa}"):  # Passando a tupla para a função
                        limits = default_limits
                        if variable in data.columns:
                            limits = (data[variable].quantile(0.1), data[variable].quantile(0.9))
                        user_restrictions[variable] = render_variable_input(variable, limits)

    for variable in unique_input_variables:
        if variable.endswith("_sqr2"):
            original_variable = variable[:-5]
            if original_variable in user_restrictions:
                min_value, max_value = user_restrictions[original_variable]
                user_restrictions[variable] = (min_value ** 2, max_value ** 2)

    return user_restrictions

def on_optimize_button_click(
    type_opt: str,
    model: Any, 
    model_predict: Any, 
    input_values: Union[Dict[str, Any], pd.DataFrame], 
    data: pd.DataFrame, 
    user_defined_restrictions: Dict[str, Any], 
    relevant_cols: List[str]) -> None:
    """
    Handle the click event of the 'Optimize' button in the Streamlit application.

    Args:
    model (Any): O modelo de machine learning usado para otimização.
    input_values (Union[Dict[str, Any], pd.DataFrame]): Valores de entrada usados para otimização.
    data (pd.DataFrame): Dados usados na otimização.
    user_defined_restrictions (Dict[str, Any]): Restrições definidas pelo usuário para a otimização.
    relevant_cols (List[str]): Colunas relevantes para o processo de otimização.
    progress_bar (st.Progress): Barra de progresso do Streamlit para feedback visual.

    This function triggers the optimization process, updates the progress bar,
    computes the best values for the given inputs, and updates the session state
    with the results of the optimization.
    """
    from libs.utils.utils import process_input_data
    input_values = process_input_data(input_values)
    
    st.session_state['optimized'] = True

    start_time = time.time()

    # Realiza a otimização
    best_values = optimize_all_columns(model_predict, input_values, data, user_defined_restrictions, relevant_cols, type_opt=type_opt)
    

    # Salva os resultados da otimização no st.session_state
    st.session_state['best_values'] = best_values
    st.session_state['input_values'] = input_values
    st.session_state['results_rendered'] = True  # Sinalizar que a otimização foi realizada

    # Round the values of variables, 4 decimal places for columns starting with "DEPRESSOR," 2 decimal places for others
    for column, value in best_values.items():
        if "DEPRESSOR" in column:
            best_values[column] = round(value, 4)
        else:
            best_values[column] = round(value, 2)
    
    # Combine the optimized values in best_values with other variables in input_values
    for column in input_values:
        if column not in best_values and not column.endswith('_sqr2'):
            # Garantir que apenas um valor único seja adicionado
            value = input_values[column]
            if isinstance(value, (pd.Series, np.ndarray, list)):
                # Se for uma série, array ou lista, pegue o primeiro elemento
                best_values[column] = value[0]
            else:
                # Se for um valor único, adicione diretamente
                best_values[column] = value

    # Convert the best_values dictionary to a DataFrame
    input_df = pd.DataFrame([best_values])

    # List the expected features by the model
    params = model_predict.get_params()

    # Extracting features from each estimator in the ensemble
    all_features = params.get('features', [])

    # Flattening the list of features and removing duplicates
    expected_features = set(feature for estimator_features in all_features for feature in estimator_features)

    # Converting the set back to a list, if you need it in list format
    expected_features = list(expected_features)

    # List the columns in input_df
    provided_features = input_df.columns

    # Find the missing columns in input_df
    missing_features = set(expected_features) - set(provided_features)
    for feature in missing_features:
        # For columns ending with "_sqr2"
        if "_sqr2" in feature:
            base_feature = feature.replace("_sqr2", "")
    # Predict using the model
    predictions = model.predict(input_df)

    # Get the predicted value for cu_teor
    cu_teor = []
    cu_teor.append(model_predict.predict(input_df)[0])
    cu_teor.append(model.predict(input_df)[0][1])
    cu_teor.append(model.predict(input_df)[0][2])
    # Round the estimated cu_teor value to 2 decimal places
    cu_teor = np.round(cu_teor, 2)

    elapsed_time = time.time() - start_time
    st.session_state.update({
        'cu_teor': cu_teor,
        'best_values': best_values,
        'elapsed_time': elapsed_time,
        'progress': 100,
        'results_displayed': True
    })    
    elapsed_time = time.time() - start_time
    st.session_state['elapsed_time'] = elapsed_time
    st.session_state['progress'] = 100

    st.session_state.optimization_results = best_values

def update_progress(progress: int) -> None:
    """
    Update the optimization progress in the Streamlit session state.

    Args:
    progress (int): The current progress value to be updated in the session state.
    """
    st.session_state['progress'] = progress
