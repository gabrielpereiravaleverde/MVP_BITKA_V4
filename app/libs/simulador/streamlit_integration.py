from typing import Dict, List, Any, Tuple
import plotly.graph_objs as go
from interpret.glassbox import ExplainableBoostingRegressor
import pandas as pd
import streamlit as st
from libs.simulador.business_logic import *

def update_chart_layout(fig: go.Figure, y_values: List[float]) -> None:
    """
    Update the layout of a waterfall chart.

    Args:
        fig (go.Figure): Plotly figure object.
        y_values (List[float]): y values for the chart to determine axis range.
    """
    
    # Ensure y_values is not empty
    if not y_values:
        raise ValueError("y_values list is empty")

    # Initialize the first value
    cumulative_sum = y_values[0]  # Assume the first value is the starting value

    y_values = [y.iloc[0] if isinstance(y, pd.Series) else y for y in y_values]

    # Calculate the cumulative sum
    cumulative_values = [y_values[0]]  # Start with the first value
    for y in y_values[1:-1]:  # Exclude the last value (predicted value)
        cumulative_values[-1] += y  # Add to the last value in the list
        cumulative_values.append(cumulative_values[-1])  # Append new cumulative sum

    # Determine the y-axis range based on the cumulative values
    min_y = min(cumulative_values)
    max_y = max(cumulative_values)

    # Update the figure layout
    fig.update_layout(
        title="", ## EDITAR TITULO ANTES DA BOOBLE BOX
        showlegend=True,
        width=1200,
        height=800
        )
    fig.update_yaxes(range=[min_y - 1, max_y + 2])

def generate_waterfall_chart(
    base_prediction: float, 
    selected_impacts: Dict[str, float], 
    other_impacts_sum: float,
    final_prediction: float
) -> go.Figure:
    x_labels, y_values = prepare_data_for_chart(base_prediction, selected_impacts, other_impacts_sum, final_prediction)
    fig = create_waterfall_chart(x_labels, y_values, final_prediction)
    update_chart_layout(fig, y_values)
    return fig


def get_selected_variables(selected_stage: str, etapas: Dict[str, List[Tuple[str, str]]]) -> List[str]:
    """
    Get the selected variables based on the chosen stage.

    Args:
        selected_stage (str): The stage selected for analysis.
        etapas (dict): Dictionary containing etapas and their variables with units.

    Returns:
        list: List of selected variable names.
    """
    if selected_stage == "Todas":
        return [variable for stage_key in etapas for variable_type, variable_tuples in etapas[stage_key].items() for variable, _ in variable_tuples]
    return [variable for variable_type, variable_tuples in etapas[selected_stage].items() for variable, _ in variable_tuples]

def update_visualization(selected_stage: str, input_df: pd.DataFrame, model: ExplainableBoostingRegressor, y: pd.Series, etapas: Dict[str, Any]):
    """
    Update and generate a visualization based on the selected stage, input data, model, and target series.

    This function calculates the individual impacts of selected variables on a model's prediction, 
    visualizes these impacts using a waterfall chart, and computes the combined impact of all other 
    non-selected variables.

    Args:
        selected_stage (str): The stage selected by the user, used to determine which variables to include.
        input_df (pd.DataFrame): The input data DataFrame for which the impacts are to be calculated.
        model (ExplainableBoostingRegressor): The model used to compute the predictions and impacts.
        y (pd.Series): The target series against which the model's predictions are compared.
        etapas (Dict[str, Any]): A dictionary defining the stages and their associated variables.

    Returns:
        A waterfall chart generated by the `generate_waterfall_chart` function, showing the impacts of 
        the selected variables, the combined impact of other variables, and the final prediction.

    The function first retrieves the variables relevant to the selected stage, calculates the base prediction 
    and the impacts of each variable on the model's prediction. It then extracts and sums the impacts of 
    the selected variables, and calculates the collective impact of the remaining variables. This data is 
    then used 
    to generate a waterfall chart that visually represents these impacts and the final prediction.
    """
    feature_names = model.explain_global().columns
    suffixes = ['_min', '_max', '_median', '_diff_min_max']

    def add_variables_with_suffix(variables, suffixes):
        extended_variables = set(variables)
        for var in variables:
            for suffix in suffixes:
                suffixed_var = f"{var}{suffix}"
                if suffixed_var in feature_names:
                    extended_variables.add(suffixed_var)
        return list(extended_variables)

    # Obter variáveis selecionadas e adicionar sufixos, se aplicável
    selected_variables = get_selected_variables(selected_stage, etapas)
    selected_variables = add_variables_with_suffix(selected_variables, suffixes)

    base_prediction, impacts_dict = calculate_impacts(input_df, model, y)

    # Impacts das variáveis selecionadas
    selected_impacts = {k: v for k, v in impacts_dict.items() if k in selected_variables}
    # A previsão final usando o modelo
    if isinstance(impacts_dict, pd.DataFrame):
        # Sum up the values of the first column if that's your intention
        impacts_sum = impacts_dict.iloc[0].sum()
    else:
        impacts_sum = impacts_dict.sum()
    sum_selected_impacts = sum(selected_impacts.values())
    # O impacto conjunto das variáveis não selecionadas é a diferença necessária
    # para que a soma dos impactos selecionados mais a base_prediction seja igual à final_prediction
    final_prediction = base_prediction + impacts_sum
    other_impacts_sum = final_prediction - (base_prediction + sum_selected_impacts)
    # other_variables = [var for var in impacts_dict.keys() if var not in selected_variables]

    # # Exibe os nomes das variáveis no Streamlit
    # st.write("Variáveis que compõem o Impacto Conjunto das Outras Variáveis:")
    # st.write(other_variables)
    if selected_stage == 'Todas':
        other_impacts_sum = 0
    # Adicionar o impacto conjunto ao dicionário de impactos selecionados para visualização
    # selected_impacts["Impacto Conjunto das Outras Variáveis"] = other_impacts_sum
    # Chama a função generate_waterfall_chart com todos os quatro argumentos necessários
    return generate_waterfall_chart(base_prediction, selected_impacts, other_impacts_sum, final_prediction)




def simulate(
    input_values_simulator: Dict[str, float], 
    selected_stage: str, 
    model: ExplainableBoostingRegressor, 
    y: pd.Series, 
    etapas: Dict[str, Any],
    session_state: st.session_state
) -> None:
    """
    Perform simulation based on user inputs and update the visualization.

    Args:
        input_values_simulator (Dict[str, float]): The input values collected from the user interface.
        selected_stage (str): The current stage selected by the user.
        model (ExplainableBoostingRegressor): The machine learning model used for prediction.
        y (pd.Series): The target variable used by the model.
        etapas (Dict[str, Any]): Dictionary containing configuration or information about the stages.
        session_state (st.session_state): Streamlit's session state object for storing session-specific variables.

    This function updates the Streamlit session state with the results of the simulation.
    """
    session_state.simulated = True
    input_df = pd.DataFrame([input_values_simulator])
    session_state.current_values = input_df
    transformed_df = process_input_data(input_df)
    predicted_value = model.predict(transformed_df)[0]
    session_state.predicted_value = predicted_value.round(2)
    session_state.last_fig = update_visualization(
        selected_stage, 
        transformed_df, 
        model, 
        y, 
        etapas
    )

def simulate(
    input_values_simulator: Dict[str, float], 
    selected_stage: str, 
    model: ExplainableBoostingRegressor, 
    y: pd.Series, 
    etapas: Dict[str, Any],
    session_state: st.session_state
) -> None:
    """
    Perform simulation based on user inputs and update the visualization.

    Args:
        input_values_simulator (Dict[str, float]): The input values collected from the user interface.
        selected_stage (str): The current stage selected by the user.
        model (ExplainableBoostingRegressor): The machine learning model used for prediction.
        y (pd.Series): The target variable used by the model.
        etapas (Dict[str, Any]): Dictionary containing configuration or information about the stages.
        session_state (st.session_state): Streamlit's session state object for storing session-specific variables.

    This function updates the Streamlit session state with the results of the simulation.
    """
    from libs.utils.utils import process_input_data
    input_df = process_input_data(input_values_simulator)

    session_state.simulated = True
    session_state.current_values = input_df
    predicted_value = model.predict(input_df)[0]
    session_state.predicted_value = predicted_value.round(2)
    session_state.last_fig = update_visualization(
        selected_stage, 
        input_df, 
        model, 
        y, 
        etapas
    )
