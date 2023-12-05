from libs.simulador.streamlit_integration import *
from interpret.glassbox import ExplainableBoostingRegressor
import pandas as pd
from typing import Dict, List, Tuple
import plotly.graph_objs as go

desired_order = [
    "Sulfetado_HG", "Sulfetado_LG", "Sulfetado_MG", "Sulfetado_SHG", 
    "FLOT_AL_MASSA", "P20_MOAGEM", "PSI_OVER_CICLO", "P99_MOAGEM", 
    "ALIM_FLOT_CU_TOT", "ALIM_FLOT_PER_SOL", "ALIM_FLOT_FE", "ALIM_FLOT_MG", 
    "ALIM_FLOT_NI", "PH_ROUGHER_COND", "VAL_DARDO_CD", "ESPUMA_ROUGHER_COND", 
    "VAZAO_AR_ROUGHER_COND", "Espumante (g/t)_CD", "CONC_ROUG_FC01_CUT(valor predito)"
]

def calculate_impacts(
    input_df: pd.DataFrame, 
    model: ExplainableBoostingRegressor,
    y: pd.Series
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the local impacts of each variable on the model prediction for the given input_df.

    Args:
        input_df (pd.DataFrame): Input data for which we want to compute the impacts.
        model (ExplainableBoostingRegressor): Trained EBM model.

    Returns:
        tuple: Prediction for the input_df and dictionary of local impacts.
    """

    # Calculate the base prediction using the model on input_df
    base_prediction = y.mean()
    local_explanation = model.explain_local(input_df)
    
    # Extract the local importances (impacts)
    local_importances_raw = dict(zip(local_explanation.data(0)["names"], local_explanation.data(0)["scores"]))

    # Handle derived variables like _lag and _sqr
    local_importances = {}
    for column, importance in local_importances_raw.items():
        # Handle lag variables
        if "_lag" in column:
            original_col = column.split("_lag")[0]
            local_importances[original_col] = local_importances.get(original_col, 0) + importance
        # Handle square variables
        elif "_sqr" in column:
            original_col = column.split("_sqr")[0]
            local_importances[original_col] = local_importances.get(original_col, 0) + importance
        # Handle other variables
        else:
            local_importances[column] = importance

    return base_prediction, local_importances

def prepare_data_for_chart(
    base_prediction: float, 
    selected_impacts: Dict[str, float],
    other_impacts_sum: float,
    final_prediction: float
) -> Tuple[List[str], List[float]]:
    """
    Prepare x labels and y values for the waterfall chart, including interactions with 'In:' prefix.

    Args:
        base_prediction (float): Base prediction.
        selected_impacts (dict): Impacts of selected variables and interactions.
        final_prediction (float): Final prediction after adding impacts.

    Returns:
        Tuple[List[str], List[float]]: x labels and y values for the chart.
    """
    ordered_impacts = {key: selected_impacts[key] for key in desired_order if key in selected_impacts}

    # Lista para rótulos e valores das interações
    x_labels_interactions = []
    y_values_interactions = []
    for key, value in selected_impacts.items():
        if "&" in key:
            x_labels_interactions.append("In: " + key)
            y_values_interactions.append(value)

    x_labels = ['Teor Médio (Histórico)'] + list(ordered_impacts.keys()) + x_labels_interactions + ['Impacto Conjunto das Outras Variáveis', 'Valor Predito']
    y_values = [base_prediction] + list(ordered_impacts.values()) + y_values_interactions + [other_impacts_sum, final_prediction]

    return x_labels, y_values

def create_waterfall_chart(x_labels: List[str], y_values: List[float], final_prediction: float) -> go.Figure:
    """
    Create a waterfall chart using x labels and y values.

    Args:
        x_labels (List[str]): x labels for the chart.
        y_values (List[float]): y values for the chart.
        final_prediction (float): Final prediction.

    Returns:
        go.Figure: Plotly figure object representing the waterfall chart.
    """
    return go.Figure(
        go.Waterfall(
            name="20",
            orientation="v",
            measure=["relative"] * (len(x_labels) - 1) + ["total"],
            x=x_labels,
            text=[f'{value:.2f}' for value in y_values],
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

