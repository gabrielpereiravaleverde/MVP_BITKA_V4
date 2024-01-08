from interpret.glassbox import ExplainableBoostingRegressor
import pandas as pd
from typing import Dict, List, Tuple, Protocol, runtime_checkable
import plotly.graph_objs as go
import sys
from src.mvvflotacao.shared_code.modeling import ConformalModel, Ensemble, Explaining
from typing import Dict, List, Any, Tuple

@runtime_checkable
class ModelProtocol(Protocol):
    def predict(self, X: pd.DataFrame) -> Any:
        """
        Make predictions using the model.

        Args:
            X (pd.DataFrame): Input data for making predictions.

        Returns:
            Any: Predicted values.
        """
        

    def explain_local(self, X: pd.DataFrame) -> Any:
        """
        Provide local explanations for model predictions.

        Args:
            X (pd.DataFrame): Input data for which explanations are to be generated.

        Returns:
            Any: Local explanations.
        """
        ...

def calculate_impacts(input_df: pd.DataFrame, model, y: pd.Series) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the local impacts of each variable on the model prediction for the given input_df.

    Args:
        input_df (pd.DataFrame): Input data for which we want to compute the impacts.
        model (ExplainableBoostingRegressor): Trained EBM model.

    Returns:
        tuple: Prediction for the input_df and dictionary of local impacts.        
    """

    teste = ConformalModel.MapieConformal(model.base_model, model.significance, model.params)
    teste.model = model
    teste.fitted = True

    base_prediction = y.mean()
    local_explanation = model.explain_local(input_df)
    local_importances = local_explanation
    return base_prediction, local_importances

def prepare_data_for_chart(base_prediction: float, selected_impacts: Dict[str, float], other_impacts_sum: float, final_prediction: float) -> Tuple[List[str], List[float]]:
    """
    Prepare x labels and y values f or the waterfall chart, including interactions with 'In:' prefix.

    Args:
        base_prediction (float): Base prediction.
        selected_impacts (dict): Impacts of selected variables and interactions.
        final_prediction (float): Final prediction after adding impacts.

    Returns:
        Tuple[List[str], List[float]]: x labels and y values for the chart.
    """
    x_labels = ['Teor Médio (Histórico)'] + list(selected_impacts.keys()) + ['Impacto Conjunto das Outras Variáveis', 'Teor de Cobre Ajustado']
    
    # Inicialize os valores para o eixo y com a previsão base
    y_values = [base_prediction]
    
    # Adicione os impactos selecionados aos valores y
    y_values.extend(selected_impacts.values())
    
    # Adicione o impacto de outras variáveis e a previsão final
    y_values.extend([other_impacts_sum, final_prediction])
    
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
    y_values = [series.iloc[0] if isinstance(series, pd.Series) else series for series in y_values]
    formatted_values = [f'{value:.2f}' if isinstance(value, (float, int)) else str(value) for value in y_values]

    fig = go.Figure(
        go.Waterfall(
            name="20",
            orientation="v",
            measure=["relative"] * (len(x_labels) - 1) + ["total"],
            x=x_labels,
            text=formatted_values,
            y=formatted_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    # Adding y-axis label
    fig.update_layout(yaxis_title="Teor de Cobre no Concentrado do CD")

    return fig