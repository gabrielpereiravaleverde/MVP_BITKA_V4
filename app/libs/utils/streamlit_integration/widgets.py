import pandas as pd
import base64

import pandas as pd
import io
import streamlit as st
from typing import Dict, List, Any, Tuple

from libs.utils.business_logic import data_processing as bl_dp
from libs.utils import utils as ut


from statistics import median, mean

def calculate_mean(values):
    return mean(values)

def calculate_min(values):
    return min(values)

def calculate_max(values):
    return max(values)

def calculate_median(values):
    return median(values)

def calculate_diff_min_max(values):
    return calculate_max(values) - calculate_min(values)

def get_table_download_link(df: pd.DataFrame, tipo: str) -> str:
    """
    Generate a downloadable link for an Excel file created from a DataFrame.

    This function converts a given Pandas DataFrame into an Excel file using the
    `to_excel` function, encodes the file in base64 format, and then creates a
    downloadable HTML link for this file. The link can be used in web applications
    to allow users to download the DataFrame as an Excel file.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be converted into an Excel file.

    Returns:
    str: A string containing the HTML anchor tag (<a>) with the downloadable link
         for the Excel file. The file is named 'template.xlsx'.

    Note:
    The function relies on the `to_excel` function to convert the DataFrame to an
    Excel file. It also uses base64 encoding to generate the link.
    """

    val = bl_dp.to_excel(df)
    b64 = base64.b64encode(val).decode()
    if tipo == "Template":
        return f'<a href="data:application/octet-stream;base64,{b64}" download="template.xlsx">Baixar Template de Planilha para Preencher os Valores das Variáveis</a>'
    if tipo == "Entrada":
        return f'<a href="data:application/octet-stream;base64,{b64}" download="template.xlsx">Baixar Dados de Entrada</a>'
    
def get_excel_download_link(excel_file: io.BytesIO, file_name: str) -> str:
    """
    Gera um link de download para um arquivo Excel.
    """
    excel_file.seek(0)
    b64 = base64.b64encode(excel_file.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Clique aqui para baixar o arquivo: {file_name}</a>'
    return href

def create_excel_file(feature_names: list) -> io.BytesIO:
    """
    Cria um DataFrame de template e o converte para um arquivo Excel em um buffer de memória.
    """
    columns = ["Variáveis", "No Horário", "1 Hora atrás", "2 Horas atrás"]
    data = {col: [] for col in columns}
    data["Variáveis"] = feature_names
    for lag_col in columns[1:]:
        data[lag_col] = [""] * len(feature_names)
    
    template_df = pd.DataFrame(data)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False)
    output.seek(0)
    return output

def render_inputs_for_stage_and_variable_type(
    etapas: Dict[str, Dict[str, list]], 
    key: str, 
    default_values: Dict[str, Any]
) -> None:
    """
    Render input fields for different stages and variable types in a Streamlit app.

    This function takes a structured dictionary defining stages and their
    associated variables, a key for session state, and a dictionary of default
    values. It renders interactive input fields in the Streamlit app based on the
    provided structure, allowing users to input or modify values. The function
    also maintains these values in the Streamlit session state.

    Parameters:
    etapas (dict): A dictionary where keys represent stages and values are dictionaries
                   of variable types, each containing tuples of variable names and units.
    key (str): The key used to store and retrieve the input values from the Streamlit
               session state.
    default_values (dict): A dictionary of default values for the variables.

    Returns:
    dict: A dictionary of the latest values for each variable, as input by the user.

    Notes:
    - The function creates tabs for each stage and expanders for each variable type.
    - It skips over any non-tuple entries or tuples not of length 2 in the input data.
    - A specific variable 'CONC_ROUG_FC01_CUT' is explicitly skipped.
    - The function ensures that each variable value is treated as a float.
    - Changes to input values are immediately reflected in the Streamlit session state.
    """
    if key not in st.session_state:
        st.session_state[key] = default_values

    input_values = st.session_state[key]
    stage_tabs = st.tabs(list(etapas.keys()))

    for stage, variables_dict in etapas.items():
        with stage_tabs[list(etapas.keys()).index(stage)]:
            for variable_type, variable_tuples in variables_dict.items():
                with st.expander(variable_type):
                    for variable_info in variable_tuples:
                        if isinstance(variable_info, tuple) and len(variable_info) == 2:
                            variable, unit = variable_info
                        else:
                            continue

                        if variable == 'CONC_ROUG_FC01_CUT':
                            continue
                        if variable == 'DYNAMIC':
                            continue
                        # Mapping for user-friendly labels
                        user_friendly_labels = {
                            'Selected': 'No horário',
                            'Lag_1hr': '1 hora atrás',
                            'Lag_2hr': '2 horas atrás'
                        }

                        cols = st.columns(3)
                        for i, lag in enumerate(['Selected', 'Lag_1hr', 'Lag_2hr']):
                            label = f"{variable} ({user_friendly_labels[lag]}) {unit}" if unit else f"{variable} ({user_friendly_labels[lag]})"
                            widget_key = f"{variable}"
                            with cols[i]:
                                # Check and initialize nested dictionary if not exists
                                if lag not in input_values:
                                    input_values[lag] = {}

                                # Default value handling
                                default_value = default_values.get(widget_key, 0.0)
                                if widget_key in input_values[lag]:
                                    value = input_values[lag][widget_key]
                                else:
                                    value = default_value
                                if value is None:
                                    default_value = default_values.get(widget_key, 0.0)
                                    value = default_value if default_value is not None else 0.0

                                value = float(value)
                                new_value = st.number_input(label, value=value,format="%.9f") #,format="%.2f")
                                input_values[lag][widget_key] = new_value

    return input_values

def is_input_changed(current_input_values: Dict[str, Any], key: str) -> bool:
    """
    Check if the current input values differ from the stored values in the Streamlit session state.

    This function compares the current input values of a form or similar input structure
    with the values stored in the Streamlit session state under a given key. It is used
    to determine if the input values have changed since the last session state update.

    Parameters:
    current_input_values (dict): The current set of input values to compare. This is typically
                                 a dictionary where keys are input field identifiers and values
                                 are the current inputs from the user.
    key (str): The key used to access the stored input values in the Streamlit session state.

    Returns:
    bool: Returns True if the current input values are different from the stored values,
          otherwise returns False. If there are no stored values under the given key, it
          returns True.

    Notes:
    - The function is useful for triggering updates or actions in a Streamlit app when
      input values change.
    - It assumes that the current input values and the stored values are both dictionaries.
    """
    if key in st.session_state:
        stored_input_values = st.session_state[key]
        return current_input_values != stored_input_values
    return True
