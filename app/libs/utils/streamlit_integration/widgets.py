import pandas as pd
import base64
from libs.utils.business_logic.data_processing import *

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

    val = to_excel(df)
    b64 = base64.b64encode(val).decode()
    if tipo == "Template":
        return f'<a href="data:application/octet-stream;base64,{b64}" download="template.xlsx">Baixar Template de Planilha para Preencher os Valores das Variáveis</a>'
    if tipo == "Entrada":
        return f'<a href="data:application/octet-stream;base64,{b64}" download="template.xlsx">Baixar Dados de Entrada</a>'
    
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
                            continue  # Se não for uma tupla válida, pula esta variável

                        # Pula a variável específica se necessário
                        if variable == 'CONC_ROUG_FC01_CUT':
                            continue

                        # Formatar o rótulo com ou sem unidade
                        label = f"{variable} ({unit})" if unit else variable

                        # Acessar o valor atual e garantir que é um float
                        value = float(input_values.get(variable, default_values.get(variable, 0.0)))

                        # Criar um input para a variável com ou sem unidade
                        new_value = st.number_input(label, value=value, format="%.2f")
                        
                        # Atualizar o valor no estado da sessão
                        input_values[variable] = new_value

    st.session_state[key] = input_values
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
