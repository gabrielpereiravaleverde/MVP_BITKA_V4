import pandas as pd
import io
import streamlit as st
from typing import Dict, List, Any, Tuple
import yaml
import sys

def load_yaml_file(file_path: str = 'app/data/00_metadata/etapas.yaml') -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def extract_todas_variaveis(data: Dict[str, Any]) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    todas_variaveis = data.get('todas_variaveis', {})
    etapas: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    for stage, variables in todas_variaveis.items():
        etapas[stage] = {}
        for category, vars in variables.items():
            etapas[stage][category] = [(var[0], var[1]) for var in vars]
    return etapas

def to_excel(df: pd.DataFrame):
    """
    Convert a Pandas DataFrame into an Excel file and return it as a byte stream.

    This function takes a Pandas DataFrame, writes it to an Excel file using
    xlsxwriter as the engine, and then returns the resulting file as a byte stream.
    This is useful for returning Excel files in web applications or APIs.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be converted to an Excel file.

    Returns:
    bytes: A byte stream representing the Excel file.

    Note:
    The Excel file is created with a single sheet named 'Sheet1'.
    The DataFrame index is not included in the output Excel file.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
    processed_data = output.getvalue()
    return processed_data

def import_excel():
    """
    Upload and import data from an Excel or CSV file using Streamlit's file uploader.

    This function provides a file uploader interface in a Streamlit app for the user
    to upload either a CSV or an Excel file. Once a file is uploaded, it reads the
    file into a Pandas DataFrame and returns it.

    Returns:
    pandas.DataFrame or None: A DataFrame containing the data from the uploaded file.
                              Returns None if no file is uploaded or if the file is not
                              in a supported format.

    Notes:
    - The function supports files with '.csv' and '.xlsx' extensions.
    - It uses `pd.read_csv` to read CSV files and `pd.read_excel` for Excel files.
    - If no file is uploaded, the function returns None.
    """
    uploaded_file = st.file_uploader("Adicione a planilha com os dados", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    
    return None

def filter_feature_names(feature_names, suffixes = ['_min', '_max', '_median', '_diff_min_max'], exclude_dynamic=True, exclude_ampersand=True):
    """
    Filtra os nomes das características removendo sufixos específicos, duplicatas,
    e opcionalmente excluindo variáveis específicas.

    Args:
    feature_names (list): Lista de nomes de características.
    suffixes (list): Lista de sufixos a serem removidos.
    exclude_dynamic (bool): Se True, exclui a variável "DYNAMIC".
    exclude_ampersand (bool): Se True, exclui variáveis que contêm "&".

    Returns:
    list: Lista de nomes de características filtrados.
    """
    def remove_suffix(name):
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name

    # Remova os sufixos
    filtered_names = [remove_suffix(name) for name in feature_names]

    # Remova duplicatas
    filtered_names = list(dict.fromkeys(filtered_names))

    # Exclua a variável "DYNAMIC", se necessário
    if exclude_dynamic:
        filtered_names = [name for name in filtered_names if name != "DYNAMIC"]

    # Exclua variáveis que contêm "&", se necessário
    if exclude_ampersand:
        filtered_names = [name for name in filtered_names if "&" not in name]

    return filtered_names
