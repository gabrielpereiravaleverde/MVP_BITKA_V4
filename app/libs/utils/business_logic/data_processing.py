import pandas as pd
import io
import streamlit as st
from typing import Dict, List, Any, Tuple

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

