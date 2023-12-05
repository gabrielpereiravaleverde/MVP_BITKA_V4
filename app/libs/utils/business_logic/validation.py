import pandas as pd
from libs.utils.business_logic.data_processing import *

def validate_uploaded_file(uploaded_df: pd.DataFrame, template_df: pd.DataFrame):
    """
    Validate the structure of an uploaded DataFrame against a template DataFrame.

    This function compares the columns of an uploaded DataFrame with those of a
    template DataFrame. It checks for missing and extra columns in the uploaded
    DataFrame, as well as the order of the columns. It compiles a list of errors
    based on these checks.

    Parameters:
    uploaded_df (pandas.DataFrame): The DataFrame created from the user-uploaded file.
    template_df (pandas.DataFrame): The DataFrame that serves as a template for the expected structure.

    Returns:
    list: A list of error messages. Each message describes a specific discrepancy
          between the uploaded DataFrame and the template. If there are no discrepancies,
          the list is empty.

    Notes:
    - The function checks for three types of discrepancies: missing columns, extra columns,
      and incorrect column order.
    - The order of columns is only checked if there are no missing or extra columns.
    """
    errors = []
    missing_cols = [col for col in template_df.columns if col not in uploaded_df.columns]
    if missing_cols:
        errors.append(f"Faltam colunas no arquivo enviado: {', '.join(missing_cols)}")
    extra_cols = [col for col in uploaded_df.columns if col not in template_df.columns]
    if extra_cols:
        errors.append(f"Colunas extras detectadas no arquivo enviado: {', '.join(extra_cols)}")
    if not missing_cols and not extra_cols:
        if list(uploaded_df.columns) != list(template_df.columns):
            errors.append("A ordem das colunas no arquivo enviado não corresponde à ordem esperada.")
    return errors