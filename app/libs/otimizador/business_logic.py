from typing import Dict, Tuple, List, Any, Union
import pandas as pd
import libs.otimizador.optimizer as opt
import io

def predict(input_data: Any, model: Any) -> Any:
    """Predict the output using the global model variable based on the provided input data.

    Args:
        input_data: Input data for prediction.

    Returns:
        The prediction result. If the prediction is a list or ndarray, the first value is returned.
    """
    pred = model.predict(input_data)
    # Ensure we are returning a single value. Adapt according to your model's output.
    return pred[0] if isinstance(pred, (list, np.ndarray)) else pred

def optimize_all_columns(model: Any, 
                         input_values,
                         data: pd.DataFrame, 
                         user_restrictions: Dict[str, Tuple[float, float]], 
                         relevant_cols: List[str], 
                         max_evals: int = 100,
                         type_opt = 'GridSearchOptimizer') -> Dict[str, float]:
    """Optimize all columns based on the given model and data, with user-defined restrictions.

    Args:
        model: Model used for prediction.
        data: Data used for optimization.
        user_restrictions: Dictionary with restrictions for output variables.
        relevant_cols: List of columns to optimize.
        max_evals: Maximum number of evaluations.

    Returns:
        Dictionary containing the best value for the columns being optimized.
    """

    r = dict(zip(relevant_cols, range(len(relevant_cols))))

    fixed_data = pd.DataFrame(input_values, index=[0])
    evaluator = opt.ModelEvaluate(model=model, fixed_data=fixed_data)

    decision_variables = {}
    for f in relevant_cols: 
        f_min = user_restrictions[f][0] if f in user_restrictions.keys() else data[f].quantile(0.1)
        f_max = user_restrictions[f][1] if f in user_restrictions.keys() else data[f].quantile(0.9)
        decision_variables[f] = {'method': 'linear', 'min': f_min, 'max': f_max, 'steps': 300}
    # Create a GridSearchOptimizer instance and set the progress callback
    ItGridSearchOptimizer = opt.ItGridSearchOptimizer(decision_variables=decision_variables, objective_function=evaluator, feature_engineering = None, max_combination_size = 10e6)
    # Perform optimization
    s = ItGridSearchOptimizer.optimize()

    r = dict(zip(s.columns, s.round(4).values.ravel()))
    del r['fitness']
    return r

def to_excel(df: pd.DataFrame) -> bytes:
    """
    Convert a pandas DataFrame to Excel format and return the result as bytes.

    Args:
        df (pd.DataFrame): The DataFrame to be converted into an Excel file.

    Returns:
        bytes: The Excel file content as bytes.

    This function is useful for converting DataFrame data into a format that can be easily downloaded or transferred as an Excel file.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def to_excel_download_link(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to Excel format and create a download link for the resulting file.

    Args:
        df (pd.DataFrame): The DataFrame to be converted into an Excel file.

    Returns:
        str: A string representing a download link for the Excel file. The link is a data URI containing the base64-encoded Excel data.

    This function combines the conversion of a DataFrame to Excel format and the creation of a downloadable link for the resulting file. It is particularly useful in web applications for providing a direct download option for data.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(excel_bytes: bytes) -> str:
    """
    Generate a downloadable link for an Excel file represented by the given bytes.

    Args:
        excel_bytes (bytes): The Excel file data in byte format.

    Returns:
        str: An HTML hyperlink element as a string that allows the Excel file to be downloaded. 
             The link points to a data URI containing the base64-encoded Excel data.

    This function is typically used in web applications (like Streamlit) to provide a user interface element for downloading Excel data.
    """
    b64 = base64.b64encode(excel_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="optimized_data.xlsx">Download Excel file</a>'
    return href