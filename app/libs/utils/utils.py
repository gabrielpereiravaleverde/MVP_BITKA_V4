import os
import yaml
import pandas as pd
from typing import Dict, List, Any, Tuple
import plotly.graph_objs as go
import base64
from datetime import datetime, timedelta
from kedro.io import DataCatalog
from libs.utils.business_logic.data_processing import *
from libs.utils.business_logic.validation import *
from libs.utils.business_logic.calculations import *
from libs.utils.streamlit_integration.visualization import *
from kedro.framework.startup import bootstrap_project
from kedro.framework.context import KedroContext
from pathlib import Path
from src.mvvflotacao.pipelines.data_processing.nodes import agg_by_3h
from src.mvvflotacao.pipelines.generate_models_inputs.nodes import build_dynamic_feature

languages  = {
    "PT": {
        "button": "Buscar",
        "instructions": "Arraste a Planilha Aqui",
        "limits": "Use o Template de Preenchimento",
    },
}

def generate_variable_dict(model_variables, yaml_data, include_all=False):
    # Lista de sufixos possíveis
    suffixes = ['_min', '_max', '_median', '_diff_min_max']

    # Função para remover os sufixos e retornar o nome base da variável
    def base_name(var):
        for suffix in suffixes:
            if suffix in var:
                return var.split(suffix)[0]
        return var

    # Mapear nomes base de variáveis para todas as suas variações
    model_vars_variations = {}
    for var in model_variables:
        if '&' in var:
            continue
        base_var_name = base_name(var)
        if base_var_name == var:    
            model_vars_variations.setdefault(base_var_name, []).append(var)

    # Inicializar o dicionário de resultado
    result_dict = {}

    # Definir a chave do YAML a ser usada
    yaml_key = 'todas_variaveis' if include_all else 'variaveis_decisao'

    # Iterar pelas categorias e subcategorias no YAML
    for category, subcategories in yaml_data[yaml_key].items():
        for subcategory, variables in subcategories.items():
            matched_vars = []
            for var_info in variables:
                var_name, var_unit = var_info
                base_var_name = base_name(var_name)
                if base_var_name in model_vars_variations:
                    # Adicionar todas as variações da variável e sua unidade
                    for var_variation in model_vars_variations[base_var_name]:
                        matched_vars.append((var_variation, var_unit))
            if matched_vars:
                result_dict.setdefault(category, {}).setdefault(
                    subcategory, []).extend(matched_vars)

    return result_dict


def load_catalog_config() -> DataCatalog:
    """
    Load and return the data catalog configuration.

    This function reads the catalog configuration from a YAML file and initializes
    a DataCatalog instance based on this configuration.

    Returns:
        DataCatalog: An instance of DataCatalog loaded with the configuration.
    """

    # Define the path to the catalog configuration file
    # current_file_path = os.path.abspath(__file__)
    # base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    path_components = [os.getcwd(), 'conf', 'web_app',
                           'catalog.yml']
    catalog_path = Path(*path_components)

    # Load the catalog configuration from the YAML file
    with open(catalog_path, 'r') as file:
        catalog_conf = yaml.safe_load(file)

    # Create and return the DataCatalog instance
    return DataCatalog.from_config(catalog_conf)



from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def process_input_data(input_values: dict) -> Any:
    """
    Process input data using a Kedro pipeline node.

    Args:
        input_values: The input data to be processed.
        context: The Kedro context.

    Returns:
        Processed data ready for the model.
    """
    measures_func = ['min','max','median','diff_min_max']
    # Obter a hora atual e arredondá-la para a hora mais próxima
    datetimes = [pd.Timestamp(datetime(2050, 12, 31, hour)) for hour in range(21, 24)]


    rows = [input_values[key] for key in ["Selected", "Lag_1hr", "Lag_2hr"]]
    input_df = pd.DataFrame(rows, index=datetimes)

    result = agg_by_3h(input_df, measures=measures_func)
    result['DYNAMIC'] = 0

    return result
