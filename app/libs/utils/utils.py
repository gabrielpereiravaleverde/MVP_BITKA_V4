import os
import yaml
import pandas as pd
from typing import Dict, List, Any, Tuple
import plotly.graph_objs as go
import base64
from kedro.io import DataCatalog
from libs.utils.business_logic.data_processing import *
from libs.utils.business_logic.validation import *
from libs.utils.business_logic.calculations import *
from libs.utils.streamlit_integration.widgets import *
from libs.utils.streamlit_integration.visualization import *

languages  = {
    "PT": {
        "button": "Buscar",
        "instructions": "Arraste a Planilha Aqui",
        "limits": "Use o Template de Preenchimento",
    },
}


stage_keys = list(etapas.keys())

def load_catalog_config() -> DataCatalog:
    """
    Load and return the data catalog configuration.

    This function reads the catalog configuration from a YAML file and initializes
    a DataCatalog instance based on this configuration.

    Returns:
        DataCatalog: An instance of DataCatalog loaded with the configuration.
    """

    # Define the path to the catalog configuration file
    current_file_path = os.path.abspath(__file__)
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    catalog_path = os.path.join(base_path, "catalog.yml")

    # Load the catalog configuration from the YAML file
    with open(catalog_path, 'r') as file:
        catalog_conf = yaml.safe_load(file)

    # Create and return the DataCatalog instance
    return DataCatalog.from_config(catalog_conf)
